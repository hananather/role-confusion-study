"""I keep one inference model across frozen jobs while a live budget lease exists.

I never create, stop or delete a GPU allocation. A job STOP ends that job; a
service STOP or expired lease closes the model. Tool execution stays on the Mac.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import threading
import time
import traceback

from .hf_backend import HFBackend, arm_vector
from .worker import execute_request, write_json


IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}\Z")


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_lease(lease, expected_pod_id, now):
    """I consume the monitor's short lease without extending its financial window."""
    if (lease.get("schema_version") != 1 or lease.get("approved") is not True
            or lease.get("expected_pod_id") != expected_pod_id
            or not isinstance(lease.get("lease_id"), str) or not lease["lease_id"]):
        raise ValueError("The live lease must identify this approved GPU allocation")
    values = {}
    for name in ("issued_unix", "lease_expires_unix", "shutdown_at_unix", "budget_remaining_usd"):
        value = lease.get(name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("The lease needs finite timing and budget fields")
        values[name] = value
    issued, expires = values["issued_unix"], values["lease_expires_unix"]
    if not 0 < expires - issued <= 180.001 or issued > now + 60:
        raise ValueError("A budget lease may authorize at most 180 seconds at a time")
    if values["budget_remaining_usd"] <= 0 or now >= min(expires, values["shutdown_at_unix"]):
        raise ValueError("The budget lease has expired or reached its financial boundary")
    return dict(lease)


def read_lease(config, now=None):
    path = Path(config["lease_path"])
    if path.is_symlink() or path.stat().st_size > 16384:
        raise ValueError("Invalid lease file")
    return validate_lease(json.loads(path.read_bytes()), config["expected_pod_id"],
                          time.time() if now is None else now)


def validate_job(job, job_id, service_config, backend, results_root):
    """I vary only a frozen arm table and cohort; model assets remain fixed."""
    if (not IDENTIFIER.fullmatch(job_id) or job.get("job_id") != job_id
            or job.get("schema_version") != 1 or job.get("execution_approved") is not True):
        raise ValueError("The job must have a frozen, approved identity")
    out = results_root / job_id
    if out.resolve() == Path(service_config["out_dir"]).resolve() or job.get("out_dir") != str(out):
        raise ValueError("A job needs its own exact output directory")
    for field in ("directions_file", "probe_file"):
        if (job.get(field) != service_config[field]
                or job.get(field + "_sha256") != backend.metadata[field + "_sha256"]):
            raise ValueError("Persistent jobs cannot replace the loaded direction/probe assets")
    for field in ("model_id", "model_revision", "attn_implementation", "max_context_tokens"):
        if field in job and job[field] != service_config.get(field, backend.metadata.get(field)):
            raise ValueError("Persistent jobs cannot change the model or inference configuration")
    arms = job.get("arms")
    if not isinstance(arms, list) or not 1 <= len(arms) <= 32:
        raise ValueError("A job needs between one and 32 frozen arms")
    table = {}
    for arm in arms:
        arm_id = arm.get("arm_id")
        if (not isinstance(arm_id, str) or not IDENTIFIER.fullmatch(arm_id) or arm_id in table
                or arm.get("layer") != 11 or arm.get("mask_mode") != "tool"):
            raise ValueError("Each unique arm must retain block11 and the tool-content mask")
        arm_vector(arm, backend.directions)
        table[arm_id] = dict(arm)
    return out, table


def run(config, *, backend_factory=HFBackend):
    if config.get("execution_approved") is not True or config.get("persistent_budget_lease_required") is not True:
        raise ValueError("This service requires explicit persistent-mode approval and a live budget lease")
    out = Path(config["out_dir"])
    results_root = Path(config.get("job_results_root", "/workspace/results/agent-steering"))
    if backend_factory is HFBackend and results_root != Path("/workspace/results/agent-steering"):
        raise ValueError("Real jobs must use the fixed agent-steering result root")
    out.mkdir(parents=True, exist_ok=True)
    for folder in ("jobs", "job-receipts"):
        (out / folder).mkdir(exist_ok=True)
    predecessor = config.get("predecessor_pid")
    if predecessor is not None:
        if type(predecessor) is not int or predecessor <= 1:
            raise ValueError("Invalid predecessor PID")
        try:
            os.kill(predecessor, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("The previous worker is still present; I do not load a second model")
    read_lease(config)
    lock_fd = os.open(out / "service.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(lock_fd, str(os.getpid()).encode()); os.close(lock_fd)
    stop, heartbeat_stop = threading.Event(), threading.Event()
    state = {"stage": "loading", "pid": os.getpid(), "started_unix": time.time(),
             "progress_unix": time.time(), "jobs_finished": 0, "active_job": None,
             "request_id": None, "allocation_action": "none"}
    original_signals = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            original_signals[sig] = signal.signal(sig, lambda *_: stop.set())

    def service_should_stop():
        if stop.is_set() or (out / "STOP").exists():
            state["stop_reason"] = "explicit_service_stop"
            return True
        try:
            lease = read_lease(config)
            state.update(lease_id=lease["lease_id"], lease_expires_unix=lease["lease_expires_unix"],
                         shutdown_at_unix=lease["shutdown_at_unix"], budget_remaining_usd=lease["budget_remaining_usd"])
        except (OSError, ValueError, TypeError, KeyError) as error:
            state.update(stop_reason="lease_unavailable_or_expired", lease_error=str(error))
            return True
        return False

    def heartbeat():
        while not heartbeat_stop.is_set():
            write_json(out / "heartbeat.json", {**state, "unix_time": time.time()})
            heartbeat_stop.wait(15)

    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    backend = None
    seen_jobs = set()
    try:
        backend = backend_factory(config)
        if service_should_stop():
            return
        write_json(out / "SERVICE-READY.json", {"schema_version": 1, "status": "ready",
                   "unix_time": time.time(), "pid": os.getpid(), "config_sha256": canonical_sha(config),
                   "backend": backend.metadata, "lease_required": True, "allocation_action": "none"})
        state.update(stage="waiting_for_job", progress_unix=time.time())
        while not service_should_stop():
            pending = sorted(p for p in (out / "jobs").glob("*.json") if p.stem not in seen_jobs)
            if not pending:
                state["progress_unix"] = time.time()
                stop.wait(.25)
                continue
            for job_path in pending:
                if service_should_stop(): break
                job_id = job_path.stem
                if job_path.is_symlink() or job_path.stat().st_size > 1024 * 1024:
                    raise ValueError("Invalid frozen job registration file")
                raw = job_path.read_bytes()
                job = json.loads(raw)
                job_sha = hashlib.sha256(raw).hexdigest()
                job_out, arms = validate_job(job, job_id, config, backend, results_root)
                job_out.mkdir(parents=True, exist_ok=False)
                for folder in ("requests", "responses", "progress"):
                    (job_out / folder).mkdir()
                write_json(job_out / "job-config.json", job)
                backend.arms = arms
                backend._metadata["arms"] = list(arms.values())
                backend.should_stop = lambda: service_should_stop() or (job_out / "STOP").exists()
                state.update(stage="serving_job", active_job=job_id, completed=0, progress_unix=time.time())
                write_json(job_out / "READY.json", {"schema_version": 1, "status": "ready", "unix_time": time.time(),
                           "pid": os.getpid(), "config_sha256": canonical_sha(job), "job_file_sha256": job_sha,
                           "service_config_sha256": canonical_sha(config), "backend": backend.metadata})
                job_error = None
                while not service_should_stop() and not (job_out / "STOP").exists():
                    queued = sorted(p for p in (job_out / "requests").glob("*.json")
                                    if not (job_out / "responses" / p.name).exists())
                    if not queued:
                        state["progress_unix"] = time.time()
                        stop.wait(.25)
                        continue
                    for request_path in queued:
                        if backend.should_stop(): break
                        state["request_id"] = request_path.stem
                        result = execute_request(backend, request_path, job_out, state)
                        state.update(completed=state["completed"] + 1, request_id=None, progress_unix=time.time())
                        if result["status"] != "ok" and result["error"]["type"] != "ContextLimitError":
                            job_error = result["error"]
                            break
                    if job_error: break
                receipt = {"schema_version": 1, "job_id": job_id, "job_file_sha256": job_sha,
                           "completed_requests": state["completed"], "finished_unix": time.time(),
                           "model_closed": False, "model_reused": True, "allocation_action": "none",
                           "status": "error" if job_error else "job_stopped",
                           "stop_reason": state.get("stop_reason", "job_stop"), "error": job_error}
                write_json(job_out / ("FAILED.json" if job_error else "DONE.json"), receipt)
                write_json(job_out / "EXIT.json", receipt)
                write_json(out / "job-receipts" / f"{job_id}.json", receipt)
                seen_jobs.add(job_id)
                state.update(active_job=None, request_id=None, stage="waiting_for_job",
                             jobs_finished=state["jobs_finished"] + 1, progress_unix=time.time())
                if job_error:
                    # I do not reuse a CUDA context after an unclassified generation failure.
                    state["stop_reason"] = "generation_infrastructure_error"
                    stop.set()
                    break
    except BaseException as error:
        state.update(stage="failed", error=repr(error), traceback=traceback.format_exc())
        write_json(out / "FAILED.json", {**state, "unix_time": time.time()})
        raise
    finally:
        model_closed = backend is None
        if backend is not None:
            try:
                backend.close(); model_closed = True
            except Exception as error:
                state.update(stage="failed", close_error=repr(error))
        heartbeat_stop.set(); thread.join(timeout=2)
        if state["stage"] != "failed": state["stage"] = "closed"
        terminal = {**state, "unix_time": time.time(), "model_closed": model_closed, "allocation_action": "none"}
        write_json(out / "heartbeat.json", terminal)
        write_json(out / "SERVICE-EXIT.json", terminal)
        for sig, previous in original_signals.items(): signal.signal(sig, previous)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))


if __name__ == "__main__": main()
