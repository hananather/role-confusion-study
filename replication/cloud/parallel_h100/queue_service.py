"""I guard the separately authorized GPU-B queue while reusing the frozen worker.

I expose inference only. The Mac runner owns the engineering gate, episode loop,
receivers and item cutoff. Loading a model does not establish that gate passed.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from replication.cloud.parallel_h100 import prime_worker as prime

FORBIDDEN_POD = "nz1bypfsiv62sc"


def validate_config(config, config_path=None):
    if config.get("purpose") != "queue_service" or config.get("execution_approved") is not True:
        raise ValueError("I require explicit GPU-B queue authorization")
    if config.get("expected_pod_id") == FORBIDDEN_POD or config.get("protected_pod_id") != FORBIDDEN_POD:
        raise ValueError("The active original pod must remain protected")
    extra = {"worker_config", "queue_job_id", "source_root"}
    priming_fields = {key: value for key, value in config.items() if key not in extra}
    # I reuse identity, cache and model checks; I do not run the priming program.
    priming_fields["purpose"] = "prime_only"
    checked = prime.validate_config(priming_fields, config_path)
    worker = config.get("worker_config")
    if not isinstance(worker, dict):
        raise ValueError("I require a frozen persistent worker configuration")
    root = Path(config["isolated_root"])
    if config.get("source_root", str(root / "source")) != str(root / "source"):
        raise ValueError("I require this lane's frozen source root")
    if config.get("queue_job_id") != config["run_id"] + "-queue":
        raise ValueError("I require this run's exact owned queue identity")
    fixed = {"execution_approved": True, "persistent_budget_lease_required": True,
             "out_dir": str(root / "model-service"), "job_results_root": "/workspace/results/agent-steering",
             "expected_pod_id": config["expected_pod_id"], "lease_path": config["lease_path"]}
    for key, value in fixed.items():
        if worker.get(key) != value:
            raise ValueError("Invalid fixed worker field: " + key)
    if (root / "model-service").resolve() != root / "model-service":
        raise ValueError("My worker directory cannot follow symlinks")
    for key in ("model_id", "model_revision", "cache_dir", "hf_home", "kernel_cache_dir", "attn_implementation"):
        if worker.get(key) != checked[key]:
            raise ValueError("The queue must reuse the verified pinned model and caches")
    for key in ("directions_file", "probe_file"):
        path = Path(worker.get(key, ""))
        if not path.is_absolute() or not isinstance(worker.get(key + "_sha256"), str) or len(worker[key + "_sha256"]) != 64:
            raise ValueError("I require frozen, hashed direction and probe assets")
    return {**checked, "purpose": "queue_service", "worker_config": dict(worker),
            "queue_job_id": config["queue_job_id"], "source_root": str(root / "source")}


def model_child(config, config_path):
    config = validate_config(config, config_path)
    prime.read_lease(config)
    from replication.cloud.agent_steering import persistent_worker
    from replication.cloud.parallel_h100.tool_raising import ToolRaisingBackend
    original_validate = persistent_worker.validate_job

    def validate_owned_job(job, job_id, service_config, backend, results_root):
        if job_id != config["queue_job_id"]:
            raise ValueError("Only my one frozen GPU-B queue may use this service")
        return original_validate(job, job_id, service_config, backend, results_root)

    # This process-local identity guard leaves the frozen worker source unchanged.
    persistent_worker.validate_job = validate_owned_job
    try:
        persistent_worker.run(config["worker_config"], backend_factory=ToolRaisingBackend)
    finally:
        persistent_worker.validate_job = original_validate
    return 0


class Guardian:
    def __init__(self, config, config_path, *, spawn=subprocess.Popen, lease_reader=prime.read_lease,
                 clock=time.monotonic, wall=time.time, sleep=time.sleep):
        self.config = validate_config(config, config_path)
        self.config_path = Path(config_path)
        self.config_bytes = self.config_path.read_bytes()
        self.root = Path(config["isolated_root"])
        self.service = self.root / "service"
        self.spawn, self.lease_reader, self.clock, self.wall, self.sleep = spawn, lease_reader, clock, wall, sleep
        self.child = None
        self.stop_requested = False
        self.stop_reason = None
        self.forced_kill = False

    def stop_child(self):
        if self.child is None or self.child.poll() is not None:
            return
        self.child.terminate()
        try:
            self.child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.forced_kill = True
            self.child.kill()
            try:
                self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.stop_reason += ":child_reap_unverified"

    def loaded(self):
        path = self.root / "model-service/SERVICE-READY.json"
        if not path.exists():
            return False
        value = json.loads(path.read_bytes())
        return (value.get("status") == "ready" and value.get("lease_required") is True
                and value.get("config_sha256") == prime.canonical_sha(self.config["worker_config"]))

    def run(self):
        if self.service.resolve() != self.service:
            raise ValueError("My service directory cannot follow symlinks")
        self.service.mkdir(parents=True, exist_ok=True)
        with (self.service / "guardian.lock").open("x") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock.write(str(os.getpid())); lock.flush()
            self.lease_reader(self.config)
            if (self.service / "STOP").exists():
                raise RuntimeError("A saved STOP prevents loading")
            command = [sys.executable, "-B", "-s", str(Path(__file__).resolve()), "--config",
                       str(self.config_path), "--model-child"]
            parent, started = os.getpid(), self.clock()
            with (self.service / "model.log").open("x") as log:
                self.child = self.spawn(command, cwd=str(Path(__file__).resolve().parents[3]),
                                       env=prime.child_environment(self.config), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                       preexec_fn=lambda: prime.parent_death_guard(parent))
                prime.write_json(self.service / "GUARDIAN-READY.json", {"guardian_pid": os.getpid(),
                                 "worker_pid": self.child.pid, "expected_pod_id": self.config["expected_pod_id"],
                                 "run_id": self.config["run_id"], "unix_time": self.wall(),
                                 "config_sha256": prime.canonical_sha(self.config), "allocation_action": "none"})
                try:
                    while self.child.poll() is None:
                        if self.stop_requested or (self.service / "STOP").exists():
                            self.stop_reason = "explicit_stop"; break
                        if self.config_path.read_bytes() != self.config_bytes:
                            self.stop_reason = "frozen_config_changed"; break
                        try:
                            prime.identity_report(self.config)
                            lease = self.lease_reader(self.config)
                        except (OSError, ValueError, TypeError, KeyError):
                            self.stop_reason = "identity_or_lease_unavailable"; break
                        loaded = self.loaded()
                        if not loaded and self.clock() - started >= self.config["startup_timeout_seconds"]:
                            self.stop_reason = "startup_timeout"; break
                        if loaded and not (self.service / "READY.json").exists():
                            prime.write_json(self.service / "READY.json", {"schema_version": 1,
                                             "status": "queue_model_loaded", "expected_pod_id": self.config["expected_pod_id"],
                                             "run_id": self.config["run_id"], "ready_unix": self.wall(),
                                             "engineering_gate_passed": False, "engineering_gate_owner": "local frozen runner",
                                             "experiment_inbox": True, "allocation_action": "none"})
                        prime.write_json(self.service / "heartbeat.json", {"stage": "serving_queue" if loaded else "loading",
                                         "guardian_pid": os.getpid(), "worker_pid": self.child.pid,
                                         "expected_pod_id": self.config["expected_pod_id"], "run_id": self.config["run_id"],
                                         "unix_time": self.wall(), "lease_expires_unix": lease["lease_expires_unix"],
                                         "shutdown_at_unix": lease["shutdown_at_unix"], "allocation_action": "none"})
                        self.sleep(self.config["poll_seconds"])
                finally:
                    self.stop_reason = self.stop_reason or "model_child_exit"
                    self.stop_child()
                    receipt = {"stop_reason": self.stop_reason, "worker_returncode": self.child.poll(),
                               "worker_exit_verified": self.child.poll() is not None, "forced_kill": self.forced_kill,
                               "expected_pod_id": self.config["expected_pod_id"], "unix_time": self.wall(), "allocation_action": "none"}
                    prime.write_json(self.service / "EXIT.json", receipt)
        return 0 if self.stop_reason in ("explicit_stop", "identity_or_lease_unavailable") else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    if args.model_child:
        return model_child(config, args.config)
    guardian = Guardian(config, args.config)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: setattr(guardian, "stop_requested", True))
    return guardian.run()


if __name__ == "__main__":
    raise SystemExit(main())
