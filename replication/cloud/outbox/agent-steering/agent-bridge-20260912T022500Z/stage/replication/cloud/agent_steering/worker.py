"""I serve a bounded, frozen file queue for inference, never tool execution."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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


REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\Z")
MAX_REQUEST_BYTES = 8 * 1024 * 1024


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_request(path):
    if path.is_symlink() or not REQUEST_ID.fullmatch(path.stem) or path.stat().st_size > MAX_REQUEST_BYTES:
        raise ValueError("Invalid request filename, symlink or size")
    raw = path.read_bytes()
    request = json.loads(raw)
    if request.get("schema_version") != 1 or request.get("request_id") != path.stem:
        raise ValueError("Request identity must match the atomic queue filename")
    if not isinstance(request.get("prompt"), str) or not request["prompt"]:
        raise ValueError("A complete prompt string is required")
    return request, hashlib.sha256(raw).hexdigest()


def execute_request(backend, path, out, progress_state=None):
    """I return an auditable error for an invalid request without executing it."""
    request_sha = None
    def progress(update):
        now = time.time()
        if progress_state is not None:
            progress_state.update(progress_unix=now, phase=update.get("phase"),
                                  generated_tokens=update.get("generated_tokens", 0))
        write_json(out / "progress" / path.name, {"schema_version": 1, "request_id": path.stem,
                                                  "request_sha256": request_sha, "unix_time": now, **update})
    try:
        request, request_sha = read_request(path)
        generation, stats = backend.generate_steered(
            request["prompt"], request["char_spans"], arm_id=request["arm_id"], seed=request["seed"],
            max_new_tokens=request.get("max_new_tokens", 4096), temperature=request.get("temperature", 1.0),
            timeout_s=request.get("timeout_s", 300), purpose=request.get("purpose", "episode"),
            on_progress=progress)
        result = {"schema_version": 1, "request_id": path.stem, "request_sha256": request_sha,
                  "status": "ok", "generation": asdict(generation), "stats": stats}
    except Exception as error:
        result = {"schema_version": 1, "request_id": path.stem, "request_sha256": request_sha,
                  "status": "error", "error": {"type": type(error).__name__, "message": str(error),
                                               "traceback": traceback.format_exc()}}
    write_json(out / "responses" / path.name, result)
    return result


def run(config, *, backend_factory=None):
    if backend_factory is None and config.get("execution_approved") is not True:
        raise ValueError("The frozen worker config must record execution approval")
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name in ("requests", "responses", "progress"):
        (out / name).mkdir(exist_ok=True)
    # I forbid a second worker or restart from silently changing a queue's model state.
    lock_fd = os.open(out / "worker.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)
    started = time.time()
    seconds = float(config.get("worker_seconds", 7200))
    idle_seconds = float(config.get("idle_timeout_s", 600))
    if not math.isfinite(seconds) or not 0 < seconds <= 86400 or not math.isfinite(idle_seconds) or idle_seconds <= 0:
        raise ValueError("Worker and idle deadlines must be finite and positive")
    deadline = min(started + seconds, float(config.get("hard_deadline_unix", started + seconds)))
    if not math.isfinite(deadline) or deadline <= started:
        raise ValueError("The compute deadline has already passed")
    stop = threading.Event()
    heartbeat_stop = threading.Event()
    state = {"stage": "loading", "started_unix": started, "deadline_unix": deadline,
             "pid": os.getpid(), "completed": 0, "request_id": None, "progress_unix": started}
    previous_signals = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_signals[sig] = signal.signal(sig, lambda *_: stop.set())

    def should_stop():
        return stop.is_set() or time.time() >= deadline or (out / "STOP").exists()

    def heartbeat():
        while not heartbeat_stop.is_set():
            write_json(out / "heartbeat.json", {**state, "unix_time": time.time()})
            heartbeat_stop.wait(15)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    backend = None
    try:
        if backend_factory is None:
            from .hf_backend import HFBackend
            backend_factory = HFBackend
        backend = backend_factory(config)
        backend.should_stop = should_stop
        ready = {"schema_version": 1, "status": "ready", "unix_time": time.time(),
                 "pid": os.getpid(), "config_sha256": hashlib.sha256(
                     json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                 "backend": backend.metadata}
        write_json(out / "READY.json", ready)
        state.update(stage="waiting", progress_unix=time.time())
        idle_since = time.time()
        while not should_stop():
            pending = sorted(path for path in (out / "requests").glob("*.json")
                             if not (out / "responses" / path.name).exists())
            if not pending:
                state["progress_unix"] = time.time()
                if time.time() - idle_since > idle_seconds:
                    state["stage"] = "idle_timeout"
                    break
                stop.wait(.25)
                continue
            for path in pending:
                if should_stop():
                    break
                state.update(stage="generating", request_id=path.stem)
                result = execute_request(backend, path, out, state)
                state["completed"] += 1
                state.update(stage="waiting", request_id=None, progress_unix=time.time())
                idle_since = time.time()
                if result["status"] != "ok" and result["error"]["type"] != "ContextLimitError":
                    state["stage"] = "request_error"
                    stop.set()
                    break
        state["stage"] = state["stage"] if state["stage"] in ("request_error", "idle_timeout") else "stopped"
        state["finished_unix"] = time.time()
    except BaseException as error:
        state.update(stage="failed", error=repr(error))
        write_json(out / "FAILED.json", {**state, "unix_time": time.time(), "traceback": traceback.format_exc()})
        raise
    finally:
        model_closed = backend is None
        if backend is not None:
            try:
                backend.close()
                model_closed = True
            except Exception as error:
                state.update(stage="failed", close_error=repr(error))
                write_json(out / "FAILED.json", {**state, "unix_time": time.time(), "model_closed": False})
        heartbeat_stop.set()
        thread.join(timeout=2)
        write_json(out / "heartbeat.json", {**state, "unix_time": time.time(), "model_closed": model_closed})
        if state["stage"] != "failed":
            write_json(out / "DONE.json", {**state, "unix_time": time.time(), "model_closed": model_closed})
        write_json(out / "EXIT.json", {**state, "unix_time": time.time(), "model_closed": model_closed,
                                       "exit_status": "error" if state["stage"] in ("failed", "request_error") else "ok"})
        for sig, previous in previous_signals.items():
            signal.signal(sig, previous)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
