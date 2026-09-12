"""I enforce renewable leases on my own model-worker child, never on a GPU allocation."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from .persistent_worker import read_lease
from .worker import write_json

SOURCE_ROOT = Path(__file__).resolve().parents[3]


def validate_config(config):
    if (config.get("execution_approved") is not True
            or config.get("persistent_budget_lease_required") is not True):
        raise ValueError("I require explicit approval for the budget-leased model service")
    pod = config.get("expected_pod_id", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", pod):
        raise ValueError("I require the exact expected pod identity")
    observed = os.environ.get("RUNPOD_POD_ID")
    if observed and observed != pod:
        raise ValueError("The service configuration names a different pod")
    if config.get("lease_path") != "/workspace/session-control/" + pod + "/lease.json":
        raise ValueError("I require this pod's own live financial lease")
    predecessor = config.get("predecessor_pid")
    if type(predecessor) is not int or predecessor <= 1:
        raise ValueError("I require the actual preceding model-worker PID")
    out = config.get("out_dir", "")
    if not isinstance(out, str) or not out.startswith("/workspace/session-control/" + pod + "/") or ".." in Path(out).parts:
        raise ValueError("I require a separate service directory on this same pod")
    wait = float(config.get("predecessor_wait_seconds", 7200))
    if not math.isfinite(wait) or not 0 < wait <= 7200:
        raise ValueError("The predecessor wait must be finite and at most two hours")
    return config


def predecessor_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def parent_death_guard(expected_parent):
    """I prevent my child from surviving an abruptly killed Linux guardian."""
    if sys.platform != "linux":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        os._exit(126)
    if os.getppid() != expected_parent:
        os._exit(125)


class Guardian:
    def __init__(self, config, config_path, directory, *, lease_reader=read_lease,
                 spawn=subprocess.Popen, exists=predecessor_alive, monotonic=time.monotonic,
                 sleep=time.sleep):
        self.config = validate_config(config)
        self.config_path = Path(config_path).resolve()
        self.config_sha256 = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lease_reader, self.spawn, self.exists = lease_reader, spawn, exists
        self.monotonic, self.sleep = monotonic, sleep
        self.child = None
        self.stop_requested = False
        self.stop_reason = None
        self.escalated = False

    def lease_valid(self):
        try:
            self.lease_reader(self.config)
            return True
        except (OSError, ValueError, TypeError, KeyError):
            self.stop_reason = "lease_unavailable_or_expired"
            return False

    def stop_owned_child(self, reason):
        """I signal only the subprocess I created; the predecessor is never a target."""
        self.stop_reason = reason
        if self.child is None or self.child.poll() is not None:
            return
        self.child.terminate()
        deadline = self.monotonic() + 5
        while self.child.poll() is None and self.monotonic() < deadline:
            self.sleep(.1)
        if self.child.poll() is None:
            self.escalated = True
            self.child.kill()
        # I bound reap time too; an uninterruptible process remains reported, not mislabelled stopped.
        try:
            self.child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.stop_reason = reason + ":child_reap_unverified"

    def run(self):
        with (self.directory / "guardian.lock").open("x") as lock:
            lock.write(str(os.getpid()))
        if not self.lease_valid():
            raise RuntimeError("No valid financial lease before model startup")
        wait_deadline = self.monotonic() + float(self.config.get("predecessor_wait_seconds", 7200))
        while self.exists(self.config["predecessor_pid"]):
            if self.stop_requested or not self.lease_valid() or self.monotonic() >= wait_deadline:
                raise RuntimeError("The preceding worker did not exit within my valid lease/wait window")
            write_json(self.directory / "WAITING.json", {"guardian_pid": os.getpid(),
                       "predecessor_pid": self.config["predecessor_pid"], "allocation_action": "none"})
            self.sleep(1)
        if not self.lease_valid():
            raise RuntimeError("The financial lease expired before model startup")
        if hashlib.sha256(self.config_path.read_bytes()).hexdigest() != self.config_sha256:
            raise ValueError("The frozen service configuration changed while I waited")
        expected_parent = os.getpid()
        command = [sys.executable, "-u", "-m", "replication.cloud.agent_steering.persistent_worker",
                   "--config", str(self.config_path)]
        # My dedicated child session plus Linux parent-death guard establish ownership.
        self.child = self.spawn(command, cwd=str(SOURCE_ROOT), stdin=subprocess.DEVNULL,
                                start_new_session=True, preexec_fn=lambda: parent_death_guard(expected_parent))
        process = {"guardian_pid": os.getpid(), "worker_pid": self.child.pid,
                   "predecessor_pid": self.config["predecessor_pid"], "expected_pod_id": self.config["expected_pod_id"],
                   "command": command, "config_sha256": self.config_sha256,
                   "source_root": str(SOURCE_ROOT), "lease_enforced": True, "allocation_action": "none"}
        write_json(self.directory / "GUARDIAN-READY.json", process)
        try:
            while self.child.poll() is None:
                if self.stop_requested or (self.directory / "GUARDIAN-STOP").exists():
                    self.stop_owned_child("explicit_guardian_stop"); break
                if (Path(self.config["out_dir"]) / "STOP").exists():
                    self.stop_owned_child("explicit_service_stop"); break
                if not self.lease_valid():
                    self.stop_owned_child("lease_unavailable_or_expired"); break
                self.sleep(1)
        finally:
            if self.child.poll() is None:
                self.stop_owned_child(self.stop_reason or "guardian_exiting")
            receipt = {**process, "worker_returncode": self.child.poll(),
                       "worker_exit_verified": self.child.poll() is not None,
                       "stop_reason": self.stop_reason, "sigkill_used": self.escalated,
                       "allocation_action": "none"}
            write_json(self.directory / "GUARDIAN-EXIT.json", receipt)
        return 0 if self.child.poll() == 0 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-dir", required=True)
    args = parser.parse_args()
    guardian = Guardian(json.loads(Path(args.config).read_text()), args.config, args.state_dir)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: setattr(guardian, "stop_requested", True))
    return guardian.run()


if __name__ == "__main__":
    raise SystemExit(main())
