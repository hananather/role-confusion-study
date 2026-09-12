#!/usr/bin/env python3
"""Keep my GPU session bounded while retaining its network volume and failed jobs.

Start as soon as the pod exists, before SSH is ready. I anchor the maximum
lifetime to --created-at; updates to the connection file never extend it.
Only a root QUEUE_DONE, STOP_SESSION, or the lifetime limit ends the session.
Per-job DONE/EXIT/FAILED markers are preserved for review and repair.

The RunPod account key is read from RUNPOD_API_KEY, never from the JSON config.
No volume deletion method is used. A pod's ordinary local volume would be lost
on deletion, so the caller must provision a separate persistent network volume.

Connection JSON (write atomically when ready):
{"ssh_host": "203.0.113.10", "ssh_port": 22022,
 "ssh_key": "/Users/hananather/.ssh/id_ed25519"}
"""

import argparse
import fcntl
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runpod_api  # noqa: E402


def utc_now():
    return datetime.now(timezone.utc)


def parse_utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include its UTC offset")
    return parsed.astimezone(timezone.utc)


class QuietApi(runpod_api.Api):
    """Keep provider response bodies and credentials out of my session log."""

    def _note(self, text):
        pass


class SessionWatchdog:
    def __init__(self, args, *, api=None, now=utc_now, monotonic=time.monotonic,
                 sleep=time.sleep, command=subprocess.run):
        self.args = args
        self.now, self.monotonic, self.sleep, self.command = now, monotonic, sleep, command
        self.created = parse_utc(args.created_at)
        if not 0 < args.hours <= 3:
            raise ValueError("The authorized session lifetime must be in (0, 3] hours")
        if self.created > now() + timedelta(seconds=60):
            raise ValueError("Pod creation time is in the future")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.pod_id):
            raise ValueError("Invalid pod ID")
        self.deadline = self.created + timedelta(hours=args.hours)
        self.deadline_mono = monotonic() + max(0, (self.deadline - now()).total_seconds())
        self.directory = Path(args.session_dir).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.outputs = self.directory / "outputs"
        self.outputs.mkdir(exist_ok=True)
        self.config = Path(args.config).expanduser().resolve() if args.config else self.directory / "connection.json"
        self.api = api if api is not None else QuietApi()
        self.connection = None
        self.terminated = False
        self.end_reason = None
        self.interrupted = False
        self.last_sync = None
        self.sync_failures = 0
        self.last_state = None
        self.config_problem = None
        self.lock = None

    def remaining(self):
        return max(0, min((self.deadline - self.now()).total_seconds(),
                          self.deadline_mono - self.monotonic()))

    def log(self, event, **fields):
        # I log only my own short fields, never subprocess output or API responses.
        record = {"time_utc": self.now().isoformat(), "event": event, **fields}
        line = json.dumps(record, sort_keys=True)
        try:
            print(line, flush=True)
        except OSError:
            pass
        try:
            with (self.directory / "session_watchdog.log").open("a") as handle:
                handle.write(line + "\n")
        except OSError:
            pass  # A full local disk must not prevent the API lifetime safeguard.

    def write_status(self, state, **fields):
        status = {"time_utc": self.now().isoformat(), "pid": os.getpid(),
                  "pod_id": self.args.pod_id, "state": state,
                  "created_at": self.created.isoformat(), "deadline_utc": self.deadline.isoformat(),
                  "remaining_seconds": round(self.remaining(), 1),
                  "last_sync_utc": self.last_sync, "consecutive_sync_failures": self.sync_failures,
                  "termination_reason": self.end_reason,
                  "network_volume_policy": "preserve; DELETE pod only", **fields}
        target = self.directory / "session_status.json"
        temporary = target.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(target)
        except OSError:
            pass
        if state != self.last_state:
            self.log("state", state=state)
            self.last_state = state

    def read_connection(self):
        if not self.config.exists():
            return self.connection
        try:
            data = json.loads(self.config.read_text())
            if not isinstance(data, dict):
                raise ValueError("Connection must be an object")
            host = data.get("ssh_host")
            if not host:
                return self.connection
            if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9.:-]+", host):
                raise ValueError("Invalid SSH host")
            port = int(data.get("ssh_port", 22))
            if not 1 <= port <= 65535:
                raise ValueError("Invalid SSH port")
            key = str(Path(data.get("ssh_key", "~/.ssh/id_ed25519")).expanduser())
            if "\n" in key or "\r" in key:
                raise ValueError("Invalid SSH key path")
            self.connection = {"host": host, "port": port, "key": key}
            self.config_problem = None
        except (OSError, ValueError, TypeError, KeyError):
            self.config_problem = "invalid_connection_config"
        return self.connection

    def sync(self):
        connection = self.read_connection()
        if not connection or self.remaining() <= 0:
            return False
        timeout = min(self.args.sync_timeout, self.remaining())
        # rsync owns SSH's stdin for its protocol; only the outer rsync stdin is closed.
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
               "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10",
               "-i", connection["key"], "-p", str(connection["port"])]
        host = connection["host"]
        if ":" in host:
            host = "[" + host + "]"
        cmd = ["rsync", "-rltz", "--partial", "--timeout=" + str(max(1, math.ceil(timeout))),
               "--exclude=.env", "--exclude=.env.*", "--exclude=self.json",
               "-e", shlex.join(ssh), f"root@{host}:/workspace/results/", str(self.outputs) + "/"]
        try:
            result = self.command(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=timeout, check=False)
            succeeded = result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            succeeded = False
        if succeeded:
            self.last_sync = self.now().isoformat()
            self.sync_failures = 0
        else:
            self.sync_failures += 1
            if self.sync_failures == 1 or self.sync_failures % 10 == 0:
                self.log("sync_unavailable", consecutive_failures=self.sync_failures)
        return succeeded

    def failures(self):
        failures = []
        for name in ("FAILED", "NEEDS_ATTENTION", "EXIT"):
            for path in self.outputs.rglob(name):
                if not path.is_file():
                    continue
                if name == "EXIT":
                    try:
                        with path.open() as handle:
                            text = handle.read(64).strip()
                        if text == "0":
                            continue
                    except OSError:
                        pass
                failures.append(str(path.relative_to(self.outputs)))
        return sorted(failures)

    def fresh_root_marker(self, name):
        marker = self.outputs / name
        # I ignore old session markers copied from a preceding session's volume.
        return marker.is_file() and marker.stat().st_mtime >= self.created.timestamp()

    def request_termination(self, reason):
        if self.terminated:
            return
        if self.end_reason is None:
            self.end_reason = reason
            self.log("terminate_requested", reason=reason, pod_id=self.args.pod_id)
        self.write_status("terminating")
        try:
            result = self.api.delete_pod(self.args.pod_id, tries=1)
            verified = bool(result.get("terminated"))
            if not verified:
                # A preceding DELETE may have succeeded before its response was lost.
                try:
                    pod = self.api.get_pod(self.args.pod_id)
                    verified = isinstance(pod, dict) and pod.get("desiredStatus") in ("TERMINATED", "EXITED")
                except RuntimeError as error:
                    verified = bool(re.search(r"\b404\b", str(error)))
            if verified:
                self.terminated = True
                self.write_status("terminated")
                self.log("pod_termination_verified", pod_id=self.args.pod_id)
            else:
                self.log("pod_termination_unverified", retry=True)
        except Exception as error:
            self.log("pod_termination_unverified", error_type=type(error).__name__, retry=True)

    def tick(self):
        if self.end_reason is not None:
            self.request_termination(self.end_reason)
            return
        if self.remaining() <= 0:
            # I never delay the lifetime limit for a final transfer or SSH readiness.
            self.request_termination("hard_deadline")
            return
        synced = self.sync()
        if self.remaining() <= 0:
            self.request_termination("hard_deadline")
        elif (self.directory / "STOP_SESSION").exists() or self.fresh_root_marker("STOP_SESSION"):
            self.request_termination("STOP_SESSION")
        elif self.interrupted:
            self.request_termination("supervisor_interrupted")
        elif synced and self.fresh_root_marker("QUEUE_DONE"):
            self.request_termination("QUEUE_DONE")
        else:
            failures = self.failures()
            state = "needs_attention" if failures or self.sync_failures or self.config_problem else (
                "running" if self.last_sync else "waiting_for_ssh")
            self.write_status(state, failure_markers=failures, config_problem=self.config_problem)

    def run(self):
        self.lock = (self.directory / "session_watchdog.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A supervisor already owns this session directory")
        self.log("supervisor_started", pod_id=self.args.pod_id, deadline_utc=self.deadline.isoformat())
        previous_handlers = {}
        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.signal(signum, lambda *_: setattr(self, "interrupted", True))
            while not self.terminated:
                try:
                    self.tick()
                except Exception as error:
                    # A malformed file or failed transfer must not stop lifetime enforcement.
                    self.log("supervision_error", error_type=type(error).__name__)
                if not self.terminated:
                    self.sleep(min(self.args.interval, max(0.1, self.remaining())) if self.end_reason is None else 10)
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            self.lock.close()
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--created-at", required=True, help="Actual pod creation time, including UTC offset")
    parser.add_argument("--hours", type=float, default=3, help="Maximum hours since creation, at most 3")
    parser.add_argument("--config", help="Connection JSON, default SESSION_DIR/connection.json")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--sync-timeout", type=float, default=20)
    args = parser.parse_args()
    if not 0 < args.interval <= 60 or not 0 < args.sync_timeout <= 60:
        parser.error("interval and sync-timeout must be in (0, 60] seconds")
    try:
        return SessionWatchdog(args).run()
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
