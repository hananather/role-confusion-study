"""I supervise one separately authorized, idle H100 without running experiments.

I retain the loaded model while its finite lease is funded. My account credential
stays local. I can delete only my explicitly owned pod, never the existing lane.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from replication.cloud.agent_steering.control import Provider, connection_command, read_json, timestamp, write_json

FORBIDDEN_POD = "nz1bypfsiv62sc"
PEER_DEADLINE = 1789192339.3502789
PEER_RATE = 3.8776712328767124
INCREMENTAL_RATE = 3.49 + 30 * 0.1 / 730
INITIAL_ALLOWANCE = 3.0
PRIOR_SIDECAR_SPEND = 0.16
SHUTDOWN_RESERVE = 0.12
MAX_DELETE_ATTEMPTS = 6


def validate_config(config):
    if config.get("version") != 1 or config.get("scope") not in ("parallel_h100_standby", "parallel_h100_queue"):
        raise ValueError("I require the separate standby-only authorization")
    for field in ("run_id", "pod_id"):
        if not isinstance(config.get(field), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", config[field]):
            raise ValueError("Invalid owned identity")
    if config["pod_id"] == FORBIDDEN_POD:
        raise ValueError("I must never own or mutate the existing H100")
    if config.get("remote_root") != "/workspace/parallel-lanes/" + config["run_id"]:
        raise ValueError("I require this run's unique remote directory")
    queued = config["scope"] == "parallel_h100_queue"
    if queued and config.get("queue_job_id") != config["run_id"] + "-queue":
        raise ValueError("My queue supervision requires the exact owned queue identity")
    exact = {"allowance_usd": 3.65 if queued else INITIAL_ALLOWANCE, "incremental_rate_usd_h": INCREMENTAL_RATE,
             "credit_reserve_usd": 2.0, "protected_peer_deadline_unix": PEER_DEADLINE,
             "protected_peer_rate_usd_h": PEER_RATE, "aggregate_cap_usd": 20.0,
             "protected_prior_projected_total_usd": 16.3444,
             "prior_sidecar_spend_usd": PRIOR_SIDECAR_SPEND}
    for key, expected in exact.items():
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-8):
            raise ValueError("My initial authorization fixes " + key)
    if config.get("protected_peer_pod_id") != FORBIDDEN_POD:
        raise ValueError("I require the recorded protected peer")
    if config.get("identity_marker_path") != "/tmp/mats-prime-" + config["run_id"] + "-identity.json":
        raise ValueError("I require this run's container-local identity marker")
    if config.get("identity_source") != "provider_api_ssh_binding":
        raise ValueError("I require the provider-verified SSH identity binding")
    if not isinstance(config.get("identity_nonce"), str) or not re.fullmatch(r"[0-9a-f]{64}", config["identity_nonce"]):
        raise ValueError("I require a fresh identity nonce")
    if config["protected_prior_projected_total_usd"] + config["allowance_usd"] > config["aggregate_cap_usd"]:
        raise ValueError("The new allowance cannot fit the aggregate cap")
    created = timestamp(config["pod_created_at"])
    if not math.isclose(float(config.get("startup_deadline_unix", -1)), created + 600, rel_tol=0, abs_tol=1e-6):
        raise ValueError("I require a fixed ten-minute startup deadline from pod creation")
    for key in ("connection_file", "local_mirror"):
        if not isinstance(config.get(key), str) or not Path(config[key]).is_absolute():
            raise ValueError("I require an absolute " + key)
    query = config.get("account_check", {}).get("query", "")
    if not isinstance(query, str) or not query.lstrip().startswith("query") or "mutation" in query.lower():
        raise ValueError("My account check must be read-only")
    return config


def verify_identity_marker(config):
    """I verify the provider-bound local marker without fabricating pod environment."""
    import json
    import os
    import stat

    actual = os.environ.get("RUNPOD_POD_ID")
    if actual is not None and actual != config["pod_id"]:
        raise ValueError("The actual pod environment disagrees with the owned identity")
    fd = os.open(config["identity_marker_path"], os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("I require an owned, private, regular identity marker")
        marker = json.load(file)
    expected = {"schema_version": 1, "identity_source": "provider_api_ssh_binding",
                "expected_pod_id": config["pod_id"], "run_id": config["run_id"],
                "identity_nonce": config["identity_nonce"]}
    if marker != expected:
        raise ValueError("The container-local marker does not match this authorization")


def remote_identity_check(config):
    """I use the same tested verifier before remote lease writes and artifact reads."""
    selected = {key: config[key] for key in ("pod_id", "run_id", "identity_marker_path", "identity_nonce")}
    return inspect.getsource(verify_identity_marker) + "\nverify_identity_marker(" + repr(selected) + ")\n"


class StandbySupervisor:
    def __init__(self, config, directory, provider, *, clock=time.time, monotonic=time.monotonic,
                 command=subprocess.run, sleep=time.sleep):
        self.config = json.loads(json.dumps(validate_config(config)))
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.provider, self.clock, self.monotonic = provider, clock, monotonic
        self.command, self.sleep = command, sleep
        self.initial_wall, self.initial_mono = clock(), monotonic()
        self.created = timestamp(config["pod_created_at"])
        if self.created > self.initial_wall + 5:
            raise ValueError("The pod creation time cannot be in the future")
        self.pod_id = config["pod_id"]
        self.pod_path = "/pods/" + self.pod_id
        self.connection = None
        self.connection_seen_at = None
        encoded = json.dumps(self.config, sort_keys=True, separators=(",", ":"))
        self.config_sha = hashlib.sha256(encoded.encode()).hexdigest()
        authorization = self.directory / "initial-authorization.json"
        if authorization.exists():
            if read_json(authorization).get("config_sha256") != self.config_sha:
                raise ValueError("I cannot reset or silently extend my immutable initial allowance")
        else:
            with authorization.open("x") as file:
                json.dump({"config_sha256": self.config_sha, "config": self.config}, file, indent=2)
                file.flush(); os.fsync(file.fileno())
        self.high_consumed = float(config["prior_sidecar_spend_usd"])
        self.last_account, self.last_account_at = None, None
        self.last_provider_ok = self.initial_wall
        self.last_lease_ok = self.initial_wall
        ledger = self.directory / "ledger.json"
        if ledger.exists():
            previous = read_json(ledger)
            if previous.get("config_sha256") != self.config_sha:
                raise ValueError("My financial ledger belongs to another authorization")
            self.high_consumed = max(self.high_consumed, float(previous["consumed_usd"]))
            self.last_account = previous.get("last_account")
            self.last_account_at = previous.get("account_checked_unix")
            self.last_provider_ok = previous.get("provider_checked_unix", self.created)
            self.last_lease_ok = previous.get("lease_published_unix", self.created)
        self.last_sync = -math.inf
        self.reason, self.ended, self.delete_verified = None, False, False
        self.delete_attempts = 0
        self.signal_seen = False
        self.ready = False
        if (self.directory / "termination.json").exists():
            receipt = read_json(self.directory / "termination.json")
            if receipt.get("pod_id") != self.pod_id:
                raise ValueError("My termination receipt belongs to another pod")
            if receipt.get("verified_404"):
                self.ended = self.delete_verified = True

    def refresh_connection(self):
        path = Path(self.config["connection_file"])
        if not path.exists():
            return self.connection is not None
        connection = read_json(path)
        if connection.get("pod_id") != self.pod_id:
            raise ValueError("My SSH connection must identify this exact new pod")
        if not re.fullmatch(r"[A-Za-z0-9.:-]+", connection.get("host", "")):
            raise ValueError("Invalid SSH host")
        if not 1 <= int(connection["port"]) <= 65535 or not Path(connection["key"]).is_absolute():
            raise ValueError("Invalid SSH connection")
        if self.connection is not None and self.connection != connection:
            raise ValueError("My verified SSH connection cannot change during this run")
        if self.connection_seen_at is None:
            self.connection_seen_at = self.wall()
        self.connection = connection
        return True

    def model_ready(self):
        path = Path(self.config["local_mirror"]) / "service/READY.json"
        if not path.exists():
            return False
        data = read_json(path)
        if self.config["scope"] == "parallel_h100_queue":
            return (data.get("expected_pod_id") == self.pod_id and data.get("run_id") == self.config["run_id"]
                    and data.get("status") == "queue_model_loaded" and data.get("experiment_inbox") is True
                    and data.get("engineering_gate_passed") is False
                    and float(data.get("ready_unix", 0)) >= self.created)
        return (data.get("expected_pod_id") == self.pod_id and data.get("status") == "primed_idle"
                and data.get("smoke_passed") is True and data.get("experiment_inbox") is False
                and float(data.get("ready_unix", 0)) >= self.created)

    def wall(self):
        return max(self.clock(), self.initial_wall + self.monotonic() - self.initial_mono)

    def record(self, event, **fields):
        row = {"time_unix": self.wall(), "pod_id": self.pod_id, "event": event, **fields}
        with (self.directory / "events.jsonl").open("a") as file:
            file.write(json.dumps(row, sort_keys=True) + "\n")
            file.flush()

    def financial_state(self, account=None):
        current = self.wall()
        if account is not None:
            balance = account.get("balance_usd")
            if isinstance(balance, bool) or not isinstance(balance, (int, float)) or not math.isfinite(balance) or balance < 0:
                raise ValueError("Invalid verified balance")
            self.last_account = {"balance_usd": float(balance), "autopay_enabled": account.get("autopay_enabled")}
            self.last_account_at = current
        rate = self.config["incremental_rate_usd_h"]
        current_allocation_spend = max(0, current - self.created) * rate / 3600
        self.high_consumed = max(self.high_consumed, self.config["prior_sidecar_spend_usd"] + current_allocation_spend)
        own_remaining = max(0, self.config["allowance_usd"] - self.high_consumed)
        peer_remaining = max(0, PEER_DEADLINE - current) * PEER_RATE / 3600
        credit_available = 0.0
        if self.last_account is not None:
            stale_seconds = max(0, current - self.last_account_at)
            peer_stale_seconds = max(0, min(current, PEER_DEADLINE) - self.last_account_at)
            projected_balance = self.last_account["balance_usd"] - (stale_seconds * rate + peer_stale_seconds * PEER_RATE) / 3600
            credit_available = projected_balance - peer_remaining - self.config["credit_reserve_usd"]
        available = max(0, min(own_remaining - SHUTDOWN_RESERVE, credit_available - SHUTDOWN_RESERVE))
        result = {"config_sha256": self.config_sha, "pod_id": self.pod_id,
                  "initial_allowance_usd": self.config["allowance_usd"], "consumed_usd": self.high_consumed,
                  "prior_sidecar_spend_usd": self.config["prior_sidecar_spend_usd"],
                  "current_allocation_estimated_spend_usd": current_allocation_spend,
                  "own_allowance_remaining_usd": own_remaining, "peer_obligation_usd": peer_remaining,
                  "shutdown_reserve_usd": SHUTDOWN_RESERVE,
                  "credit_available_for_own_pod_usd": credit_available, "budget_remaining_usd": available,
                  "shutdown_at_unix": current + available / rate * 3600,
                  "incremental_rate_usd_h": rate, "last_account": self.last_account,
                  "account_checked_unix": self.last_account_at, "provider_checked_unix": self.last_provider_ok,
                  "lease_published_unix": self.last_lease_ok, "shared_volume_charged_here": False}
        write_json(self.directory / "ledger.json", result)
        return result

    def sync(self):
        if self.connection is None:
            self.record("sync", succeeded=False, reason="awaiting_own_connection")
            return False
        mirror = Path(self.config["local_mirror"])
        mirror.mkdir(parents=True, exist_ok=True)
        # I verify the container-local identity before rsync reads this lane.
        script = remote_identity_check(self.config) + "import os,sys\nos.execvp('rsync',['rsync']+sys.argv[1:])\n"
        remote_rsync = shlex.join(["/workspace/venv-probes/bin/python", "-c", script])
        cmd = ["rsync", "-rltz", "--partial", "--timeout=15", "--exclude=.env", "--exclude=.env.*",
               "--include=/service/***", "--include=/model-service/***", "--include=/logs/***", "--include=/results/***", "--exclude=*",
               "--rsync-path", remote_rsync, "-e", shlex.join(connection_command(self.connection)),
               "root@" + self.connection["host"] + ":" + self.config["remote_root"] + "/", str(mirror) + "/"]
        try:
            result = self.command(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=20, check=False)
            succeeded = result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            succeeded = False
        if self.config["scope"] == "parallel_h100_queue":
            job_id = self.config["queue_job_id"]
            destination = mirror / "results" / job_id
            destination.mkdir(parents=True, exist_ok=True)
            queue_command = ["rsync", "-rltz", "--partial", "--timeout=15", "--exclude=.env", "--exclude=.env.*",
                             "--rsync-path", remote_rsync, "-e", shlex.join(connection_command(self.connection)),
                             "root@" + self.connection["host"] + ":/workspace/results/agent-steering/" + job_id + "/",
                             str(destination) + "/"]
            try:
                result = self.command(queue_command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, timeout=20, check=False)
                queue_synced = result.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                queue_synced = False
            self.record("queue_sync", succeeded=queue_synced, queue_job_id=job_id)
            succeeded = succeeded and queue_synced
        self.last_sync = self.wall()
        self.record("sync", succeeded=succeeded)
        return succeeded

    def publish_lease(self, finance):
        if self.connection is None:
            raise RuntimeError("I cannot publish a lease before the owned connection is available")
        current = self.wall()
        lease = {"schema_version": 1, "run_id": self.config["run_id"], "lease_id": self.config["run_id"],
                 "pod_id": self.pod_id, "expected_pod_id": self.pod_id, "approved": True,
                 "scope": "approved_gpu_b_queue" if self.config["scope"] == "parallel_h100_queue" else "model_load_and_idle_only", "issued_unix": current,
                 "lease_expires_unix": min(current + 180, finance["shutdown_at_unix"]),
                 "shutdown_at_unix": finance["shutdown_at_unix"],
                 "budget_remaining_usd": finance["budget_remaining_usd"]}
        target = self.config["remote_root"] + "/control/lease.json"
        script = remote_identity_check(self.config) + ("import json,os,pathlib,sys; "
                  "d=json.load(sys.stdin); "
                  "p=pathlib.Path(" + repr(target) + "); "
                  "assert str(p.resolve())==str(p); "
                  "os.umask(0o077); p.parent.mkdir(parents=True,exist_ok=True); "
                  "t=p.with_suffix('.tmp'); t.write_text(json.dumps(d)); t.replace(p)")
        remote = shlex.join(["/workspace/venv-probes/bin/python", "-c", script])
        cmd = connection_command(self.connection) + ["root@" + self.connection["host"], remote]
        result = self.command(cmd, input=json.dumps(lease).encode(), stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=20, check=False)
        if result.returncode != 0:
            raise RuntimeError("Lease publication failed")
        self.last_lease_ok = current
        write_json(self.directory / "lease.json", lease)
        self.record("lease", expires_unix=lease["lease_expires_unix"])
        return lease

    def terminate(self, reason):
        if self.reason is None:
            self.reason = reason
            (self.directory / "STOP").write_text(reason + "\n")
            self.record("termination_requested", reason=reason)
            try:
                self.sync()
            except Exception:
                self.record("sync", succeeded=False, reason="final_sync_error")
        self.delete_attempts += 1
        status = None
        try:
            self.provider.request("DELETE", self.pod_path)
        except Exception:
            self.record("delete_error", attempt=self.delete_attempts)
        try:
            status, _ = self.provider.request("GET", self.pod_path)
            self.delete_verified = status == 404
        except Exception:
            self.record("delete_verification_error", attempt=self.delete_attempts)
        receipt = {"pod_id": self.pod_id, "reason": self.reason, "verified_404": self.delete_verified,
                   "delete_attempts": self.delete_attempts, "checked_at_unix": self.wall(),
                   "protected_peer_untouched": True, "last_get_status": status}
        write_json(self.directory / "termination.json", receipt)
        self.record("termination_check", verified_404=self.delete_verified, attempt=self.delete_attempts)
        if self.delete_verified:
            self.ended = True
        elif self.delete_attempts >= MAX_DELETE_ATTEMPTS:
            write_json(self.directory / "NEEDS_ATTENTION.json", {**receipt, "action": "Verify and terminate only this owned pod; supervised cleanup continues every 60 seconds"})

    def tick(self):
        if self.ended:
            return
        if self.reason or self.signal_seen or (self.directory / "STOP").exists():
            self.terminate(self.reason or ("process_signal" if self.signal_seen else "explicit_own_STOP"))
            return
        error, finance, live_ok = None, None, False
        try:
            status, pod = self.provider.request("GET", self.pod_path)
            if status == 404:
                self.ended = self.delete_verified = True
                write_json(self.directory / "termination.json", {"pod_id": self.pod_id, "verified_404": True,
                           "reason": "owned_pod_already_absent", "checked_at_unix": self.wall()})
                return
            if status != 200 or not isinstance(pod, dict) or pod.get("id") != self.pod_id:
                raise RuntimeError("Owned pod identity unavailable")
            rate = float(pod.get("costPerHr", float("nan")))
            if not math.isfinite(rate) or rate <= 0 or rate > 3.49 + 1e-8:
                self.terminate("live_rate_outside_authorization")
                return
            account = self.provider.account(self.config["account_check"])
            finance = self.financial_state(account)
            if account.get("autopay_enabled") is not False:
                raise RuntimeError("Auto-Pay not verified disabled")
            self.last_provider_ok = self.wall()
            live_ok = True
        except Exception as exc:
            error = type(exc).__name__
            self.record("provider_check_error", error_type=error)
        finance = self.financial_state()
        if finance["own_allowance_remaining_usd"] <= SHUTDOWN_RESERVE:
            self.terminate("initial_lifetime_allowance_exhausted")
            return
        if self.last_account is not None and finance["budget_remaining_usd"] <= 0:
            self.terminate("protected_peer_and_credit_reserve_floor")
            return
        lease = None
        try:
            connected = self.refresh_connection()
        except Exception:
            self.terminate("owned_connection_invalid")
            return
        if live_ok and connected:
            try:
                lease = self.publish_lease(finance)
            except Exception as exc:
                error = type(exc).__name__
                self.record("lease_error", error_type=error)
        if self.wall() - self.last_provider_ok > 120:
            self.terminate("provider_verification_stale")
            return
        if connected and self.wall() - max(self.last_lease_ok, self.connection_seen_at) > 120:
            self.terminate("lease_publication_stale")
            return
        if connected and self.wall() - self.last_sync >= 90:
            self.sync()
        if self.wall() >= self.config["startup_deadline_unix"] and not self.model_ready():
            self.terminate("startup_readiness_deadline")
            return
        write_json(self.directory / "latest-monitor.json", {"pod_id": self.pod_id, "state": "retained_standby",
                   "checked_at_unix": self.wall(), "live_check_passed": live_ok, "finance": finance,
                   "lease": lease, "error_type": error, "ready_and_idle_do_not_terminate": True})
        self.financial_state()
        if lease and not self.ready:
            self.ready = True
            write_json(self.directory / "SUPERVISOR_READY.json", {"pod_id": self.pod_id,
                       "run_id": self.config["run_id"], "pid": os.getpid(), "lease_published": True,
                       "checked_at_unix": self.wall(), "model_loaded": "not established by supervisor"})

    def run(self):
        with (self.directory / "supervisor.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            signal.signal(signal.SIGTERM, lambda *_: setattr(self, "signal_seen", True))
            signal.signal(signal.SIGINT, lambda *_: setattr(self, "signal_seen", True))
            while not self.ended:
                start = self.monotonic()
                try:
                    self.tick()
                except Exception as exc:
                    self.record("supervisor_error", error_type=type(exc).__name__)
                    self.terminate("unhandled_supervisor_error")
                if not self.ended:
                    interval = (60 if self.delete_attempts >= MAX_DELETE_ATTEMPTS else 10) if self.reason else 30
                    self.sleep(max(1, interval - (self.monotonic() - start)))
        return 0 if self.delete_verified else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-dir", required=True)
    args = parser.parse_args()
    config = validate_config(read_json(args.config))
    return StandbySupervisor(config, args.state_dir, Provider()).run()


if __name__ == "__main__":
    raise SystemExit(main())
