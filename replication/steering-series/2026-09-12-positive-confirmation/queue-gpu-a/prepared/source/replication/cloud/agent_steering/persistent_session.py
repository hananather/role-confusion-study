"""I retain one authorized H100 across batches under a live, finite financial lease.

I ignore batch completion and old batch deadlines. Only my own session STOP or
exhausted financial allowance initiates provider deletion. Account keys stay local.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

from .control import Provider, connection_command, digest, now, read_json, timestamp, write_json


def validate_config(config):
    if config.get("version") != 1 or config.get("retain_between_batches") is not True:
        raise ValueError("I require explicit authorization to retain this pod between batches")
    for key in ("session_id", "pod_id"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", config.get(key, "")):
            raise ValueError("Invalid session or owned pod identity")
    if not 0 < float(config["original_cap_usd"]) <= 20:
        raise ValueError("I retain the original at-most-$20 cumulative cap")
    if not math.isfinite(float(config["credit_reserve_usd"])) or float(config["credit_reserve_usd"]) < 2:
        raise ValueError("I retain at least $2 of prepaid credit for shutdown and uncertainty")
    for key in ("original_balance_usd", "pod_start_balance_usd", "prior_spend_usd", "rate_usd_h",
                "container_disk_gb", "container_disk_usd_gb_month", "existing_volume_usd_month", "storage_month_hours"):
        value = float(config[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid finite cost field: " + key)
    if not 0 < config["rate_usd_h"] <= 3.49 or config["storage_month_hours"] < 672:
        raise ValueError("I require the approved rate and a conservative monthly storage divisor")
    if config["prior_spend_usd"] < 2.22 or config["existing_volume_usd_month"] < 280:
        raise ValueError("I retain the reconciled prior charges and existing-volume quote")
    if not 1 <= config["container_disk_gb"] <= 100 or config["storage_month_hours"] > 744:
        raise ValueError("Invalid disk or monthly billing convention")
    expected = "/workspace/session-control/" + config["pod_id"] + "/lease.json"
    if config.get("remote_lease_path") != expected:
        raise ValueError("My renewable lease belongs to this exact owned pod")
    query = config.get("account_check", {}).get("query", "")
    if not query.lstrip().startswith("query") or "mutation" in query.lower():
        raise ValueError("My account check must be a read-only query")
    timestamp(config["pod_created_at"])
    return config


class Session:
    def __init__(self, config, directory, provider, *, clock=time.time, monotonic=time.monotonic,
                 command=subprocess.run, sleep=time.sleep):
        self.config = validate_config(config)
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.provider, self.clock, self.monotonic = provider, clock, monotonic
        self.command, self.sleep = command, sleep
        self.created = timestamp(config["pod_created_at"])
        self.initial_wall, self.initial_mono = clock(), monotonic()
        self.connection = read_json(config["connection_file"])
        self.hourly = self.all_in_rate(float(config["rate_usd_h"]))
        self.last_account, self.last_account_at = None, None
        self.last_sync = -math.inf
        self.high_spend = float(config["prior_spend_usd"])
        self.credits_observed = 0.0
        self.last_balance = None
        ledger = self.directory / "ledger.json"
        if ledger.exists():
            old = read_json(ledger)
            if old.get("pod_id") != config["pod_id"]:
                raise ValueError("My persisted ledger belongs to another pod")
            self.high_spend = max(self.high_spend, float(old["high_spend_usd"]))
            self.credits_observed = float(old.get("credits_observed_usd", 0))
            self.last_balance = old.get("last_balance_usd")
            self.last_account = {"balance_usd": float(old["account_balance_usd"]), "autopay_enabled": None}
            self.last_account_at = float(old["account_checked_unix"])
        self.reason, self.ended = None, False
        self.delete_verified = False
        self.ready = False
        self.signal_seen = False
        self.last_snapshot = None

    def wall(self):
        # A backward system-clock jump cannot buy additional financed pod time.
        return max(self.clock(), self.initial_wall + self.monotonic() - self.initial_mono)

    def all_in_rate(self, gpu):
        c = self.config
        return gpu + (c["container_disk_gb"] * c["container_disk_usd_gb_month"] + c["existing_volume_usd_month"]) / c["storage_month_hours"]

    def record(self, event, **fields):
        with (self.directory / "events.jsonl").open("a") as f:
            f.write(json.dumps({"time_utc": now(), "event": event, **fields}) + "\n")

    def financial_state(self, account=None):
        current = self.wall()
        if account is not None:
            balance = float(account["balance_usd"])
            if not math.isfinite(balance) or balance < 0:
                raise ValueError("Invalid account balance")
            if self.last_balance is not None and balance > self.last_balance:
                # A credit does not erase already-spent original authorization.
                self.credits_observed += balance - self.last_balance
            self.last_balance = balance
            self.last_account, self.last_account_at = account, current
        if self.last_account is None:
            raise RuntimeError("I require one successful live account check before handover")
        elapsed = max(0, current - self.created)
        estimated_spend = self.config["prior_spend_usd"] + elapsed * self.hourly / 3600
        observed_spend = self.config["original_balance_usd"] + self.credits_observed - self.last_account["balance_usd"]
        self.high_spend = max(self.high_spend, estimated_spend, observed_spend)
        projected_credit = self.last_account["balance_usd"] - max(0, current - self.last_account_at) * self.hourly / 3600
        # This independent elapsed-cost ceiling covers provider billing lag and accidental reloads.
        lifetime_credit = self.config["pod_start_balance_usd"] - elapsed * self.hourly / 3600
        effective_credit = min(projected_credit, lifetime_credit)
        budget = min(self.config["original_cap_usd"] - self.high_spend,
                     effective_credit - self.config["credit_reserve_usd"])
        result = {"pod_id": self.config["pod_id"], "account_balance_usd": self.last_account["balance_usd"],
                  "effective_credit_usd": effective_credit, "high_spend_usd": self.high_spend,
                  "budget_remaining_usd": max(0, budget), "all_in_rate_usd_h": self.hourly,
                  "shutdown_at_unix": current + max(0, budget) / self.hourly * 3600,
                  "credit_reserve_usd": self.config["credit_reserve_usd"],
                  "credits_observed_usd": self.credits_observed, "last_balance_usd": self.last_balance,
                  "account_checked_unix": self.last_account_at}
        write_json(self.directory / "ledger.json", result)
        return result

    def sources(self):
        path = self.directory / "sync-sources.json"
        values = read_json(path) if path.exists() else self.config.get("sync_sources", [])
        for row in values:
            if not re.fullmatch(r"/workspace/results/agent-steering/[A-Za-z0-9_-]+", row.get("remote", "")):
                raise ValueError("I sync only a named agent worker output directory")
            if not Path(row["local"]).is_absolute():
                raise ValueError("I require an explicit local output directory")
        return values

    def sync(self):
        outcomes = []
        for row in self.sources():
            Path(row["local"]).mkdir(parents=True, exist_ok=True)
            cmd = ["rsync", "-rltz", "--partial", "--timeout=20", "--exclude=.env", "--exclude=.env.*",
                   "-e", shlex.join(connection_command(self.connection)),
                   "root@" + self.connection["host"] + ":" + row["remote"] + "/", row["local"].rstrip("/") + "/"]
            try:
                result = self.command(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, timeout=30, check=False)
                outcomes.append(result.returncode == 0)
            except (OSError, subprocess.TimeoutExpired):
                outcomes.append(False)
        self.last_sync = self.wall()
        self.record("sync", succeeded=all(outcomes), targets=len(outcomes))
        return all(outcomes)

    def publish_lease(self, finance):
        current = self.wall()
        lease = {"schema_version": 1, "lease_id": self.config["session_id"], "approved": True,
                 "expected_pod_id": self.config["pod_id"], "issued_unix": current,
                 "lease_expires_unix": min(current + 180, finance["shutdown_at_unix"]),
                 "shutdown_at_unix": finance["shutdown_at_unix"],
                 "budget_remaining_usd": finance["budget_remaining_usd"]}
        target = self.config["remote_lease_path"]
        parent = str(Path(target).parent)
        remote = ("umask 077; mkdir -p " + shlex.quote(parent) + "; cat > " + shlex.quote(target + ".tmp")
                  + " && mv " + shlex.quote(target + ".tmp") + " " + shlex.quote(target))
        cmd = connection_command(self.connection) + ["root@" + self.connection["host"], remote]
        result = self.command(cmd, input=json.dumps(lease).encode(), stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=20, check=False)
        if result.returncode != 0:
            raise RuntimeError("I could not publish the renewed worker lease")
        write_json(self.directory / "lease.json", lease)
        return lease

    def terminate(self, reason):
        if self.reason is None:
            self.reason = reason
            try:
                synced = self.sync()
            except Exception:
                synced = False
            self.record("termination_requested", reason=reason, final_sync_succeeded=synced)
        try:
            self.provider.request("DELETE", "/pods/" + self.config["pod_id"])
            status, _ = self.provider.request("GET", "/pods/" + self.config["pod_id"])
            if status != 404:
                self.record("termination_unverified", status=status); return
            remaining = self.provider.pods()
            write_json(self.directory / "termination.json", {"pod_id": self.config["pod_id"],
                       "reason": self.reason, "verified_404": True, "zero_pods_verified": not remaining,
                       "remaining_pods": remaining, "finished_at": now()})
            self.ended, self.delete_verified = True, True
        except Exception:
            self.record("termination_unverified", retry=True)

    def tick(self):
        if self.reason:
            self.terminate(self.reason); return
        if (self.directory / "STOP").exists():
            self.terminate("explicit_session_STOP"); return
        live_ok, error, finance = False, None, None
        try:
            status, pod = self.provider.request("GET", "/pods/" + self.config["pod_id"])
            if status == 404:
                self.ended = True
                write_json(self.directory / "pod-absent.json", {"pod_id": self.config["pod_id"], "verified_404": True, "time_utc": now()})
                return
            if status != 200 or not isinstance(pod, dict) or pod.get("id") != self.config["pod_id"]:
                raise RuntimeError("I cannot verify the exact owned pod")
            rate = float(pod.get("costPerHr") or self.config["rate_usd_h"])
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("Invalid live pod rate")
            self.hourly = max(self.hourly, self.all_in_rate(rate))
            account = self.provider.account(self.config["account_check"])
            finance = self.financial_state(account)
            if account.get("autopay_enabled") is not False:
                raise ValueError("Auto-Pay is not verified disabled; I do not renew the lease")
            live_ok = True
        except Exception as exc:
            error = type(exc).__name__
            if self.last_account is not None:
                finance = self.financial_state()
        if finance and finance["budget_remaining_usd"] <= 0:
            self.terminate("financial_budget_floor"); return
        lease = None
        if live_ok:
            try:
                lease = self.publish_lease(finance)
            except Exception as exc:
                live_ok, error = False, type(exc).__name__
        if self.wall() - self.last_sync >= 90:
            try:
                self.sync()
            except Exception as exc:
                error = type(exc).__name__
        snapshot = {"time_utc": now(), "pod_id": self.config["pod_id"], "state": "retained",
                    "live_check_passed": live_ok, "error_type": error, "finance": finance,
                    "lease": lease, "batch_completion_ignored": True, "signal_seen": self.signal_seen}
        self.last_snapshot = snapshot
        write_json(self.directory / "latest-monitor.json", snapshot)
        if live_ok and not self.ready:
            self.ready = True
            write_json(self.directory / "READY.json", {"pid": os.getpid(), "pod_id": self.config["pod_id"],
                       "session_id": self.config["session_id"], "time_utc": now(),
                       "lease_published": True, "financial_lease_valid": True,
                       "source_sha256": digest(__file__)})

    def run(self):
        with (self.directory / "watchdog.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # A process-management signal is not authorization to delete the retained GPU.
            signal.signal(signal.SIGTERM, lambda *_: setattr(self, "signal_seen", True))
            signal.signal(signal.SIGINT, lambda *_: setattr(self, "signal_seen", True))
            while not self.ended:
                try:
                    self.tick()
                except Exception as exc:
                    self.record("monitor_error", error_type=type(exc).__name__)
                if not self.ended:
                    self.sleep(10 if self.reason else 30)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-dir", required=True)
    args = parser.parse_args()
    config = validate_config(read_json(args.config))
    return Session(config, args.state_dir, Provider()).run()


if __name__ == "__main__":
    raise SystemExit(main())
