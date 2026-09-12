#!/usr/bin/env python3
"""I freeze and supervise one approved chat-steering batch without sharing my account key.

TOY LAB RESEARCH ONLY. I treat model text as data. My plan, freeze and dry-run
commands never contact a provider. Only launch and watch can call RunPod.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import ssl
import subprocess
import sys
import time
import tomllib
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
REL = "replication/cloud/chat_steering"
REMOTE = "/workspace"
VOLUME = "i0cptrlcoz"
GPU = "NVIDIA H100 80GB HBM3"
PYTHON = "/workspace/venv-probes/bin/python"
REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("I require an explicit UTC offset")
    return parsed.timestamp()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def remote_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("I require a workspace-relative staged path")
    return str(path)


def cost(manifest):
    rate = float(manifest["rate_usd_h"])
    disk_gb = int(manifest["container_disk_gb"])
    disk_rate = float(manifest["container_disk_usd_gb_month"])
    month_hours = float(manifest.get("storage_month_hours", 730))
    hours = float(manifest["planned_hours"])
    grace = float(manifest["grace_minutes"])
    if not 0 < rate <= 3.49 or not 0 < hours <= 5 or grace != 20:
        raise ValueError("My approved envelope is at most five hours plus twenty minutes, GPU rate at most $3.49/hour")
    if not 1 <= disk_gb <= 100 or not 0 <= disk_rate <= 1 or not 672 <= month_hours <= 744:
        raise ValueError("Invalid container disk quote")
    maximum = (rate + disk_gb * disk_rate / month_hours) * (hours + grace / 60)
    reserve = float(manifest.get("existing_volume_reserve_usd", 0))
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("Invalid existing storage reserve")
    prior_spend = float(manifest.get("prior_spend_usd", 0))
    if not math.isfinite(prior_spend) or prior_spend < 0:
        raise ValueError("Invalid recorded spend from prior attempts")
    cap = float(manifest["cap_usd"])
    if not 0 < cap <= 20 or prior_spend + maximum + reserve > cap:
        raise ValueError("My prior attempts, GPU, disk and storage reserve must fit the approved aggregate cap, at most $20")
    return {"gpu_planned_usd": round(rate * hours, 4),
            "gpu_with_grace_usd": round(rate * (hours + grace / 60), 4),
            "container_disk_with_grace_usd": round(maximum - rate * (hours + grace / 60), 4),
            "maximum_incremental_usd": round(maximum, 4), "existing_volume_reserve_usd": reserve,
            "prior_spend_usd": prior_spend,
            "maximum_total_usd": round(prior_spend + maximum + reserve, 4), "cap_usd": cap,
            "existing_network_volume": "retained and separately billed; reserve its charges in available balance"}


def validate_manifest(path, *, freeze=False, on_pod=False):
    manifest = read_json(path)
    if manifest.get("version") != 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", manifest.get("run_id", "")):
        raise ValueError("Invalid manifest version or run ID")
    fresh = manifest.get("asset_mode", "existing_volume") == "fresh_ephemeral"
    if fresh:
        if manifest.get("network_volume_id") is not None or manifest.get("scope") != "released_example_pilot":
            raise ValueError("My fresh ephemeral scope never creates or attaches a network volume")
        if manifest.get("container_disk_gb") != 100:
            raise ValueError("My fresh ephemeral pilot uses exactly 100 GB of container disk")
    elif manifest.get("network_volume_id") != VOLUME or manifest.get("data_center") != "EU-FR-1":
        raise ValueError("I may attach only the existing approved EU-FR-1 network volume")
    if not re.fullmatch(r"[A-Z0-9-]+", manifest.get("data_center", "")):
        raise ValueError("I require an explicit regional deployment target")
    if manifest.get("gpu_type") != GPU or manifest.get("cloud") != "SECURE":
        raise ValueError("I require one secure H100 SXM")
    if not isinstance(manifest.get("image"), str) or not manifest["image"]:
        raise ValueError("I require a frozen image name")
    cost(manifest)
    command = manifest.get("command")
    prefix = [PYTHON, "-u", "-m", "replication.cloud.chat_steering.batch"]
    if (not isinstance(command, list) or command[:4] != prefix or len(command) != 7
            or command[4] not in ("run", "baseline", "examples") or command[5] != "--config"):
        raise ValueError("My command must run the frozen batch module with exactly one config")
    config_relative = remote_relative(command[6].removeprefix(REMOTE + "/"))
    if command[6] != REMOTE + "/" + config_relative:
        raise ValueError("My batch config must be inside /workspace")
    names = set()
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("I require a frozen source and input file list")
    for entry in files:
        relative = remote_relative(entry["remote"])
        if relative in names or Path(relative).name.startswith(".env"):
            raise ValueError("Duplicate or private staged filename")
        names.add(relative)
        source = Path(REMOTE, relative) if on_pod else Path(entry["source"]).expanduser().resolve()
        if not source.is_file():
            raise ValueError("A frozen input is missing: " + relative)
        actual = digest(source)
        if freeze:
            entry["source"] = str(source)
            entry["sha256"] = actual
        elif actual != entry.get("sha256"):
            raise ValueError("A frozen input changed: " + relative)
    required = {REL + "/" + name for name in ("control.py", "run.sh", "batch.py", "protocol.py", "runtime.py")}
    required |= {"replication/cloud/jobcommon.py", "replication/cloud/job_steering.py"}
    if fresh:
        required.add(REL + "/setup_fresh.sh")
    if not (required | {config_relative}).issubset(names):
        raise ValueError("My freeze must include controller, wrapper, batch, protocol, runtime and config")
    query = manifest.get("account_check", {}).get("query", "")
    if not query.lstrip().startswith("query") or "mutation" in query.lower():
        raise ValueError("I require a verified read-only account query")
    return manifest


def validate_corpus(manifest):
    """I validate the exact frozen corpus and its semantic review without loading a model."""
    import importlib.util
    mapping = {REMOTE + "/" + row["remote"]: row["source"] for row in manifest["files"]}
    config = read_json(mapping[manifest["command"][6]])
    base_only = manifest.get("scope") == "base_only"
    released_pilot = manifest.get("scope") == "released_example_pilot"
    small_scope = base_only or released_pilot
    if config.get("execution_approved") is not True or (not small_scope and config.get("protocol_decisions_complete") is not True):
        raise ValueError("The batch's execution and protocol decisions are not approved")
    if released_pilot and (config.get("protocol_decisions_complete") is not True or config.get("examples_execution_aligned") is not True):
        raise ValueError("The released-example pilot's protocol and scope need explicit alignment")
    needed = (("strongreject_csv",) if base_only else (
        ("strongreject_csv", "harmless_csv", "released_examples", "released_examples_source") if released_pilot else
        ("strongreject_csv", "policies_csv", "harmless_csv", "corpus_review_receipt")))
    for key in needed:
        if config.get(key) not in mapping:
            raise ValueError("My frozen inputs must include " + key)
        config[key] = mapping[config[key]]
    for key, value in list(config.items()):
        if isinstance(value, str) and value in mapping:
            config[key] = mapping[value]
    expected_out = "/workspace/results/chat-steering/" + manifest["run_id"]
    if config.get("out_dir") != expected_out:
        raise ValueError("The frozen batch output directory differs from the supervised directory")
    spec = importlib.util.spec_from_file_location("_frozen_chat_protocol", mapping[REMOTE + "/" + REL + "/protocol.py"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if small_scope:
        expected_action = "baseline" if base_only else "examples"
        if manifest["command"][4] != expected_action:
            raise ValueError("My approved scope must match the frozen batch mode")
        for field in ("model_free_validation", "runtime_review_receipt"):
            if manifest.get(field) not in mapping:
                raise ValueError("I require the frozen " + field + " receipt")
        validation = read_json(mapping[manifest["model_free_validation"]])
        review = read_json(mapping[manifest["runtime_review_receipt"]])
        if validation.get("passed") is not True or len(validation.get("samples", [])) != 5:
            raise ValueError("My five-prompt tokenizer validation has not passed")
        runtime_hash = next(e["sha256"] for e in manifest["files"] if e["remote"] == REL + "/runtime.py")
        if review.get("passed") is not True or review.get("runtime_sha256") != runtime_hash:
            raise ValueError("My current runtime has not passed source review")
        # The batch's base-only path validates all 313 real questions before loading weights.
        if released_pilot:
            if not hasattr(module, "load_examples_inputs"):
                raise ValueError("The released-example pilot's source validator is not ready")
            module.load_examples_inputs(config)
        elif hasattr(module, "load_baseline_inputs"):
            module.load_baseline_inputs(config)
        else:
            import csv
            source = Path(config["strongreject_csv"])
            rows = list(csv.DictReader(source.open()))
            if digest(source) != module.DATA_SHA256 or len(rows) != 313 or len({r["forbidden_prompt"] for r in rows}) != 313:
                raise ValueError("My base-only batch requires the exact 313-prompt StrongREJECT corpus")
    else:
        module.load_inputs(config, require_policies=True)


def launch_command(manifest_path, receipt_path):
    return shlex.join([sys.executable, str(Path(__file__).resolve()), "launch", "--manifest",
                       str(Path(manifest_path).resolve()), "--approval", str(Path(receipt_path).resolve())])


def authorize(manifest_path, receipt_path):
    receipt = read_json(receipt_path)
    if receipt.get("approved") is not True or receipt.get("approved_by") != "Hanan":
        raise ValueError("I require Hanan's explicit approval receipt; plan and dry-run do not authorize launch")
    if receipt.get("manifest_sha256") != digest(manifest_path):
        raise ValueError("The approval does not bind this exact manifest")
    if receipt.get("command") != launch_command(manifest_path, receipt_path):
        raise ValueError("The approval does not bind this exact launch command")
    if receipt.get("cap_usd") != read_json(manifest_path).get("cap_usd"):
        raise ValueError("The approval must record this manifest's exact cost cap")
    timestamp(receipt["approved_at"])


class Provider:
    """I expose selected account fields and HTTP status, never credentials or response errors."""
    def __init__(self):
        key = os.environ.get("RUNPOD_API_KEY")
        if not key:
            cfg = tomllib.loads((Path.home() / ".runpod/config.toml").read_text())
            key = cfg.get("apikey") or cfg.get("api_key")
        if not key:
            raise ValueError("My RunPod account credential is unavailable")
        self.key = key
        try:
            import certifi
            self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            raise ValueError("I require the existing certifi package for verified provider TLS") from None

    def request(self, method, path, body=None, *, graphql=False):
        url = GRAPHQL if graphql else REST + path
        request = Request(url, method=method, data=None if body is None else json.dumps(body).encode(),
                          headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json",
                                   "User-Agent": "mats-preflight/1.0"})
        try:
            with urlopen(request, timeout=15, context=self.ssl_context) as response:
                raw = response.read()
                return response.status, json.loads(raw) if raw else None
        except HTTPError as error:
            return error.code, None
        except Exception:
            raise RuntimeError("Provider transport failed; I have not retried any mutation") from None

    def pods(self):
        status, data = self.request("GET", "/pods")
        if status != 200 or not isinstance(data, list):
            raise RuntimeError("I cannot verify the live pod inventory")
        return [{k: pod.get(k) for k in ("id", "name", "desiredStatus", "costPerHr")} for pod in data]

    def account(self, specification):
        status, data = self.request("POST", "", {"query": specification["query"]}, graphql=True)
        if status != 200 or not isinstance(data, dict) or data.get("errors"):
            raise RuntimeError("I cannot verify account balance and Auto-Pay")
        def field(keys):
            value = data
            for key in keys:
                value = value[key]
            return value
        try:
            balance = field(specification["balance_path"])
            enabled = field(specification["autopay_path"])
            if not isinstance(enabled, bool) or isinstance(balance, bool):
                raise ValueError()
            balance = float(balance)
            if not math.isfinite(balance):
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("Account response does not establish balance and Boolean Auto-Pay state") from None
        return {"checked_at": now(), "balance_usd": balance, "autopay_enabled": enabled}

    def quote_and_volume(self, manifest):
        region = manifest["data_center"]
        query = ('query { gpuTypes(input: { id: "NVIDIA H100 80GB HBM3" }) { id '
                 'lowestPrice(input: { gpuCount: 1, secureCloud: true, dataCenterId: ' + json.dumps(region) + ' }) '
                 '{ stockStatus uninterruptablePrice availableGpuCounts } } }')
        status, response = self.request("POST", "", {"query": query}, graphql=True)
        try:
            if status != 200 or response.get("errors"):
                raise ValueError()
            items = response["data"]["gpuTypes"]
            item = next(row for row in items if row["id"] == GPU)
            quote = item["lowestPrice"]
            rate = float(quote["uninterruptablePrice"])
            counts = quote["availableGpuCounts"]
            if (not math.isfinite(rate) or rate <= 0 or not quote["stockStatus"]
                    or (counts is not None and 1 not in counts)):
                raise ValueError()
        except (KeyError, TypeError, ValueError, StopIteration):
            raise ValueError("The regional " + region + " H100 SXM quote or capacity is unavailable") from None
        volume_id = manifest.get("network_volume_id")
        if volume_id:
            status, volume = self.request("GET", "/networkvolumes/" + volume_id)
            if (status != 200 or not isinstance(volume, dict) or volume.get("id") != VOLUME
                    or volume.get("dataCenterId") != "EU-FR-1" or volume.get("size") != 2000):
                raise ValueError("I cannot verify the existing 2 TB EU-FR-1 network volume")
        return {"rate_usd_h": rate, "data_center": region, "gpu": GPU,
                "network_volume_id": volume_id, "stock_status": quote["stockStatus"]}


def preflight(provider, manifest):
    pods = provider.pods()
    account = provider.account(manifest["account_check"])
    if pods:
        raise ValueError("I require zero existing pods before creation")
    if account["autopay_enabled"]:
        raise ValueError("Auto-Pay must be off before launch")
    quote = provider.quote_and_volume(manifest)
    if quote["rate_usd_h"] > float(manifest["rate_usd_h"]):
        raise ValueError("The live regional price exceeds the approved rate")
    reserve = float(manifest.get("existing_volume_reserve_usd", 0))
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("Invalid existing storage reserve")
    required = cost(manifest)["maximum_incremental_usd"] + reserve
    if account["balance_usd"] < required:
        raise ValueError("My live balance does not cover the batch and existing storage reserve")
    return {**account, "existing_pods": [], "required_balance_usd": required, "quote": quote}


def connection_command(connection):
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10", "-i", connection["key"], "-p", str(connection["port"])]


class Watchdog:
    """I retain the network volume and keep trying until the pod GET returns 404."""
    def __init__(self, state_dir, provider, *, clock=time.time, monotonic=time.monotonic,
                 command=subprocess.run, sleep=time.sleep):
        self.directory = Path(state_dir)
        self.meta = read_json(self.directory / "pod.json")
        self.clock, self.monotonic, self.command, self.sleep = clock, monotonic, command, sleep
        self.provider = provider
        self.created = timestamp(self.meta["created_at"])
        self.deadline = self.created + (float(self.meta.get("planned_hours", 5)) * 3600 + 20 * 60)
        self.mono_deadline = monotonic() + max(0, self.deadline - clock())
        self.results = self.directory / "results"
        self.results.mkdir(exist_ok=True)
        self.reason = None
        self.terminated = False
        self.zero_pods_verified = False
        self.final_synced = False
        self.interrupted = False
        self.last_sync = None

    def remaining(self):
        return max(0, min(self.deadline - self.clock(), self.mono_deadline - self.monotonic()))

    def record(self, event, **fields):
        data = {"time_utc": now(), "event": event, **fields}
        try:
            with (self.directory / "watchdog.jsonl").open("a") as handle:
                handle.write(json.dumps(data) + "\n")
        except OSError:
            pass

    def sync(self):
        path = self.directory / "connection.json"
        if not path.exists() or self.remaining() <= 0:
            return False
        try:
            connection = read_json(path)
            timeout = min(30, self.remaining())
            cmd = ["rsync", "-rltz", "--partial", "--timeout=20", "--exclude=.env", "--exclude=.env.*",
                   "-e", shlex.join(connection_command(connection)),
                   f"root@{connection['host']}:{self.meta['remote_results']}/", str(self.results) + "/"]
            result = self.command(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=timeout, check=False)
            ok = result.returncode == 0
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
            ok = False
        if ok:
            self.last_sync = self.clock()
        self.record("sync", succeeded=ok)
        return ok

    def heartbeat_age(self):
        path = self.results / "heartbeat.json"
        try:
            heartbeat = read_json(path)
            stamp = float(heartbeat["progress_unix"])
            if not math.isfinite(stamp):
                return None
            if stamp < self.created or stamp > self.clock() + 60:
                return None
            return self.clock() - stamp
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def terminate(self, reason):
        if self.reason is None:
            self.reason = reason
            self.final_synced = self.sync()
            self.record("termination_requested", reason=reason, final_sync_succeeded=self.final_synced)
        try:
            self.provider.request("DELETE", "/pods/" + self.meta["id"])
            status, _ = self.provider.request("GET", "/pods/" + self.meta["id"])
            if status != 404:
                self.record("termination_unverified", status=status)
                return
            pods = self.provider.pods()
            self.zero_pods_verified = len(pods) == 0
            result = {"verified_404": True, "zero_pods_verified": len(pods) == 0,
                      "remaining_pods": pods, "reason": self.reason,
                      "final_sync_succeeded": self.final_synced, "finished_at": now(),
                      "actual_hours": max(0, (self.clock() - self.created) / 3600)}
            result["actual_gpu_usd_estimate"] = result["actual_hours"] * self.meta["rate_usd_h"]
            self.terminated = True
            try:
                write_json(self.directory / "termination.json", result)
            except OSError:
                pass
            self.record("terminated", **result)
        except Exception:
            self.record("termination_unverified", retry=True)

    def tick(self):
        if self.reason:
            self.terminate(self.reason)
            return
        if self.remaining() <= 0:
            self.terminate("hard_deadline")
            return
        synced = self.sync()
        if self.remaining() <= 0:
            self.terminate("hard_deadline")
        elif self.interrupted or (self.directory / "STOP").exists():
            self.terminate("STOP" if not self.interrupted else "supervisor_interrupted")
        elif synced and any((self.results / name).exists() for name in ("DONE.json", "FAILED.json", "EXIT.json")):
            self.terminate("batch_exited")
        else:
            age = self.heartbeat_age()
            if age is not None and age > 600:
                self.terminate("stale_heartbeat")
            elif age is None and self.clock() - self.created > 1200:
                self.terminate("missing_heartbeat")

    def run(self):
        with (self.directory / "watchdog.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            signal.signal(signal.SIGTERM, lambda *_: setattr(self, "interrupted", True))
            signal.signal(signal.SIGINT, lambda *_: setattr(self, "interrupted", True))
            write_json(self.directory / "watchdog-ready.json", {"pid": os.getpid(), "started_at": now()})
            while not self.terminated:
                try:
                    self.tick()
                except Exception:
                    self.record("supervision_error")
                if not self.terminated:
                    self.sleep(10 if self.reason else min(90, max(0.1, self.remaining())))
        return 0 if self.zero_pods_verified else 3


def launch(manifest_path, receipt_path):
    manifest_path, receipt_path = Path(manifest_path).resolve(), Path(receipt_path).resolve()
    authorize(manifest_path, receipt_path)
    manifest = validate_manifest(manifest_path)
    validate_corpus(manifest)
    record = Path(manifest["execution_record"]).read_text()
    if manifest["run_id"] not in record:
        raise ValueError("My run ID must be recorded in EXECUTION.md before a pod exists")
    directory = ROOT / "replication/cloud/outbox/chat-steering" / manifest["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "creation-intent.json").exists():
            raise ValueError("I must reconcile the previous creation intent before any new launch")
        # I snapshot before making provider calls; later workspace edits cannot alter this batch.
        stage = directory / "stage"
        for entry in manifest["files"]:
            target = stage / entry["remote"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry["source"], target)
            if digest(target) != entry["sha256"]:
                raise ValueError("An input changed while I was freezing the launch")
        frozen_manifest = stage / REL / "launch-manifest.json"
        shutil.copyfile(manifest_path, frozen_manifest)
        shutil.copyfile(receipt_path, directory / "approval.json")
        provider = Provider()
        write_json(directory / "account-before.json", preflight(provider, manifest))
        created = now()
        name = "chat-steering-" + manifest["run_id"]
        remote_results = "/workspace/results/chat-steering/" + manifest["run_id"]
        spec = {"name": name, "cloudType": "SECURE", "gpuTypeIds": [GPU], "gpuCount": 1,
                "dataCenterIds": [manifest["data_center"]], "imageName": manifest["image"],
                "containerDiskInGb": manifest["container_disk_gb"], "volumeInGb": 0,
                "volumeMountPath": REMOTE, "ports": ["22/tcp"], "supportPublicIp": True,
                "interruptible": False,
                "env": {"PUBLIC_KEY": (Path.home() / ".ssh/id_ed25519.pub").read_text().strip()}}
        if manifest.get("network_volume_id"):
            spec["networkVolumeId"] = manifest["network_volume_id"]
        write_json(directory / "creation-intent.json", {"created_at": created, "name": name,
                   "manifest_sha256": digest(manifest_path), "cost": cost(manifest)})
        try:
            status, pod = provider.request("POST", "/pods", spec)
            if status not in (200, 201) or not isinstance(pod, dict) or not pod.get("id"):
                raise RuntimeError("Creation did not return a confirmed pod")
        except Exception:
            matches = [p for p in provider.pods() if p.get("name") == name]
            if len(matches) != 1:
                raise RuntimeError("Creation is ambiguous; inspect live inventory before retrying") from None
            pod = matches[0]
        meta = {"id": pod["id"], "created_at": created, "remote_results": remote_results,
                "rate_usd_h": float(pod.get("costPerHr") or manifest["rate_usd_h"]),
                "planned_hours": manifest["planned_hours"],
                "manifest_sha256": digest(manifest_path)}
        write_json(directory / "pod.json", meta)
        # I arm independent supervision immediately, before readiness checks, setup or uploads.
        log = (directory / "supervisor-console.log").open("ab", buffering=0)
        watch = [sys.executable, str(stage / REL / "control.py"), "watch", "--state-dir", str(directory)]
        if shutil.which("caffeinate"):
            watch = ["caffeinate", "-i"] + watch
        try:
            supervisor = subprocess.Popen(watch, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True, close_fds=True)
            for _ in range(50):
                if (directory / "watchdog-ready.json").exists():
                    break
                if supervisor.poll() is not None:
                    raise RuntimeError("My independent watchdog did not start")
                time.sleep(0.1)
            else:
                raise RuntimeError("My independent watchdog did not acknowledge readiness")
            if meta["rate_usd_h"] > manifest["rate_usd_h"]:
                raise ValueError("The pod rate exceeds the approved rate")
            endpoint = None
            for _ in range(90):
                status, current = provider.request("GET", "/pods/" + pod["id"])
                if status == 200 and current and current.get("publicIp") and (current.get("portMappings") or {}).get("22"):
                    endpoint = {"host": current["publicIp"], "port": int(current["portMappings"]["22"]),
                                "key": str(Path.home() / ".ssh/id_ed25519")}
                    break
                time.sleep(10)
            if not endpoint:
                raise RuntimeError("My pod did not expose SSH within fifteen minutes")
            ssh = connection_command(endpoint) + ["root@" + endpoint["host"]]
            # I install only rsync, and I install it before any rsync push.
            cached_check = "" if manifest.get("asset_mode") == "fresh_ephemeral" else "test -x /workspace/venv-probes/bin/python; "
            setup = "set -e; command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync); " + cached_check + "mkdir -p " + shlex.quote(remote_results)
            subprocess.run(ssh + [setup], stdin=subprocess.DEVNULL, check=True, timeout=180,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            write_json(directory / "connection.json", endpoint)
            subprocess.run(["rsync", "-rltz", "--partial", "--timeout=30", "-e", shlex.join(connection_command(endpoint)),
                            str(stage) + "/", "root@" + endpoint["host"] + ":/workspace/"],
                           stdin=subprocess.DEVNULL, check=True, timeout=180,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            remote_command = ["bash", REMOTE + "/" + REL + "/run.sh", "--manifest",
                              REMOTE + "/" + REL + "/launch-manifest.json", "--created-at", created]
            detached = "nohup " + shlex.join(remote_command) + " > " + shlex.quote(remote_results + "/pod.log") + " 2>&1 < /dev/null &"
            subprocess.run(ssh + [detached], stdin=subprocess.DEVNULL, check=True, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(json.dumps({"started": True, "pod_id": pod["id"], "outbox": str(directory), "watchdog_pid": supervisor.pid}))
        except Exception:
            (directory / "STOP").touch()
            if not (directory / "watchdog-ready.json").exists():
                fallback = Watchdog(directory, provider)
                while not fallback.terminated:
                    fallback.terminate("launch_failed_before_watchdog")
                    if not fallback.terminated:
                        time.sleep(10)
            raise


def batch_environment(environ=None):
    """I preserve the separate kernel cache; the runtime names the local model snapshot explicitly."""
    env = dict(os.environ if environ is None else environ)
    env.pop("HF_HUB_CACHE", None)
    env.update(HF_HOME="/workspace/hf/home", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    # My account key is never provisioned; I remove API-key variables from the evaluated process as well.
    for key in list(env):
        if key.endswith("API_KEY") or key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            env.pop(key, None)
    return env


def pod_run(manifest_path, created_at):
    manifest = validate_manifest(manifest_path, on_pod=True)
    results = Path("/workspace/results/chat-steering") / manifest["run_id"]
    results.mkdir(parents=True, exist_ok=True)
    env = batch_environment()
    deadline = timestamp(created_at) + float(manifest["planned_hours"]) * 3600
    if manifest.get("asset_mode") == "fresh_ephemeral":
        setup_env = env.copy()
        setup_env.pop("HF_HUB_OFFLINE", None); setup_env.pop("TRANSFORMERS_OFFLINE", None)
        # The frozen setup script owns its 30-second heartbeat; I avoid competing writers.
        subprocess.run(["bash", REMOTE + "/" + REL + "/setup_fresh.sh", "--results", str(results)], env=setup_env,
                       stdin=subprocess.DEVNULL, check=True, timeout=max(1, min(1800, deadline - time.time())))
    seconds = deadline - time.time()
    if seconds <= 0:
        raise ValueError("My compute window has already ended")
    proc = subprocess.Popen(manifest["command"], cwd=REMOTE, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        code = proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        code = 124
    write_json(results / "EXIT.json", {"returncode": code, "time_utc": now(), "manifest_sha256": digest(manifest_path)})
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("freeze", "plan", "dry-run", "launch"):
        child = sub.add_parser(action)
        child.add_argument("--manifest", required=True)
        child.add_argument("--approval", default=str(ROOT / "replication/cloud/chat_steering/approval.json"))
        if action == "freeze":
            child.add_argument("--output", required=True)
    child = sub.add_parser("watch")
    child.add_argument("--state-dir", required=True)
    child = sub.add_parser("pod-run")
    child.add_argument("--manifest", required=True)
    child.add_argument("--created-at", required=True)
    args = parser.parse_args()
    try:
        if args.action == "freeze":
            manifest = validate_manifest(args.manifest, freeze=True)
            target = Path(args.output)
            if target.exists():
                raise ValueError("I never overwrite a frozen manifest")
            write_json(target, manifest)
            print(json.dumps({"manifest": str(target.resolve()), "sha256": digest(target), "cost": cost(manifest)}))
        elif args.action in ("plan", "dry-run"):
            manifest = validate_manifest(args.manifest)
            print(json.dumps({"mode": args.action, "network_calls": 0, "manifest_sha256": digest(args.manifest),
                              "launch_command": launch_command(args.manifest, args.approval), "cost": cost(manifest),
                              "account_check": "balance, Auto-Pay off and zero pods rechecked live only after approval",
                              "runner_command": manifest["command"], "pilot": "first five prompts, enforced by frozen batch"}, indent=2))
        elif args.action == "launch":
            launch(args.manifest, args.approval)
        elif args.action == "watch":
            return Watchdog(args.state_dir, Provider()).run()
        elif args.action == "pod-run":
            return pod_run(args.manifest, args.created_at)
    except Exception as error:
        if args.action == "pod-run":
            try:
                manifest = read_json(args.manifest)
                directory = Path("/workspace/results/chat-steering") / manifest["run_id"]
                write_json(directory / "FAILED.json", {"time_utc": now(), "error_type": type(error).__name__,
                                                       "stage": "pod_wrapper"})
            except Exception:
                pass
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
