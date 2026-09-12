"""I dispatch one frozen follow-up after bridge checks, without changing GPU allocation."""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import statistics
import subprocess
import time

PACKAGE = "replication.cloud.agent_steering."
BRIDGE_ARMS = {"none", "zero", "role_a16", "reverse_a16", "random_0_a16", "random_1_a16", "random_2_a16"}
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]+")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(label + " must be finite and positive")
    return value


def absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("I require an absolute path without parent traversal")
    return value


def frozen_table(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("I require an explicit nonempty frozen-file table")
    result = {}
    for row in rows:
        path = absolute(row["path"])
        digest = row["sha256"]
        if path in result or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Frozen paths must be unique and SHA-256 bound")
        result[path] = digest
    return result


def verify_files(rows):
    table = frozen_table(rows)
    for path, digest in table.items():
        if Path(path).is_symlink() or sha(path) != digest:
            raise ValueError("A frozen file changed: " + path)
    return table


def validate_launch(launch, module, remote=False):
    for field in ("cwd", "logpath"):
        absolute(launch[field])
    argv = launch["argv"]
    expected_length = 8 if module == "service_guardian" else 6
    if (not isinstance(argv, list) or len(argv) != expected_length
            or argv[1:5] != ["-u", "-m", PACKAGE + module, "--config"]):
        raise ValueError("I permit only the exact registered module entrypoint")
    absolute(argv[0]); absolute(argv[5])
    if remote:
        if argv[0] != "/workspace/venv-probes/bin/python" or not launch["cwd"].startswith("/workspace/"):
            raise ValueError("I require the existing cached interpreter and frozen remote source root")
        if argv[6] != "--state-dir":
            raise ValueError("The guardian needs a separate state directory")
        absolute(argv[7])
    return launch


def validate_config(config):
    if config.get("version") != 1 or config.get("execution_approved") is not True:
        raise ValueError("I require approval for this exact follow-up registration")
    for field in ("expected_pod_id", "job_id"):
        if not isinstance(config.get(field), str) or not IDENTIFIER.fullmatch(config[field]):
            raise ValueError("The follow-up must identify one pod and one job")
    for field in ("current_run_dir", "lease_path", "connection_file", "registration_file", "local_results"):
        absolute(config[field])
    if type(config.get("current_runner_pid")) is not int or config["current_runner_pid"] <= 1:
        raise ValueError("I require the actual preceding runner PID")
    if type(config.get("predecessor_pid")) is not int or config["predecessor_pid"] <= 1:
        raise ValueError("I require the actual preceding remote model-worker PID")
    positive(config["all_in_rate_usd_h"], "All-in hourly rate")
    if not 0 < positive(config.get("ready_timeout_seconds", 900), "Readiness timeout") <= 1800:
        raise ValueError("Readiness waiting must be bounded by 1800 seconds")
    if config["all_in_rate_usd_h"] < 3.49 + 280 / 730:
        raise ValueError("The rate must include the H100 and retained-volume quote")
    remote_results = "/workspace/results/agent-steering/" + config["job_id"]
    if config.get("remote_results") != remote_results:
        raise ValueError("The remote result directory must match the exact job ID")
    local_launch = validate_launch(config["local_launch"], "new_page_runner")
    remote_launch = validate_launch(config["remote_launch"], "service_guardian", remote=True)
    local = frozen_table(config["local_frozen_files"])
    remote = frozen_table(config["remote_frozen_files"])
    required_local = [config["registration_file"], config["connection_file"], local_launch["argv"][5],
                      str(Path(local_launch["cwd"]) / "replication/cloud/agent_steering/new_page_runner.py")]
    required_remote = [remote_launch["argv"][5],
                       str(Path(remote_launch["cwd"]) / "replication/cloud/agent_steering/service_guardian.py")]
    if not set(required_local).issubset(local) or not set(required_remote).issubset(remote):
        raise ValueError("The commands, registration, transport and entrypoint sources must be frozen")
    service_prefix = "/workspace/session-control/" + config["expected_pod_id"] + "/"
    registration = config["remote_registration_path"]
    absolute(registration)
    if not registration.startswith(service_prefix) or not registration.endswith("/jobs/" + config["job_id"] + ".json"):
        raise ValueError("The registration must target this pod's own service job queue")
    if not remote_launch["argv"][7].startswith(service_prefix):
        raise ValueError("The guardian state must belong to this pod's service")
    if Path(config["local_results"]).resolve() == Path(config["current_run_dir"]).resolve():
        raise ValueError("The follow-up needs a new local output directory")
    if config.get("sync_sources_file"):
        absolute(config["sync_sources_file"]); absolute(config["local_sync_results"])
        if config["local_sync_results"] == config["local_results"]:
            raise ValueError("The worker artifact mirror must be separate from local runner output")
    return config


def prerequisite(directory):
    """I use completion and engineering checks, without selecting behavioral outcomes."""
    directory = Path(directory)
    if not (directory / "FINISHED.json").exists():
        return None
    final = read_json(directory / "FINISHED.json")
    status = read_json(directory / "status.json")
    summary = read_json(directory / "summary.json")
    for item in (final, status):
        if item.get("status") != "completed" or item.get("recorded_episodes") != 35 or item.get("planned_episodes") != 35:
            raise ValueError("The bridge did not complete its entire 35-episode queue")
    if summary.get("recorded_episodes") != 35 or summary.get("total_planned_episodes") != 35:
        raise ValueError("The bridge summary does not account for all 35 episodes")
    rows = read_json(directory / "episode-index.json")
    pairs = [(r["case_id"], r["arm_id"]) for r in rows]
    case_ids = {case for case, _ in pairs}
    if len(rows) != 35 or len(set(pairs)) != 35 or len(case_ids) != 5 or set(pairs) != {(c, a) for c in case_ids for a in BRIDGE_ARMS}:
        raise ValueError("Timing must cover every arm on each of the five bridge cases")
    elapsed = [positive(row["elapsed_s"], "Episode elapsed seconds") for row in rows]
    pilot = read_json(directory / "engineering-pilot/PASSED.json")
    samples = pilot.get("samples", [])
    required_checks = {"zero_token_identity", "zero_edits", "mask_edit_counts", "matched_vector_norm", "finite_downstream_readout", "no_timeout"}
    if (pilot.get("passed") is not True or len(samples) != 5 or {s.get("case_id") for s in samples} != case_ids
            or any(any(s.get(k) is not True for k in required_checks) for s in samples)):
        raise ValueError("The exact five-case engineering gate must have passed")
    identity = read_json(directory / "full-zero-identity.json")
    comparisons = identity.get("comparisons", [])
    if (identity.get("pairs_available") != 5 or identity.get("planned_pairs") != 5
            or identity.get("unexplained_identity_failure") is not False or len(comparisons) != 5
            or {p.get("case_id") for p in comparisons} != case_ids):
        raise ValueError("I require all five full none/zero pairs without unexplained identity failure")
    for pair in comparisons:
        generations = pair.get("generations", [])
        comparable = [g for g in generations if g.get("both_reached") is True and g.get("prompts_identical") is True]
        if pair.get("unexplained_identity_failure") is not False or not comparable:
            raise ValueError("Each none/zero pair needs actual comparable generations")
        for row in comparable:
            if row.get("tokens_identical") is not True and not (row.get("timing_censored") is True
                    and row.get("shared_prefix_identical") is True and row.get("censored_length_difference") is True):
                raise ValueError("An identical prompt has unexplained token divergence")
    mean = statistics.mean(elapsed)
    return {"bridge_episodes": 35, "mean_episode_seconds": mean,
            "followup_episodes": 40, "safety_multiplier": 1.3, "setup_and_pilot_seconds": 600,
            "required_seconds": mean * 40 * 1.3 + 600,
            "timing_source_sha256": sha(directory / "episode-index.json"),
            "outcome_selection": False}


def budget_gate(lease, pod_id, required_seconds, rate, now):
    if (lease.get("schema_version") != 1 or lease.get("approved") is not True
            or lease.get("expected_pod_id") != pod_id or not lease.get("lease_id")):
        raise ValueError("The lease must authorize this exact pod")
    issued, expires, shutdown, budget = [positive(lease.get(k), k) for k in
        ("issued_unix", "lease_expires_unix", "shutdown_at_unix", "budget_remaining_usd")]
    if not 0 < expires - issued <= 180.001 or issued > now + 1 or now >= min(expires, shutdown):
        raise ValueError("The local financial lease is stale or invalid")
    remaining = budget - max(0, now-issued) * rate / 3600
    cost = required_seconds * rate / 3600
    if required_seconds > shutdown-now or cost > remaining:
        raise ValueError("The timing-only follow-up estimate exceeds the current financial lease")
    return {"required_seconds": required_seconds, "estimated_cost_usd": cost,
            "budget_remaining_after_elapsed_usd": remaining, "financial_seconds_remaining": shutdown-now,
            "lease_id": lease["lease_id"], "lease_expires_unix": expires, "checked_unix": now}


# I send only this fixed bootstrap as shell code. All configuration and registration bytes use stdin.
REMOTE_BOOTSTRAP = r'''
import base64, hashlib, json, math, os, pathlib, subprocess, sys, time
p = json.load(sys.stdin)
observed = os.environ.get("RUNPOD_POD_ID")
if observed and observed != p["expected_pod_id"]: raise ValueError("Wrong pod")
for row in p["files"]:
    f = pathlib.Path(row["path"])
    if f.is_symlink() or hashlib.sha256(f.read_bytes()).hexdigest() != row["sha256"]: raise ValueError("Frozen remote file changed")
frozen = {r["path"] for r in p["files"]}
if any(str(f) not in frozen for f in pathlib.Path(p["cwd"]).rglob("*.py")): raise ValueError("Unfrozen Python source in remote root")
service = json.loads(pathlib.Path(p["argv"][5]).read_bytes())
if service["expected_pod_id"] != p["expected_pod_id"] or service.get("execution_approved") is not True: raise ValueError("Wrong service")
jobpath = pathlib.Path(p["registration_path"])
if jobpath != pathlib.Path(service["out_dir"]) / "jobs" / (p["job_id"]+".json"): raise ValueError("Wrong job queue")
if service.get("persistent_budget_lease_required") is not True or service.get("predecessor_pid") != p["predecessor_pid"]: raise ValueError("Wrong predecessor/lease policy")
lease = json.loads(pathlib.Path(service["lease_path"]).read_bytes())
now = time.time()
if any(type(lease.get(k)) not in (int,float) or not math.isfinite(lease[k]) for k in ("lease_expires_unix","issued_unix","shutdown_at_unix","budget_remaining_usd")): raise ValueError("Invalid lease numbers")
if (service["lease_path"] != "/workspace/session-control/"+p["expected_pod_id"]+"/lease.json"
    or lease.get("schema_version") != 1 or lease.get("expected_pod_id") != p["expected_pod_id"] or lease.get("approved") is not True
    or not 0 < lease["lease_expires_unix"]-lease["issued_unix"] <= 180.001
    or lease["issued_unix"] > now+1 or now >= min(lease["lease_expires_unix"],lease["shutdown_at_unix"])
    or lease["budget_remaining_usd"] <= 0): raise ValueError("No fresh remote financial lease")
if (p["required_seconds"] > min(lease["shutdown_at_unix"],p["work_deadline_unix"])-now
    or p["required_seconds"]*p["rate_usd_h"]/3600 > lease["budget_remaining_usd"]-max(0,now-lease["issued_unix"])*p["rate_usd_h"]/3600): raise ValueError("Insufficient renewed remote budget")
raw = base64.b64decode(p["registration_base64"], validate=True)
if hashlib.sha256(raw).hexdigest() != p["registration_sha256"]: raise ValueError("Registration bytes changed")
job = json.loads(raw)
if job.get("job_id") != p["job_id"] or job.get("out_dir") != p["remote_results"] or job.get("execution_approved") is not True: raise ValueError("Wrong job")
if pathlib.Path(p["remote_results"]).exists() or jobpath.exists(): raise ValueError("Job already exists")
state = pathlib.Path(p["argv"][7]); state.mkdir(parents=True, exist_ok=False)
jobpath.parent.mkdir(parents=True, exist_ok=True)
temporary = jobpath.with_suffix(".uploading")
with temporary.open("xb") as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
os.replace(temporary, jobpath)
log = pathlib.Path(p["logpath"]); log.parent.mkdir(parents=True, exist_ok=True)
env = dict(os.environ)
for key in list(env):
    if "API_KEY" in key or key.endswith("HF_TOKEN") or key in ("HUGGING_FACE_HUB_TOKEN","HUGGINGFACE_TOKEN","HF_HUB_CACHE","PYTHONPATH","PYTHONHOME"):
        env.pop(key,None)
env.update(HF_HOME="/workspace/hf/home",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",PYTHONUNBUFFERED="1")
with log.open("xb") as output:
    child = subprocess.Popen(p["argv"], cwd=p["cwd"], stdin=subprocess.DEVNULL,
                             stdout=output, stderr=subprocess.STDOUT, start_new_session=True, env=env)
receipt = {"guardian_pid":child.pid,"job_id":p["job_id"],"expected_pod_id":p["expected_pod_id"],"registration_sha256":p["registration_sha256"],"allocation_action":"none"}
(state / "DISPATCHED.json").write_text(json.dumps(receipt)+"\n")
print(json.dumps(receipt), flush=True)
'''


REMOTE_STATUS = r'''
import json, os, pathlib, sys
p = json.load(sys.stdin)
observed = os.environ.get("RUNPOD_POD_ID")
if observed and observed != p["expected_pod_id"]: raise ValueError("Wrong pod")
def read(path):
    f = pathlib.Path(path)
    if not f.exists(): return None
    if f.is_symlink() or f.stat().st_size > 1048576: raise ValueError("Invalid readiness receipt")
    return json.loads(f.read_bytes())
service = pathlib.Path(p["registration_path"]).parent.parent
guardian = pathlib.Path(p["argv"][7])
job = pathlib.Path(p["remote_results"])
print(json.dumps({"guardian":read(guardian/"GUARDIAN-READY.json"),"service":read(service/"SERVICE-READY.json"),
 "job":read(job/"READY.json"),"terminals":{str(f):read(f) for f in [guardian/"GUARDIAN-EXIT.json",service/"SERVICE-EXIT.json",job/"FAILED.json",job/"EXIT.json"]}}))
'''


def ssh_json(connection, payload, program):
    if not re.fullmatch(r"[A-Za-z0-9.:-]+", connection["host"]):
        raise ValueError("Invalid frozen SSH host")
    port = int(connection["port"])
    if not 0 < port <= 65535:
        raise ValueError("Invalid SSH port")
    command = ["ssh", "-i", absolute(connection["key"]), "-p", str(port),
               "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
               "root@" + connection["host"], "python3 -c " + shlex.quote(program)]
    result = subprocess.run(command, input=json.dumps(payload, allow_nan=False).encode(), capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError("Frozen remote dispatch failed: " + result.stderr.decode(errors="replace")[:500])
    return json.loads(result.stdout)


def remote_launch(connection, payload):
    return ssh_json(connection, payload, REMOTE_BOOTSTRAP)


def remote_status(connection, payload):
    return ssh_json(connection, payload, REMOTE_STATUS)


def process_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def register_sync(config):
    if not config.get("sync_sources_file"):
        return
    path = Path(config["sync_sources_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".dispatch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = read_json(path) if path.exists() else config.get("sync_sources_initial")
        if not isinstance(rows, list) or not rows:
            raise ValueError("I require the existing artifact mirror list before appending a target")
        for row in rows:
            if not re.fullmatch(r"/workspace/results/agent-steering/[A-Za-z0-9_-]+", row.get("remote", "")):
                raise ValueError("Invalid existing worker artifact mirror")
            absolute(row["local"])
        new = {"remote": config["remote_results"], "local": config["local_sync_results"]}
        if any(row["remote"] == new["remote"] and row != new for row in rows):
            raise ValueError("This job already has a different artifact mirror")
        if new not in rows:
            write_json(path, [*rows, new])


class Dispatcher:
    def __init__(self, config_path, state_dir, *, remote=remote_launch, status=remote_status, spawn=subprocess.Popen,
                 exists=process_alive, now=time.time, sleep=time.sleep):
        self.config_path = Path(config_path).resolve()
        self.config_sha256 = sha(self.config_path)
        self.config = validate_config(read_json(self.config_path))
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.remote, self.spawn, self.exists, self.now, self.sleep = remote, spawn, exists, now, sleep
        self.status = status
        self.stop_requested = False

    def check_stop(self):
        if self.stop_requested or (self.state_dir / "STOP").exists():
            raise ValueError("Explicit dispatcher STOP; GPU allocation is retained")

    def check_packet(self):
        c = self.config
        if sha(self.config_path) != self.config_sha256:
            raise ValueError("The frozen dispatcher configuration changed while waiting")
        files = verify_files(c["local_frozen_files"])
        if any(str(path) not in files for path in Path(c["local_launch"]["cwd"]).rglob("*.py")):
            raise ValueError("Every Python source within the local frozen root must be declared")
        local = read_json(c["local_launch"]["argv"][5])
        if (local.get("execution_approved") is not True or local.get("out_dir") != c["local_results"]
                or local.get("remote_results") != c["remote_results"] or local.get("expected_pod_id") != c["expected_pod_id"]
                or local.get("registration_file") != c["registration_file"] or local.get("lease_path") != c["lease_path"]
                or local.get("connection_file") != c["connection_file"]):
            raise ValueError("The follow-up runner configuration differs from this registration")
        for field in ("plan_file", "diagnostics_file", "model_free_validation", "registration_file"):
            if local.get(field) not in files:
                raise ValueError("Every runner input must be frozen: " + field)
        for field, digest in (("plan_file", "plan_sha256"), ("diagnostics_file", "diagnostics_sha256"),
                              ("model_free_validation", "model_free_validation_sha256"), ("registration_file", "registration_sha256")):
            if files[local[field]] != local.get(digest):
                raise ValueError("The runner input hash is inconsistent: " + field)
        plan = read_json(local["plan_file"])
        if (plan.get("scope") != "new_page_descriptive_extension" or plan.get("page_count") != 5
                or plan.get("expected_episode_count") != 40 or len(plan.get("jobs", [])) != 40
                or plan.get("selected_bank_indices") != list(range(5))):
            raise ValueError("I dispatch only the exact first-five-page, 40-episode extension")
        for case in plan.get("cases", []):
            if files.get(case["fixture_path"]) != case["fixture_sha256"]:
                raise ValueError("Every selected page fixture must be frozen")
        registration = read_json(c["registration_file"])
        if (registration.get("schema_version") != 1 or registration.get("execution_approved") is not True
                or registration.get("job_id") != c["job_id"] or registration.get("out_dir") != c["remote_results"]
                or registration.get("plan_sha256") != local["plan_sha256"] or registration.get("arms") != plan.get("engine_arms")):
            raise ValueError("The exact persistent job registration differs from the frozen plan")
        if Path(c["local_results"]).exists():
            raise ValueError("The follow-up output directory already exists")
        return local

    def wait_ready(self, payload, remote_receipt, registration):
        deadline = self.now() + self.config.get("ready_timeout_seconds", 900)
        while self.now() < deadline:
            self.check_stop()
            budget_gate(read_json(self.config["lease_path"]), self.config["expected_pod_id"],
                        payload["required_seconds"], payload["rate_usd_h"], self.now())
            state = self.status(read_json(self.config["connection_file"]), payload)
            if any(value is not None for value in state.get("terminals", {}).values()):
                write_json(self.state_dir / "REMOTE-TERMINAL.json", state)
                raise ValueError("The guardian or service exited before follow-up readiness")
            if state.get("job") is not None:
                job, service, guardian = state["job"], state.get("service") or {}, state.get("guardian") or {}
                if (job.get("status") != "ready" or job.get("job_file_sha256") != payload["registration_sha256"]
                        or guardian.get("guardian_pid") != remote_receipt["guardian_pid"]
                        or guardian.get("expected_pod_id") != self.config["expected_pod_id"]
                        or guardian.get("config_sha256") != frozen_table(self.config["remote_frozen_files"])[payload["argv"][5]]
                        or service.get("status") != "ready" or type(service.get("pid")) is not int
                        or service["pid"] <= 1 or job.get("pid") != service["pid"] or guardian.get("worker_pid") != service["pid"]):
                    raise ValueError("The ready job/service/guardian identities do not match this launch")
                for key in ("directions_file_sha256", "probe_file_sha256"):
                    if not registration.get(key) or job.get("backend", {}).get(key) != registration[key] or service.get("backend", {}).get(key) != registration[key]:
                        raise ValueError("The ready model's assets differ from the frozen job")
                write_json(self.state_dir / "REMOTE-READY.json", state)
                return state
            write_json(self.state_dir / "heartbeat.json", {"stage": "waiting_for_registered_job_ready", "unix": self.now(),
                       "job_id": self.config["job_id"], "allocation_action": "none"})
            self.sleep(10)
        raise TimeoutError("The registered job did not become ready within the bounded handover window")

    def run(self):
        with (self.state_dir / "dispatcher.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                if (self.state_dir / "LAUNCHED.json").exists():
                    if read_json(self.state_dir / "LAUNCHED.json").get("config_sha256") != self.config_sha256:
                        raise ValueError("A different registration already used this dispatcher directory")
                    return 0
                if (self.state_dir / "LAUNCH-INTENT.json").exists():
                    raise ValueError("An uncertain earlier launch requires review; I never replay it automatically")
                while True:
                    self.check_stop()
                    local = self.check_packet()
                    estimate = prerequisite(self.config["current_run_dir"])
                    if estimate is not None:
                        break
                    if not self.exists(self.config["current_runner_pid"]):
                        raise ValueError("The preceding runner exited without a complete bridge receipt")
                    write_json(self.state_dir / "heartbeat.json", {"stage": "waiting_for_bridge35", "unix": self.now(),
                               "pid": os.getpid(), "job_id": self.config["job_id"], "allocation_action": "none"})
                    self.sleep(30)
                self.check_stop()
                budget = budget_gate(read_json(self.config["lease_path"]), self.config["expected_pod_id"],
                                     estimate["required_seconds"], self.config["all_in_rate_usd_h"], self.now())
                if positive(local["work_deadline_unix"], "Work deadline") - self.now() < estimate["required_seconds"]:
                    raise ValueError("The frozen follow-up work deadline is too short for the timing estimate")
                self.check_packet()
                self.check_stop()
                intent = {"config_sha256": self.config_sha256, "job_id": self.config["job_id"],
                          "expected_pod_id": self.config["expected_pod_id"], "estimate": estimate,
                          "budget": budget, "unix": self.now(), "allocation_action": "none"}
                write_json(self.state_dir / "LAUNCH-INTENT.json", intent)
                c = self.config
                raw = Path(c["registration_file"]).read_bytes()
                payload = {**c["remote_launch"], "files": c["remote_frozen_files"], "expected_pod_id": c["expected_pod_id"],
                           "job_id": c["job_id"], "remote_results": c["remote_results"],
                           "registration_path": c["remote_registration_path"], "registration_base64": base64.b64encode(raw).decode(),
                           "registration_sha256": hashlib.sha256(raw).hexdigest(), "predecessor_pid": c["predecessor_pid"],
                           "required_seconds": estimate["required_seconds"], "rate_usd_h": c["all_in_rate_usd_h"],
                           "work_deadline_unix": local["work_deadline_unix"]}
                remote_receipt = self.remote(read_json(c["connection_file"]), payload)
                write_json(self.state_dir / "REMOTE-LAUNCHED.json", remote_receipt)
                if (remote_receipt.get("job_id") != c["job_id"] or remote_receipt.get("expected_pod_id") != c["expected_pod_id"]
                        or remote_receipt.get("registration_sha256") != payload["registration_sha256"]
                        or type(remote_receipt.get("guardian_pid")) is not int or remote_receipt["guardian_pid"] <= 1):
                    raise ValueError("Remote dispatch receipt does not identify this registered job")
                register_sync(c)
                self.wait_ready(payload, remote_receipt, read_json(c["registration_file"]))
                self.check_stop()
                self.check_packet()
                budget_gate(read_json(c["lease_path"]), c["expected_pod_id"], estimate["required_seconds"], c["all_in_rate_usd_h"], self.now())
                if local["work_deadline_unix"] - self.now() < estimate["required_seconds"]:
                    raise ValueError("The frozen work deadline became too short during model readiness")
                launch = c["local_launch"]
                Path(launch["logpath"]).parent.mkdir(parents=True, exist_ok=True)
                with Path(launch["logpath"]).open("xb") as output:
                    child = self.spawn(launch["argv"], cwd=launch["cwd"], stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                write_json(self.state_dir / "LAUNCHED.json", {**intent, "remote": remote_receipt, "local_runner_pid": child.pid})
                write_json(self.state_dir / "heartbeat.json", {"stage": "launched", "unix": self.now(),
                           "job_id": c["job_id"], "local_runner_pid": child.pid, "allocation_action": "none"})
                return 0
            except Exception as error:
                write_json(self.state_dir / "BLOCKED.json", {"error": str(error), "unix": self.now(),
                           "job_id": self.config["job_id"], "allocation_action": "none",
                           "launch_uncertain": (self.state_dir / "LAUNCH-INTENT.json").exists()})
                write_json(self.state_dir / "heartbeat.json", {"stage": "blocked", "unix": self.now(),
                           "error": str(error), "allocation_action": "none"})
                return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-dir", required=True)
    args = parser.parse_args()
    dispatcher = Dispatcher(args.config, args.state_dir)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: setattr(dispatcher, "stop_requested", True))
    return dispatcher.run()


if __name__ == "__main__":
    raise SystemExit(main())
