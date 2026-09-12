"""I allocate only my second H100 and prime its isolated cached model setup.

I preserve the existing pod and its funded window. I run no experiment queue.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from replication.cloud.agent_steering.control import Provider, connection_command, now, timestamp, write_json

PROTECTED = "nz1bypfsiv62sc"


def emit(stage, **fields):
    print(json.dumps({"time": now(), "stage": stage, **fields}), flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, *, timeout=30, input=None):
    return subprocess.run(command, input=input, stdin=subprocess.DEVNULL if input is None else None,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True)


def verify_endpoint(row, pod_id, run_id, connection):
    if (row.get("id") != pod_id or row.get("name") != run_id
            or row.get("desiredStatus") != "RUNNING"
            or row.get("publicIp") != connection["host"]
            or int((row.get("portMappings") or {}).get("22", -1)) != connection["port"]):
        raise ValueError("My SSH endpoint no longer matches the owned live provider allocation")


def terminate_owned(provider, pod_id):
    if not pod_id or pod_id == PROTECTED:
        raise ValueError("I cannot terminate the protected pod")
    for _ in range(12):
        provider.request("DELETE", "/pods/" + pod_id)
        status, _ = provider.request("GET", "/pods/" + pod_id)
        if status == 404:
            return True
        time.sleep(5)
    return False


def reconcile_creation(provider, name, state):
    for attempt in range(24):
        try:
            matches = [x for x in provider.pods() if x.get("name") == name]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                break
        except Exception:
            pass
        time.sleep(5)
    write_json(state / "NEEDS_ATTENTION.json", {"reason": "creation_unconfirmed", "name": name,
                                               "time": now(), "second_creation_attempted": False})
    raise RuntimeError("Creation remains unconfirmed; reconcile this exact name before any retry")


def main(packet):
    packet = Path(packet).resolve()
    plan = json.loads((packet / "plan.json").read_text())
    if plan["purpose"] != "prime_only" or plan["protected_pod_id"] != PROTECTED:
        raise ValueError("I require my setup-only scope and protected first pod")
    if plan["allowance_usd"] != 3 or plan["aggregate_cap_usd"] != 20:
        raise ValueError("I preserve the frozen initial three-dollar allowance and original aggregate cap")
    prior_spend = float(plan.get("prior_sidecar_spend_usd", 0))
    if not 0 <= prior_spend < plan["allowance_usd"] - 0.12:
        raise ValueError("I require a positive remaining cumulative setup allowance")
    if plan["remote_root"] != "/workspace/parallel-lanes/" + plan["run_id"]:
        raise ValueError("I require a dedicated remote root")
    for entry in plan["files"]:
        path = packet / "source" / entry["path"]
        if sha(path) != entry["sha256"]:
            raise ValueError("A frozen source changed: " + entry["path"])
    if plan["run_id"] not in Path(plan["execution_record"]).read_text():
        raise ValueError("I require the recorded launch before allocation")
    if plan["protected_prior_projected_total_usd"] + plan["allowance_usd"] > plan["aggregate_cap_usd"]:
        raise ValueError("My full two-pod reservation exceeds the aggregate cap")
    state = packet / "state"
    state.mkdir(exist_ok=True)
    with (state / "launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (state / "creation-intent.json").exists():
            raise ValueError("I must reconcile the previous creation intent before retrying")
        p = Provider()
        pods = p.pods()
        if len(pods) != 1 or pods[0]["id"] != PROTECTED or pods[0]["desiredStatus"] != "RUNNING":
            raise ValueError("I require exactly the known running peer and no unaccounted allocation")
        account = p.account(plan["account_check"])
        quote = p.quote_and_volume({"data_center": "EU-FR-1", "network_volume_id": "i0cptrlcoz"})
        peer_future = max(0, plan["protected_peer_deadline_unix"] - time.time()) / 3600 * plan["protected_peer_rate_usd_h"]
        required = plan["allowance_usd"] - prior_spend + plan["credit_reserve_usd"] + peer_future
        check = {"time": now(), "account": account, "quote": quote, "protected_remaining_usd": peer_future,
                 "required_balance_usd": required, "ready": account["balance_usd"] >= required,
                 "protected_pod_id": PROTECTED, "allowance_usd": plan["allowance_usd"],
                 "prior_sidecar_spend_usd": prior_spend}
        write_json(state / "preflight.json", check)
        if account["autopay_enabled"] or quote["rate_usd_h"] > plan["gpu_rate_usd_h"]:
            raise ValueError("Auto-Pay or the quote differs from the declared setup")
        if not check["ready"]:
            emit("awaiting_funds", balance=account["balance_usd"], required=required)
            return 4
        spec = {"name": plan["run_id"], "cloudType": "SECURE", "gpuTypeIds": ["NVIDIA H100 80GB HBM3"],
                "gpuCount": 1, "dataCenterIds": ["EU-FR-1"], "imageName": plan["image"],
                "containerDiskInGb": 30, "volumeInGb": 0, "networkVolumeId": "i0cptrlcoz",
                "volumeMountPath": "/workspace", "ports": ["22/tcp"], "supportPublicIp": True,
                "interruptible": False, "env": {"PUBLIC_KEY": (Path.home() / ".ssh/id_ed25519.pub").read_text().strip()}}
        created = now()
        write_json(state / "creation-intent.json", {"created_at": created, "name": plan["run_id"], "plan_sha256": sha(packet / "plan.json")})
        pod = None
        try:
            status, candidate = p.request("POST", "/pods", spec)
            if status in (200, 201) and isinstance(candidate, dict) and candidate.get("id"):
                pod = candidate
            else:
                pod = reconcile_creation(p, plan["run_id"], state)
        except Exception:
            if pod is None:
                pod = reconcile_creation(p, plan["run_id"], state)
        pod_id = pod["id"]
        if pod_id == PROTECTED:
            raise ValueError("Provider returned the protected pod")
        write_json(state / "pod.json", {"id": pod_id, "created_at": created, "costPerHr": pod.get("costPerHr"), "remote_root": plan["remote_root"]})
        try:
            identity = {"schema_version": 1, "identity_source": "provider_api_ssh_binding",
                        "expected_pod_id": pod_id, "run_id": plan["run_id"], "identity_nonce": secrets.token_hex(32)}
            identity_path = "/tmp/mats-prime-" + plan["run_id"] + "-identity.json"
            config = {"version": 1, "scope": "parallel_h100_standby", "run_id": plan["run_id"], "pod_id": pod_id,
                      "pod_created_at": created, "remote_root": plan["remote_root"],
                      "connection_file": str(state / "connection.json"), "local_mirror": str(packet / "mirror"),
                      "allowance_usd": 3, "incremental_rate_usd_h": plan["incremental_rate_usd_h"],
                      "prior_sidecar_spend_usd": prior_spend,
                      "identity_marker_path": identity_path, "identity_nonce": identity["identity_nonce"],
                      "identity_source": identity["identity_source"],
                      "credit_reserve_usd": 2, "protected_peer_pod_id": PROTECTED,
                      "protected_peer_deadline_unix": plan["protected_peer_deadline_unix"],
                      "protected_peer_rate_usd_h": plan["protected_peer_rate_usd_h"], "aggregate_cap_usd": 20,
                      "protected_prior_projected_total_usd": plan["protected_prior_projected_total_usd"],
                      "account_check": plan["account_check"], "startup_deadline_unix": timestamp(created) + 600}
            write_json(state / "supervisor-config.json", config)
            command = [sys.executable, "-B", str(packet / "source/replication/cloud/parallel_h100/standby_supervisor.py"),
                       "--config", str(state / "supervisor-config.json"), "--state-dir", str(state / "supervisor")]
            if shutil.which("caffeinate"):
                command = ["caffeinate", "-i"] + command
            with (state / "supervisor.log").open("ab", buffering=0) as log:
                guardian = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True, close_fds=True)
            write_json(state / "supervisor-launch.json", {"pid": guardian.pid, "argv": command, "time": now()})
            time.sleep(1)
            if guardian.poll() is not None:
                raise RuntimeError("My independent supervisor exited")
            emit("allocated", pod_id=pod_id, guardian_pid=guardian.pid, allowance_usd=3)
            connection = None
            for _ in range(90):
                status, row = p.request("GET", "/pods/" + pod_id)
                if status == 200 and row and row.get("publicIp") and (row.get("portMappings") or {}).get("22"):
                    connection = {"pod_id": pod_id, "host": row["publicIp"], "port": int(row["portMappings"]["22"]),
                                  "key": str(Path.home() / ".ssh/id_ed25519")}
                    break
                if guardian.poll() is not None:
                    raise RuntimeError("Supervisor exited while waiting for SSH")
                time.sleep(5)
            if connection is None:
                raise RuntimeError("No SSH endpoint was assigned")
            ssh = connection_command(connection) + ["root@" + connection["host"]]
            for attempt in range(30):
                try:
                    run(ssh + ["test -x /workspace/venv-probes/bin/python"], timeout=15)
                    break
                except (subprocess.SubprocessError, OSError):
                    if attempt == 29:
                        raise
                    time.sleep(3)
            status, bound_pod = p.request("GET", "/pods/" + pod_id)
            if status != 200:
                raise RuntimeError("I cannot verify my provider allocation before provisioning identity")
            verify_endpoint(bound_pod, pod_id, plan["run_id"], connection)
            marker_script = ("import json,os,sys; d=json.load(sys.stdin); "
                             "actual=os.environ.get('RUNPOD_POD_ID'); "
                             "assert not actual or actual==d['identity']['expected_pod_id']; "
                             "fd=os.open(d['path'],os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); "
                             "f=os.fdopen(fd,'w'); json.dump(d['identity'],f); f.close(); "
                             "print(json.dumps({'identity_source':'provider_api_ssh_binding','provider_environment_id_present':bool(actual)}))")
            marker_result = run(ssh + [shlex.join(["/workspace/venv-probes/bin/python", "-B", "-c", marker_script])],
                                input=json.dumps({"path": identity_path, "identity": identity}).encode())
            write_json(state / "identity-binding.json", {**identity, "marker_path": identity_path,
                       "endpoint": {"host": connection["host"], "port": connection["port"]},
                       "provider_verified_at": now(), **json.loads(marker_result.stdout)})
            remote = plan["remote_root"]
            setup = "set -e; test -x /workspace/venv-probes/bin/python; command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync); mkdir -p " + shlex.quote(remote + "/control") + " " + shlex.quote(remote + "/service")
            run(ssh + [setup], timeout=180)
            write_json(state / "connection.json", connection)
            run(["rsync", "-rltz", "--partial", "--timeout=30", "-e", shlex.join(connection_command(connection)),
                 str(packet / "source") + "/", "root@" + connection["host"] + ":" + remote + "/source/"], timeout=180)
            verifier = ("import hashlib,json,pathlib,sys; d=json.load(sys.stdin); root=pathlib.Path(d['root']); "
                        "bad=[x['path'] for x in d['files'] if hashlib.sha256((root/x['path']).read_bytes()).hexdigest()!=x['sha256']]; "
                        "print(json.dumps({'verified_files':len(d['files']),'mismatches':bad})); sys.exit(bool(bad))")
            verified = run(ssh + [shlex.join(["/workspace/venv-probes/bin/python", "-B", "-c", verifier])],
                           input=json.dumps({"root": remote + "/source", "files": plan["files"]}).encode())
            write_json(state / "remote-source-verification.json", json.loads(verified.stdout))
            worker = {"schema_version": 1, "purpose": "prime_only", "execution_approved": True,
                      "run_id": plan["run_id"], "expected_pod_id": pod_id, "protected_pod_id": PROTECTED,
                      "identity_marker_path": identity_path, "identity_nonce": identity["identity_nonce"],
                      "identity_source": identity["identity_source"],
                      "isolated_root": remote, "lease_path": remote + "/control/lease.json",
                      "model_id": "openai/gpt-oss-20b", "model_revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
                      "cache_dir": "/workspace/hf", "hf_home": "/workspace/hf/home", "kernel_cache_dir": "/workspace/hf/home/hub",
                      "attn_implementation": "kernels-community/vllm-flash-attn3", "startup_timeout_seconds": 600, "poll_seconds": 1}
            write_json(state / "worker-config.json", worker)
            run(ssh + ["cat > " + shlex.quote(remote + "/worker-config.json")], input=json.dumps(worker).encode())
            for _ in range(30):
                try:
                    lease=json.loads(run(ssh + ["cat " + shlex.quote(worker["lease_path"])]).stdout)
                    if lease.get("expected_pod_id") == pod_id and lease.get("approved") and lease["lease_expires_unix"] > time.time()+30:
                        break
                except (subprocess.SubprocessError, ValueError, KeyError):
                    pass
                time.sleep(2)
            else:
                raise RuntimeError("No fresh financial lease exists for model loading")
            argv = ["/workspace/venv-probes/bin/python", "-B", "-s", remote + "/source/replication/cloud/parallel_h100/prime_worker.py", "--config", remote + "/worker-config.json"]
            detached = "nohup " + shlex.join(argv) + " > " + shlex.quote(remote + "/service/console.log") + " 2>&1 < /dev/null &"
            run(ssh + [detached])
            write_json(state / "worker-launch.json", {"argv": argv, "time": now()})
            emit("priming", pod_id=pod_id, remote_root=remote)
            return 0
        except BaseException as error:
            (state / "supervisor" / "STOP").parent.mkdir(exist_ok=True)
            (state / "supervisor" / "STOP").touch()
            verified = terminate_owned(p, pod_id)
            write_json(state / "launch-failure.json", {"error_type": type(error).__name__, "owned_pod_id": pod_id,
                                                       "delete_verified": verified, "time": now()})
            emit("setup_failed", error_type=type(error).__name__, owned_pod_deleted=verified)
            return 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True)
    args = ap.parse_args()
    raise SystemExit(main(args.packet))
