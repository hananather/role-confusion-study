"""I write a new launch registration after the prior batch and service close; I never launch it."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shutil
import time
from prepare_inputs import PACKET, save, sha

def register(args):
    plan = json.loads((PACKET / "plan.json").read_text())
    if time.time() >= plan["launch_cutoff_unix"]: raise ValueError("The registered launch cutoff has passed")
    closure = json.loads(args.closure_receipt.read_text())
    if (not all(closure.get(k) is True for k in ("passed", "audit_passed", "sync_verified", "item4_completed", "previous_service_exit_verified"))
            or closure.get("previous_run_id") != "agent-newpages-20260912T025000Z" or closure.get("completed_jobs") != 40
            or closure.get("previous_worker_pid") != args.predecessor_pid or args.predecessor_pid <= 1):
        raise ValueError("I require the verified prior40 closure and exact previous worker identity")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.job_id): raise ValueError("Invalid frozen job ID")
    if not str(args.remote_root).startswith("/workspace/continuations/") or '..' in args.remote_root.parts:
        raise ValueError("The successor needs a new isolated remote continuation root")
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.closure_receipt, out / "closure-receipt.json")
    series = PACKET.parents[1]
    active = json.loads((series / "newpage-001/launch-packet/local-config.json").read_text())
    service_out = "/workspace/session-control/nz1bypfsiv62sc/queue-a-model-service"
    remote_results = "/workspace/results/agent-steering/" + args.job_id
    service = {"execution_approved": True, "persistent_budget_lease_required": True, "expected_pod_id": "nz1bypfsiv62sc",
        "lease_path": "/workspace/session-control/nz1bypfsiv62sc/lease.json",
        "out_dir": service_out, "job_results_root": "/workspace/results/agent-steering", "predecessor_pid": args.predecessor_pid,
        "predecessor_wait_seconds": 60, "model_id": "openai/gpt-oss-20b", "model_revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
        "cache_dir": "/workspace/hf", "kernel_cache_dir": "/workspace/hf/home/hub",
        "attn_implementation": "kernels-community/vllm-flash-attn3", "max_context_tokens": 65536,
        "directions_file": str(args.remote_root / "inputs/block11.npz"), "directions_file_sha256": plan["directions_sha256"],
        "probe_file": str(args.remote_root / "inputs/probes.npz"), "probe_file_sha256": plan["probe_sha256"], "arms": [plan["none_arm"]]}
    save(out / "service-config.json", service)
    job = {"schema_version": 1, "job_id": args.job_id, "execution_approved": True, "out_dir": remote_results,
        "plan_sha256": sha(PACKET / "plan.json"), **{k: service[k] for k in ("directions_file", "directions_file_sha256", "probe_file", "probe_file_sha256", "arms")}}
    save(out / "registration.json", job)
    local = {"execution_approved": True, "expected_pod_id": "nz1bypfsiv62sc", "plan_file": str(PACKET / "plan.json"),
        "plan_sha256": sha(PACKET / "plan.json"), "registration_file": str(out / "registration.json"),
        "registration_sha256": sha(out / "registration.json"), "closure_receipt_file": str(out / "closure-receipt.json"),
        "closure_receipt_sha256": sha(out / "closure-receipt.json"), "remote_results": remote_results,
        "out_dir": str(args.local_results.resolve()), "connection_file": active["connection_file"],
        "lease_file": active["lease_path"], "sandbox_image_id": active["sandbox_image_id"], "frozen_files": str(out / "frozen-files.json")}
    local["sync_sources_file"] = str(series / "persistent-session/state/sync-sources.json")
    local["local_sync_results"] = str(args.local_results.resolve().parent / "remote-results")
    files = {str(p): sha(p) for p in sorted(PACKET.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    for p in out.glob('*.json'): files[str(p)] = sha(p)
    files[active["connection_file"]] = sha(active["connection_file"])
    save(out / "frozen-files.json", files)
    local["frozen_files_sha256"] = sha(out / "frozen-files.json")
    save(out / "local-config.json", local)
    save(out / "launch.json", {"launched": False, "remote_root": str(args.remote_root), "service_out": service_out,
        "local_config_sha256": sha(out / "local-config.json"), "service_config_sha256": sha(out / "service-config.json"),
        "frozen_files_sha256": sha(out / "frozen-files.json"),
        "remote_service_config": str(args.remote_root / "registration/service-config.json"),
        "remote_job_file": service_out + "/jobs/" + args.job_id + ".json", "remote_results": remote_results,
        "guardian_argv": ["/workspace/venv-probes/bin/python", "-u", "-m", "replication.cloud.agent_steering.service_guardian",
            "--config", str(args.remote_root / "registration/service-config.json"), "--state-dir", service_out + "/guardian"],
        "guardian_cwd": str(args.remote_root / "source"),
        "local_argv": [str(PACKET.parents[3] / '.venv/bin/python'), '-u', str(PACKET / 'code/run_queue.py'), '--config', str(out / 'local-config.json')],
        "requires": "Upload exact source/inputs/config bytes to the new root, upload job atomically to its new service queue, start guardian, verify exact READY identities, then start local runner. Never signal the former process from this registration."})
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--closure-receipt', required=True, type=Path)
    ap.add_argument('--predecessor-pid', required=True, type=int); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--remote-root', required=True, type=Path); ap.add_argument('--job-id', required=True)
    ap.add_argument('--local-results', required=True, type=Path)
    print(register(ap.parse_args()))
