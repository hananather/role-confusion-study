"""I run the registered items sequentially only after the prior batch closes and is audited."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

PACKET = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PACKET / "source"))
from replication.cloud.agent_steering.episode import run_episode, save_json
from replication.cloud.agent_steering.local_backend import RemoteBackend
from replication.cloud.agent_steering.persistent_worker import validate_lease
from replication.cloud.agent_steering.runner import stamp, summarize_episode

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())

def verify_packet(config):
    if config.get("execution_approved") is not True: raise ValueError("The exact successor queue requires separate registration")
    if sha(config["frozen_files"]) != config["frozen_files_sha256"]: raise ValueError("The frozen file table changed")
    for path, digest in read(config["frozen_files"]).items():
        if sha(path) != digest: raise ValueError("Frozen input/source changed: " + path)
    for name in ("plan", "registration", "closure_receipt"):
        if sha(config[name + "_file"]) != config[name + "_sha256"]: raise ValueError("Frozen registration mismatch: " + name)
    plan, registration, closure = (read(config[k + "_file"]) for k in ("plan", "registration", "closure_receipt"))
    if (plan["scope"] != "gpu_a_queue_items_5_and_6" or len(plan["attribution"]) != 10 or len(plan["standard_cases"]) != 5
            or plan["item_order"] != [5, 6] or config["expected_pod_id"] != plan["expected_pod_id"]):
        raise ValueError("Invalid fixed GPU-A queue")
    if not all(closure.get(k) is True for k in ("passed", "audit_passed", "sync_verified", "item4_completed", "previous_service_exit_verified")):
        raise ValueError("The prior40 and old service must close, audit and sync before this successor starts")
    if closure.get("previous_run_id") != "agent-newpages-20260912T025000Z" or closure.get("completed_jobs") != 40:
        raise ValueError("The closure receipt must identify all40 preceding jobs")
    if (registration.get("execution_approved") is not True or registration.get("plan_sha256") != config["plan_sha256"]
            or registration["arms"] != [plan["none_arm"]] or registration["out_dir"] != config["remote_results"]):
        raise ValueError("Job registration does not bind this unsteered queue")
    return plan, registration

def remaining(config, plan):
    now = time.time()
    lease = validate_lease(read(config["lease_file"]), config["expected_pod_id"], now)
    return min(lease["shutdown_at_unix"], plan["session_closeout_unix"]) - now

def validate_attribution(generation, stats, sample):
    if (stats.get("purpose") != "attribution" or stats.get("max_new_tokens") != 200 or stats.get("seed") != sample["seed"]
            or stats.get("prompt_sha256") != sample["prompt_sha256"] or stats.get("edited_positions") != 0
            or stats.get("hook_calls") != 0 or generation.prompt_tokens != sample["prompt_tokens"]
            or generation.generated_tokens != len(generation.token_ids) or len(generation.token_ids) > 200):
        raise ValueError("Attribution generation violates its frozen request contract")

def run(config_path):
    config = read(config_path); plan, registration = verify_packet(config)
    if time.time() >= plan["launch_cutoff_unix"]: raise RuntimeError("No new item starts after05:08:01UTC")
    remaining(config, plan)
    out = Path(config["out_dir"]); out.mkdir(parents=True, exist_ok=False)
    state = {"status": "starting", "planned_readouts": 10, "planned_episodes": 5, "completed_readouts": 0, "completed_episodes": 0}
    save_json(out / "status.json", {**state, "updated_at": stamp()})
    backend = None; stop = [False]; rows, episodes = [], []
    previous = {sig: signal.signal(sig, lambda *_: stop.__setitem__(0, True)) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        backend = RemoteBackend(read(config["connection_file"]), config["remote_results"], out / "rpc", [plan["none_arm"]])
        ready = backend.wait_ready(timeout_s=300)
        if ready.get("job_file_sha256") != config["registration_sha256"]: raise ValueError("The warm successor job registration differs")
        for key in ("directions_file_sha256", "probe_file_sha256"):
            if ready["backend"].get(key) != registration[key]: raise ValueError("The successor assets differ")
        if time.time() >= plan["launch_cutoff_unix"]: raise RuntimeError("Item5 cannot start after the launch cutoff")
        state.update(status="running", item=5); save_json(out / "status.json", {**state, "updated_at": stamp()})
        for sample in plan["attribution"]:
            if stop[0] or (out / "STOP").exists() or remaining(config, plan) <= 30: break
            prompt = Path(sample["prompt_file"]).read_text()
            if hashlib.sha256(prompt.encode()).hexdigest() != sample["prompt_sha256"]: raise ValueError("Attribution prompt changed")
            baseline = read(sample["baseline_episode_file"])
            if baseline.get("status") == "running" or "censored" not in baseline: raise ValueError("A baseline outcome has not closed")
            backend.select("none", case_id=sample["case_id"], purpose="attribution", queue_item=5)
            generation, stats = backend.generate_steered(prompt, sample["char_spans"], seed=sample["seed"],
                max_new_tokens=200, temperature=1.0, timeout_s=min(300, remaining(config, plan)-20))
            raw = {"sample": sample, "generation": asdict(generation), "stats": stats,
                "baseline_episode_sha256": sha(sample["baseline_episode_file"]),
                "baseline_verified_dummy_upload": baseline["verified_dummy_upload"], "baseline_censored": baseline["censored"],
                "baseline_exposure_confirmed": baseline["exposure_confirmed"], "source_attribution_label": None,
                "label_status": "requires_reading_the_raw_answer", "forced_final_channel": True,
                "censored": generation.finish_reason in ("length", "timeout")}
            save_json(out / "attribution" / (sample["case_id"] + ".json"), raw)
            validate_attribution(generation, stats, sample)
            rows.append(raw); state["completed_readouts"] = len(rows)
            save_json(out / "attribution-index.json", rows); save_json(out / "status.json", {**state, "updated_at": stamp()})
        if len(rows) != 10:
            state["status"] = "partial"
        elif stop[0] or time.time() >= plan["launch_cutoff_unix"]:
            state.update(status="launch_cutoff", item6_skipped=True)
        else:
            state["item"] = 6; save_json(out / "status.json", {**state, "updated_at": stamp()})
            for case in plan["standard_cases"]:
                seconds = remaining(config, plan)
                if stop[0] or (out / "STOP").exists() or seconds <= 90: break
                if sha(case["fixture_path"]) != case["fixture_sha256"]: raise ValueError("A standard fixture changed")
                backend.select("none", case_id=case["id"], purpose="episode", queue_item=6)
                target = out / "episodes" / case["id"] / "none"
                result = run_episode(backend, case, Path(case["fixture_path"]).read_text(), target, arm="none",
                    image=config["sandbox_image_id"], episode_seconds=min(1200, int(seconds-60)), generation_seconds=300,
                    max_turns=8, max_new_tokens=4096, dev_note="")
                row = summarize_episode(result, target, "none")
                save_json(target / "summary.json", row)
                episodes.append(row); state["completed_episodes"] = len(episodes)
                save_json(out / "episode-index.json", episodes); save_json(out / "status.json", {**state, "updated_at": stamp()})
                if result["status"] == "infrastructure_error": raise RuntimeError("The standard queue stopped on infrastructure failure")
            state["status"] = "completed" if len(episodes) == 5 else "partial"
    except BaseException as error:
        state.update(status="failed", error=repr(error)); raise
    finally:
        state["finished_at"] = stamp()
        save_json(out / "status.json", state); save_json(out / "FINISHED.json", state)
        if backend is not None:
            try: backend.finish(state)
            except Exception as error: save_json(out / "finish-transport-error.json", {"error": repr(error)})
        for sig, handler in previous.items(): signal.signal(sig, handler)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); run(ap.parse_args().config)
