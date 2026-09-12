"""Run the experiment on this GPU host; Modal supplies only isolated CPU agents.

Example: python run.py --config config.json --run-id first-20260904 \
    --results-dir /workspace/results --cache-dir /workspace/model-cache --hours 12

This runner never rents a GPU. With --stop-pod it requests stopping its own
RunPod after execution. Keep results and caches on the Pod's persistent volume.
"""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
MAX_SECONDS = 12 * 60 * 60
SOURCE_FILES = ("run.py", "sandbox.py", "model.py", "experiment.py", "prepare.py",
                "requirements-gpu.txt")


def checkpoint(result_dir):
    """Flush local results to the mounted disk after every completed episode."""
    for path in result_dir.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    descriptor = os.open(result_dir, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def stop_pod(pod_id, api_key, result_dir):
    """Request a stop, never termination: docs.runpod.io/pods/manage-pods."""
    record = {"pod_id": pod_id, "time_utc": datetime.now(timezone.utc).isoformat(),
              "status": "requesting_stop"}

    def log():
        print(json.dumps(record), flush=True)
        try:
            with (result_dir / "stop-requests.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            print("Could not persist Pod stop-request record", flush=True)

    log()  # Persist intent before the Pod can stop and end this process.
    try:
        request = Request(f"https://rest.runpod.io/v1/pods/{pod_id}/stop", method="POST",
                          headers={"Authorization": f"Bearer {api_key}"})
        with urlopen(request, timeout=30) as response:
            record.update(status="stop_request_accepted", http_status=response.status)
    except Exception as error:
        record.update(status="stop_request_failed", error=type(error).__name__,
                      http_status=getattr(error, "code", None))
    log()


def run_job(config, result_dir):
    """Preserve immutable inputs and source; append a record for each attempt."""
    import modal
    from sandbox import MarkerTrial, SandboxTrial, sandbox_image

    marker_study = config.get("protocol") == "permission-swap"
    if marker_study:
        from permission import run
    else:
        from experiment import run

    if importlib.metadata.version("modal") != "1.5.5":
        raise RuntimeError("Install requirements-gpu.txt; Modal 1.5.5 is required")
    limit = int(config["wall_seconds"])
    if not 60 <= limit <= MAX_SECONDS:
        raise ValueError("wall_seconds must be between 60 and 43200")
    started = time.time()
    config_bytes = json.dumps(config, sort_keys=True, indent=2).encode()
    config_path = result_dir / "config.json"
    source_dir = result_dir / "source"
    source_files = SOURCE_FILES + (("permission.py",) if marker_study else ())
    source_payloads = {name: (ROOT / name).read_bytes() for name in source_files}
    source_hashes = {name: hashlib.sha256(payload).hexdigest()
                     for name, payload in source_payloads.items()}
    runtime_path = result_dir / "runtime.json"
    if config_path.exists():
        original_config = json.loads(config_path.read_text())
        logical = lambda value: {key: item for key, item in value.items()
                                 if key != "wall_seconds"}
        if logical(original_config) != logical(config):
            raise ValueError("Resume configuration differs; use a new run_id")
        saved_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in source_dir.iterdir() if path.is_file()}
        if saved_hashes != source_hashes:
            raise ValueError("Resume source differs; use a new run_id")
        if not runtime_path.exists():
            raise ValueError("Existing run lacks its runtime manifest; use a new run_id")
        manifest = json.loads(runtime_path.read_text())
        if manifest.get("source_sha256") != source_hashes:
            raise ValueError("Saved source hashes disagree with runtime manifest")
    else:
        config_path.write_bytes(config_bytes)
        source_dir.mkdir()
        for name, payload in source_payloads.items():
            (source_dir / name).write_bytes(payload)
        manifest = {
            "run_id": config["run_id"],
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "source_sha256": source_hashes, "modal_version": modal.__version__,
            "gpu_execution": "local process on persistent GPU host",
            "agent_execution": "Modal CPU Sandbox; block_network=True",
            "attempts": [],
        }
    attempt = {
        "attempt": len(manifest["attempts"]) + 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": limit, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        "status": "running",
    }
    manifest["attempts"].append(attempt)
    manifest["status"] = "running"

    def save_manifest():
        temporary = result_dir / "runtime.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(runtime_path)
        checkpoint(result_dir)

    def deadline(signum, frame):
        raise TimeoutError("Configured batch wall-time limit reached")

    def terminated(signum, frame):
        raise InterruptedError("GPU host requested termination")

    previous_alarm = signal.signal(signal.SIGALRM, deadline)
    previous_term = signal.signal(signal.SIGTERM, terminated)
    signal.alarm(limit)
    save_manifest()
    try:
        app = modal.App("role-steering-cpu-agents", include_source=False)
        image = sandbox_image()
        trial = MarkerTrial if marker_study else SandboxTrial
        with modal.enable_output(), app.run():
            output = run(
                config, result_dir,
                sandbox_factory=lambda case: trial(app=app, image=image, case=case),
                checkpoint=lambda: checkpoint(result_dir),
            )
        attempt["status"] = "complete"
        return {"run_id": config["run_id"], "result_dir": str(result_dir), "output": output}
    except BaseException as error:
        attempt["status"] = "failed"
        attempt["error"] = f"{type(error).__name__}: {error}"
        (result_dir / f"failure-{attempt['attempt']:03d}.txt").write_text(traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGTERM, previous_term)
        attempt["elapsed_seconds"] = time.time() - started
        attempt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = attempt["status"]
        save_manifest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--run-id", help="Stable result directory name; reuse only to resume identical work")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=Path("model-cache"))
    parser.add_argument("--protocol", choices=("transfer", "permission-swap"), default="transfer")
    parser.add_argument("--manifest", type=Path, help="Frozen permission-study manifest")
    parser.add_argument("--assets-from", type=Path, help="Completed source run supplying its exact learned assets")
    parser.add_argument("--hours", type=float, default=12,
                        help="Maximum execution time for this attempt, up to 12 hours")
    parser.add_argument("--stop-pod", action="store_true",
                        help="Request stopping this RunPod after success, failure or deadline")
    args = parser.parse_args()
    pod_id, api_key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if args.stop_pod and (not pod_id or not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id) or not api_key):
        parser.error("--stop-pod requires RUNPOD_POD_ID and RUNPOD_API_KEY")
    if not 1 / 60 <= args.hours <= 12:
        parser.error("--hours must be between 1/60 and 12")
    path = args.config.expanduser().resolve()
    config = json.loads(path.read_text())
    inputs_path = Path(config.pop("inputs_path", "data/inputs.json"))
    if not inputs_path.is_absolute():
        inputs_path = path.parent / inputs_path
    config["inputs"] = json.loads(inputs_path.read_text())
    config["model_revision"] = config["inputs"]["model_revision"]
    if args.protocol == "permission-swap":
        if args.manifest is None or args.assets_from is None:
            parser.error("permission-swap requires --manifest and --assets-from")
        manifest_path = args.manifest.expanduser().resolve()
        manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(inputs_path.read_bytes()).hexdigest() != manifest["base_inputs_sha256"]:
            parser.error("Permission manifest refers to a different input bundle")
        if manifest["model_revision"] != config["model_revision"]:
            parser.error("Permission manifest refers to a different model revision")
        config.update(protocol=args.protocol, protocol_manifest=manifest,
                      generation_seeds=manifest["generation_seeds"],
                      assets_source=str(args.assets_from.expanduser().resolve()))
    elif args.manifest is not None or args.assets_from is not None:
        parser.error("--manifest and --assets-from apply only to permission-swap")
    config["run_id"] = args.run_id or config.get("run_id") or datetime.now(timezone.utc).strftime("first-%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", config["run_id"]):
        parser.error("--run-id must contain only letters, numbers, underscores, periods or hyphens")
    config["wall_seconds"] = int(args.hours * 3600)
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    os.environ["HF_MODULES_CACHE"] = str(cache_dir / "modules")
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir / "triton")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    result_dir = args.results_dir.expanduser().resolve() / config["run_id"]
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This run_id is already active on this filesystem")
        print(json.dumps({"run_id": config["run_id"], "result_dir": str(result_dir),
                          "max_hours": args.hours}), flush=True)
        try:
            print(json.dumps(run_job(config, result_dir), indent=2, default=str))
        finally:
            if args.stop_pod:
                stop_pod(pod_id, api_key, result_dir)


if __name__ == "__main__":
    main()
