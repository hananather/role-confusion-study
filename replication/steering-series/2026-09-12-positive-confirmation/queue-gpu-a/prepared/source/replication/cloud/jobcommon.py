"""Shared pieces for the one-shot cloud jobs (steering and conversations).

Everything here is CUDA-free so it can be unit-checked on the Mac with the system python3:
manifest parsing, sharding, deterministic seeds, JSONL checkpoint and resume, heartbeat,
provenance records, the --hours alarm, and the stop-file check. The model loader lives here too
but only imports torch/transformers when called.

Reused from role-steering/experiment/first-20260904/source/run.py and model.py: fsync checkpoint,
intent-then-result stop record, resume guards on config and source hashes, attempt manifest,
SIGALRM/SIGTERM handlers, flock, cache env vars, pinned-version and MXFP4-retained checks.
Changes named in GPU-PLAYBOOK.md: the alarm is armed after model load (fix 6), the pod is
terminated (DELETE) not stopped, with retries (fix 4), the seed carries the item identity
(COMPUTE-PLAN.md 3.1), and a STOP file is honoured between batches.
"""

import fcntl
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
MAX_HOURS = 4.0  # COMPUTE-PLAN.md 4.5: no single pod is planned beyond 2 h; hard cap 4 h
SOURCE_FILES = ("jobcommon.py", "job_steering.py", "job_conversations.py", "requirements-pod.txt")
PINS = {"transformers": "4.57.5", "torch": "2.9.1+cu128", "triton": "3.5.1", "kernels": "0.11.5"}
MODEL_ID = "openai/gpt-oss-20b"
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"  # same as probes/ and e8/steer.py
RUNPOD_REST = "https://rest.runpod.io/v1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


# ----------------------------------------------------------------------------- manifest

def load_manifest(path):
    """JSON (preferred) or YAML manifest. Fills defaults and validates the fields every job needs."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml  # only on pods that have it; JSON needs nothing
        manifest = yaml.safe_load(text)
    else:
        manifest = json.loads(text)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a mapping")
    required = ("experiment_id", "job")
    for key in required:
        if key not in manifest:
            raise ValueError(f"Manifest lacks required field {key!r}")
    if manifest["job"] not in ("steering", "conversations"):
        raise ValueError("job must be 'steering' or 'conversations'")
    manifest.setdefault("shard_index", 0)
    manifest.setdefault("shard_count", 1)
    manifest.setdefault("shuffle_seed", 123)
    manifest.setdefault("replicate", 0)
    manifest.setdefault("checkpoint_every", 20)
    manifest.setdefault("model_id", MODEL_ID)
    manifest.setdefault("model_revision", MODEL_REVISION)
    manifest.setdefault("attn_implementation", "eager")
    manifest.setdefault("batch_size", 8)
    idx, count = int(manifest["shard_index"]), int(manifest["shard_count"])
    if count < 1 or not 0 <= idx < count:
        raise ValueError("Require 0 <= shard_index < shard_count and shard_count >= 1")
    manifest["shard_index"], manifest["shard_count"] = idx, count
    manifest["_path"] = str(path.resolve())
    manifest["_sha256"] = sha256_bytes(text.encode())
    return manifest


def resolve_path(manifest, value):
    """Manifest-relative paths resolve against the manifest's directory, then against cloud/'s parent."""
    if value is None:
        return None
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    base = Path(manifest["_path"]).parent
    for root in (base, HERE.parent, HERE):
        candidate = root / p
        if candidate.exists():
            return candidate
    return base / p


# ----------------------------------------------------------------------------- sharding and seeds

def shard_items(items, shard_index, shard_count, shuffle_seed, key=None):
    """One fixed shuffle, then contiguous blocks. Identity, not position, decides everything else."""
    import random
    order = list(items)
    rng = random.Random(shuffle_seed)
    rng.shuffle(order)
    if key is not None:
        order = order  # shuffle is by position; the caller has already deduplicated keys
    n = len(order)
    per = -(-n // shard_count)  # ceil
    return order[shard_index * per:(shard_index + 1) * per]


def item_seed(experiment_id, item_id, arm, replicate=0):
    """blake2b(experiment|item|arm|replicate) as an unsigned 64-bit int (COMPUTE-PLAN.md 3.1)."""
    digest = hashlib.blake2b(f"{experiment_id}|{item_id}|{arm}|{replicate}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def torch_seed_from(value):
    """torch.manual_seed accepts values up to 2**64 - 1; keep it in range."""
    return int(value) % (2 ** 63 - 1)


# ----------------------------------------------------------------------------- results dir, checkpoint, heartbeat

def fsync_dir(result_dir):
    for path in Path(result_dir).rglob("*"):
        if path.is_file():
            try:
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            except OSError:
                pass
    descriptor = os.open(str(result_dir), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Jsonl:
    """Append-only JSONL with resume by identity key and fsync every `every` rows."""

    def __init__(self, path, key_fields, every=20):
        self.path = Path(path)
        self.key_fields = tuple(key_fields)
        self.every = int(every)
        self.done = set()
        self.rows_since_sync = 0
        if self.path.exists():
            repaired = []
            for line in self.path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break  # tail repair: a torn last line is dropped, everything before it kept
                repaired.append(line)
                self.done.add(self.key(row))
            if len(repaired) != len(self.path.read_text().splitlines()):
                self.path.write_text("\n".join(repaired) + ("\n" if repaired else ""))
        self.handle = self.path.open("a")

    def key(self, row):
        return tuple(row[k] for k in self.key_fields)

    def has(self, row_or_key):
        key = row_or_key if isinstance(row_or_key, tuple) else self.key(row_or_key)
        return key in self.done

    def write(self, row):
        key = self.key(row)
        if key in self.done:
            return False
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.handle.flush()
        self.done.add(key)
        self.rows_since_sync += 1
        if self.rows_since_sync >= self.every:
            self.sync()
        return True

    def sync(self):
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.rows_since_sync = 0

    def close(self):
        self.sync()
        self.handle.close()


def heartbeat(result_dir, extra=None):
    """The Mac-side watchdog terminates the pod if this file stops changing (playbook fix 5)."""
    record = {"time_utc": utc_now(), "pid": os.getpid(), **(extra or {})}
    target = Path(result_dir) / "heartbeat"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(record) + "\n")
    tmp.replace(target)


def stop_requested(result_dir):
    """A review that says stop must reach a process that can stop: the watchdog writes STOP."""
    return (Path(result_dir) / "STOP").exists()


# ----------------------------------------------------------------------------- provenance and deadline

class Run:
    """Provenance record set: runtime.json with attempts, config and source hashes, failures."""

    def __init__(self, manifest, result_dir, hours):
        self.manifest = manifest
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.hours = float(hours)
        if not 1 / 60 <= self.hours <= MAX_HOURS:
            raise ValueError(f"--hours must be between 1/60 and {MAX_HOURS}")
        self.started = time.time()
        self.runtime_path = self.result_dir / "runtime.json"
        self.lock = (self.result_dir / ".run.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This results directory is already active on this filesystem")
        sources = {name: (HERE / name) for name in SOURCE_FILES if (HERE / name).exists()}
        self.source_sha = {name: sha256_file(path) for name, path in sources.items()}
        logical = {k: v for k, v in manifest.items() if not k.startswith("_")}
        self.config_sha = sha256_bytes(json.dumps(logical, sort_keys=True).encode())
        config_path = self.result_dir / "manifest.json"
        if self.runtime_path.exists():
            self.runtime = json.loads(self.runtime_path.read_text())
            if self.runtime.get("config_sha256") != self.config_sha:
                raise SystemExit("Resume manifest differs from the saved one; use a new results directory")
            if self.runtime.get("source_sha256") != self.source_sha:
                raise SystemExit("Resume source differs from the saved one; use a new results directory")
        else:
            config_path.write_text(json.dumps(logical, indent=2, sort_keys=True))
            source_dir = self.result_dir / "source"
            source_dir.mkdir(exist_ok=True)
            for name, path in sources.items():
                (source_dir / name).write_bytes(path.read_bytes())
            self.runtime = {"experiment_id": manifest["experiment_id"], "job": manifest["job"],
                            "shard_index": manifest["shard_index"], "shard_count": manifest["shard_count"],
                            "started_utc": utc_now(), "config_sha256": self.config_sha,
                            "manifest_file_sha256": manifest.get("_sha256"),
                            "source_sha256": self.source_sha, "attempts": []}
        self.attempt = {"attempt": len(self.runtime["attempts"]) + 1, "started_utc": utc_now(),
                        "hours": self.hours, "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
                        "python": sys.version.split()[0], "status": "running"}
        self.runtime["attempts"].append(self.attempt)
        self.runtime["status"] = "running"
        self.save()
        self._prev = {}

    def save(self):
        tmp = self.runtime_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.runtime, indent=2, default=str))
        tmp.replace(self.runtime_path)

    def arm_alarm(self):
        """Arm the wall-clock cap only after assets are ready (playbook fix 6)."""

        def deadline(signum, frame):
            raise TimeoutError("Configured wall-time limit reached")

        def terminated(signum, frame):
            raise InterruptedError("Host requested termination")

        self._prev["alrm"] = signal.signal(signal.SIGALRM, deadline)
        self._prev["term"] = signal.signal(signal.SIGTERM, terminated)
        signal.alarm(int(self.hours * 3600))
        self.attempt["alarm_armed_utc"] = utc_now()
        self.save()

    def finish(self, status, error=None):
        signal.alarm(0)
        for sig, key in ((signal.SIGALRM, "alrm"), (signal.SIGTERM, "term")):
            if key in self._prev:
                signal.signal(sig, self._prev[key])
        self.attempt["status"] = status
        if error is not None:
            self.attempt["error"] = f"{type(error).__name__}: {error}"
            (self.result_dir / f"failure-{self.attempt['attempt']:03d}.txt").write_text(traceback.format_exc())
        self.attempt["elapsed_seconds"] = time.time() - self.started
        self.attempt["finished_utc"] = utc_now()
        self.runtime["status"] = status
        self.save()
        fsync_dir(self.result_dir)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
        except (OSError, ValueError):
            pass


# ----------------------------------------------------------------------------- pod self-termination

def terminate_self(result_dir, dry_run=False):
    """DELETE this pod with retries. Intent is persisted before the call (run.py stop_pod order)."""
    pod_id, api_key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    record = {"pod_id": pod_id, "time_utc": utc_now(), "status": "requesting_terminate", "dry_run": dry_run}
    log_path = Path(result_dir) / "stop-requests.jsonl"

    def log():
        print(json.dumps(record), flush=True)
        try:
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            print("Could not persist terminate record", flush=True)

    log()
    if not pod_id or not api_key:
        record.update(status="skipped_no_pod_env")
        log()
        return False
    if dry_run:
        record.update(status="dry_run", request=f"DELETE {RUNPOD_REST}/pods/{pod_id}")
        log()
        return True
    for attempt in range(3):
        try:
            request = Request(f"{RUNPOD_REST}/pods/{pod_id}", method="DELETE",
                              headers={"Authorization": f"Bearer {api_key}"})
            with urlopen(request, timeout=30) as response:
                record.update(status="terminate_accepted", http_status=response.status, tries=attempt + 1)
                log()
                return True
        except Exception as error:  # noqa: BLE001
            record.update(status="terminate_failed", error=type(error).__name__,
                          http_status=getattr(error, "code", None), tries=attempt + 1)
            log()
            time.sleep(5 * (attempt + 1))
    return False


# ----------------------------------------------------------------------------- model (CUDA only)

def set_cache_env(cache_dir):
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("HF_MODULES_CACHE", str(cache_dir / "modules"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_dir / "triton"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return cache_dir


def load_model(manifest, strict_pins=True):
    """gpt-oss-20b in transformers with the authors' pins; MXFP4 must be retained (model.py check).

    Returns (tokenizer, model, meta). The tokenizer is left-padded as in the authors' NB01/NB02
    loaders. attn_implementation: 'eager' (works on every card; what the Sept 4 H100 run used) or
    'kernels-community/vllm-flash-attn3' (the authors' loader, Hopper only).
    """
    import importlib.metadata
    import re
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("This job requires a CUDA GPU")
    versions = {}
    for name in ("torch", "transformers", "triton", "kernels", "accelerate", "numpy", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if strict_pins:
        for name, pin in PINS.items():
            if versions.get(name) != pin:
                raise RuntimeError(f"{name}=={versions.get(name)} but the authors' pin is {pin}; "
                                   "install requirements-pod.txt or pass --no-strict-pins")
    revision = manifest["model_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("model_revision must be a full immutable commit SHA")
    tokenizer = AutoTokenizer.from_pretrained(manifest["model_id"], revision=revision, use_fast=True,
                                              add_eos_token=False, add_bos_token=False,
                                              padding_side="left", trust_remote_code=False)
    if not tokenizer.is_fast:
        raise RuntimeError("Offset mappings need a fast tokenizer")
    attn = manifest.get("attn_implementation", "eager")
    model = AutoModelForCausalLM.from_pretrained(manifest["model_id"], revision=revision, dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation=attn,
                                                 trust_remote_code=False)
    model.eval().requires_grad_(False)
    quant = getattr(model.config, "quantization_config", None)
    quant = quant.to_dict() if hasattr(quant, "to_dict") else quant
    if not quant or quant.get("quant_method") != "mxfp4" or quant.get("dequantize", False):
        raise RuntimeError("MXFP4 was not retained; check the Triton/kernels installation and GPU capability")
    experts = model.model.layers[0].mlp.experts
    meta = {"model_id": manifest["model_id"], "model_revision": revision, "versions": versions,
            "gpu": torch.cuda.get_device_name(), "capability": ".".join(map(str, torch.cuda.get_device_capability())),
            "quantization": quant, "attention": model.config._attn_implementation,
            "expert_module": type(experts).__module__ + "." + type(experts).__name__,
            "n_layers": len(model.model.layers), "hidden": int(model.config.hidden_size),
            "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
            "generation_config": model.generation_config.to_dict(), "hf_home": os.environ.get("HF_HOME")}
    return tokenizer, model, meta


def block_output(output):
    """gpt_oss decoder layers return a tensor in 4.57.5; tolerate the tuple form of other versions."""
    return output[0] if isinstance(output, tuple) else output


def replace_block_output(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden,) + tuple(output[1:])
    return new_hidden
