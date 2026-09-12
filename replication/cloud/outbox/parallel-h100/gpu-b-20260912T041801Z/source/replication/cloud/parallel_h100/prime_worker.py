"""I prime one isolated H100 and keep its model idle under a renewable lease.

I perform one fixed, benign setup generation. I have no experiment inbox, tool
execution, provider client, install step, or allocation action. My parent process
enforces the lease while my own child imports CUDA and loads the cached model.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
BASE_ROOT = Path("/workspace/parallel-lanes")
MARKER_DIR = Path("/tmp")
MODEL_ID = "openai/gpt-oss-20b"
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
ATTENTION = "kernels-community/vllm-flash-attn3"
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}\Z")
SMOKE_PROMPT = ("<|start|>system<|message|>You are a helpful assistant.<|end|>"
                "<|start|>user<|message|>Reply with the word ready.<|end|>"
                "<|start|>assistant<|channel|>final<|message|>")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def identity_report(config):
    """I verify the container-local binding without inventing provider environment.

    The launcher creates this nonce-bound marker exclusively through an SSH
    endpoint it has just verified against the provider API. The marker is not
    on the shared volume. This is a provider/SSH binding, not hardware attestation.
    """
    observed = os.environ.get("RUNPOD_POD_ID")
    if observed is not None and observed != config["expected_pod_id"]:
        raise ValueError("The actual pod environment disagrees with this new pod")
    fields = {"identity_marker_path", "identity_nonce", "identity_source"}
    present = fields & config.keys()
    if not present:
        if observed != config["expected_pod_id"]:
            raise ValueError("I require the actual pod environment or a provider-bound local marker")
        return {"identity_source": "provider_environment", "expected_pod_id": observed}
    if present != fields:
        raise ValueError("The provider-bound marker configuration is incomplete")
    path = MARKER_DIR / ("mats-prime-" + config["run_id"] + "-identity.json")
    nonce = config["identity_nonce"]
    if (config["identity_marker_path"] != str(path)
            or config["identity_source"] != "provider_api_ssh_binding"
            or not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{64}", nonce)
            or path.parent.resolve() != path.parent):
        raise ValueError("Invalid container-local identity marker configuration")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1
                or not 0 < info.st_size <= 16384):
            raise ValueError("The identity marker must be a private, UID-owned regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(16385)
        marker = json.loads(raw)
        expected = {"schema_version": 1, "identity_source": "provider_api_ssh_binding",
                    "expected_pod_id": config["expected_pod_id"], "run_id": config["run_id"],
                    "identity_nonce": nonce}
        if (not isinstance(marker, dict) or type(marker.get("schema_version")) is not int
                or any(marker.get(key) != value for key, value in expected.items())):
            raise ValueError("The local identity marker disagrees with the provider-bound configuration")
        return {"identity_source": "provider_api_ssh_binding", "expected_pod_id": config["expected_pod_id"],
                "run_id": config["run_id"], "marker_path": str(path),
                "marker_sha256": hashlib.sha256(raw).hexdigest(), "provider_environment_present": observed is not None}
    finally:
        os.close(descriptor)


def validate_config(config, config_path=None):
    required = {"schema_version", "purpose", "execution_approved", "run_id",
                "expected_pod_id", "protected_pod_id", "isolated_root", "lease_path"}
    optional = {"model_id", "model_revision", "cache_dir", "hf_home", "kernel_cache_dir",
                "attn_implementation", "startup_timeout_seconds", "poll_seconds",
                "identity_marker_path", "identity_nonce", "identity_source"}
    if not isinstance(config, dict) or not required <= config.keys() or config.keys() - required - optional:
        raise ValueError("I require the exact priming-only configuration fields")
    if (type(config["schema_version"]) is not int or config["schema_version"] != 1
            or config["purpose"] != "prime_only" or config["execution_approved"] is not True):
        raise ValueError("I require explicit setup-only authorization")
    for field in ("run_id", "expected_pod_id", "protected_pod_id"):
        if not isinstance(config[field], str) or not IDENTIFIER.fullmatch(config[field]):
            raise ValueError("Invalid isolated run or pod identity")
    if config["expected_pod_id"] == config["protected_pod_id"]:
        raise ValueError("I may not load a model on the protected existing pod")
    identity_report(config)
    root = BASE_ROOT / config["run_id"]
    if config["isolated_root"] != str(root) or root.resolve() != root:
        raise ValueError("I require a unique, non-symlink isolated run root")
    if config["lease_path"] != str(root / "control/lease.json"):
        raise ValueError("The renewable financial lease must belong to this isolated root")
    if config_path is not None:
        path = Path(config_path)
        if not path.is_absolute() or path.resolve() != path or not path.is_relative_to(root):
            raise ValueError("My frozen configuration must be inside this isolated root")
    fixed = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
             "cache_dir": "/workspace/hf", "hf_home": "/workspace/hf/home",
             "kernel_cache_dir": "/workspace/hf/home/hub", "attn_implementation": ATTENTION}
    for key, value in fixed.items():
        if config.get(key, value) != value:
            raise ValueError("I reuse only the verified pinned offline model and kernels: " + key)
    result = {**fixed, **config}
    for key, default, maximum in (("startup_timeout_seconds", 600, 1200), ("poll_seconds", 1, 5)):
        value = result.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError("Invalid finite bound: " + key)
        result[key] = value
    return result


def read_lease(config, now=None):
    path = Path(config["lease_path"])
    if path.is_symlink() or path.resolve() != path or path.stat().st_size > 16384:
        raise ValueError("Invalid financial lease path")
    lease = json.loads(path.read_bytes())
    current = time.time() if now is None else now
    if (lease.get("schema_version") != 1 or lease.get("approved") is not True
            or lease.get("expected_pod_id") != config["expected_pod_id"]
            or lease.get("run_id") != config["run_id"]
            or not isinstance(lease.get("lease_id"), str) or not lease["lease_id"]):
        raise ValueError("The lease must authorize this exact pod")
    for key in ("issued_unix", "lease_expires_unix", "shutdown_at_unix", "budget_remaining_usd"):
        if type(lease.get(key)) not in (int, float) or not math.isfinite(lease[key]):
            raise ValueError("Invalid financial lease number")
    issued, expires = lease["issued_unix"], lease["lease_expires_unix"]
    if (not 0 < expires - issued <= 180 or issued > current + 1
            or current >= min(expires, lease["shutdown_at_unix"])
            or lease["budget_remaining_usd"] <= 0):
        raise ValueError("The renewable financial lease is expired or out of budget")
    return lease


def child_environment(config):
    """I pass no account credentials and keep writable caches in my own root."""
    inherited = ("PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH",
                 "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "RUNPOD_POD_ID")
    env = {key: os.environ[key] for key in inherited if key in os.environ}
    cache = Path(config["isolated_root"]) / "cache"
    destinations = {"HF_MODULES_CACHE": "hf-modules", "TRITON_CACHE_DIR": "triton",
                    "CUDA_CACHE_PATH": "cuda", "TORCH_HOME": "torch",
                    "TORCHINDUCTOR_CACHE_DIR": "inductor", "XDG_CACHE_HOME": "xdg",
                    "TMPDIR": "tmp"}
    for key, suffix in destinations.items():
        path = cache / suffix
        if path.resolve() != path:
            raise ValueError("My writable runtime cache may not follow symlinks")
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(HF_HOME=config["hf_home"], HF_HUB_CACHE=config["kernel_cache_dir"],
               KERNELS_CACHE=config["kernel_cache_dir"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_ENABLE_HF_TRANSFER="0",
               TOKENIZERS_PARALLELISM="false", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0")
    return env


def validate_metadata(meta):
    quant = meta.get("quantization", {})
    if (meta.get("model_id") != MODEL_ID or meta.get("model_revision") != MODEL_REVISION
            or "H100" not in meta.get("gpu", "") or meta.get("capability") != "9.0"
            or meta.get("attention") != ATTENTION or meta.get("n_layers") != 24
            or meta.get("hidden") != 2880 or quant.get("quant_method") != "mxfp4"
            or quant.get("dequantize", False) or "Mxfp4" not in meta.get("expert_module", "")):
        raise RuntimeError("The loaded H100, pinned model, attention or MXFP4 metadata differs")


def parent_death_guard(parent_pid):
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent_pid:
            os._exit(126)


def model_child(config, config_path):
    """I load once, smoke once, and wait; I never inspect an experiment inbox."""
    config = validate_config(config, config_path)
    service = Path(config["isolated_root"]) / "service"
    read_lease(config)
    stop = {"requested": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    def should_stop():
        if stop["requested"] or (service / "STOP").exists(): return True
        try: read_lease(config)
        except (OSError, ValueError, TypeError, KeyError): return True
        return False
    runtime = None
    try:
        # I add only the frozen source tree; user-site and PYTHONPATH are disabled.
        source_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(source_root))
        tempfile.tempdir = os.environ["TMPDIR"]
        from replication.cloud.chat_steering.runtime import Runtime
        runtime = Runtime(config); runtime.output_dir = service
        runtime._load()
        if should_stop(): return 0
        validate_metadata(runtime.metadata)
        torch = runtime.torch
        if (torch.cuda.device_count() != 1 or runtime.device.type != "cuda"
                or torch.cuda.get_device_properties(0).total_memory < 70 * 1024**3):
            raise RuntimeError("I require exactly one visible H100 with its model on CUDA")
        placement = getattr(runtime.model, "hf_device_map", {})
        if any(str(value) in ("cpu", "disk", "meta") for value in placement.values()):
            raise RuntimeError("The model must not offload to CPU or disk")
        runtime.metadata.update(cuda_device_count=torch.cuda.device_count(),
                                cuda_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                                cuda_allocated_bytes=torch.cuda.memory_allocated(0),
                                cuda_reserved_bytes=torch.cuda.memory_reserved(0))
        from transformers import StoppingCriteria, StoppingCriteriaList
        class LeaseStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return torch.full((input_ids.shape[0],), should_stop(), dtype=torch.bool, device=input_ids.device)
        inputs = runtime.tokenizer(SMOKE_PROMPT, add_special_tokens=False, return_tensors="pt").to(runtime.device)
        started = time.monotonic()
        with torch.inference_mode():
            output = runtime.model.generate(**inputs, do_sample=False, max_new_tokens=8,
                                            logits_to_keep=1, use_cache=True,
                                            stopping_criteria=StoppingCriteriaList([LeaseStop()]))
        torch.cuda.synchronize()
        if should_stop(): return 0
        generated = output[0, inputs["input_ids"].shape[1]:].tolist()
        if not generated or len(generated) > 8:
            raise RuntimeError("The tiny engineering generation returned invalid token counts")
        smoke = {"purpose": "setup_smoke_only", "prompt": SMOKE_PROMPT,
                 "token_ids": generated, "text": runtime.tokenizer.decode(generated, skip_special_tokens=False),
                 "max_new_tokens": 8, "do_sample": False, "elapsed_seconds": time.monotonic()-started}
        write_json(service / "smoke.json", smoke)
        write_json(service / "READY.json", {"schema_version": 1, "status": "primed_idle",
                   "pid": os.getpid(), "expected_pod_id": config["expected_pod_id"],
                   "config_sha256": canonical_sha(config), "ready_unix": time.time(),
                   "metadata": runtime.metadata, "identity": identity_report(config),
                   "smoke_passed": True, "experiment_inbox": False,
                   "later_experiments_require": "a reviewed attached runner or a model-child restart",
                   "allocation_action": "none"})
        while not should_stop():
            write_json(service / "model-heartbeat.json", {"stage": "primed_idle", "pid": os.getpid(),
                       "unix_time": time.time(), "experiment_inbox": False, "allocation_action": "none"})
            time.sleep(config["poll_seconds"])
        return 0
    except BaseException as error:
        write_json(service / "MODEL-FAILED.json", {"error_type": type(error).__name__,
                   "error": str(error), "unix_time": time.time(), "allocation_action": "none"})
        return 2
    finally:
        if runtime is not None:
            runtime.model = None; runtime.tokenizer = None; runtime.probe = None
            gc.collect()
            if hasattr(runtime, "torch"): runtime.torch.cuda.empty_cache()
        write_json(service / "MODEL-EXIT.json", {"pid": os.getpid(), "unix_time": time.time(),
                   "model_references_released": True, "allocation_action": "none"})


class Guardian:
    """I enforce leases during blocking load, smoke generation, and idle retention."""
    def __init__(self, config, config_path, *, spawn=subprocess.Popen, lease_reader=read_lease,
                 clock=time.monotonic, sleep=time.sleep):
        self.config = validate_config(config, config_path)
        self.config_path = Path(config_path)
        self.config_bytes = self.config_path.read_bytes()
        self.root = Path(config["isolated_root"])
        self.service = self.root / "service"
        self.spawn, self.lease_reader, self.clock, self.sleep = spawn, lease_reader, clock, sleep
        self.child = None; self.stop_requested = False; self.stop_reason = None
        self.forced_kill = False

    def stop_child(self):
        if self.child is None or self.child.poll() is not None: return
        self.child.terminate()
        try: self.child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.forced_kill = True; self.child.kill()
            try: self.child.wait(timeout=5)
            except subprocess.TimeoutExpired: self.stop_reason += ":child_reap_unverified"

    def run(self):
        if self.service.resolve() != self.service:
            raise ValueError("My service output may not follow symlinks")
        self.service.mkdir(parents=True, exist_ok=True)
        with (self.service / "guardian.lock").open("x") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock.write(str(os.getpid())); lock.flush()
            self.lease_reader(self.config)
            if (self.service / "STOP").exists():
                raise RuntimeError("An existing STOP blocks model loading")
            env = child_environment(self.config)
            parent = os.getpid()
            command = [sys.executable, "-B", "-s", str(Path(__file__).resolve()), "--config",
                       str(self.config_path), "--model-child"]
            started = self.clock()
            with (self.service / "model.log").open("x") as log:
                self.child = self.spawn(command, cwd=str(Path(__file__).resolve().parents[3]), env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, preexec_fn=lambda: parent_death_guard(parent))
                write_json(self.service / "GUARDIAN-READY.json", {"guardian_pid": os.getpid(),
                           "worker_pid": self.child.pid, "expected_pod_id": self.config["expected_pod_id"],
                           "command": command, "config_sha256": canonical_sha(self.config),
                           "identity": identity_report(self.config),
                           "unix_time": time.time(), "allocation_action": "none"})
                try:
                    while self.child.poll() is None:
                        if self.stop_requested or (self.service / "STOP").exists():
                            self.stop_reason = "explicit_stop"; break
                        if self.config_path.read_bytes() != self.config_bytes:
                            self.stop_reason = "frozen_config_changed"; break
                        try: identity_report(self.config)
                        except (OSError, ValueError, TypeError, KeyError):
                            self.stop_reason = "container_identity_unavailable_or_changed"; break
                        try: lease = self.lease_reader(self.config)
                        except (OSError, ValueError, TypeError, KeyError):
                            self.stop_reason = "lease_unavailable_or_expired"; break
                        ready = (self.service / "READY.json").exists()
                        if not ready and self.clock()-started >= self.config["startup_timeout_seconds"]:
                            self.stop_reason = "startup_timeout"; break
                        write_json(self.service / "heartbeat.json", {"stage": "primed_idle" if ready else "priming",
                                   "guardian_pid": os.getpid(), "worker_pid": self.child.pid, "unix_time": time.time(),
                                   "expected_pod_id": self.config["expected_pod_id"],
                                   "lease_expires_unix": lease["lease_expires_unix"],
                                   "shutdown_at_unix": lease["shutdown_at_unix"], "experiment_inbox": False,
                                   "allocation_action": "none"})
                        self.sleep(self.config["poll_seconds"])
                finally:
                    self.stop_reason = self.stop_reason or "model_child_exit"
                    self.stop_child()
                    receipt = {"stop_reason": self.stop_reason, "worker_returncode": self.child.poll(),
                               "worker_exit_verified": self.child.poll() is not None, "forced_kill": self.forced_kill,
                               "expected_pod_id": self.config["expected_pod_id"], "unix_time": time.time(),
                               "allocation_action": "none"}
                    write_json(self.service / "EXIT.json", receipt)
                    write_json(self.service / "heartbeat.json", {**receipt, "stage": "closed"})
        return 0 if self.stop_reason in ("explicit_stop", "lease_unavailable_or_expired") else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    if args.model_child: return model_child(config, args.config)
    guardian = Guardian(config, args.config)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: setattr(guardian, "stop_requested", True))
    return guardian.run()


if __name__ == "__main__": raise SystemExit(main())
