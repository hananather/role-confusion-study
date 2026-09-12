"""I keep inference independent of tool execution and preserve all raw tokens."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .protocol import STOP_TOKENS

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class Generation:
    text: str
    token_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    elapsed_s: float
    prompt_elapsed_s: float | None
    peak_memory_gb: float
    finish_reason: str
    stop_token: str | None
    repetition_detected: bool


class InferenceBackend(Protocol):
    """I can replace MLX with CUDA without changing the environment or runner."""

    @property
    def metadata(self) -> dict[str, Any]: ...

    def count_tokens(self, prompt: str) -> int: ...

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        max_new_tokens: int = 4096,
        temperature: float = 1.0,
        timeout_s: float = 300,
        on_progress: ProgressCallback | None = None,
    ) -> Generation: ...

    def close(self) -> None: ...


class ContextLimitError(ValueError):
    """I reject an oversized request instead of silently truncating the prompt."""


class _GenerationDeadline(Exception):
    pass


def has_repetition(token_ids: list[int], *, repeats: int = 4) -> bool:
    """I flag four adjacent identical token spans of length 8–128 after generation.

    This conservative diagnostic can miss paraphrased loops and can flag valid
    repetition. I never use it to change sampling or terminate an episode.
    """
    if repeats < 2:
        raise ValueError("repeats must be at least two")
    for width in range(8, min(128, len(token_ids) // repeats) + 1):
        for end in range(width * repeats, len(token_ids) + 1):
            start = end - width * repeats
            span = token_ids[start : start + width]
            if all(
                token_ids[start + i * width : start + (i + 1) * width] == span
                for i in range(1, repeats)
            ):
                return True
    return False


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _local_snapshot(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"A cached local model directory is required: {path}")
    if not list(path.glob("*.safetensors")):
        raise FileNotFoundError(f"Cached model weights are missing: {path}")
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Cached tokenizer file is missing: {path / filename}")
    return path


class MLXBackend:
    """I load only local weights and let MLX run natively on Apple silicon.

    The instance is single-owner and single-threaded. I do not share its MLX
    random state or model with another concurrent generation.
    """

    def __init__(
        self,
        model_path: str | Path,
        max_context_tokens: int = 65536,
        prefill_step_size: int = 512,
        *,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> None:
        self.model_path = _local_snapshot(model_path)
        if max_context_tokens < 2 or prefill_step_size < 1:
            raise ValueError("Context and prefill limits must be positive")
        if top_k < 0 or not 0 < top_p <= 1:
            raise ValueError("top_k must be nonnegative and top_p must be in (0, 1]")
        self.max_context_tokens = max_context_tokens
        self.prefill_step_size = prefill_step_size
        self.top_k = top_k
        self.top_p = top_p
        self._closed = False
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler

        self._mx = mx
        self._stream_generate = stream_generate
        self._make_sampler = make_sampler
        started = time.monotonic()
        self._model, self._tokenizer, config = load(
            str(self.model_path),
            tokenizer_config={
                "local_files_only": True,
                "trust_remote_code": False,
                "add_bos_token": False,
                "add_eos_token": False,
            },
            return_config=True,
        )
        self._stop_ids = {}
        for token in STOP_TOKENS:
            encoded = self._tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) != 1 or self._decode(encoded) != token:
                raise ValueError(f"Tokenizer does not encode a Harmony control token: {token}")
            self._stop_ids[int(encoded[0])] = token
        # I override defaults so endoftext does not become an extra stopping rule.
        self._tokenizer.eos_token_ids = set(self._stop_ids)
        config_limit = config.get("max_position_embeddings")
        if isinstance(config_limit, int):
            self.max_context_tokens = min(max_context_tokens, config_limit)
        quantization = config.get("quantization", config.get("quantization_config", {}))
        module_modes = Counter(
            f"{value.get('mode', 'affine')}:{value.get('bits')}"
            for value in quantization.values()
            if isinstance(value, dict)
        )
        config_hashes = {
            name: hashlib.sha256((self.model_path / name).read_bytes()).hexdigest()
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
            if (self.model_path / name).is_file()
        }
        self._metadata = {
            "backend": "mlx",
            "model_path": str(self.model_path),
            "model_revision": self.model_path.name if self.model_path.parent.name == "snapshots" else None,
            "model_repository": self.model_path.parent.parent.name.removeprefix("models--").replace("--", "/")
            if self.model_path.parent.name == "snapshots" else None,
            "model_type": config.get("model_type"),
            "model_max_position_embeddings": config_limit,
            "max_context_tokens": self.max_context_tokens,
            "prefill_step_size": self.prefill_step_size,
            "quantization_default": {k: v for k, v in quantization.items() if not isinstance(v, dict)},
            "quantization_module_modes": dict(module_modes),
            "config_sha256": config_hashes,
            "runtime": {name: _version(name) for name in ("mlx", "mlx-lm", "transformers", "tokenizers")},
            "platform": platform.platform(),
            "machine": platform.machine(),
            "load_elapsed_s": time.monotonic() - started,
            "stop_token_ids": {label: token_id for token_id, label in self._stop_ids.items()},
            "sampling": {"top_k": top_k, "top_p": top_p, "repetition_penalty": None},
            "kv_cache_quantization": None,
            "prompt_truncation": False,
            "generation_preserves_stop_token": True,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._metadata))

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("The backend is closed")

    def _decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def count_tokens(self, prompt: str) -> int:
        self._require_open()
        return len(self._tokenizer.encode(prompt, add_special_tokens=False))

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        max_new_tokens: int = 4096,
        temperature: float = 1.0,
        timeout_s: float = 300,
        on_progress: ProgressCallback | None = None,
        steering=None,
    ) -> Generation:
        self._require_open()
        if max_new_tokens < 1 or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Generation token and time limits must be positive")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            raise ValueError("The prompt must contain at least one token")
        if len(prompt_ids) + max_new_tokens > self.max_context_tokens:
            raise ContextLimitError(
                f"{len(prompt_ids)} prompt tokens + {max_new_tokens} requested generation "
                f"tokens exceeds context limit {self.max_context_tokens}"
            )
        if steering is not None:
            from .intervention import install_source_hook
            hook = install_source_hook(self._model, layer=steering.layer)
            spec = steering
            if spec.prompt_length == 0:
                spec.prompt_length = len(prompt_ids)
            hook.reset(spec)
        self._mx.random.seed(seed)
        self._mx.reset_peak_memory()
        started = time.monotonic()
        deadline = started + timeout_s
        token_ids: list[int] = []
        prompt_elapsed_s = None
        finish_reason = "length"
        stop_token = None

        def check_deadline() -> None:
            if time.monotonic() >= deadline:
                raise _GenerationDeadline

        def progress(processed: int, total: int) -> None:
            check_deadline()
            if on_progress is not None:
                on_progress({
                    "phase": "prefill", "elapsed_s": time.monotonic() - started,
                    "prompt_tokens": total, "processed_prompt_tokens": processed,
                    "generated_tokens": 0,
                })

        stream = self._stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt_ids,
            max_tokens=max_new_tokens,
            sampler=self._make_sampler(temp=temperature, top_k=self.top_k, top_p=self.top_p),
            prefill_step_size=self.prefill_step_size,
            prompt_progress_callback=progress,
        )
        try:
            for response in stream:
                token_ids.append(int(response.token))
                if prompt_elapsed_s is None and response.prompt_tps > 0:
                    prompt_elapsed_s = len(prompt_ids) / response.prompt_tps
                if response.token in self._stop_ids:
                    stop_token = self._stop_ids[response.token]
                    finish_reason = "stop"
                elif response.finish_reason is not None:
                    finish_reason = response.finish_reason
                if on_progress is not None and (
                    len(token_ids) == 1 or len(token_ids) % 32 == 0 or response.finish_reason
                ):
                    on_progress({
                        "phase": "generation", "elapsed_s": time.monotonic() - started,
                        "prompt_tokens": len(prompt_ids), "generated_tokens": len(token_ids),
                        "prompt_elapsed_s": prompt_elapsed_s,
                        "peak_memory_gb": self._mx.get_peak_memory() / 1e9,
                        "token_ids": list(token_ids), "text": self._decode(token_ids),
                    })
                check_deadline()
                if stop_token is not None or response.finish_reason is not None:
                    break
        except _GenerationDeadline:
            finish_reason = "timeout"
        finally:
            stream.close()
        elapsed_s = time.monotonic() - started
        return Generation(
            text=self._decode(token_ids), token_ids=token_ids,
            prompt_tokens=len(prompt_ids), generated_tokens=len(token_ids),
            elapsed_s=elapsed_s, prompt_elapsed_s=prompt_elapsed_s,
            peak_memory_gb=self._mx.get_peak_memory() / 1e9,
            finish_reason=finish_reason, stop_token=stop_token,
            repetition_detected=has_repetition(token_ids),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._model = None
        self._tokenizer = None
        gc.collect()
        self._mx.clear_cache()
