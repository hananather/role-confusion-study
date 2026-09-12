"""I preserve the agent generation contract while using the cached CUDA model.

This adapter performs inference only. It never executes a generated tool call.
The initial bridge reuses the frozen MLX direction file and records that transfer.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .frozen_harness.backend import Generation, ContextLimitError, has_repetition


ROLES = ("system", "user", "cot", "assistant", "tool")
STOP_TOKENS = ("<|call|>", "<|return|>")
STEER_LAYER, PROBE_LAYER = 11, 12


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def span_masks(prompt, offsets, char_spans):
    """I preserve positive-overlap masks over all accumulated tool contents."""
    if not isinstance(char_spans, dict) or set(char_spans) - {"page", "header", "payload"}:
        raise ValueError("Only page, header and payload character spans are accepted")
    result = {}
    for name in ("page", "header", "payload"):
        mask = np.zeros(len(offsets), dtype=bool)
        for span in char_spans.get(name, []):
            if (not isinstance(span, (list, tuple)) or len(span) != 2
                    or any(type(x) is not int for x in span)):
                raise ValueError("Character spans must be integer pairs")
            a, b = span
            if not 0 <= a < b <= len(prompt):
                raise ValueError("Character span is outside the complete prompt")
            for i, (ts, te) in enumerate(offsets):
                if ts < b and te > a and te > ts:
                    mask[i] = True
        result[name] = mask
    return result


def arm_vector(arm, directions):
    """I use the original Tool-minus-CoT gap for every bridge control."""
    direction = arm.get("direction", "none")
    alpha = float(arm.get("alpha", 0))
    if not math.isfinite(alpha):
        raise ValueError("Nonfinite alpha")
    if arm.get("mask_mode", "tool") != "tool":
        raise ValueError("This bridge exposes only the frozen tool-content mask")
    hooks_enabled = arm.get("hooks_enabled", True)
    if type(hooks_enabled) is not bool:
        raise ValueError("hooks_enabled must be Boolean")
    gap = float(directions["gap_tool_cot"])
    if not math.isfinite(gap) or gap <= 0:
        raise ValueError("Invalid frozen Tool-minus-CoT gap")
    if direction == "none":
        if alpha != 0:
            raise ValueError("An unsteered direction requires alpha zero")
        return None, gap
    if direction not in {"tool_minus_cot", "reverse", "random_0", "random_1", "random_2"}:
        raise ValueError("Direction is outside the frozen bridge")
    if not hooks_enabled and alpha != 0:
        raise ValueError("An unhooked arm cannot request a nonzero edit")
    unit = np.asarray(directions["tool_minus_cot" if direction == "reverse" else direction], dtype=np.float32)
    if unit.shape != (2880,) or not np.isfinite(unit).all() or not np.isclose(np.linalg.norm(unit), 1, atol=1e-5):
        raise ValueError("A direction must be a finite 2880-dimensional unit vector")
    if direction == "reverse":
        unit = -unit
    # I preserve the MLX adapter's float32 vector before converting to residual dtype.
    return None if alpha == 0 else ((alpha * gap) * unit).astype(np.float32), gap


class TokenRecorder:
    """I retain the actual sampled token IDs, including the Harmony stop token."""

    def __init__(self, prompt_ids, tokenizer, started, callback, peak_memory):
        self.prompt_ids, self.tokenizer = list(prompt_ids), tokenizer
        self.started, self.callback, self.peak_memory = started, callback, peak_memory
        self.token_ids = []
        self.saw_prompt = False
        self.prompt_elapsed_s = None

    def put(self, value):
        values = value.detach().cpu().reshape(-1).tolist()
        if not self.saw_prompt:
            if values != self.prompt_ids:
                raise RuntimeError("The streamer did not receive the full original prompt first")
            self.saw_prompt = True
            return
        self.token_ids.extend(int(x) for x in values)
        if self.prompt_elapsed_s is None:
            self.prompt_elapsed_s = time.monotonic() - self.started
        if len(self.token_ids) == 1 or len(self.token_ids) % 32 == 0:
            self.report()

    def report(self):
        if self.callback:
            self.callback({"phase": "generation", "elapsed_s": time.monotonic() - self.started,
                           "prompt_tokens": len(self.prompt_ids), "generated_tokens": len(self.token_ids),
                           "prompt_elapsed_s": self.prompt_elapsed_s, "peak_memory_gb": self.peak_memory(),
                           "token_ids": list(self.token_ids),
                           "text": self.tokenizer.decode(self.token_ids, skip_special_tokens=False,
                                                         clean_up_tokenization_spaces=False)})

    def end(self):
        self.report()


class HFBackend:
    def __init__(self, config):
        self.config = dict(config)
        self.should_stop = lambda: False
        self.max_context_tokens = int(config.get("max_context_tokens", 65536))
        if not 4096 < self.max_context_tokens <= 65536:
            raise ValueError("The bridge context ceiling must be at most 65536 tokens")
        self.arms = {}
        for arm in config["arms"]:
            key = arm["arm_id"]
            if not isinstance(key, str) or not key or key in self.arms:
                raise ValueError("Arm IDs must be unique nonempty strings")
            self.arms[key] = dict(arm)
        if not self.arms:
            raise ValueError("An approved arm table is required")
        directions_file = Path(config["directions_file"])
        with np.load(directions_file, allow_pickle=False) as data:
            self.directions = {name: data[name].copy() for name in data.files}
        for arm in self.arms.values():
            arm_vector(arm, self.directions)
        hashes = {name: sha256_file(config[name]) for name in ("directions_file", "probe_file")}
        for name, actual in hashes.items():
            if config.get(name + "_sha256") and config[name + "_sha256"] != actual:
                raise ValueError(f"Frozen asset checksum mismatch: {name}")
        # I import the existing loader only after its cache environment can be set.
        from ..chat_steering.runtime import Runtime
        self.runtime = Runtime(config)
        self.runtime.output_dir = Path(config["out_dir"])
        self.runtime.output_dir.mkdir(parents=True, exist_ok=True)
        self.runtime._load()
        self.torch, self.model, self.tokenizer = self.runtime.torch, self.runtime.model, self.runtime.tokenizer
        self.device, self.probe = self.runtime.device, self.runtime.probe
        if self.probe is None:
            raise ValueError("The bridge requires the frozen layer-12 probe")
        self.max_context_tokens = min(self.max_context_tokens, int(self.model.config.max_position_embeddings))
        self.stop_ids = {}
        for token in STOP_TOKENS:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1 or self.tokenizer.decode(ids, skip_special_tokens=False) != token:
                raise RuntimeError("The tokenizer must preserve each Harmony stopping token")
            self.stop_ids[int(ids[0])] = token
        self._metadata = {**self.runtime.metadata, "backend": "cuda-transformers-agent-bridge",
                          "steering_layer_zero_based": STEER_LAYER, "probe_layer_zero_based": PROBE_LAYER,
                          "probe_site": "layers[12].post_attention_layernorm output; downstream of block11 output",
                          "direction_source_backend": "mlx-community/gpt-oss-20b-MXFP4-Q8",
                          "direction_transfer": "Frozen MLX dev-10 vectors and gap; no CUDA recomputation",
                          "directions_file_sha256": hashes["directions_file"],
                          "probe_file_sha256": hashes["probe_file"],
                          "gap_tool_cot": float(self.directions["gap_tool_cot"]),
                          "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0,
                                       "do_sample": True, "repetition_penalty": 1.0},
                          "max_new_tokens": 4096, "max_context_tokens": self.max_context_tokens,
                          "stop_token_ids": {v: k for k, v in self.stop_ids.items()},
                          "generation_eos_ids": sorted(self.stop_ids), "generation_preserves_stop_token": True,
                          "prefill": "one complete CUDA prefill per agent turn; fresh KV cache",
                          "mask": "all accumulated tool-response content spans, prefill only",
                          "backend_parity": "I require CUDA none/zero identity; I do not assume MLX/CUDA token identity",
                          "arms": list(self.arms.values()), "source_sha256": sha256_file(__file__)}
        self.closed = False

    @property
    def metadata(self):
        return json.loads(json.dumps(self._metadata))

    def count_tokens(self, prompt):
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    def generate_steered(self, prompt, char_spans, *, arm_id, seed, max_new_tokens=4096,
                         temperature=1.0, timeout_s=300, on_progress=None, purpose="episode"):
        if self.closed:
            raise RuntimeError("The backend is closed")
        valid_cap = (purpose == "episode" and max_new_tokens == 4096) or (purpose == "engineering_pilot" and max_new_tokens == 64)
        if (type(seed) is not int or not 0 <= seed < 2**63 or not valid_cap
                or temperature != 1.0 or not math.isfinite(timeout_s) or not 0 < timeout_s <= 1200):
            raise ValueError("Generation must preserve the frozen seed, cap, temperature and bounded timeout")
        arm = self.arms[arm_id]
        vector, gap = arm_vector(arm, self.directions)
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        prompt_ids = list(encoded["input_ids"])
        n = len(prompt_ids)
        if not n or n + max_new_tokens > self.max_context_tokens:
            raise ContextLimitError(f"{n} prompt tokens + {max_new_tokens} exceeds {self.max_context_tokens}")
        masks = span_masks(prompt, encoded["offset_mapping"], char_spans)
        torch = self.torch
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        indices = {key: np.flatnonzero(mask).tolist() for key, mask in masks.items()}
        delta = None if vector is None else torch.tensor(vector, dtype=torch.float32, device=self.device)
        calls = {"steer": 0, "probe": 0, "positions": 0, "edited": 0}
        probe_means = {key: {"n_tokens": int(mask.sum())} for key, mask in masks.items()}

        def edit(module, args, output):
            from ..jobcommon import block_output, replace_block_output
            hidden = block_output(output)
            calls["steer"] += 1
            first = calls["steer"] == 1
            if tuple(hidden.shape[:2]) != (1, n if first else 1):
                raise RuntimeError("Unexpected CUDA prefill/decode shape")
            calls["positions"] += int(hidden.shape[1])
            if first and delta is not None and indices["page"]:
                hidden = hidden.clone()
                hidden[:, indices["page"], :] += delta.to(hidden.dtype)
                calls["edited"] += len(indices["page"])
                return replace_block_output(output, hidden)
            return None

        def probe(module, args, output):
            calls["probe"] += 1
            if calls["probe"] != 1:
                return
            if tuple(output.shape[:2]) != (1, n):
                raise RuntimeError("The probe did not see the complete prefill")
            weight, bias = self.probe
            for name, selected in indices.items():
                if selected:
                    probs = (output[0, selected].float() @ weight.T + bias).softmax(-1).mean(0).cpu().tolist()
                    if not np.isfinite(probs).all():
                        raise RuntimeError("The downstream role probe returned nonfinite probabilities")
                    probe_means[name].update({f"p_{role}": float(value) for role, value in zip(ROLES, probs)})

        started = time.monotonic()
        deadline = started + timeout_s
        stopped = {"timeout": False}
        class Deadline:
            def __call__(inner, input_ids, scores, **kwargs):
                stopped["timeout"] = bool(time.monotonic() >= deadline or self.should_stop())
                return torch.full((1,), stopped["timeout"], dtype=torch.bool, device=input_ids.device)

        from transformers import StoppingCriteriaList
        torch.cuda.reset_peak_memory_stats(self.device)
        peak = lambda: torch.cuda.max_memory_allocated(self.device) / 1e9
        recorder = TokenRecorder(prompt_ids, self.tokenizer, started, on_progress, peak)
        handles = []
        hooks_enabled = arm.get("hooks_enabled", True)
        if on_progress:
            on_progress({"phase": "prefill", "elapsed_s": 0, "prompt_tokens": n, "generated_tokens": 0})
        try:
            if hooks_enabled:
                handles = [self.model.model.layers[STEER_LAYER].register_forward_hook(edit),
                           self.model.model.layers[PROBE_LAYER].post_attention_layernorm.register_forward_hook(probe)]
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                             do_sample=True, temperature=1.0, top_k=50, top_p=1.0,
                                             repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                                             eos_token_id=list(self.stop_ids), pad_token_id=self.tokenizer.pad_token_id,
                                             use_cache=True, logits_to_keep=1, return_dict_in_generate=False,
                                             num_beams=1, streamer=recorder,
                                             stopping_criteria=StoppingCriteriaList([Deadline()]))
            ids = output[0, n:].tolist()
        finally:
            for handle in handles:
                handle.remove()
        if ids != recorder.token_ids or not recorder.saw_prompt:
            raise RuntimeError("Streamed tokens differ from the returned generation")
        if hooks_enabled and not (calls["steer"] == calls["probe"] == len(ids) > 0):
            raise RuntimeError("Each generated token must correspond to one audited forward")
        expected_edits = len(indices["page"]) if hooks_enabled and vector is not None else 0
        if calls["edited"] != expected_edits:
            raise RuntimeError("Actual edited positions differ from the frozen tool-content mask")
        stop_token = self.stop_ids.get(ids[-1]) if ids else None
        finish = "timeout" if stopped["timeout"] else ("stop" if stop_token else "length")
        generation = Generation(text=self.tokenizer.decode(ids, skip_special_tokens=False,
                                                            clean_up_tokenization_spaces=False),
                                token_ids=ids, prompt_tokens=n, generated_tokens=len(ids),
                                elapsed_s=time.monotonic() - started, prompt_elapsed_s=recorder.prompt_elapsed_s,
                                peak_memory_gb=peak(), finish_reason=finish, stop_token=stop_token,
                                repetition_detected=has_repetition(ids))
        stats = {"prompt_tokens": n, "page_span_tokens": len(indices["page"]),
                 "payload_span_tokens": len(indices["payload"]), "hook_calls": calls["steer"],
                 "hook_positions_seen": calls["positions"], "edited_positions": calls["edited"],
                 "edited_positions_prompt": calls["edited"], "edited_positions_generated": 0,
                 "offset_mismatch_calls": 0, "probe_means": probe_means,
                 "hooks_enabled": hooks_enabled, "seed": seed, "arm_id": arm_id,
                 "direction": arm.get("direction", "none"), "alpha": float(arm.get("alpha", 0)),
                 "purpose": purpose, "max_new_tokens": max_new_tokens,
                 "gap": gap, "magnitude": float(arm.get("alpha", 0)) * gap,
                 "delta_norm_float32": 0.0 if vector is None else float(np.linalg.norm(vector)),
                 "mask_mode": "tool", "mask_token_indices": indices,
                 "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        return generation, stats

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.model = self.runtime.model = self.probe = self.runtime.probe = None
        self.torch.cuda.empty_cache()
