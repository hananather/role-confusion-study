"""I verify the bridge's masks, raw tokens and hook order without model weights."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from .hf_backend import HFBackend, TokenRecorder, arm_vector, span_masks


class Tensor:
    """I use a small NumPy tensor double; this is not a CUDA execution test."""
    def __init__(self, data, dtype=None): self.a = np.asarray(data, dtype=dtype)
    @property
    def shape(self): return self.a.shape
    @property
    def dtype(self): return self.a.dtype
    @property
    def device(self): return "cpu"
    @property
    def T(self): return Tensor(self.a.T)
    def __getitem__(self, key): return Tensor(self.a[key])
    def __setitem__(self, key, value): self.a[key] = value.a if isinstance(value, Tensor) else value
    def __add__(self, other): return Tensor(self.a + (other.a if isinstance(other, Tensor) else other))
    def __matmul__(self, other): return Tensor(self.a @ other.a)
    def clone(self): return Tensor(self.a.copy())
    def to(self, dtype): return Tensor(self.a.astype(dtype))
    def float(self): return self.to(np.float32)
    def cpu(self): return self
    def detach(self): return self
    def reshape(self, *shape): return Tensor(self.a.reshape(*shape))
    def tolist(self): return self.a.tolist()
    def mean(self, axis): return Tensor(self.a.mean(axis))
    def softmax(self, axis):
        values = np.exp(self.a - self.a.max(axis, keepdims=True))
        return Tensor(values / values.sum(axis, keepdims=True))


class Layer:
    def __init__(self): self.hooks = []
    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))
    def call(self, x):
        for hook in self.hooks:
            changed = hook(self, (), x)
            if changed is not None: x = changed
        return x


class Model:
    def __init__(self):
        self.model = SimpleNamespace(layers=[Layer() for _ in range(13)])
        self.model.layers[12].post_attention_layernorm = Layer()
        self.prefills, self.decodes, self.kwargs = [], [], None
    def generate(self, **kw):
        self.kwargs = kw
        ids = kw["input_ids"].tolist()[0]
        kw["streamer"].put(Tensor([ids]))
        generated = []
        for turn in range(2):
            hidden = Tensor(np.zeros((1, len(ids) if turn == 0 else 1, 2880), np.float32))
            hidden = self.model.layers[11].call(hidden)
            (self.prefills if turn == 0 else self.decodes).append(hidden.a.copy())
            self.model.layers[12].post_attention_layernorm.call(hidden)
            token = 99 if turn else (21 if np.any(hidden.a) else 20)
            generated.append(token)
            kw["streamer"].put(Tensor([token]))
            if any(bool(rule(Tensor([ids + generated]), None).a[0]) for rule in kw["stopping_criteria"]): break
        kw["streamer"].end()
        return Tensor([ids + generated])


class Tokenizer:
    pad_token_id = 0
    def __call__(self, prompt, **kw):
        return {"input_ids": list(range(1, len(prompt)+1)),
                "offset_mapping": [(i, i+1) for i in range(len(prompt))]}
    def decode(self, ids, **kw): return "".join("<|return|>" if x == 99 else str(x)+"," for x in ids)


def fake_backend():
    backend = object.__new__(HFBackend)
    vector = np.zeros(2880, np.float32); vector[0] = 1
    backend.directions = {"tool_minus_cot": vector, "gap_tool_cot": np.float32(38.507904)}
    for i in range(3): backend.directions[f"random_{i}"] = np.roll(vector, i+1)
    backend.arms = {
        "none": {"direction": "none", "alpha": 0, "hooks_enabled": False},
        "zero": {"direction": "tool_minus_cot", "alpha": 0},
        "role": {"direction": "tool_minus_cot", "alpha": 16},
    }
    backend.closed = False
    backend.should_stop = lambda: False
    backend.max_context_tokens = 65536
    backend.device = "cpu"
    backend.tokenizer, backend.model = Tokenizer(), Model()
    weights = np.zeros((5, 2880), np.float32); weights[4, 0] = .01
    backend.probe = Tensor(weights), Tensor(np.zeros(5))
    backend.stop_ids = {98: "<|call|>", 99: "<|return|>"}
    backend.torch = SimpleNamespace(
        tensor=lambda x, dtype=None, device=None: Tensor(x, dtype), long=np.int64, bool=np.bool_, float32=np.float32,
        ones_like=lambda x: Tensor(np.ones_like(x.a)),
        full=lambda shape, value, dtype=None, device=None: Tensor(np.full(shape, value, dtype)),
        random=SimpleNamespace(fork_rng=lambda **kw: nullcontext()), inference_mode=nullcontext,
        manual_seed=lambda _: None,
        cuda=SimpleNamespace(current_device=lambda: 0, manual_seed_all=lambda _: None,
                             reset_peak_memory_stats=lambda _: None, max_memory_allocated=lambda _: 0))
    return backend


class Tests(unittest.TestCase):
    def test_positive_overlap_and_union_preserve_wrappers(self):
        masks = span_masks("abcdefgh", [(0, 0), (0, 2), (2, 4), (4, 6), (6, 8)],
                           {"page": [[2, 5], [4, 6]], "payload": [[3, 4]]})
        self.assertEqual(np.flatnonzero(masks["page"]).tolist(), [2, 3])
        self.assertEqual(np.flatnonzero(masks["payload"]).tolist(), [2])
        self.assertFalse(masks["header"].any())

    def test_invalid_character_ranges_fail(self):
        for span in [[-1, 3], [0, 9], [2, 2], [True, 3], [1.0, 3]]:
            with self.subTest(span=span), self.assertRaises(ValueError):
                span_masks("abcdefgh", [(0, 1)], {"page": [span]})

    def test_reverse_and_random_use_exact_original_gap(self):
        d = fake_backend().directions
        a, gap = arm_vector({"direction": "tool_minus_cot", "alpha": 16}, d)
        reverse, _ = arm_vector({"direction": "reverse", "alpha": 16}, d)
        negative, _ = arm_vector({"direction": "tool_minus_cot", "alpha": -16}, d)
        self.assertTrue(np.array_equal(reverse, -a))
        self.assertTrue(np.array_equal(reverse, negative))
        self.assertEqual(gap, float(d["gap_tool_cot"]))
        for i in range(3):
            random, _ = arm_vector({"direction": f"random_{i}", "alpha": 16}, d)
            self.assertAlmostEqual(float(np.linalg.norm(a)), float(np.linalg.norm(random)), places=4)

    def test_nonzero_without_hooks_is_rejected(self):
        with self.assertRaises(ValueError):
            arm_vector({"direction": "tool_minus_cot", "alpha": 16, "hooks_enabled": False}, fake_backend().directions)

    def generate(self, backend, arm, **kw):
        with patch.dict("sys.modules", {"transformers": SimpleNamespace(StoppingCriteriaList=lambda x: x)}):
            return backend.generate_steered("abcdef", {"page": [[1, 3], [4, 5]], "payload": [[2, 3]]},
                                            arm_id=arm, seed=1235, max_new_tokens=64,
                                            purpose="engineering_pilot", **kw)

    def test_zero_identity_nonzero_mask_decode_and_downstream_probe(self):
        b = fake_backend()
        nohook, ns = self.generate(b, "none")
        zero, zs = self.generate(b, "zero")
        role, rs = self.generate(b, "role")
        self.assertEqual(nohook.token_ids, zero.token_ids)
        self.assertEqual(ns["hook_calls"], 0)
        self.assertEqual(zs["hook_calls"], 2)
        self.assertEqual(zs["edited_positions"], 0)
        self.assertEqual(ns["delta_norm_float32"], 0.0)
        self.assertEqual(zs["delta_norm_float32"], 0.0)
        self.assertAlmostEqual(rs["delta_norm_float32"], abs(rs["magnitude"]), delta=.01)
        self.assertAlmostEqual(rs["delta_norm_float32"], 616.12646484375, delta=.01)
        self.assertEqual(rs["edited_positions"], 3)
        self.assertEqual(rs["edited_positions_generated"], 0)
        edited = np.flatnonzero(np.any(b.model.prefills[-1][0] != 0, axis=1)).tolist()
        self.assertEqual(edited, [1, 2, 4])
        self.assertFalse(b.model.decodes[-1].any())
        self.assertGreater(rs["probe_means"]["page"]["p_tool"], zs["probe_means"]["page"]["p_tool"])
        self.assertEqual(role.stop_token, "<|return|>")
        self.assertTrue(role.text.endswith("<|return|>"))
        self.assertEqual(role.generated_tokens, len(role.token_ids))
        self.assertFalse(b.model.model.layers[11].hooks)
        self.assertFalse(b.model.model.layers[12].post_attention_layernorm.hooks)
        settings = b.model.kwargs
        self.assertEqual((settings["temperature"], settings["top_k"], settings["top_p"]), (1., 50, 1.))
        self.assertEqual(settings["eos_token_id"], [98, 99])

    def test_empty_page_has_zero_edits(self):
        b = fake_backend()
        with patch.dict("sys.modules", {"transformers": SimpleNamespace(StoppingCriteriaList=lambda x: x)}):
            generation, stats = b.generate_steered("abcdef", {"page": []}, arm_id="role", seed=1235,
                                                  max_new_tokens=64, purpose="engineering_pilot")
        self.assertEqual(stats["edited_positions"], 0)
        self.assertEqual(generation.token_ids, [20, 99])

    def test_pilot_cap_cannot_leak_into_episode(self):
        with self.assertRaises(ValueError):
            fake_backend().generate_steered("abcdef", {}, arm_id="none", seed=1235, max_new_tokens=64)

    def test_timeout_retains_partial_and_removes_hooks(self):
        b = fake_backend(); b.should_stop = lambda: True
        generation, stats = self.generate(b, "role")
        self.assertEqual(generation.finish_reason, "timeout")
        self.assertEqual(len(generation.token_ids), 1)
        self.assertFalse(b.model.model.layers[11].hooks)

    def test_hook_error_still_removes_handles(self):
        b = fake_backend()
        b.model.generate = lambda **kw: (_ for _ in ()).throw(RuntimeError("fixture failure"))
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            self.generate(b, "role")
        self.assertFalse(b.model.model.layers[11].hooks)
        self.assertFalse(b.model.model.layers[12].post_attention_layernorm.hooks)

    def test_streamer_requires_exact_prompt(self):
        rec = TokenRecorder([1, 2], Tokenizer(), time.monotonic(), None, lambda: 0)
        with self.assertRaises(RuntimeError): rec.put(Tensor([1, 3]))


if __name__ == "__main__":
    unittest.main()
