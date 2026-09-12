"""I check the registered direction, temporary alias, and two-dose stopping rule."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from . import tool_raising as t


@dataclass
class Generation:
    finish_reason: str = "length"
    generated_tokens: int = 64
    token_ids: tuple = (1,)


class FakeRemote:
    def __init__(self, samples, probabilities):
        self.samples = {s["prompt"]: s for s in samples}
        self.probabilities = probabilities; self.calls = []
    def select(self, arm_id, **context): self.arm = arm_id; self.context = context
    def generate_steered(self, prompt, char_spans, **kwargs):
        sample = self.samples[prompt]
        alpha = next(a["alpha"] for a in t.ARMS if a["arm_id"] == self.arm)
        ptool, puser = self.probabilities[self.arm]
        self.calls.append((self.arm, prompt))
        stats = {"direction": "tool_raising", "arm_id": self.arm,
                 "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                 "seed": sample["seed"], "purpose": "engineering_pilot", "max_new_tokens": 64,
                 "edited_positions": sample["expected_page_tokens"], "edited_positions_generated": 0,
                 "offset_mismatch_calls": 0, "delta_norm_float32": t.BASE_MAGNITUDE*alpha/16,
                 "probe_means": {"page": {"n_tokens": sample["expected_page_tokens"],
                    "p_tool": ptool, "p_user": puser, "p_cot": 1-ptool-puser, "p_assistant": 0., "p_system": 0.}}}
        return Generation(), stats


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.samples = [{"case_id": f"{i:03d}-forgery", "prompt": f"prompt {i}", "char_spans": {},
                         "seed": 100+i, "expected_page_tokens": 50+i} for i in range(5)]
        self.proof = {"samples": [{"case_id": s["case_id"], "prompt_sha256": hashlib.sha256(s["prompt"].encode()).hexdigest(),
                                    "page_tokens": s["expected_page_tokens"]} for s in self.samples]}

    def test_derivation_preserves_all_original_arrays_and_exact_scale(self):
        rng = np.random.default_rng(42)
        original = {"mean_" + role: rng.normal(size=2880).astype(np.float32) for role in ("tool", "user", "cot")}
        original.update(gap_tool_cot=np.asarray(38.507904, np.float32), unrelated=np.arange(10, dtype=np.int16))
        source, target = self.root / "old.npz", self.root / "new.npz"
        np.savez(source, **original); source_hash = t.sha(source)
        receipt = t.derive(source, target)
        self.assertEqual(source_hash, t.sha(source)); self.assertTrue(receipt["source_arrays_unchanged"])
        with np.load(target) as result:
            for key, array in original.items():
                np.testing.assert_array_equal(result[key], array)
                self.assertEqual(result[key].dtype, array.dtype)
            raw = original["mean_tool"].astype(np.float64) - (original["mean_user"].astype(np.float64)+original["mean_cot"].astype(np.float64))/2
            np.testing.assert_array_equal(result["tool_raising"], (raw/np.linalg.norm(raw)).astype(np.float32))
            self.assertEqual(float(result["gap_tool_raising"])*16, t.BASE_MAGNITUDE)
        with self.assertRaises(FileExistsError): t.derive(source, target)

    def backend_without_model(self):
        backend = object.__new__(t.ToolRaisingBackend)
        backend._vector_lock = threading.RLock()
        backend.arms = {a["arm_id"]: dict(a) for a in t.ARMS}
        backend.arms["role_a16"] = {"arm_id": "role_a16", "direction": "tool_minus_cot", "alpha": 16}
        backend.directions = {"tool_minus_cot": np.array([1., 0.]), "gap_tool_cot": 38.507904,
                              "tool_raising": np.array([0., 1.]), "gap_tool_raising": t.BASE_MAGNITUDE/16}
        return backend

    def test_alias_uses_new_unit_and_scale_then_restores_original_even_on_error(self):
        backend = self.backend_without_model()
        unit, gap = backend.directions["tool_minus_cot"], backend.directions["gap_tool_cot"]
        def generate(obj, *args, **kwargs):
            self.assertIs(obj.directions["tool_minus_cot"], obj.directions["tool_raising"])
            self.assertEqual(obj.directions["gap_tool_cot"]*16, t.BASE_MAGNITUDE)
            return Generation(), {"direction": "tool_minus_cot"}
        with patch.object(t.HFBackend, "generate_steered", generate):
            _, stats = backend.generate_steered("prompt", {}, arm_id="tool_raising_a16")
        self.assertEqual(stats["direction"], "tool_raising")
        self.assertIs(backend.directions["tool_minus_cot"], unit)
        self.assertEqual(backend.directions["gap_tool_cot"], gap)
        with patch.object(t.HFBackend, "generate_steered", side_effect=RuntimeError("CUDA failure")):
            with self.assertRaises(RuntimeError): backend.generate_steered("prompt", {}, arm_id="tool_raising_a32")
        self.assertIs(backend.directions["tool_minus_cot"], unit)
        self.assertEqual(backend.directions["gap_tool_cot"], gap)

    def test_original_role_arm_uses_original_unit_without_alias(self):
        backend = self.backend_without_model(); original = backend.directions["tool_minus_cot"]
        def generate(obj, *args, **kwargs):
            self.assertIs(obj.directions["tool_minus_cot"], original)
            return Generation(), {"direction": "tool_minus_cot"}
        with patch.object(t.HFBackend, "generate_steered", generate):
            _, stats = backend.generate_steered("prompt", {}, arm_id="role_a16")
        self.assertEqual(stats["direction"], "tool_minus_cot")

    def test_changed_registered_dose_or_mask_is_rejected(self):
        for change in ({"alpha": 48}, {"mask_mode": "all"}, {"vector_key": "random_0"}, {"hooks_enabled": False}):
            with self.subTest(change=change), self.assertRaises(ValueError): t.checked_arm({**t.ARMS[0], **change})

    def run_fake_gate(self, low, high):
        backend = FakeRemote(self.samples, {"tool_raising_a16": low, "tool_raising_a32": high})
        receipt = t.run_gate(backend, self.samples, self.root / "gate", self.proof)
        return backend, receipt

    def test_base_pass_launches_no_double_dose(self):
        backend, receipt = self.run_fake_gate((.6, .2), (.7, .1))
        self.assertEqual(receipt["selected_arm"], "tool_raising_a16")
        self.assertEqual(len(backend.calls), 5)

    def test_threshold_miss_doubles_once_on_same_five_prompts(self):
        backend, receipt = self.run_fake_gate((.5, .2), (.6, .2))
        self.assertEqual(receipt["selected_arm"], "tool_raising_a32")
        self.assertEqual(len(backend.calls), 10)
        self.assertEqual([x[1] for x in backend.calls[:5]], [x[1] for x in backend.calls[5:]])

    def test_second_failure_stops_without_selecting_arm(self):
        backend, receipt = self.run_fake_gate((.5, .2), (.6, .3))
        self.assertIsNone(receipt["selected_arm"])
        self.assertEqual(receipt["status"], "thresholds_failed_both_registered_doses")
        self.assertEqual(len(backend.calls), 10)

    def test_invalid_probe_does_not_trigger_dose_increase(self):
        backend = FakeRemote(self.samples, {"tool_raising_a16": (float("nan"), .2)})
        with self.assertRaises(RuntimeError): t.run_gate(backend, self.samples, self.root / "gate", self.proof)
        self.assertEqual(len(backend.calls), 1)

    def test_wrong_frozen_prompt_and_cutoff_prevent_generation(self):
        backend = FakeRemote(self.samples, {"tool_raising_a16": (.6, .2)})
        wrong = [{**s, "prompt": "changed"} for s in self.samples]
        with self.assertRaises(ValueError): t.run_gate(backend, wrong, self.root / "gate", self.proof)
        result = t.run_gate(backend, self.samples, self.root / "gate", self.proof, can_start=lambda: False)
        self.assertEqual(result["status"], "cutoff"); self.assertFalse(backend.calls)


if __name__ == "__main__": unittest.main()
