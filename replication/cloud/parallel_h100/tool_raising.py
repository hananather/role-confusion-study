"""I derive and gate the registered Tool-minus-mean(User, CoT) intervention.

I preserve the frozen CUDA implementation and every original direction on disk.
My two new arms use an explicitly recorded compatibility alias only inside one
locked generation call. No source fixture or model weight is changed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from ..agent_steering.hf_backend import HFBackend
from ..agent_steering.worker import write_json

BASE_MAGNITUDE = 616.1265
DIRECTION = "tool_raising"
ARMS = [{"arm_id": "tool_raising_a" + str(alpha), "direction": "tool_minus_cot",
         "vector_key": DIRECTION, "intervention": DIRECTION, "alpha": alpha,
         "hooks_enabled": True, "layer": 11, "mask_mode": "tool", "scale_direction": DIRECTION}
        for alpha in (16, 32)]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate_record(path, value):
    def safe(item):
        if isinstance(item, dict): return {key: safe(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)): return [safe(val) for val in item]
        if isinstance(item, float) and not math.isfinite(item): return {"invalid_nonfinite": repr(item)}
        return item
    write_json(path, safe(value))


def derive(source, destination):
    """I append my direction while keeping all source arrays exactly unchanged."""
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError("I never overwrite an existing direction or receipt")
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    if DIRECTION in arrays or "gap_tool_raising" in arrays:
        raise ValueError("My source must be the unextended block11 class-mean file")
    means = [np.asarray(arrays["mean_" + role], dtype=np.float64) for role in ("tool", "user", "cot")]
    if any(value.shape != (2880,) or not np.isfinite(value).all() for value in means):
        raise ValueError("I require finite 2880-dimensional block11 role means")
    raw = means[0] - (means[1] + means[2]) / 2
    norm = float(np.linalg.norm(raw))
    if not math.isfinite(norm) or norm <= 0: raise ValueError("The new direction has no finite length")
    arrays[DIRECTION] = (raw / norm).astype(np.float32)
    arrays["gap_tool_raising"] = np.asarray(BASE_MAGNITUDE / 16, dtype=np.float64)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream: np.savez(stream, **arrays)
    with np.load(source, allow_pickle=False) as before, np.load(destination, allow_pickle=False) as after:
        unchanged = all(before[key].dtype == after[key].dtype and np.array_equal(before[key], after[key])
                        for key in before.files)
    if not unchanged: raise RuntimeError("A source array changed during derivation")
    old_magnitude = 16 * float(arrays["gap_tool_cot"])
    receipt = {"schema_version": 1, "source": str(source.resolve()), "source_sha256": sha(source),
               "destination": str(destination.resolve()), "directions_sha256": sha(destination),
               "direction": "Tool - mean(User, CoT)", "source_backend": "MLX dev-10 block-output means",
               "site": "zero-based block11 output", "unit_key": DIRECTION, "scale_key": "gap_tool_raising",
               "raw_difference_norm": norm, "unit_norm_float32": float(np.linalg.norm(arrays[DIRECTION])),
               "base_magnitude": BASE_MAGNITUDE, "double_magnitude": 2 * BASE_MAGNITUDE,
               "historical_control_magnitude": old_magnitude,
               "registered_rounding_difference": BASE_MAGNITUDE-old_magnitude,
               "source_arrays_unchanged": unchanged, "arms": ARMS,
               "gate": {"prompts": 5, "p_tool_strictly_above": .5, "p_user_strictly_below": .3,
                        "aggregation": "each prompt's page-token mean must pass", "double_once_only": True},
               "comparison_boundary": "The doubled arm has no existing norm-matched random control."}
    write_json(destination.with_suffix(".json"), receipt)
    return receipt


def checked_arm(arm):
    expected = next((row for row in ARMS if row["arm_id"] == arm.get("arm_id")), None)
    marked = arm.get("vector_key") == DIRECTION or arm.get("intervention") == DIRECTION
    if expected is None:
        if marked: raise ValueError("Unregistered Tool-raising arm identity")
        return False
    if any(arm.get(key) != value for key, value in expected.items()):
        raise ValueError("A registered Tool-raising arm changed after preparation")
    return True


class ToolRaisingBackend(HFBackend):
    """I extend only the registered vector lookup around the unchanged CUDA hook."""
    def __init__(self, config):
        for arm in config["arms"]: checked_arm(arm)
        super().__init__(config)
        self._vector_lock = threading.RLock()
        unit = self.directions[DIRECTION]
        if (unit.shape != (2880,) or not np.isfinite(unit).all()
                or not np.isclose(np.linalg.norm(unit), 1, atol=1e-5)
                or float(self.directions["gap_tool_raising"]) != BASE_MAGNITUDE / 16):
            raise ValueError("The registered Tool-raising direction or scale changed")
        self._metadata.update(additional_direction=DIRECTION, additional_scale_key="gap_tool_raising",
                              additional_base_magnitude=BASE_MAGNITUDE,
                              additional_direction_source="MLX block11 Tool-minus-mean(User, CoT)",
                              additional_direction_compatibility_alias="tool_minus_cot inside one locked call only",
                              tool_raising_source_sha256=sha(__file__))

    def generate_steered(self, prompt, char_spans, *, arm_id, **kwargs):
        with self._vector_lock:
            arm = self.arms[arm_id]
            if not checked_arm(arm):
                return super().generate_steered(prompt, char_spans, arm_id=arm_id, **kwargs)
            old_unit, old_gap = self.directions["tool_minus_cot"], self.directions["gap_tool_cot"]
            try:
                self.directions["tool_minus_cot"] = self.directions[DIRECTION]
                self.directions["gap_tool_cot"] = self.directions["gap_tool_raising"]
                generation, stats = super().generate_steered(prompt, char_spans, arm_id=arm_id, **kwargs)
            finally:
                self.directions["tool_minus_cot"], self.directions["gap_tool_cot"] = old_unit, old_gap
            stats.update(direction=DIRECTION, vector_key=DIRECTION, scale_key="gap_tool_raising",
                         bridge_direction_alias="tool_minus_cot", registered_magnitude=BASE_MAGNITUDE*(arm["alpha"]/16))
            return generation, stats


def gate_check(sample, generation, stats, arm):
    page = stats.get("probe_means", {}).get("page", {})
    probabilities = [page.get("p_" + role) for role in ("system", "user", "cot", "assistant", "tool")]
    finite = all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in probabilities)
    expected_magnitude = BASE_MAGNITUDE * arm["alpha"] / 16
    valid = (finite and abs(sum(probabilities)-1) < 1e-4
             and stats.get("direction") == DIRECTION and stats.get("arm_id") == arm["arm_id"]
             and stats.get("prompt_sha256") == hashlib.sha256(sample["prompt"].encode()).hexdigest()
             and stats.get("seed") == sample["seed"] and stats.get("purpose") == "engineering_pilot"
             and stats.get("max_new_tokens") == 64
             and page.get("n_tokens") == sample["expected_page_tokens"]
             and stats.get("edited_positions") == sample["expected_page_tokens"]
             and stats.get("edited_positions_generated") == 0 and stats.get("offset_mismatch_calls") == 0
             and abs(stats.get("delta_norm_float32", -1)-expected_magnitude) < .01
             and generation.finish_reason != "timeout" and 0 < generation.generated_tokens <= 64)
    return {"case_id": sample["case_id"], "arm_id": arm["arm_id"], "implementation_valid": bool(valid),
            "p_tool": page.get("p_tool"), "p_user": page.get("p_user"), "p_cot": page.get("p_cot"),
            "page_tokens": page.get("n_tokens"), "delta_norm_float32": stats.get("delta_norm_float32"),
            "threshold_passed": bool(valid and page["p_tool"] > .5 and page["p_user"] < .3)}


def run_gate(backend, diagnostics, directory, proof, *, can_start=lambda: True):
    """I gate dose16, then dose32 once only when the first dose misses thresholds."""
    expected = {row["case_id"]: row for row in proof["samples"]}
    if (len(diagnostics) != 5 or len(expected) != 5 or len({s["case_id"] for s in diagnostics}) != 5
            or {s["case_id"] for s in diagnostics} != set(expected)):
        raise ValueError("I require the same five historical engineering prompts")
    for sample in diagnostics:
        match = expected[sample["case_id"]]
        if (hashlib.sha256(sample["prompt"].encode()).hexdigest() != match["prompt_sha256"]
                or sample["expected_page_tokens"] != match["page_tokens"] or match["page_tokens"] <= 0):
            raise ValueError("An engineering prompt or token mask changed")
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    doses = []
    for arm in ARMS:
        checks = []
        for sample in diagnostics:
            if not can_start():
                receipt = {"status": "cutoff", "selected_arm": None, "doses": doses, "partial_checks": checks}
                gate_record(directory / "gate-result.json", receipt); return receipt
            backend.select(arm["arm_id"], case_id=sample["case_id"], purpose="engineering_pilot")
            generation, stats = backend.generate_steered(sample["prompt"], sample["char_spans"],
                seed=sample["seed"], max_new_tokens=64, temperature=1.0, timeout_s=300)
            check = gate_check(sample, generation, stats, arm); checks.append(check)
            gate_record(directory / (sample["case_id"] + "-" + arm["arm_id"] + ".json"),
                       {"generation": asdict(generation), "stats": stats, "check": check})
            if not check["implementation_valid"]:
                receipt = {"status": "implementation_failure", "selected_arm": None,
                           "doses": doses, "partial_checks": checks}
                gate_record(directory / "gate-result.json", receipt)
                raise RuntimeError("I do not increase the dose to conceal an invalid engineering measurement")
        passed = all(row["threshold_passed"] for row in checks)
        doses.append({"arm_id": arm["arm_id"], "passed": passed, "checks": checks})
        if passed:
            receipt = {"status": "passed", "selected_arm": arm["arm_id"], "doses": doses,
                       "unix_time": time.time(), "claim": "A probe manipulation check, not a mitigation result."}
            gate_record(directory / "gate-result.json", receipt); return receipt
    receipt = {"status": "thresholds_failed_both_registered_doses", "selected_arm": None,
               "doses": doses, "unix_time": time.time()}
    gate_record(directory / "gate-result.json", receipt); return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.source, args.out), indent=2))
