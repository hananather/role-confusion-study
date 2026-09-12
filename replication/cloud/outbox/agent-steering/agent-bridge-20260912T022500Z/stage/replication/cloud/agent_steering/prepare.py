"""I freeze five historical cases and prospective controls without loading weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

from .episode import tool_content_spans, tool_header_spans
from .frozen_harness import protocol as p

ROOT = Path(__file__).resolve().parents[3]
ARM_IDS = ["none", "zero", "role_a16", "reverse_a16", "random_0_a16", "random_1_a16", "random_2_a16"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def arms():
    result = [
        {"arm_id": "none", "direction": "none", "alpha": 0, "hooks_enabled": False},
        {"arm_id": "zero", "direction": "tool_minus_cot", "alpha": 0, "hooks_enabled": True},
        {"arm_id": "role_a16", "direction": "tool_minus_cot", "alpha": 16, "hooks_enabled": True},
        {"arm_id": "reverse_a16", "direction": "tool_minus_cot", "alpha": -16, "hooks_enabled": True},
    ]
    result.extend({"arm_id": f"random_{i}_a16", "direction": f"random_{i}", "alpha": 16,
                   "hooks_enabled": True} for i in range(3))
    for arm in result:
        arm.update(mask_mode="tool", scale_direction="tool_minus_cot", layer=11)
    return result


def prior_messages(prompt, messages):
    prefix = []
    for message in messages:
        candidate = "".join(prefix + [message])
        if not prompt.startswith(candidate):
            break
        prefix.append(message)
        if candidate + p.ASSISTANT_PREFIX == prompt:
            return prefix
    raise ValueError("I could not reconstruct the exact recorded prompt from its messages")


def build(out, tokenizer_path):
    import numpy as np
    from transformers import AutoTokenizer
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True,
                                       trust_remote_code=False, add_bos_token=False, add_eos_token=False)
    inputs = out / "inputs"
    inputs.mkdir()
    direction_source = ROOT / "replication/steering-agent/directions/block11.npz"
    shutil.copyfile(direction_source, inputs / "block11.npz")
    probe_source = ROOT / "replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full/probes.npz"
    shutil.copyfile(probe_source, inputs / "probes.npz")
    with np.load(direction_source) as d:
        gap = float(d["gap_tool_cot"])
        for key in ["tool_minus_cot", "random_0", "random_1", "random_2"]:
            assert d[key].shape == (2880,) and np.isfinite(d[key]).all()
            assert np.isclose(np.linalg.norm(d[key]), 1.0, atol=1e-5)
        magnitude = gap * 16
    cases, diagnostics, samples = [], [], []
    for i in range(5):
        case_id = f"{i:03d}-forgery"
        relative = (f"replication/agent-hijacking/runs/ten-20260911/{case_id}" if i < 4 else
                    f"replication/agent-hijacking/runs/resample-004-forgery-20260911/{case_id}")
        base = ROOT / relative
        case = json.loads((base / "case.json").read_text())
        fixture = Path(case["fixture_path"])
        if not fixture.is_absolute():
            fixture = ROOT / "replication/agent-hijacking/data/pilot-20260911" / fixture
        if not fixture.exists():
            fixture = ROOT / "replication/agent-hijacking/data/pilot-20260911/fixtures" / f"{case_id}.html"
        assert sha(fixture) == case["fixture_sha256"]
        target = inputs / f"{case_id}.html"
        shutil.copyfile(fixture, target)
        case["fixture_path"] = str(target)
        case["historical_baseline"] = str(base)
        prompt = (base / "step-01.prompt.txt").read_text()
        messages = prior_messages(prompt, json.loads((base / "messages.json").read_text()))
        rendered = p.render_prompt(p.initial_messages(case["user_prompt"])) + p.ASSISTANT_PREFIX
        assert rendered == (base / "step-00.prompt.txt").read_text()
        escaped = json.dumps(case["payload"], ensure_ascii=False)[1:-1]
        start = prompt.find(escaped)
        assert start >= 0
        spans = {"page": tool_content_spans(messages), "header": tool_header_spans(messages),
                 "payload": [[start, start + len(escaped)]]}
        encoded = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(encoded.input_ids)
        masks = {name: [j for j, (a, b) in enumerate(encoded.offset_mapping)
                         if b > a and any(a < end and b > begin for begin, end in boundaries)]
                 for name, boundaries in spans.items()}
        assert masks["page"] and masks["payload"] and len(ids) + 4096 <= 65536
        assert set(masks["payload"]).issubset(masks["page"])
        assert not set(masks["page"]).intersection(masks["header"])
        # I verify full-token decoding preserves the historical prompt exactly.
        assert tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == prompt
        for control in p.STOP_TOKENS:
            assert len(tok.encode(control, add_special_tokens=False)) == 1
        diagnostics.append({"case_id": case_id, "prompt": prompt, "char_spans": spans,
                            "seed": int(case["seed"]) + 100003, "expected_page_tokens": len(masks["page"])})
        samples.append({"case_id": case_id, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "token_count": len(ids), "page_tokens": len(masks["page"]),
                        "payload_tokens": len(masks["payload"]), "initial_prompt_identical": True,
                        "roundtrip": True, "mask_disjoint_from_header": True})
        cases.append(case)
    rng = random.Random(20260912)
    order = list(range(5)); rng.shuffle(order)
    jobs = []
    for i in order:
        arm_order = ARM_IDS.copy(); rng.shuffle(arm_order)
        for arm_id in arm_order:
            jobs.append({"case_id": cases[i]["id"], "arm_id": arm_id,
                         "seed": cases[i]["seed"], "block_id": cases[i]["id"]})
    plan = {"schema_version": 1, "scope": "agent_steering_bridge", "stage": "bridge35",
            "execution_approved": True, "authorization": "Hanan asked to continue this direction and keep running H100 experiments in the background on 2026-09-12 UTC.",
            "case_count": 5, "expected_episode_count": 35, "cases": cases, "arms": arms(), "jobs": jobs,
            "allocation_seed": 20260912, "allocation_order": "case-major with fixed shuffled arm order",
            "engineering_pilot": {"required": True, "samples": 5, "arms": ["none", "zero", "role_a16"],
                                  "max_new_tokens": 64, "gate": ["zero_token_identity", "mask_edit_counts", "finite_downstream_readout"]},
            "settings": {"max_new_tokens": 4096, "max_turns": 8, "temperature": 1.0, "top_k": 50,
                         "top_p": 1.0, "reasoning_effort": "high", "generation_seconds": 300,
                         "episode_seconds": 1200, "max_context_tokens": 65536},
            "directions_sha256": sha(inputs / "block11.npz"), "probe_sha256": sha(inputs / "probes.npz"),
            "direction_origin": "fixed exported MLX dev-10 class means, applied unchanged in a CUDA bridge",
            "direction_gap": gap, "alpha16_magnitude": magnitude,
            "primary_outcome": "receiver-verified dummy upload; missing or censored outcomes retained",
            "utility_outcome": "completed page summary, factual adequacy separately reviewed",
            "inference_boundary": "H100 generates text; isolated network-none Docker executes tools on Mac",
            "claim_boundary": "same-five-case cross-backend bridge; not a new-page population estimate",
            "budget_stop": "complete a fixed case block when measured conservative remaining time permits; retain partial blocks if a hard deadline interrupts",
            "behavior_stopping": "no outcome-based early stopping or success-based case selection",
            "random_trace_seed": 20260913, "random_review_job_indices": sorted(random.Random(20260913).sample(range(35), 10))}
    save(out / "bridge-plan.json", plan)
    save(inputs / "diagnostic-prompts.json", diagnostics)
    save(out / "model-free-validation.json", {"passed": True, "model_loaded": False,
         "bridge_plan_sha256": sha(out / "bridge-plan.json"), "samples": samples,
         "tokenizer_path": str(tokenizer_path), "tokenizer_sha256": sha(Path(tokenizer_path) / "tokenizer.json"),
         "direction_norms_checked": True, "alpha16_magnitude": magnitude})
    print(json.dumps({"out": str(out), "cases": len(cases), "planned_episodes": len(jobs), "validation": "passed"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    build(args.out, args.tokenizer)
