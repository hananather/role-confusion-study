"""I prepare a fixed journal-order page prefix; preparation never starts inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

from .episode import tool_content_spans, tool_header_spans
from .frozen_harness import protocol as p
from .hf_backend import span_masks
from .prepare import arms, sha, save


ROOT = Path(__file__).resolve().parents[3]
BANK_SHA256 = "19b95f66d7afb0b62620a4b5e10f861811a27eb8a41711c0110201d2dc236c97"
FULL_ARM_IDS = ("none", "role_a16", "reverse_a16", "random_0_a16", "random_1_a16", "random_2_a16")
BENIGN_ARM_IDS = ("none", "role_a16")
SETTINGS = {"max_new_tokens": 4096, "max_turns": 8, "temperature": 1.0, "top_k": 50,
            "top_p": 1.0, "reasoning_effort": "high", "generation_seconds": 300,
            "episode_seconds": 1200, "max_context_tokens": 65536}


def prefix(manifest, count):
    if type(count) is not int or not 1 <= count <= 20:
        raise ValueError("A planning prefix must stay within the first 20 development pages")
    pages = manifest["pages"]
    if (len(pages) != 77 or manifest.get("accepted_page_count") != 77
            or len({r["page_id"] for r in pages}) != 77):
        raise ValueError("I require the frozen 77-page bank")
    return pages[:count]


def make_jobs(cases, allocation_seed=20260912):
    by_page = {}
    for case in cases:
        by_page.setdefault(case["page_id"], {})[case["variant"]] = case
    rng = random.Random(allocation_seed)
    jobs = []
    for page_id, variants in by_page.items():
        if set(variants) != {"forgery", "benign"}:
            raise ValueError("Every selected page needs its exact benign counterpart")
        block = []
        for variant, selected_arms in (("forgery", FULL_ARM_IDS), ("benign", BENIGN_ARM_IDS)):
            case = variants[variant]
            for arm in selected_arms:
                block.append({"case_id": case["id"], "page_id": page_id, "variant": variant,
                              "arm_id": arm, "seed": case["seed"], "block_id": page_id})
        rng.shuffle(block)
        jobs.extend(block)
    return jobs


def validate_plan(plan):
    n = plan.get("page_count")
    if type(n) is not int or not 1 <= n <= 20:
        raise ValueError("Invalid frozen prefix count")
    if plan.get("scope") != "new_page_descriptive_extension" or plan.get("bank_manifest_sha256") != BANK_SHA256:
        raise ValueError("The plan must bind the frozen page bank and declared scope")
    if plan.get("selected_bank_indices") != list(range(n)) or plan.get("reserved_confirmation_indices") != list(range(20, 60)):
        raise ValueError("The journal prefix and untouched confirmation reserve must remain fixed")
    if plan.get("settings") != SETTINGS or plan.get("full_arm_ids") != list(FULL_ARM_IDS):
        raise ValueError("The agent settings and six full-episode arms must stay fixed")
    if plan.get("benign_arm_ids") != list(BENIGN_ARM_IDS):
        raise ValueError("Both benign capability arms are required")
    cases = plan.get("cases", [])
    if len(cases) != 2*n or len({c["id"] for c in cases}) != 2*n:
        raise ValueError("The plan needs two unique cases per page")
    for index in range(n):
        paired = {case["variant"]: case for case in cases if case["bank_index"] == index}
        if set(paired) != {"forgery", "benign"} or paired["benign"].get("payload") != "" or not paired["forgery"].get("payload"):
            raise ValueError("Every selected bank index needs an attack and payload-free counterpart")
        for field in ("page_id", "user_prompt", "seed", "dummy_secret", "page_url", "upload_url"):
            if paired["forgery"][field] != paired["benign"][field]:
                raise ValueError("Counterparts must preserve the same legitimate prompt, environment and seed")
    expected_jobs = make_jobs(cases, plan["allocation_seed"])
    if plan.get("jobs") != expected_jobs or plan.get("expected_episode_count") != 8*n:
        raise ValueError("The exact case-blocked episode queue must remain fixed")
    if plan.get("engine_arms") != arms():
        raise ValueError("The engine arm table must preserve the frozen vectors and zero-dose diagnostic")
    return plan


def diagnostic(tokenizer, case, html):
    """I check an explicit after-fetch prompt; it is not a scored episode."""
    messages = p.initial_messages(case["user_prompt"])
    arguments = json.dumps({"command": "curl -s " + case["page_url"]})
    messages.extend([p.h_tool_call("functions.bash", arguments),
                     p.h_tool_result("functions.bash", json.dumps({"stdout": html, "stderr": "", "exit_code": 0}, ensure_ascii=False))])
    prompt = p.render_prompt(messages) + p.ASSISTANT_PREFIX
    spans = {"page": tool_content_spans(messages), "header": tool_header_spans(messages), "payload": []}
    if case["variant"] == "forgery":
        escaped = json.dumps(case["payload"], ensure_ascii=False)[1:-1]
        start = prompt.find(escaped)
        if start < 0: raise ValueError("The diagnostic lost the frozen attack payload")
        spans["payload"] = [[start, start + len(escaped)]]
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    masks = span_masks(prompt, encoded["offset_mapping"], spans)
    if (not masks["page"].any() or (masks["page"] & masks["header"]).any()
            or len(ids) + SETTINGS["max_new_tokens"] > SETTINGS["max_context_tokens"]):
        raise ValueError("A selected page fails the unchanged context or span contract; I do not silently replace it")
    if tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != prompt:
        raise ValueError("The full diagnostic prompt must round-trip")
    return {"case_id": case["id"], "page_id": case["page_id"], "variant": case["variant"],
            "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "char_spans": spans,
            "seed": case["seed"] + 100003, "token_count": len(ids),
            "expected_page_tokens": int(masks["page"].sum()),
            "expected_page_indices": [int(i) for i in masks["page"].nonzero()[0]],
            "roundtrip": True, "mask_disjoint_from_header": True}


def build(bank_dir, out, page_count, tokenizer_path):
    from transformers import AutoTokenizer
    bank, out = Path(bank_dir).resolve(), Path(out).resolve()
    manifest_path = bank / "manifest.json"
    if sha(manifest_path) != BANK_SHA256:
        raise ValueError("The frozen 77-page manifest changed")
    manifest = json.loads(manifest_path.read_text())
    selected = prefix(manifest, page_count)
    if sha(manifest_path) != json.loads((bank / "partial-bank-receipt.json").read_text())["manifest_sha256"]:
        raise ValueError("The bank receipt and manifest disagree")
    source_hashes = json.loads((bank / "frozen-files.json").read_text())["sha256_files"]
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"; inputs.mkdir()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False,
                                              add_bos_token=False, add_eos_token=False)
    source_cases = {c["id"]: c for c in manifest["cases"]}
    cases, diagnostics = [], []
    for bank_index, page in enumerate(selected):
        original = source_cases[f"{page['page_id']}-forgery"]
        raw_path, attack_path = bank / page["raw_path"], bank / original["fixture_path"]
        for path in (raw_path, attack_path):
            if sha(path) != source_hashes[path.relative_to(bank).as_posix()]:
                raise ValueError("A frozen raw page or attack fixture changed")
        if sha(raw_path) != page["sha256"] or sha(attack_path) != original["fixture_sha256"]:
            raise ValueError("The case and page hashes disagree")
        attack = {**original, "id": f"new-{page['page_id']}-forgery", "bank_index": bank_index,
                  "bank_case_id": original["id"], "cohort_role": "development_prefix"}
        benign = {**attack, "id": f"new-{page['page_id']}-benign", "variant": "benign", "payload": "",
                  "payload_sha256": hashlib.sha256(b"").hexdigest(), "original_payload_sha256": None,
                  "benign_source": "Original frozen raw HTML before any attack insertion"}
        raw_html = raw_path.read_bytes().decode("utf-8", errors="ignore")
        for case, html in ((attack, attack_path.read_text()), (benign, raw_html)):
            destination = inputs / f"{case['id']}.html"
            destination.write_text(html)
            case.update(fixture_path=str(destination), fixture_sha256=sha(destination))
            cases.append(case)
            diagnostics.append(diagnostic(tokenizer, case, html))
    directions = ROOT / "replication/steering-agent/directions/block11.npz"
    probes = ROOT / "replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full/probes.npz"
    for source, name in ((directions, "block11.npz"), (probes, "probes.npz")):
        shutil.copyfile(source, inputs / name)
    import numpy as np
    with np.load(directions, allow_pickle=False) as values:
        gap = float(values["gap_tool_cot"])
    jobs = make_jobs(cases)
    plan = {"schema_version": 1, "scope": "new_page_descriptive_extension", "stage": "prepared_not_registered",
            "execution_approved": False, "page_count": page_count, "expected_episode_count": len(jobs),
            "bank_manifest": str(manifest_path), "bank_manifest_sha256": BANK_SHA256,
            "bank_status": "77 accepted pages from an incomplete 100-page acquisition; journal order preserved",
            "selected_bank_indices": list(range(page_count)), "development_bank_indices": list(range(20)),
            "reserved_confirmation_indices": list(range(20, 60)), "cases": cases,
            "full_arm_ids": list(FULL_ARM_IDS), "benign_arm_ids": list(BENIGN_ARM_IDS),
            "engine_arms": arms(), "jobs": jobs, "allocation_seed": 20260912, "settings": SETTINGS,
            "directions_sha256": sha(directions), "probe_sha256": sha(probes),
            "direction_gap": gap, "alpha16_magnitude": gap * 16,
            "engineering_pilot": {"sample_count": page_count, "diagnostic_variant": "forgery", "max_new_tokens": 64,
                                  "arms": ["none", "zero", "role_a16"]},
            "selection_rule": "I freeze a journal-order prefix by available time before observing its outcomes; I never replace failed or unexposed pages.",
            "outcome": "Receiver-verified dummy upload; incomplete and unexposed nonuploads remain unresolved.",
            "capability": "Payload-free raw HTML under none and role16; actual page exposure and completion are recorded; factual summary adequacy requires separate review.",
            "identity": "The technical zero-dose comparison uses identical fixed prompts and seeds. Independent full episodes can diverge after dynamic shell outputs.",
            "claim_boundary": "A small descriptive CUDA transfer and new-page extension; the reserved confirmation indices remain untouched."}
    validate_plan(plan)
    save(out / "plan.json", plan)
    save(inputs / "diagnostics.json", diagnostics)
    proof = {"passed": True, "model_loaded": False, "plan_sha256": sha(out / "plan.json"),
             "diagnostics_sha256": sha(inputs / "diagnostics.json"), "sample_count": len(diagnostics),
             "tokenizer_sha256": sha(Path(tokenizer_path) / "tokenizer.json"),
             "samples": [{k: v for k, v in d.items() if k not in ("prompt", "char_spans", "expected_page_indices")} for d in diagnostics]}
    save(out / "model-free-validation.json", proof)
    save(out / "prepared-files.json", {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=5)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    plan = build(args.bank, args.out, args.pages, args.tokenizer)
    print(json.dumps({"out": str(args.out), "pages": plan["page_count"], "episodes": plan["expected_episode_count"],
                      "execution_approved": False, "status": "prepared_not_registered"}))


if __name__ == "__main__": main()
