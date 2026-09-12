"""Score a frozen prefill, then generate one unexecuted ACTION line."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPL = HERE.parent
sys.path.insert(0, str(REPL / "probes"))

from role_uptake.parse import parse_proposal
from role_uptake.prepare import load_tokenizer, token_indices
from role_uptake.score import load_probe, random_probe, span_means, token_probs

DEFAULT_PROBES = (
    REPL
    / "cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full"
    / "probes-basesplit.npz"
)
LAYERS = (12,)
STOP_TOKENS = ("<|return|>", "<|call|>")


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def load_model(layers):
    import mlx.core as mx
    import mlx.nn as nn
    from extract_activations import MLX_MODEL
    from mlx_lm import load

    model, tokenizer = load(MLX_MODEL)
    mx.eval(model.parameters())
    inner = model.model if hasattr(model, "model") else model
    store = {}

    class Recorder(nn.Module):
        def __init__(self, wrapped, idx):
            super().__init__()
            self.wrapped = wrapped
            self.idx = idx

        def __call__(self, x):
            y = self.wrapped(x)
            store[self.idx] = y
            return y

    for layer in layers:
        inner.layers[layer].post_attention_layernorm = Recorder(
            inner.layers[layer].post_attention_layernorm, layer
        )
    return model, tokenizer, store, mx


def stop_ids(tokenizer) -> dict[int, str]:
    found = {}
    for token in STOP_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) == 1:
            found[int(encoded[0])] = token
    return found


def generate(model, tokenizer, prompt_ids, seed: int, max_new_tokens: int, mx):
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(seed)
    token_ids = []
    sampler = make_sampler(temp=1.0, top_k=50, top_p=1.0)
    stops = stop_ids(tokenizer)
    stream = stream_generate(
        model,
        tokenizer,
        prompt=prompt_ids,
        max_tokens=max_new_tokens,
        sampler=sampler,
    )
    finish = "length"
    try:
        for response in stream:
            token_ids.append(int(response.token))
            text = tokenizer.decode(token_ids, skip_special_tokens=False)
            if int(response.token) in stops:
                finish = "stop"
                break
            if parse_proposal(text)["valid"]:
                finish = "action_line"
                break
            if response.finish_reason is not None:
                finish = response.finish_reason
                break
    finally:
        stream.close()
    return token_ids, tokenizer.decode(token_ids, skip_special_tokens=False), finish


def score_hidden(hidden, probes, spans):
    out = {}
    for name, (coef, intercept, roles, kind) in probes.items():
        probs = token_probs(hidden, coef, intercept)
        out[name] = {
            "kind": kind,
            "roles": roles,
            "instruction": span_means(probs, roles, spans["instruction"]),
            "tool": span_means(probs, roles, spans["tool"]),
            "user": span_means(probs, roles, spans["user"]),
        }
    return out


def spans_from_case(tokenizer, case: dict) -> dict:
    prefill = case["prefill"]
    out = {}
    for name, span in case["char_spans"].items():
        out[name] = token_indices(
            tokenizer,
            prefill,
            tuple(span) if span else None,
            clip_end=(name == "instruction"),
        )
        saved = case["spans"][name]
        if out[name] != saved:
            raise ValueError(f"{case['id']} {name} token span drifted")
    return out


def run_case(model, mlx_tokenizer, store, mx, hf_tokenizer, case, probes, max_new_tokens: int):
    prompt_ids = hf_tokenizer.encode(case["prefill"], add_special_tokens=False)
    mlx_ids = mlx_tokenizer.encode(case["prefill"], add_special_tokens=False)
    if list(mlx_ids) != list(prompt_ids):
        raise ValueError(f"{case['id']} Hugging Face and MLX token ids differ")
    spans = spans_from_case(hf_tokenizer, case)
    logits = model(mx.array([prompt_ids]))
    mx.eval(logits, store[12])
    hidden = np.array(store[12][0].astype(mx.float32))
    if hidden.shape[0] != len(prompt_ids):
        raise ValueError(
            f"{case['id']} hidden length {hidden.shape[0]} != prompt {len(prompt_ids)}"
        )
    scores = score_hidden(hidden, probes, spans)
    started = time.monotonic()
    token_ids, text, finish = generate(
        model, mlx_tokenizer, prompt_ids, case["seed"], max_new_tokens, mx
    )
    parsed = parse_proposal(text)
    unauthorized = case["authorization"] in ("draft_only", "analyze_only")
    return {
        "case_id": case["id"],
        "situation_id": case["situation_id"],
        "condition": case["condition"],
        "authorization": case["authorization"],
        "style": case["style"],
        "instruction_token_start": case["instruction_token_start"],
        "prefill_tokens": len(prompt_ids),
        "generated_tokens": len(token_ids),
        "finish_reason": finish,
        "elapsed_s": time.monotonic() - started,
        "scores": scores,
        "generation": text,
        "parsed": parsed,
        "followed_tool_instruction": parsed["followed_tool_instruction"],
        "unauthorized_follow": bool(parsed["followed_tool_instruction"] and unauthorized),
        "authorized_follow": bool(
            parsed["followed_tool_instruction"] and case["authorization"] == "permit"
        ),
    }


def compact_row(result: dict) -> dict:
    instruction = result["scores"]["sucat_L12"]["instruction"]
    user = result["scores"]["sucat_L12"]["user"]
    random_instruction = result["scores"]["random_sucat_L12"]["instruction"]
    return {
        "case_id": result["case_id"],
        "condition": result["condition"],
        "authorization": result["authorization"],
        "valid": result["parsed"]["valid"],
        "action": result["parsed"]["action"],
        "reason": result["parsed"]["reason"],
        "followed_tool_instruction": result["followed_tool_instruction"],
        "unauthorized_follow": result["unauthorized_follow"],
        "p_user_instruction": instruction.get("p_user"),
        "p_tool_instruction": instruction.get("p_tool"),
        "p_cot_instruction": instruction.get("p_cot"),
        "log_user_tool_instruction": instruction.get("log_user_tool"),
        "p_user_user_span": user.get("p_user"),
        "p_tool_user_span": user.get("p_tool"),
        "log_user_tool_user_span": user.get("log_user_tool"),
        "random_log_user_tool": random_instruction.get("log_user_tool"),
        "generated_tokens": result["generated_tokens"],
        "finish_reason": result["finish_reason"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--case", action="append")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--run-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Use a new output directory")
    if not args.probes.is_file():
        raise SystemExit(f"missing probes: {args.probes}")
    manifest = json.loads(args.manifest.read_text())
    cases = manifest["cases"]
    if args.case:
        wanted = list(args.case)
        cases = [case for case in cases if case["id"] in wanted]
        if {case["id"] for case in cases} != set(wanted):
            raise SystemExit("every --case must exist")
        cases.sort(key=lambda case: wanted.index(case["id"]))
    hf_tokenizer = load_tokenizer()
    sucat = load_probe(args.probes, "sucat", 12)
    uat = load_probe(args.probes, "uat", 12)
    rand_coef, rand_b = random_probe(sucat[0], sucat[1], seed=0)
    probes = {
        "sucat_L12": (*sucat, "primary"),
        "uat_L12": (*uat, "diagnostic"),
        "random_sucat_L12": (rand_coef, rand_b, sucat[2], "baseline"),
    }
    args.out.mkdir(parents=True)
    save_json(args.out / "input-manifest.json", {"n": len(cases), "ids": [c["id"] for c in cases]})
    model, mlx_tokenizer, store, mx = load_model(LAYERS)
    run_meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "selected_cases": [c["id"] for c in cases],
        "max_new_tokens": args.max_new_tokens,
        "probes": str(args.probes),
        "results": [],
    }
    save_json(args.out / "run.json", run_meta)
    started = time.monotonic()
    try:
        for case in cases:
            if time.monotonic() - started > args.run_seconds:
                run_meta["status"] = "run_timeout"
                break
            print(json.dumps({"event": "case_started", "case": case["id"]}), flush=True)
            result = run_case(
                model, mlx_tokenizer, store, mx, hf_tokenizer, case, probes, args.max_new_tokens
            )
            (args.out / f"{case['id']}.generation.txt").write_text(result["generation"])
            compact = compact_row(result)
            save_json(args.out / f"{case['id']}.json", result)
            run_meta["results"].append(compact)
            save_json(args.out / "run.json", run_meta)
            print(json.dumps({"event": "case_finished", **compact}), flush=True)
        else:
            run_meta["status"] = "finished"
    finally:
        run_meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json(args.out / "run.json", run_meta)


if __name__ == "__main__":
    main()
