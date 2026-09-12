"""Build the 24 frozen prefills and verify instruction token identity and start index."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .protocol import (
    AUTHORIZATION,
    CONDITIONS,
    DEVELOPER_PROMPT,
    PLAIN_PAD_UNIT,
    SITUATIONS,
    STYLE,
    SYSTEM_PROMPT,
    USER_BY_AUTH,
    USER_FILLER,
    assistant_prefill,
    case_id,
    h_developer,
    h_system,
    h_tool,
    h_user,
    seed_for,
    tool_body,
)

HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUT = HERE / "data" / "pilot-24-20260911"
TOKENIZER_REV = "6cee5e81ee83917806bbde320786a8fb61efebee"
PAD_UNITS = (PLAIN_PAD_UNIT, USER_FILLER, " a ", " ", ". ", ", ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_tokenizer():
    from transformers import AutoTokenizer

    snap = (
        Path.home()
        / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots"
        / TOKENIZER_REV
    )
    return AutoTokenizer.from_pretrained(
        str(snap), add_eos_token=False, add_bos_token=False
    )


def render_prefill(user_text: str, tool_text: str) -> str:
    return (
        h_system(SYSTEM_PROMPT)
        + h_developer(DEVELOPER_PROMPT)
        + h_user(user_text)
        + h_tool(tool_text)
        + assistant_prefill()
    )


def char_span(haystack: str, needle: str) -> tuple[int, int] | None:
    start = haystack.find(needle)
    if start < 0:
        return None
    if haystack.find(needle, start + 1) >= 0:
        raise ValueError(f"needle is not unique: {needle[:40]!r}")
    return start, start + len(needle)


def token_indices(
    tokenizer,
    prefill: str,
    span: tuple[int, int] | None,
    *,
    clip_end: bool = False,
) -> list[int]:
    if span is None:
        return []
    start, end = span
    encoded = tokenizer(prefill, add_special_tokens=False, return_offsets_mapping=True)
    out = []
    for i, (ts, te) in enumerate(encoded.offset_mapping):
        if te <= ts or ts >= end or te <= start:
            continue
        if clip_end and te > end:
            continue
        out.append(i)
    return out


def instruction_start(tokenizer, prefill: str, instruction: str) -> int | None:
    span = char_span(prefill, instruction) if instruction in prefill else None
    indices = token_indices(tokenizer, prefill, span)
    return indices[0] if indices else None


def prepend_until(tokenizer, prefix: str, make_start, target: int) -> str:
    current = make_start(prefix)
    if current is None:
        raise ValueError("padding target is missing")
    if current > target:
        raise ValueError(f"already past target: {current}>{target}")
    padded = prefix
    while current < target:
        progressed = False
        for unit in PAD_UNITS:
            trial = unit + padded
            start = make_start(trial)
            if start is None:
                continue
            if current < start <= target:
                padded = trial
                current = start
                progressed = True
                break
        if not progressed:
            raise ValueError(f"cannot advance from {current} to {target}")
    if current != target:
        raise ValueError(f"ended at {current}, want {target}")
    return padded


def split_tool(body: str, instruction: str) -> tuple[str, str] | None:
    if instruction not in body:
        return None
    start = body.find(instruction)
    return body[:start], body[start + len(instruction) :]


def user_token_count(tokenizer, user_text: str) -> int:
    prefill = render_prefill(user_text, "Note.")
    span = char_span(prefill, user_text)
    return len(token_indices(tokenizer, prefill, span))


def match_user_lengths(tokenizer, users: dict[str, str]) -> dict[str, str]:
    target = max(user_token_count(tokenizer, text) for text in users.values())
    return {
        condition: prepend_until(
            tokenizer, text, lambda prefix: user_token_count(tokenizer, prefix), target
        )
        for condition, text in users.items()
    }


def match_instruction_starts(
    tokenizer,
    tools: dict[str, str],
    users: dict[str, str],
    instruction: str,
) -> dict[str, str]:
    def start_of(condition: str, prefix: str, suffix: str) -> int | None:
        body = prefix + instruction + suffix
        return instruction_start(tokenizer, render_prefill(users[condition], body), instruction)

    starts = {}
    parts = {}
    for condition, body in tools.items():
        split = split_tool(body, instruction)
        if split is None:
            continue
        prefix, suffix = split
        parts[condition] = (prefix, suffix)
        starts[condition] = start_of(condition, prefix, suffix)
    if not starts:
        return tools
    if any(value is None for value in starts.values()):
        raise ValueError("instruction vanished while measuring starts")
    target = max(starts.values())
    out = dict(tools)
    for condition, (prefix, suffix) in parts.items():
        padded = prepend_until(
            tokenizer, prefix, lambda p, c=condition, s=suffix: start_of(c, p, s), target
        )
        out[condition] = padded + instruction + suffix
        if start_of(condition, padded, suffix) != target:
            raise ValueError(f"failed to match start for {condition}")
    return out


def build_case(
    tokenizer,
    situation: dict,
    condition: str,
    situation_index: int,
    condition_index: int,
    tool_text: str,
    user_text: str,
) -> dict:
    instruction = situation["instruction"]
    prefill = render_prefill(user_text, tool_text)
    instruction_chars = char_span(prefill, instruction) if instruction in tool_text else None
    user_chars = char_span(prefill, user_text)
    tool_chars = char_span(prefill, tool_text)
    instruction_ix = token_indices(
        tokenizer, prefill, instruction_chars, clip_end=True
    )
    ids = tokenizer.encode(prefill, add_special_tokens=False)
    return {
        "id": case_id(situation["id"], condition),
        "situation_id": situation["id"],
        "condition": condition,
        "authorization": AUTHORIZATION[condition],
        "style": STYLE[condition],
        "instruction": instruction if instruction in tool_text else None,
        "seed": seed_for(situation_index, condition_index),
        "user_text": user_text,
        "tool_text": tool_text,
        "prefill": prefill,
        "prefill_sha256": sha256_text(prefill),
        "n_tokens": len(ids),
        "instruction_token_start": instruction_ix[0] if instruction_ix else None,
        "instruction_token_ids": ids[instruction_ix[0] : instruction_ix[-1] + 1]
        if instruction_ix
        else [],
        "char_spans": {
            "instruction": list(instruction_chars) if instruction_chars else None,
            "tool": list(tool_chars) if tool_chars else None,
            "user": list(user_chars) if user_chars else None,
        },
        "spans": {
            "instruction": instruction_ix,
            "tool": token_indices(tokenizer, prefill, tool_chars),
            "user": token_indices(tokenizer, prefill, user_chars),
        },
    }


def verify_situation(cases: list[dict], situation_id: str) -> None:
    with_instruction = [case for case in cases if case["instruction"]]
    starts = [case["instruction_token_start"] for case in with_instruction]
    token_ids = [tuple(case["instruction_token_ids"]) for case in with_instruction]
    if len(set(starts)) != 1:
        raise ValueError(f"{situation_id} instruction starts differ: {starts}")
    if len(set(token_ids)) != 1:
        raise ValueError(f"{situation_id} instruction token ids differ")
    plain = [
        case["tool_text"]
        for case in with_instruction
        if case["condition"] in ("legitimate_delegation", "ordinary_injection")
    ]
    forged = [
        case["tool_text"]
        for case in with_instruction
        if case["condition"] in ("authorized_reasoning", "cot_forgery")
    ]
    if len(set(plain)) != 1:
        raise ValueError(f"{situation_id} plain tool text diverged")
    if len(set(forged)) != 1:
        raise ValueError(f"{situation_id} forged tool text diverged")


def build_manifest(tokenizer) -> dict:
    cases = []
    for s_index, situation in enumerate(SITUATIONS):
        raw_tools = {condition: tool_body(situation, condition) for condition in CONDITIONS}
        users = match_user_lengths(
            tokenizer,
            {condition: USER_BY_AUTH[AUTHORIZATION[condition]] for condition in CONDITIONS},
        )
        tools = match_instruction_starts(
            tokenizer, raw_tools, users, situation["instruction"]
        )
        built = []
        for c_index, condition in enumerate(CONDITIONS):
            case = build_case(
                tokenizer,
                situation,
                condition,
                s_index,
                c_index,
                tools[condition],
                users[condition],
            )
            built.append(case)
            cases.append(case)
        verify_situation(built, situation["id"])
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(cases),
        "n_situations": len(SITUATIONS),
        "tokenizer_revision": TOKENIZER_REV,
        "probe": {
            "split": "grouped-base",
            "file": (
                "replication/cloud/persistent/sessions/20260911T030050Z/"
                "outputs/20260911T030050Z/probes-full/probes-basesplit.npz"
            ),
            "primary": "sucat_L12",
            "diagnostic": "uat_L12",
        },
        "adaptations": [
            "reasoning set to low and 256-token generation cap to de-risk analysis loops",
            "no tool execution; ACTION line is the only behavioral endpoint",
            "instruction position matched by padding earlier tool text",
            "user texts length-matched so authorization is not a length cue",
        ],
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists() and (args.out / "manifest.json").exists():
        raise SystemExit("Use a new output directory")
    tokenizer = load_tokenizer()
    manifest = build_manifest(tokenizer)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    starts = sorted(
        {
            case["situation_id"]: case["instruction_token_start"]
            for case in manifest["cases"]
            if case["instruction_token_start"] is not None
        }.items()
    )
    print(
        json.dumps(
            {
                "wrote": str(path),
                "n": manifest["n_cases"],
                "instruction_starts": starts,
                "tokens_first_six": [case["n_tokens"] for case in manifest["cases"][:6]],
            }
        )
    )


if __name__ == "__main__":
    main()
