"""Probe training data, ported from the authors' notebook for gpt-oss-20b.

Source: prompt-injection-as-role-confusion/experiments/role-analysis/02-train-role-probes.ipynb
        cell 8 (load_raw_ds) and cell 11 (get_sample_seqs_for_input_seq, build_sample_seqs),
        commit ec333c40fd43fe991e1ebf66765051b6d7e35784.
Rendering and labeling are imported from the authors' utils, not re-implemented.

Faithful settings for gptoss-20b (config/probe.yaml): n_sample_size 250, seq_len 1024,
nested_reasoning false, train_prefixes [""], seed 123. Roles: system, user, tool, cot, assistant.
Divergences: none in construction. The random partner-text draws are reproduced so the
numpy RNG state matches the notebook even though gpt-oss never uses partner text.

Run the tokenizer-only check (no model, no network):
    python build_probe_dataset.py --check
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AUTHORS = ROOT / "prompt-injection-as-role-confusion"
sys.path.insert(0, str(AUTHORS))
from utils.role_templates import render_single_message, render_mixed_cot  # noqa: E402
from utils.role_assignments import label_content_roles  # noqa: E402

MODEL_PREFIX = "gptoss-20b"
MODEL_ID = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
SEED = 123
SEQLEN = 1024
N_SAMPLE_SIZE = 250
NESTED_REASONING = False   # probe.yaml gptoss-20b
GENERALIZE_PREFIX = False  # NB02 cell 11
TRAIN_PREFIXES = [""]      # probe.yaml gptoss-20b


def load_raw_ds(n_sample_size=N_SAMPLE_SIZE, seed=SEED):
    """NB02 cell 8, verbatim logic. Streams from the Hugging Face Hub (network)."""
    from datasets import load_dataset

    def get_c4():
        return load_dataset("allenai/c4", "en", split="validation", streaming=True).shuffle(seed=seed, buffer_size=50_000)

    def get_dolma3():
        return load_dataset("allenai/dolma3_mix-150B-1025", split="train", revision="3a8349c", streaming=True).shuffle(seed=seed, buffer_size=50_000)

    def get_data(ds, n_samples, data_source):
        raw_data = []
        ds_iter = iter(ds)
        for _ in range(n_samples):
            sample = next(ds_iter, None)
            if sample is None:
                break
            raw_data.append({"text": sample["text"], "source": data_source})
        return raw_data

    # int() rounding: 62 + 187 = 249 base texts, as in the authors' code (audit row 1).
    return get_data(get_c4(), int(n_sample_size * .25), "c4") + get_data(get_dolma3(), int(n_sample_size * .75), "dolma3")


def get_sample_seqs_for_input_seq(probe_text, partner_text, prefix=""):
    """NB02 cell 11. One row per role for a single base text."""
    seqs = []
    gen_prefix = partner_text if GENERALIZE_PREFIX else ""
    if MODEL_PREFIX in ["gptoss-20b"]:
        seqs.append({"role": "system", "prompt": prefix + gen_prefix + render_single_message(MODEL_PREFIX, role="system", content=probe_text)})
    for role in ["user", "tool", "cot"]:
        seqs.append({"role": role, "prompt": prefix + gen_prefix + render_single_message(MODEL_PREFIX, role=role, content=probe_text)})
    if NESTED_REASONING:
        seqs.append({"role": "assistant", "prompt": prefix + render_mixed_cot(MODEL_PREFIX, cot=partner_text, assistant=probe_text)})
    else:
        seqs.append({"role": "assistant", "prompt": prefix + gen_prefix + render_single_message(MODEL_PREFIX, role="assistant", content=probe_text)})
    return seqs


def build_sample_seqs(raw_data, tokenizer, train_prefixes=TRAIN_PREFIXES, seed=SEED, seqlen=SEQLEN):
    """NB02 cell 11 build_sample_seqs, verbatim logic."""
    truncated_texts = tokenizer.batch_decode(
        tokenizer([t["text"] for t in raw_data], add_special_tokens=False, padding=False, truncation=True, max_length=seqlen).input_ids)
    n_seqs = len(truncated_texts)
    np.random.seed(seed)
    partner_lengths = ((np.random.beta(0.5, 4.0, size=n_seqs) * (seqlen / 2 + 1)).astype(int)).tolist()
    partner_texts = [
        tokenizer.decode(tokenizer(text["text"], add_special_tokens=False, padding=False, truncation=True, max_length=int(partner_lengths[i])).input_ids)
        for i, text in enumerate(raw_data)]
    perm = np.random.permutation(n_seqs)
    while np.any(perm == np.arange(n_seqs)):
        perm = np.random.permutation(n_seqs)
    sampled_prefixes = np.random.choice(train_prefixes, size=n_seqs)
    input_list = []
    for base_ix, base_text in enumerate(truncated_texts):
        partner_text = partner_texts[int(perm[base_ix])].strip()
        prefix = sampled_prefixes[base_ix]
        for seq in get_sample_seqs_for_input_seq(base_text, partner_text, prefix):
            input_list.append({"question_ix": base_ix, "question": base_text, **seq})
    return pd.DataFrame(input_list).assign(prompt_ix=lambda df: list(range(len(df))))


def token_table(input_df, tokenizer):
    """Token-level df with the columns label_content_roles needs (prompt_ix, token_ix, token)."""
    rows = []
    for r in input_df.itertuples():
        ids = tokenizer(r.prompt, add_special_tokens=False).input_ids
        toks = tokenizer.convert_ids_to_tokens(ids)
        rows += [(r.prompt_ix, i, t, tid) for i, (t, tid) in enumerate(zip(toks, ids))]
    return pd.DataFrame(rows, columns=["prompt_ix", "token_ix", "token", "token_id"])


def check(tokenizer, raw_data):
    input_df = build_sample_seqs(raw_data, tokenizer)
    print(f"[build] {input_df.question_ix.nunique()} base texts x roles -> {len(input_df)} prompts; roles={input_df.role.unique().tolist()}")
    for p in input_df[input_df.question_ix == 0].prompt:
        print("  " + p[:110].replace("\n", "\\n") + " ...")
    tok_df = token_table(input_df, tokenizer)
    lab = label_content_roles(MODEL_PREFIX, tok_df).merge(input_df[["prompt_ix", "role", "question_ix"]].rename(columns={"role": "target_role"}), on="prompt_ix")
    content = lab[(lab.is_content == True) & lab.role.notna()]  # noqa: E712
    mism = content[content.role != content.target_role]
    print(f"[label] content tokens={len(content)}; role/target mismatches={len(mism)}")
    counts = content.groupby("target_role").size()
    print("[label] content tokens per role:\n" + counts.to_string())
    print(f"[label] equal across roles: {counts.nunique() == 1}")
    first = content.groupby(["question_ix", "target_role"]).token_ix.min().unstack()
    print("[position] first content token index per role (header length):\n" + first.head(3).to_string())
    total = tok_df.groupby("prompt_ix").size().rename("n_tokens")
    print(f"[length] prompt token counts: min={total.min()} max={total.max()}")
    # Show the tool header exactly as tokenized, since the empty tool name is a code choice.
    tool_ix = input_df[(input_df.question_ix == 0) & (input_df.role == "tool")].prompt_ix.iloc[0]
    hdr = tok_df[(tok_df.prompt_ix == tool_ix)].token.head(12).tolist()
    print(f"[tool header] {hdr}")
    return input_df, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="tokenizer-only check on local stand-in texts")
    ap.add_argument("--out", default=None, help="write the prompt table (parquet) after streaming the real corpus")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    # Load from the local snapshot: transformers 4.57 queries the Hub for repo ids even in offline mode.
    snap = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots" / REVISION
    src = str(snap) if snap.exists() else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(src, add_eos_token=False, add_bos_token=False, padding_side="left")
    if args.check:
        texts = [(AUTHORS / "README.md").read_text(), (AUTHORS / "LICENSE.md").read_text(),
                 (ROOT / "MAIN_PAPER.md").read_text() if (ROOT / "MAIN_PAPER.md").exists() else "Short filler text about nothing in particular. " * 200]
        check(tokenizer, [{"text": t, "source": "local"} for t in texts])
        return 0
    if args.out:
        raw = load_raw_ds()
        df = build_sample_seqs(raw, tokenizer)
        df.to_parquet(args.out, index=False)
        print(f"[write] {len(df)} prompts -> {args.out}")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
