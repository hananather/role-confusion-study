"""E4 step 2: filter the regenerated conversations and render the four tagging conditions.

Port of the authors' NB02 cells 25-26 (experiments/role-analysis/02-train-role-probes.ipynb),
commit ec333c40fd43fe991e1ebf66765051b6d7e35784:
  cell 25 filter_convs   drop conversations with any missing user_query/assistant/cot; total length < 10,000
                         chars; every message >= 100 chars; at most 2 user turns; no message that is a substring
                         of another; then sample max_samples with random_state = 123 and renumber conv_id.
                         Messages per turn: user, cot, assistant.
  cell 26 conditions     test_prefix (BOS + canonical system prompt, probe.yaml) + separators (probe.yaml):
                           untagged     prefix + " " + contents joined by " "
                           tagged       prefix + " " + custom gptoss.j2 template over fold_cot_into_final(conv)
                           user_tagged  prefix + " " + render_single_message('user', contents joined by " ")
                           tool_tagged  prefix + " " + render_single_message('tool', contents joined by " ")
Output: prompts.parquet with prompt_ix, conv_type, conv_id, dataset, messages (JSON), prompt. Row order is
conv_type-major as NB02 cell 29 (untagged block, then tagged, user_tagged, tool_tagged).

max_samples: the shipped notebook uses 30 with the comment "100 for full test"; the paper says 200
(audit row 19). Default here is 100; pass --max-samples to change.

Usage:
  python e4/build_conditions.py --convs e4/data/conversations/gptoss-20b.csv --out e4/data/prompts.parquet
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "probes"))
from build_probe_dataset import AUTHORS, MODEL_PREFIX  # noqa: E402
from extract_activations import get_tokenizer  # noqa: E402
from utils.role_templates import fold_cot_into_final, load_chat_template, render_single_message  # noqa: E402

SEED = 123  # NB02 cell 1
CONV_TYPES = ["untagged", "tagged", "user_tagged", "tool_tagged"]  # NB02 cell 26 insertion order


def probe_config():
    return yaml.safe_load(open(AUTHORS / "experiments/role-analysis/config/probe.yaml"))[MODEL_PREFIX]


def filter_convs(user_queries_df, max_samples, max_total_len=10000, min_msg_len=100, max_user_messages_per_conv=2, seed=SEED, log=print):
    """NB02 cell 25, verbatim logic."""
    content_cols = ["user_query", "assistant", "cot"]
    log(f"Starting length: {user_queries_df['conv_id'].nunique()} convs")
    user_queries_df = user_queries_df.pipe(lambda df: df[~df["conv_id"].isin(df[df[content_cols].isna().any(axis=1)]["conv_id"].unique())])
    log(f"After dropping missing: {user_queries_df['conv_id'].nunique()} convs")
    user_queries_df = (
        user_queries_df
        .pipe(lambda df: df.assign(**{c: df[c].astype(str).str.strip() for c in content_cols}))
        .assign(
            row_total_len=lambda df: df[content_cols].apply(lambda r: r.str.len()).sum(axis=1),
            row_min_len=lambda df: df[content_cols].apply(lambda r: r.str.len()).min(axis=1),
        )
        .assign(
            total_len=lambda df: df.groupby("conv_id")["row_total_len"].transform("sum"),
            conv_min_len=lambda df: df.groupby("conv_id")["row_min_len"].transform("min"),
            user_turns=lambda df: df.groupby("conv_id")["user_query_ix"].transform("count"),
        )
        .query("total_len < @max_total_len and conv_min_len >= @min_msg_len and user_turns <= @max_user_messages_per_conv")
        .drop(columns=["row_total_len", "row_min_len", "total_len", "conv_min_len", "user_turns"])
    )
    log(f"After dropping long convs: {user_queries_df['conv_id'].nunique()} convs")

    def has_subtr_messages(g):
        msgs = sum([g[c].tolist() for c in ["user_query", "assistant", "cot"]], [])
        msgs = sorted(msgs, key=len)
        return any((i != j) and (a in b) for i, a in enumerate(msgs) for j, b in enumerate(msgs) if len(a) <= len(b) and (i < j))

    bad_conv_ids = (user_queries_df.groupby("conv_id", sort=False).apply(has_subtr_messages, include_groups=False)
                    .reset_index(name="bad").query("bad")["conv_id"].unique())
    user_queries_df = user_queries_df.pipe(lambda df: df[~df["conv_id"].isin(bad_conv_ids)])
    log(f"After dropping substr: {user_queries_df['conv_id'].nunique()} convs")

    def rows_to_messages(g):
        msgs = []
        for _, row in g.iterrows():
            msgs.append({"role": "user", "content": row["user_query"]})
            msgs.append({"role": "cot", "content": row["cot"]})
            msgs.append({"role": "assistant", "content": row["assistant"]})
        return msgs

    convs_df = (
        user_queries_df
        .sort_values(["conv_id", "user_query_ix"])
        .assign(conv_id=lambda df: df.groupby("conv_id", sort=False).ngroup())
        .groupby("conv_id", sort=False)
        .apply(lambda g: pd.Series({"dataset": g["dataset"].values[0], "messages": rows_to_messages(g)}), include_groups=False)
        .reset_index()
        [["conv_id", "dataset", "messages"]]
        .pipe(lambda df: df.sample(n=min(max_samples, len(df)), random_state=seed))
        .reset_index(drop=True)
        .assign(conv_id=lambda df: range(len(df)))
    )
    log(f"Final: {convs_df['conv_id'].nunique()} convs")
    return convs_df["messages"].tolist(), convs_df


def build_conditions(convs, tokenizer, test_prefix, seps):
    """NB02 cell 26, verbatim logic."""
    def prep_untagged_conv(conv, start_sep, role_seperator):
        return test_prefix + start_sep + role_seperator.join([x["content"] for x in conv])

    def prep_mistagged_conv(conv, start_sep, role_seperator, role):
        return test_prefix + start_sep + render_single_message(MODEL_PREFIX, role, role_seperator.join([x["content"] for x in conv]))

    out = {}
    out["untagged"] = [prep_untagged_conv(c, seps["untagged_start_sep"], seps["untagged_role_sep"]) for c in convs]
    out["tagged"] = [test_prefix + seps["tagged_start_sep"] + x
                     for x in tokenizer.apply_chat_template(fold_cot_into_final(convs), tokenize=False, add_generation_prompt=False)]
    out["user_tagged"] = [prep_mistagged_conv(c, seps["tagged_start_sep"], seps["untagged_role_sep"], "user") for c in convs]
    out["tool_tagged"] = [prep_mistagged_conv(c, seps["tagged_start_sep"], seps["untagged_role_sep"], "tool") for c in convs]
    assert all(len(v) == len(convs) for v in out.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", default=str(HERE / "data/conversations" / f"{MODEL_PREFIX}.csv"))
    ap.add_argument("--out", default=str(HERE / "data/prompts.parquet"))
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--show", type=int, default=1, help="print this conversation index in all four conditions")
    args = ap.parse_args()

    cfg = probe_config()
    test_prefix, seps = cfg["test_prefix"], cfg["test_seperators"]
    tokenizer = get_tokenizer()
    tokenizer.chat_template = load_chat_template(str(AUTHORS / "utils/chat_templates"), MODEL_PREFIX)

    convs, convs_df = filter_convs(pd.read_csv(args.convs), max_samples=args.max_samples, seed=args.seed)
    by_type = build_conditions(convs, tokenizer, test_prefix, seps)
    rows = []
    for conv_type in CONV_TYPES:
        for cid, prompt in enumerate(by_type[conv_type]):
            rows.append({"conv_type": conv_type, "conv_id": cid, "dataset": convs_df.dataset[cid],
                         "messages": json.dumps(convs_df.messages[cid], ensure_ascii=False), "prompt": prompt})
    df = pd.DataFrame(rows).assign(prompt_ix=lambda d: range(len(d)))[["prompt_ix", "conv_type", "conv_id", "dataset", "messages", "prompt"]]
    df.to_parquet(args.out, index=False)
    n_tok = df.prompt.map(lambda p: len(tokenizer(p, add_special_tokens=False).input_ids))
    print(f"[prompts] {len(df)} prompts = {len(convs)} convs x {len(CONV_TYPES)} conditions; tokens per prompt "
          f"min {n_tok.min()} mean {n_tok.mean():.0f} max {n_tok.max()}; total {n_tok.sum()}; wrote {args.out}", flush=True)
    ix = min(args.show, len(convs) - 1)
    for conv_type in CONV_TYPES:
        print(f"------------------- {conv_type.upper()} (conv {ix}) ------------------")
        p = by_type[conv_type][ix]
        print(p if len(p) < 1200 else p[:700] + "\n   [...]\n" + p[-400:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
