"""E4 step 1: build the real-conversation user-turn set and regenerate assistant turns with gpt-oss-20b.

Port of the authors' NB01 (experiments/role-analysis/01-get-conversations-data.ipynb), commit
ec333c40fd43fe991e1ebf66765051b6d7e35784:
  cell 3  load_raw_ds        ToxicChat prompts (lmsys/toxic-chat toxicchat1123 train), shuffle(seed 1234), 100-500 chars,
                             first 100. Gated dataset: loaded by default with the machine's HF login; if the download
                             fails the run falls back to OASST only and says so (--no-toxicchat skips it on purpose)
  cell 4  load_oasst_conversations  OpenAssistant/oasst1 train, lang en, tree_state ready_for_export,
                             first-child path from root, every prompter turn 100-500 chars, first 100 trees
  cell 5  user_queries_df    conv_id, dataset, user_query_ix, user_query; truncated to 2 user turns
  cell 8  generation         reasoning effort medium, max 4000 new tokens, temperature 1, top_p 1, top_k 0;
                             turn-2 history carries the turn-1 FINAL text only (not the CoT);
                             finish_reason == 'length' or empty final -> failed turn -> whole conversation dropped
  output  {model_prefix}.csv with columns conv_id, dataset, user_query_ix, user_query, cot, assistant, model, model_prefix

Divergences (logged in generation-stats.json):
  - backend: mlx_lm on mlx-community/gpt-oss-20b-MXFP4-Q8 (authors: OpenRouter provider nebius/fp4).
  - prompt: the tokenizer's native chat template (canonical "You are ChatGPT" system prompt, today's date,
    "Reasoning: medium"), tokenized without a BOS token, as the authors' local fallback (cell 11) does.
  - sampling: mlx_lm make_sampler(temp=1.0, top_p=1.0); mx.random.seed is set per generation batch and the
    seed is recorded per conversation turn (column `seed`). With --batch-size 1 the seed is per turn.
  - extra columns: seed, finish_reason, n_gen_tokens (dropped by nothing downstream; filter_convs ignores them).

Usage (smoke test, 2 conversations):
  python e4/generate_conversations.py --out e4/data/conversations --smoke 2
Full run (100 OASST conversations, as NB01 cell 4):
  python e4/generate_conversations.py --out e4/data/conversations --n-convs 100 --batch-size 8
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "probes"))
from build_probe_dataset import MODEL_PREFIX  # noqa: E402
from extract_activations import MLX_MODEL  # noqa: E402

SEED = 1234                      # NB01 cell 1
STD_PARAMS = {"temperature": 1.0, "top_p": 1.0, "top_k": 0}  # NB01 cell 1
MAX_TOKS = 4_000                 # NB01 cell 8
REASONING_EFFORT = "medium"      # NB01 cell 8
MODEL_NAME = "openai/gpt-oss-20b"


# ----------------------------------------------------------------------------- data (NB01 cells 3-5)
def load_toxicchat(n_samples=100, seed=SEED):
    """NB01 cell 3, verbatim logic. Gated dataset: needs HF auth."""
    from datasets import load_dataset
    ds = load_dataset("lmsys/toxic-chat", "toxicchat1123", split="train").shuffle(seed=seed)
    raw_data = []
    for sample in ds:
        user_text = sample["user_input"]
        if len(user_text) >= 100 and len(user_text) <= 500:
            raw_data.append([user_text])
        if len(raw_data) >= n_samples:
            break
    return raw_data


def load_oasst_conversations(n_samples=100):
    """NB01 cell 4, verbatim logic (no shuffle; first-child path; all prompter turns 100-500 chars)."""
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=False)
    rows = [
        {"message_tree_id": ex["message_tree_id"], "message_id": ex["message_id"], "parent_id": ex["parent_id"],
         "role": ex["role"], "text": ex["text"]}
        for ex in ds if ex["lang"] == "en" and ex["tree_state"] == "ready_for_export"
    ]
    trees = {}
    for r in rows:
        trees.setdefault(r["message_tree_id"], []).append(r)
    conversations = []
    for tid in list(trees.keys()):
        msgs = trees[tid]
        id2msg = {m["message_id"]: m for m in msgs}
        children = {}
        for m in msgs:
            if m["parent_id"]:
                children.setdefault(m["parent_id"], []).append(m["message_id"])
        roots = [m for m in msgs if not m["parent_id"]]
        if not roots:
            continue
        root = roots[0]
        path_ids = [root["message_id"]]
        cur_id = root["message_id"]
        while cur_id in children:
            child_id = children[cur_id][0]
            path_ids.append(child_id)
            cur_id = child_id
        conv = [id2msg[mid] for mid in path_ids]
        is_valid = True
        for msg in conv:
            if msg["role"] == "prompter" and not (100 <= len(msg["text"]) <= 500):
                is_valid = False
                break
        if is_valid:
            conversations.append(conv)
        if n_samples is not None and len(conversations) >= n_samples:
            break
    return conversations


def build_user_queries_df(oasst_user_only, toxicchat_ds):
    """NB01 cell 5, verbatim logic."""
    parts = [pd.DataFrame({"convs": oasst_user_only}).assign(dataset="oasst")]
    if toxicchat_ds:
        parts.append(pd.DataFrame({"convs": toxicchat_ds}).assign(dataset="toxicchat"))
    return (pd.concat(parts, ignore_index=True)
            .assign(conv_id=lambda df: range(len(df)))
            .explode("convs")
            .assign(user_query_ix=lambda df: df.groupby("conv_id").cumcount())
            .pipe(lambda df: df[df["user_query_ix"] <= 1])
            .rename(columns={"convs": "user_query"})
            [["conv_id", "dataset", "user_query_ix", "user_query"]]
            .reset_index(drop=True))


# ----------------------------------------------------------------------------- generation (NB01 cell 8, local backend)
ANALYSIS_RE = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?=<\|end\|>|<\|start\|>|<\|return\|>|$)", re.S)
FINAL_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?=<\|return\|>|<\|end\|>|$)", re.S)


def parse_harmony(text, finish_reason):
    """Split a raw Harmony completion into (reasoning, output), mirroring NB01 _validate_and_extract_response:
    early stop (finish_reason 'length') or an empty final channel -> (None, None)."""
    if finish_reason == "length":
        return {"reasoning": None, "output": None}
    finals = FINAL_RE.findall(text)
    output = finals[-1] if finals else None
    if output is None or output.strip() == "":
        return {"reasoning": None, "output": None}
    analyses = [a for a in ANALYSIS_RE.findall(text) if a.strip()]
    return {"reasoning": "\n\n".join(analyses) if analyses else None, "output": output}


def build_prompt_ids(tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, reasoning_effort=REASONING_EFFORT)
    return tokenizer.encode(text, add_special_tokens=False), text


def generate_batch(model, tokenizer, prompt_ids, sampler, seed, batch_size, max_tokens):
    """Generate completions for a list of token-id prompts. Returns (token lists, finish reasons, stats dict)."""
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator, stream_generate

    mx.random.seed(seed)
    t0 = time.time()
    if batch_size == 1:
        toks, fins, gen_tokens = [], [], 0
        for ids in prompt_ids:
            out, fin = [], None
            for r in stream_generate(model, tokenizer, prompt=ids, max_tokens=max_tokens, sampler=sampler):
                fin = r.finish_reason
                if fin != "stop":
                    out.append(r.token)
            toks.append(out)
            fins.append(fin)
            gen_tokens += len(out)
        el = time.time() - t0
        return toks, fins, {"generation_tokens": gen_tokens, "seconds": el, "prompt_tokens": sum(len(p) for p in prompt_ids),
                            "peak_gb": mx.get_peak_memory() / 1e9}
    gen = BatchGenerator(model, stop_tokens=[[t] for t in tokenizer.eos_token_ids], sampler=sampler,
                         completion_batch_size=batch_size, prefill_batch_size=min(batch_size, 4))
    uids = gen.insert(prompt_ids, [max_tokens] * len(prompt_ids))
    results = {u: [] for u in uids}
    fins = {u: None for u in uids}
    with gen.stats() as stats:
        while responses := gen.next_generated():
            for r in responses:
                if r.finish_reason is not None:
                    fins[r.uid] = r.finish_reason
                if r.finish_reason != "stop":
                    results[r.uid].append(r.token)
    gen.close()
    el = time.time() - t0
    return [results[u] for u in uids], [fins[u] for u in uids], {
        "generation_tokens": int(stats.generation_tokens), "seconds": el, "prompt_tokens": int(stats.prompt_tokens),
        "generation_tps_reported": float(stats.generation_tps), "prompt_tps_reported": float(stats.prompt_tps),
        "peak_gb": float(stats.peak_memory)}


def chunk(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data/conversations"))
    ap.add_argument("--n-convs", type=int, default=100, help="OASST conversations to take (NB01: 100); ToxicChat adds up to 100 more")
    ap.add_argument("--no-toxicchat", action="store_true", help="skip the 100 ToxicChat prompts (gated dataset; loaded by default)")
    ap.add_argument("--smoke", type=int, default=0, help="keep only the first N conversations")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_TOKS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-resume", action="store_true", help="ignore responses.jsonl from an earlier run")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "generate.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    # ---- user turns
    t0 = time.time()
    oasst_convs = load_oasst_conversations(args.n_convs)
    oasst_user_only = [[m["text"] for m in conv if m["role"] == "prompter"] for conv in oasst_convs]
    toxic = []
    if args.no_toxicchat:
        log("[data] ToxicChat skipped (--no-toxicchat): OASST only")
    else:
        try:
            toxic = load_toxicchat(100, args.seed)
            log(f"[data] ToxicChat loaded: {len(toxic)} prompts")
        except Exception as e:
            log(f"[data] WARNING: ToxicChat download failed ({type(e).__name__}: {str(e)[:200]}). FALLING BACK TO OASST ONLY.")
    uq = build_user_queries_df(oasst_user_only, toxic)
    if args.smoke:
        uq = uq[uq.conv_id < args.smoke].reset_index(drop=True)
    uq.to_csv(out / "user_queries.csv", index=False)
    n_turns = uq.groupby("conv_id").user_query_ix.max().add(1).value_counts().sort_index().to_dict()
    log(f"[data] {uq.conv_id.nunique()} conversations, {len(uq)} user turns, by dataset "
        f"{uq.groupby('dataset').conv_id.nunique().to_dict()}, turns per conv {n_turns} ({time.time() - t0:.0f}s)")

    # ---- model
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler
    model, tokenizer = load(MLX_MODEL)
    mx.eval(model.parameters())
    sampler = make_sampler(temp=STD_PARAMS["temperature"], top_p=STD_PARAMS["top_p"], top_k=STD_PARAMS["top_k"])
    log(f"[model] {MLX_MODEL} loaded, eos ids {sorted(tokenizer.eos_token_ids)}, mlx {mx.__version__}")

    # ---- resume
    ckpt = out / "responses.jsonl"
    responses = {}
    if ckpt.exists() and not args.no_resume:
        for line in ckpt.read_text().splitlines():
            r = json.loads(line)
            responses[(r["conv_id"], r["user_query_ix"])] = r
        log(f"[resume] {len(responses)} turns loaded from {ckpt}")
    ckpt_f = open(ckpt, "a")

    query_lookup = uq.set_index(["conv_id", "user_query_ix"])["user_query"].to_dict()
    max_rounds = int(uq.user_query_ix.max()) + 1
    failed = {cid for (cid, _), r in responses.items() if r["output"] is None}
    totals = {"generation_tokens": 0, "seconds": 0.0, "prompt_tokens": 0, "batches": 0, "peak_gb": 0.0}
    first_logged = False

    for round_ix in range(max_rounds):
        cur = uq[uq.user_query_ix == round_ix]
        todo = []
        for row in cur.itertuples():
            cid = int(row.conv_id)
            if (cid, round_ix) in responses:
                continue
            if cid in failed:
                responses[(cid, round_ix)] = {"conv_id": cid, "user_query_ix": round_ix, "reasoning": None, "output": None,
                                              "finish_reason": "prior_failure", "seed": None, "n_gen_tokens": 0}
                continue
            messages = []
            for prev_ix in range(round_ix):
                messages.append({"role": "user", "content": query_lookup[(cid, prev_ix)]})
                messages.append({"role": "assistant", "content": responses[(cid, prev_ix)]["output"]})  # final text only
            messages.append({"role": "user", "content": row.user_query})
            todo.append((cid, messages))
        log(f"[round {round_ix}] {len(todo)} turns to generate, {len(cur) - len(todo)} already done or failed")
        for b_ix, batch in enumerate(chunk(todo, args.batch_size)):
            ids_texts = [build_prompt_ids(tokenizer, m) for _, m in batch]
            prompt_ids = [i for i, _ in ids_texts]
            if not first_logged:
                log(f"[prompt] first prompt {len(prompt_ids[0])} tokens, starts {tokenizer.decode(prompt_ids[0][:8])!r}")
                first_logged = True
            seed = args.seed * 1000 + round_ix * 100000 + batch[0][0]  # unique per (round, first conv in batch)
            toks, fins, st = generate_batch(model, tokenizer, prompt_ids, sampler, seed, args.batch_size, args.max_new_tokens)
            for (cid, _), t, fin in zip(batch, toks, fins):
                raw = tokenizer.decode(t, skip_special_tokens=False)
                parsed = parse_harmony(raw, fin)
                rec = {"conv_id": cid, "user_query_ix": round_ix, **parsed, "finish_reason": fin, "seed": seed, "n_gen_tokens": len(t),
                       "raw": raw}
                responses[(cid, round_ix)] = rec
                ckpt_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if parsed["output"] is None:
                    failed.add(cid)
            ckpt_f.flush()
            for k in ("generation_tokens", "seconds", "prompt_tokens"):
                totals[k] += st[k]
            totals["batches"] += 1
            totals["peak_gb"] = max(totals["peak_gb"], st["peak_gb"])
            tps = st["generation_tokens"] / max(st["seconds"], 1e-9)
            log(f"[round {round_ix} batch {b_ix}] {len(batch)} prompts, {st['generation_tokens']} new tokens in {st['seconds']:.0f}s "
                f"= {tps:.1f} tok/s (batch throughput), finish {dict(pd.Series(fins).value_counts())}, "
                f"cumulative {totals['generation_tokens'] / max(totals['seconds'], 1e-9):.1f} tok/s, failed convs {len(failed)}")

    # ---- assemble (NB01 cell 8 tail)
    resp_df = pd.DataFrame([{k: v for k, v in r.items() if k != "raw"} for r in responses.values()]).rename(
        columns={"output": "assistant", "reasoning": "cot"})
    res = uq.merge(resp_df, on=["conv_id", "user_query_ix"], how="left").assign(model=MODEL_NAME, model_prefix=MODEL_PREFIX)
    res.to_csv(out / f"{MODEL_PREFIX}-raw.csv", index=False)
    is_invalid = res["assistant"].isna() | (res["assistant"].astype(str).str.strip() == "")
    poisoned = res.loc[is_invalid, "conv_id"].unique()
    clean = res[~res.conv_id.isin(poisoned)].copy()
    cols = ["conv_id", "dataset", "user_query_ix", "user_query", "cot", "assistant", "model", "model_prefix", "seed", "finish_reason", "n_gen_tokens"]
    clean[cols].to_csv(out / f"{MODEL_PREFIX}.csv", index=False)
    gen_tps = totals["generation_tokens"] / max(totals["seconds"], 1e-9)
    stats = {"model": MLX_MODEL, "toxicchat_included": bool(toxic), "reasoning_effort": REASONING_EFFORT, "max_new_tokens": args.max_new_tokens, "sampling": STD_PARAMS,
             "batch_size": args.batch_size, "seed_base": args.seed, "n_convs_requested": int(uq.conv_id.nunique()),
             "n_convs_clean": int(clean.conv_id.nunique()), "n_convs_dropped": int(len(poisoned)),
             "turns_generated_this_run": int(sum(1 for r in responses.values() if r.get("finish_reason") not in (None, "prior_failure"))),
             "generation_tokens_this_run": totals["generation_tokens"], "generation_seconds_this_run": round(totals["seconds"], 1),
             "generation_tok_per_s": round(gen_tps, 2), "prompt_tokens_this_run": totals["prompt_tokens"], "peak_gb": round(totals["peak_gb"], 2),
             "mean_new_tokens_per_turn": round(float(resp_df.n_gen_tokens.mean()), 1) if len(resp_df) else None,
             "cot_missing_turns": int(clean.cot.isna().sum()),
             "divergences": ["mlx_lm local generation on MXFP4-Q8 build (authors: OpenRouter nebius/fp4)",
                             "native chat template with today's date, no BOS", "seed per batch, recorded per turn"],
             "source": "NB01 cells 3-5, 8; commit ec333c40fd43fe991e1ebf66765051b6d7e35784"}
    (out / "generation-stats.json").write_text(json.dumps(stats, indent=2))
    log(f"[done] clean {stats['n_convs_clean']} convs ({len(clean)} turns), dropped {stats['n_convs_dropped']}; "
        f"{totals['generation_tokens']} new tokens in {totals['seconds'] / 60:.1f} min = {gen_tps:.1f} tok/s; "
        f"mean {stats['mean_new_tokens_per_turn']} new tokens/turn; wrote {out / f'{MODEL_PREFIX}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
