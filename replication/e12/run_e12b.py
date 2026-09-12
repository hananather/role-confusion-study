"""E12b: controlled test of "progressive role confusion" (EXPERIMENTS.md section E12, advocate X1,
critic sections 2 and 4). Local MLX only. No judge, no forgeries, no harmful text.

Three subcommands, run in order:

  generate  Sample 50 benign Alpaca instructions (cached HF parquet, offline). For each, generate the
            model's own chain of thought with gpt-oss-20b (canonical system prompt, Reasoning: medium,
            greedy, at most 900 new tokens) and keep the analysis-channel text.
  extract   For each prompt build user-turn readings: SEGMENT inside <|start|>user<|message|>...<|end|>
            after the canonical system prompt, where SEGMENT is
              own_cot   the prompt's own CoT truncated at L tokens
              shuffled  the same L tokens, token-shuffled (seeded per prompt)
              reversed  the same text, sentence order reversed
              neutral   length-matched held-out C4/Dolma text (base-split held-out texts of runs/full-249)
              shifted   100 tokens of neutral filler, then the own-CoT segment (position shift)
              genuine   the same CoT in its native assistant analysis channel after the question (reference)
            Forward pass with a Recorder on post_attention_layernorm (the authors' pre-MLP site, same hook
            as probes/extract_activations.py), project every token with the suca and sucat probes
            (prompt-split and base-split families when present) and store per-token role probabilities.
  analyze   Mean CoTness by position-in-segment per condition (mean per prompt first, then across prompts
            with a bootstrap CI), the same minus the paired neutral curve, the pre-registered window
            statistics and decision rules, and one figure per (layer, space, family).

Truncation modes for extract:
  --mode causal      (default) one forward pass per condition at L_max = min(800, n_cot). Attention is
                     causal, so the reading at position p does not depend on tokens after p; every shorter
                     L is a prefix of the same pass. Six times cheaper than per-length.
  --mode per-length  one pass per (condition, L) for L in --lengths with L <= n_cot, plus the full CoT.
                     The shuffled and reversed segments are then re-drawn per L. `analyze` reports the
                     max |delta p| across lengths at shared positions as a causal-consistency check.

Pre-registered decision rule (EXPERIMENTS.md E12b): the direction survives only if the own-CoT segment
keeps rising past 100 tokens after subtracting the neutral null and the shuffled control does not.
Operationalised in analyze(): rise = mean over positions [150, 200) minus mean over [60, 100) of the
paired null-subtracted CoTness, per prompt, bootstrap CI over prompts. Survives at a layer iff the
own_cot CI excludes zero from above, the shuffled CI does not exclude zero from above, and the paired
own_cot minus shuffled difference CI excludes zero from above. Also reported: the critic's climb
statistic (mean [60, 100) minus mean [0, 20), raw), the last-20-token value, a per-prompt OLS slope
over positions >= 100, and n_prompts per window.

Smoke test (2 prompts, two lengths, dev probes):
  python e12/run_e12b.py generate --run e12/runs/smoke --n-prompts 2
  python e12/run_e12b.py extract  --run e12/runs/smoke --probe-run runs/dev-10 --layers 8,12 \
        --mode per-length --lengths 50,100
  python e12/run_e12b.py analyze  --run e12/runs/smoke
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPL = HERE.parent
sys.path.insert(0, str(REPL / "probes"))

MLX_MODEL = "mlx-community/gpt-oss-20b-MXFP4-Q8"
HF_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
# probe.yaml test_prefix for gptoss-20b (authors' canonical system prompt, date pinned).
CANON_SYSTEM = ("You are ChatGPT, a large language model trained by OpenAI.\n"
                "Knowledge cutoff: 2024-06\nCurrent date: 2025-12-31\n\nReasoning: medium\n\n"
                "# Valid channels: analysis, commentary, final. Channel must be included for every message.")
BOS, START, END, MESSAGE, CHANNEL, RETURN, CALL = 199998, 200006, 200007, 200008, 200005, 200002, 200012
LENGTHS = [25, 50, 100, 200, 400, 800]
CONDITIONS = ["own_cot", "shuffled", "reversed", "neutral", "shifted", "genuine"]
FILLER_TOKENS = 100
SEED = 123
ROLE_CHAR = {"s": "system", "u": "user", "c": "cot", "a": "assistant", "t": "tool"}
ROLES_ALL = ["system", "user", "cot", "assistant", "tool"]
COLORS = {"own_cot": "#fd9a00", "shuffled": "#90a1b9", "reversed": "#62748e", "neutral": "#00a6f4",
          "shifted": "#0084d1", "genuine": "#00d492"}
STYLES = {"own_cot": "-", "shuffled": "--", "reversed": ":", "neutral": "-", "shifted": "-.", "genuine": "-"}
LABELS = {"own_cot": "own CoT in user turn", "shuffled": "token-shuffled", "reversed": "sentence-reversed",
          "neutral": "neutral text (null)", "shifted": "100 filler tokens + own CoT", "genuine": "genuine analysis channel"}
WINDOWS = {"early": (0, 20), "plateau": (60, 100), "late": (150, 200), "later": (350, 400)}


# ----------------------------------------------------------------------------- shared helpers
def log_to(run):
    run.mkdir(parents=True, exist_ok=True)
    f = open(run / "e12b.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


def get_tokenizer():
    from transformers import AutoTokenizer
    snap = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots" / HF_REVISION
    return AutoTokenizer.from_pretrained(str(snap), add_eos_token=False, add_bos_token=False)


def enc(tok, s):
    return tok.encode(s, add_special_tokens=False)


def system_ids(tok, bos):
    return ([BOS] if bos else []) + enc(tok, "<|start|>system<|message|>" + CANON_SYSTEM + "<|end|>")


def load_model():
    import mlx.core as mx
    from mlx_lm import load
    t0 = time.time()
    model, mlx_tok = load(MLX_MODEL)
    mx.eval(model.parameters())
    return model, mlx_tok, time.time() - t0


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def write_jsonl(p, rows):
    Path(p).write_text("".join(json.dumps(r) + "\n" for r in rows))


# ----------------------------------------------------------------------------- generate
def load_alpaca_prompts(n, seed):
    hub = Path.home() / ".cache/huggingface/hub/datasets--tatsu-lab--alpaca/snapshots"
    files = sorted(hub.glob("*/data/*.parquet"))
    if not files:
        raise FileNotFoundError("tatsu-lab/alpaca not in the HF cache; pass --prompts-file (jsonl with 'instruction')")
    df = pd.read_parquet(files[0], columns=["instruction", "input"])
    df = df[(df.input == "") & df.instruction.str.len().between(20, 160)].drop_duplicates("instruction")
    rng = np.random.default_rng(seed)
    pick = df.iloc[rng.choice(len(df), size=n, replace=False)]
    return [{"prompt_id": i, "source": f"tatsu-lab/alpaca:{files[0].parent.parent.name}", "instruction": s}
            for i, s in enumerate(pick.instruction.tolist())]


def cmd_generate(args):
    run = Path(args.run)
    log = log_to(run)
    if args.prompts_file:
        prompts = [{"prompt_id": i, "source": args.prompts_file, **r} for i, r in enumerate(read_jsonl(args.prompts_file))][: args.n_prompts]
    else:
        prompts = load_alpaca_prompts(args.n_prompts, args.seed)
    write_jsonl(run / "prompts.jsonl", prompts)
    log(f"[prompts] {len(prompts)} benign prompts -> {run / 'prompts.jsonl'}")

    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    tok = get_tokenizer()
    model, mlx_tok, dt = load_model()
    log(f"[load] {MLX_MODEL} in {dt:.0f}s")
    sys_ids = system_ids(tok, args.bos)
    # The MLX build ships its own tokenizer files; make sure ids agree with the pinned HF tokenizer.
    probe = "<|start|>user<|message|>Give three tips.<|end|><|start|>assistant<|channel|>analysis<|message|>We need to answer.<|end|>"
    assert mlx_tok.encode(probe, add_special_tokens=False) == enc(tok, probe), "mlx tokenizer disagrees with HF tokenizer"
    hdr = [CHANNEL] + enc(tok, "analysis") + [MESSAGE]
    sampler = make_sampler(temp=0.0)
    out_path = run / "generations.jsonl"
    done = {r["prompt_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    rows = read_jsonl(out_path) if out_path.exists() else []
    tot_tok, tot_sec = 0, 0.0
    for p in prompts:
        if p["prompt_id"] in done:
            continue
        prefix = sys_ids + enc(tok, "<|start|>user<|message|>" + p["instruction"] + "<|end|><|start|>assistant")
        gen, t0 = [], time.time()
        for r in stream_generate(model, mlx_tok, prompt=prefix, max_tokens=args.max_new, sampler=sampler):
            t = int(r.token)
            gen.append(t)
            if t in (END, RETURN, CALL):
                break
        sec = time.time() - t0
        has_analysis = gen[: len(hdr)] == hdr
        if has_analysis:
            body = gen[len(hdr):]
            closed = bool(body) and body[-1] == END
            cot_ids = body[:-1] if closed else body
        else:
            cot_ids, closed = [], False
        row = {**p, "gen_ids": gen, "cot_ids": cot_ids, "cot_text": tok.decode(cot_ids), "n_cot": len(cot_ids),
               "has_analysis": has_analysis, "cot_closed": closed, "n_generated": len(gen), "gen_seconds": round(sec, 2)}
        rows.append(row)
        write_jsonl(out_path, rows)
        tot_tok += len(gen)
        tot_sec += sec
        log(f"[gen] prompt {p['prompt_id']}: {len(gen)} new tokens in {sec:.1f}s ({len(gen) / sec:.1f} tok/s), "
            f"cot {len(cot_ids)} tokens, analysis={has_analysis} closed={closed}: {row['cot_text'][:90]!r}")
    meta = {"model": MLX_MODEL, "mlx": mx.__version__, "hf_revision": HF_REVISION, "system_prompt": CANON_SYSTEM, "bos": args.bos,
            "decoding": "greedy", "max_new_tokens": args.max_new, "stop_on": ["<|end|>", "<|return|>", "<|call|>"],
            "n_prompts": len(rows), "gen_tokens_per_sec": round(tot_tok / tot_sec, 1) if tot_sec else None,
            "n_cot_summary": pd.Series([r["n_cot"] for r in rows]).describe().round(1).to_dict()}
    (run / "generate-metadata.json").write_text(json.dumps(meta, indent=2))
    log("[done] " + json.dumps({k: v for k, v in meta.items() if k != "system_prompt"}))


# ----------------------------------------------------------------------------- extract
def load_probes(probe_run, spaces, layers, log):
    """Return {(family, space, layer): (coef, intercept, roles)} from every probes*.npz in probe_run."""
    from sklearn.model_selection import train_test_split  # noqa: F401  (same dependency as the fitting script)
    found, files = {}, {}
    for f in sorted(Path(probe_run).glob("probes*.npz")):
        family = "base" if "basesplit" in f.name else "prompt"
        z = np.load(f)
        for space in spaces:
            for l in layers:
                k = (family, space, l)
                if k not in found and f"{space}_L{l:02d}__coef" in z:
                    found[k] = (z[f"{space}_L{l:02d}__coef"].astype(np.float32), z[f"{space}_L{l:02d}__intercept"].astype(np.float32),
                                [ROLE_CHAR[c] for c in space])
                    files[k] = f.name
    for k, fn in files.items():
        log(f"[probe] {k[0]:6s} {k[1]:5s} L{k[2]:02d} <- {fn}")
    return found


def heldout_neutral_pool(tok, source_run):
    """Base-split held-out texts of the probe corpus (same split as fit_probes --split-by base)."""
    from sklearn.model_selection import train_test_split
    df = pd.read_parquet(Path(source_run) / "prompts.parquet", columns=["question_ix", "question"]).drop_duplicates("question_ix").sort_values("question_ix")
    _, te = train_test_split(df.question_ix.unique(), test_size=0.1, random_state=SEED)
    te = sorted(int(q) for q in te)
    texts = dict(zip(df.question_ix, df.question))
    return [(q, enc(tok, texts[q])) for q in te]


def take_neutral(pool, rng, n, avoid=()):
    """Concatenate held-out texts (random start text, in order, cyclic) until n tokens."""
    order = [i for i in range(len(pool)) if pool[i][0] not in avoid]
    start = int(rng.integers(len(order)))
    ids, used = [], []
    k = 0
    while len(ids) < n:
        q, t = pool[order[(start + k) % len(order)]]
        ids += t
        used.append(q)
        k += 1
    return ids[:n], used


def reverse_sentences(tok, ids):
    text = tok.decode(ids)
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return enc(tok, " ".join(reversed(sents)))


def build_segments(tok, gens, lengths, mode, pool, bos, log):
    """One row per forward pass: prompt_id, condition, length, and the chunk list [(ids, region)]."""
    sys_ids = system_ids(tok, bos)
    u_open, u_close = enc(tok, "<|start|>user<|message|>"), [END]
    a_open = enc(tok, "<|start|>assistant<|channel|>analysis<|message|>")
    rows = []
    for g in gens:
        cot = g["cot_ids"]
        n_cot = len(cot)
        rng = np.random.default_rng(SEED + g["prompt_id"])
        if mode == "causal":
            Ls = [min(max(lengths), n_cot)]
        else:
            Ls = [L for L in lengths if L <= n_cot] + ([n_cot] if n_cot < max(lengths) else [])
        Ls = sorted(set(Ls))
        # one neutral draw and one filler draw per prompt; both truncated per length
        neutral_full, neutral_src = take_neutral(pool, rng, max(Ls))
        filler, filler_src = take_neutral(pool, rng, FILLER_TOKENS, avoid=neutral_src)
        q_ids = enc(tok, g["instruction"])
        for L in Ls:
            seg = cot[:L]
            variants = {
                "own_cot": [(u_open, "header"), (seg, "segment"), (u_close, "close")],
                "shuffled": [(u_open, "header"), ([int(x) for x in rng.permutation(seg)], "segment"), (u_close, "close")],
                "reversed": [(u_open, "header"), (reverse_sentences(tok, seg), "segment"), (u_close, "close")],
                "neutral": [(u_open, "header"), (neutral_full[:L], "segment"), (u_close, "close")],
                "shifted": [(u_open, "header"), (filler, "filler"), (seg, "segment"), (u_close, "close")],
                "genuine": [(u_open, "header"), (q_ids, "question"), (u_close, "close"), (a_open, "header"), (seg, "segment"), (u_close, "close")],
            }
            for cond, chunks in variants.items():
                chunks = [(sys_ids, "system")] + chunks
                rows.append({"prompt_id": g["prompt_id"], "condition": cond, "length": L, "n_cot": n_cot,
                             "n_seg": sum(len(c) for c, r in chunks if r == "segment"), "chunks": chunks,
                             "neutral_src": neutral_src, "filler_src": filler_src})
    log(f"[segments] {len(rows)} forward passes for {len(gens)} prompts, mode={mode}, "
        f"{sum(sum(len(c) for c, _ in r['chunks']) for r in rows)} tokens total")
    return rows


def flatten(chunks):
    ids, region, pos = [], [], []
    for c, r in chunks:
        ids += c
        region += [r] * len(c)
        if r == "segment":
            pos += list(range(len(c)))
        elif r == "filler":
            pos += list(range(-len(c), 0))
        else:
            pos += [None] * len(c)
    return ids, region, pos


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def cmd_extract(args):
    run = Path(args.run)
    log = log_to(run)
    layers = [int(x) for x in args.layers.split(",")]
    spaces = args.spaces.split(",")
    lengths = [int(x) for x in args.lengths.split(",")]
    probes = load_probes(args.probe_run, spaces, layers, log)
    missing = [(s, l) for s in spaces for l in layers if not any(k[1] == s and k[2] == l for k in probes)]
    if missing and not args.allow_missing:
        raise SystemExit(f"no probe for {missing} in {args.probe_run}; fit them first or pass --allow-missing")
    if missing:
        log(f"[probe] WARNING missing {missing}")
    tok = get_tokenizer()
    gens = [g for g in read_jsonl(run / "generations.jsonl") if g["has_analysis"] and g["n_cot"] >= args.min_cot]
    if args.select == "longest":
        gens = sorted(gens, key=lambda g: -g["n_cot"])
    gens = sorted(gens[: args.n_prompts] if args.n_prompts else gens, key=lambda g: g["prompt_id"])
    log(f"[gens] {len(gens)} prompts with an analysis channel of >= {args.min_cot} tokens (select={args.select}); "
        f"n_cot min/median/max {min(g['n_cot'] for g in gens)}/{int(np.median([g['n_cot'] for g in gens]))}/{max(g['n_cot'] for g in gens)}")
    pool = heldout_neutral_pool(tok, args.neutral_run)
    log(f"[neutral] {len(pool)} held-out base texts from {args.neutral_run}, {sum(len(t) for _, t in pool)} tokens")
    segs = build_segments(tok, gens, lengths, args.mode, pool, args.bos, log)
    pd.DataFrame([{**{k: v for k, v in s.items() if k != "chunks"},
                   "ids": json.dumps(flatten(s["chunks"])[0]), "text": tok.decode(flatten(s["chunks"])[0])} for s in segs]
                 ).to_parquet(run / "segments.parquet", index=False)

    import mlx.core as mx
    import mlx.nn as nn
    model, _, dt = load_model()
    log(f"[load] {MLX_MODEL} in {dt:.0f}s")
    inner = model.model if hasattr(model, "model") else model
    store = {}

    class Recorder(nn.Module):
        def __init__(self, wrapped, idx):
            super().__init__()
            self.wrapped, self.idx = wrapped, idx

        def __call__(self, x):
            y = self.wrapped(x)
            store[self.idx] = y
            return y

    for i, layer in enumerate(inner.layers):
        if i in layers:
            layer.post_attention_layernorm = Recorder(layer.post_attention_layernorm, i)

    parts = run / "readings"
    parts.mkdir(exist_ok=True)
    by_prompt = {}
    for s in segs:
        by_prompt.setdefault(s["prompt_id"], []).append(s)
    tot_tok, t0 = 0, time.time()
    for k, (pid, rows) in enumerate(sorted(by_prompt.items())):
        part = parts / f"part-{pid:04d}.parquet"
        if part.exists():
            log(f"[fwd] prompt {pid} already extracted, skipping")
            continue
        frames = []
        for s in rows:
            ids, region, pos = flatten(s["chunks"])
            logits = model(mx.array([ids]))
            mx.eval(logits, *[store[l] for l in layers])
            n = len(ids)
            tot_tok += n
            base = pd.DataFrame({"prompt_id": np.int16(pid), "condition": s["condition"], "length": np.int16(s["length"]),
                                 "n_cot": np.int16(s["n_cot"]), "n_seg": np.int16(s["n_seg"]), "abs_pos": np.arange(n, dtype=np.int16),
                                 "pos_in_seg": pd.array(pos, dtype="Int16"), "region": region})
            for l in layers:
                hs = np.array(store[l][0].astype(mx.float32))
                for (family, space, ll), (coef, intercept, roles) in probes.items():
                    if ll != l:
                        continue
                    probs = softmax(hs @ coef.T + intercept)
                    df = base.copy()
                    df["layer"], df["space"], df["family"] = np.int8(l), space, family
                    for r in ROLES_ALL:
                        df[f"p_{r}"] = probs[:, roles.index(r)].astype(np.float32) if r in roles else np.float32(np.nan)
                    frames.append(df)
        pd.concat(frames, ignore_index=True).to_parquet(part, index=False)
        el = time.time() - t0
        log(f"[fwd] {k + 1}/{len(by_prompt)} prompts, {tot_tok} tokens, {tot_tok / el:.0f} tok/s, {el / 60:.1f} min, "
            f"peak {mx.get_peak_memory() / 1e9:.1f} GB")
    meta = {"model": MLX_MODEL, "mlx": mx.__version__, "site": "post_attention_layernorm output (pre-MLP)", "layers": layers,
            "spaces": spaces, "probe_run": str(args.probe_run), "probes": {f"{k[0]}_{k[1]}_L{k[2]:02d}": True for k in probes},
            "mode": args.mode, "lengths": lengths, "bos": args.bos, "n_prompts": len(by_prompt), "n_passes": len(segs),
            "neutral_source": f"{args.neutral_run} base-split held-out texts (seed {SEED})", "filler_tokens": FILLER_TOKENS,
            "fwd_tokens": tot_tok, "fwd_tokens_per_sec": round(tot_tok / (time.time() - t0), 1) if tot_tok else None,
            "conditions": CONDITIONS}
    (run / "extract-metadata.json").write_text(json.dumps(meta, indent=2))
    log("[done] " + json.dumps(meta))


# ----------------------------------------------------------------------------- analyze
def boot_ci(mat, rng, n_boot):
    """mat: prompts x buckets with NaN. Returns mean, lo, hi, n per bucket (bootstrap over prompts)."""
    n = mat.shape[0]
    mean = np.nanmean(mat, 0)
    cnt = np.sum(~np.isnan(mat), 0)
    if n < 2:
        return mean, mean, mean, cnt
    idx = rng.integers(0, n, (n_boot, n))
    with np.errstate(all="ignore"):
        bs = np.nanmean(mat[idx], 1)
    return mean, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0), cnt


def per_prompt_matrix(df, value, index_col="pos_bucket"):
    """Pivot prompt x bucket (mean within cell)."""
    return df.pivot_table(index="prompt_id", columns=index_col, values=value, aggfunc="mean")


def window_stats(pp, windows):
    """pp: prompt x pos (token level). Returns DataFrame prompt x window means and last20."""
    out = {}
    cols = pp.columns.to_numpy()
    for name, (a, b) in windows.items():
        sel = (cols >= a) & (cols < b)
        w = pp.loc[:, sel]
        full = w.notna().sum(1) >= (b - a)  # only prompts whose segment covers the whole window
        out[name] = w.mean(1).where(full)
    last = []
    for pid, row in pp.iterrows():
        v = row.dropna()
        last.append(v.iloc[-20:].mean() if len(v) >= 20 else np.nan)
    out["last20"] = pd.Series(last, index=pp.index)
    slopes = []
    for pid, row in pp.iterrows():
        v = row.dropna()
        v = v[v.index >= 100]
        slopes.append(np.polyfit(v.index.to_numpy(float), v.to_numpy(float), 1)[0] * 100 if len(v) >= 50 else np.nan)
    out["slope_per100_after100"] = pd.Series(slopes, index=pp.index)
    return pd.DataFrame(out)


def ci_row(s, rng, n_boot):
    v = s.dropna().to_numpy()
    if len(v) == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    m, lo, hi, n = boot_ci(v[:, None], rng, n_boot)
    return {"mean": float(m[0]), "lo": float(lo[0]), "hi": float(hi[0]), "n": int(n[0])}


def cmd_analyze(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(args.run)
    log = log_to(run)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    df = pd.concat([pd.read_parquet(p) for p in sorted((run / "readings").glob("part-*.parquet"))], ignore_index=True)
    seg = df[df.region.isin(["segment", "filler"])].copy()
    seg["pos_in_seg"] = seg.pos_in_seg.astype(int)
    log(f"[load] {len(df)} token rows, {seg.prompt_id.nunique()} prompts, layers {sorted(seg.layer.unique())}, "
        f"spaces {sorted(seg.space.unique())}, families {sorted(seg.family.unique())}, lengths {sorted(seg.length.unique())}")

    # Causal-consistency check: same prompt, condition, position read under different lengths (per-length mode only).
    multi = seg[seg.condition.isin(["own_cot", "neutral", "genuine"])]
    g = multi.groupby(["family", "space", "layer", "condition", "prompt_id", "pos_in_seg"]).p_cot.agg(["max", "min", "count"])
    g = g[g["count"] > 1]
    if len(g):
        log(f"[causal-check] max |delta p_cot| across lengths at shared positions: {(g['max'] - g['min']).max():.2e} over {len(g)} cells")
    # Keep one reading per (prompt, condition, position): the longest run.
    seg = seg.sort_values("length", ascending=False).drop_duplicates(["family", "space", "layer", "condition", "prompt_id", "pos_in_seg"])
    seg["pos_bucket"] = (seg.pos_in_seg // args.bucket) * args.bucket

    rng = np.random.default_rng(0)
    curves, windows, decisions = [], [], []
    keys = seg[["family", "space", "layer"]].drop_duplicates().sort_values(["family", "space", "layer"]).itertuples(index=False)
    for family, space, layer in keys:
        sub = seg[(seg.family == family) & (seg.space == space) & (seg.layer == layer)]
        # per-prompt token-level matrices, prompt x pos_in_seg
        tokmat = {c: per_prompt_matrix(sub[sub.condition == c], "p_cot", "pos_in_seg") for c in CONDITIONS if (sub.condition == c).any()}
        null = tokmat.get("neutral")
        for c, m in tokmat.items():
            # raw curve by bucket
            bm = per_prompt_matrix(sub[sub.condition == c], "p_cot", "pos_bucket")
            mean, lo, hi, n = boot_ci(bm.to_numpy(float), rng, args.n_boot)
            for b, (mu, l_, h_, nn_) in zip(bm.columns, zip(mean, lo, hi, n)):
                curves.append({"family": family, "space": space, "layer": layer, "condition": c, "kind": "raw", "pos_bucket": int(b),
                               "mean": mu, "lo": l_, "hi": h_, "n_prompts": int(nn_)})
            # other role arms (critic 1.6: CoTness must not just mirror a Userness fall)
            for r in ROLES_ALL:
                if sub[f"p_{r}"].notna().any() and r != "cot":
                    rm = per_prompt_matrix(sub[sub.condition == c], f"p_{r}", "pos_bucket")
                    rmean, rlo, rhi, rn = boot_ci(rm.to_numpy(float), rng, args.n_boot)
                    for b, (mu, l_, h_, nn_) in zip(rm.columns, zip(rmean, rlo, rhi, rn)):
                        curves.append({"family": family, "space": space, "layer": layer, "condition": c, "kind": f"raw_{r}", "pos_bucket": int(b),
                                       "mean": mu, "lo": l_, "hi": h_, "n_prompts": int(nn_)})
            # null-subtracted (paired per prompt and position)
            if null is not None and c != "neutral":
                d = m.subtract(null.reindex(index=m.index, columns=m.columns))
                dbm = d.T.groupby((d.columns // args.bucket) * args.bucket).mean().T
                dmean, dlo, dhi, dn = boot_ci(dbm.to_numpy(float), rng, args.n_boot)
                for b, (mu, l_, h_, nn_) in zip(dbm.columns, zip(dmean, dlo, dhi, dn)):
                    curves.append({"family": family, "space": space, "layer": layer, "condition": c, "kind": "minus_null", "pos_bucket": int(b),
                                   "mean": mu, "lo": l_, "hi": h_, "n_prompts": int(nn_)})
            # window statistics, raw and null-subtracted
            ws_raw = window_stats(m, WINDOWS)
            ws_raw["climb_critic"] = ws_raw.plateau - ws_raw.early
            stats = {"raw": ws_raw}
            if null is not None and c != "neutral":
                ws_d = window_stats(m.subtract(null.reindex(index=m.index, columns=m.columns)), WINDOWS)
                ws_d["rise_past_100"] = ws_d.late - ws_d.plateau
                ws_d["rise_past_100_later"] = ws_d.later - ws_d.plateau
                stats["minus_null"] = ws_d
            for kind, ws in stats.items():
                for col in ws.columns:
                    windows.append({"family": family, "space": space, "layer": layer, "condition": c, "kind": kind, "stat": col,
                                    **ci_row(ws[col], rng, args.n_boot)})
        # decision rule per (family, space, layer)
        if null is not None and "own_cot" in tokmat and "shuffled" in tokmat:
            wd = {c: window_stats(tokmat[c].subtract(null.reindex(index=tokmat[c].index, columns=tokmat[c].columns)), WINDOWS) for c in ["own_cot", "shuffled"]}
            own = ci_row(wd["own_cot"].late - wd["own_cot"].plateau, rng, args.n_boot)
            shf = ci_row(wd["shuffled"].late - wd["shuffled"].plateau, rng, args.n_boot)
            diff = ci_row((wd["own_cot"].late - wd["own_cot"].plateau) - (wd["shuffled"].late - wd["shuffled"].plateau), rng, args.n_boot)
            evaluable = own["n"] >= args.min_n
            survives = bool(evaluable and own["lo"] > 0 and not (shf["lo"] > 0) and diff["lo"] > 0)
            verdict = "not evaluable (too few prompts reach 200 tokens)" if not evaluable else ("SURVIVES" if survives else "DOES NOT SURVIVE")
            decisions.append({"family": family, "space": space, "layer": int(layer), "rule": "rise past 100 = mean[150,200) - mean[60,100), null-subtracted, paired over prompts",
                              "own_cot": own, "shuffled": shf, "own_minus_shuffled": diff, "min_n": args.min_n, "evaluable": evaluable,
                              "survives": survives, "verdict": verdict, "note": "n counts prompts whose CoT covers 200 tokens"})
            log(f"[decision] {family} {space} L{layer:02d}: own rise {own['mean']:+.3f} [{own['lo']:+.3f},{own['hi']:+.3f}] n={own['n']}; "
                f"shuffled {shf['mean']:+.3f} [{shf['lo']:+.3f},{shf['hi']:+.3f}]; diff {diff['mean']:+.3f} [{diff['lo']:+.3f},{diff['hi']:+.3f}] -> {verdict}")

    curves = pd.DataFrame(curves)
    windows = pd.DataFrame(windows)
    curves.to_csv(out / "curves.csv", index=False)
    windows.to_csv(out / "windows.csv", index=False)
    (out / "decision.json").write_text(json.dumps({"rule_source": "EXPERIMENTS.md E12b (2026-09-10)", "windows": WINDOWS,
                                                    "bucket": args.bucket, "n_boot": args.n_boot, "decisions": decisions}, indent=2))
    summ = windows[(windows.stat.isin(["climb_critic", "last20", "rise_past_100", "slope_per100_after100"]))]
    summ = summ.assign(cell=lambda d: d["mean"].round(3).astype(str) + " [" + d.lo.round(3).astype(str) + ", " + d.hi.round(3).astype(str) + "] n=" + d.n.astype(str))
    log("[windows]\n" + summ.pivot_table(index=["family", "space", "layer", "condition"], columns=["kind", "stat"], values="cell", aggfunc="first").to_string())

    # figures: one per (layer, space, family), two panels (raw, minus null)
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.5, "axes.axisbelow": True, "figure.dpi": 150})
    for (family, space, layer), cv in curves.groupby(["family", "space", "layer"]):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        for ax, kind, title in zip(axes, ["raw", "minus_null"], ["P(cot) by position in segment", "minus paired neutral-text null"]):
            for c in CONDITIONS:
                d = cv[(cv.condition == c) & (cv.kind == kind)].sort_values("pos_bucket")
                d = d[d.n_prompts >= args.min_n_plot]
                if d.empty:
                    continue
                x = d.pos_bucket + args.bucket / 2
                ax.plot(x, d["mean"], STYLES[c], color=COLORS[c], lw=1.6, label=LABELS[c])
                ax.fill_between(x, d.lo, d.hi, color=COLORS[c], alpha=0.15, lw=0)
                ax.annotate(c.replace("_", " "), (x.iloc[-1], d["mean"].iloc[-1]), xytext=(3, 0), textcoords="offset points", fontsize=7, color="#334155", va="center")
            ax.axvline(0, color="#94a3b8", lw=0.6)
            ax.axvline(100, color="#94a3b8", lw=0.6, ls=":")
            ax.set_title(title)
            ax.set_xlabel("token position in segment (filler at negative positions)")
        axes[0].set_ylabel(f"{space} probe P(cot), layer {layer}")
        axes[0].set_ylim(0, 1)
        axes[1].axhline(0, color="#94a3b8", lw=0.6)
        n_by = cv[(cv.condition == "own_cot") & (cv.kind == "raw")].set_index("pos_bucket").n_prompts
        ns = ", ".join(f"{p}: {int(n_by.get(p, 0))}" for p in [0, 100, 200, 400, 790] if p in n_by.index)
        fig.suptitle(f"E12b: CoTness of the model's own CoT pasted into a user turn, gpt-oss-20b, layer {layer}, {space} probe ({family} split). "
                     f"Bands: 95% bootstrap CI over prompts. n prompts at bucket {ns}", fontsize=8.5, y=1.02)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
        fig.tight_layout()
        fp = out / f"fig-L{layer:02d}-{space}-{family}.png"
        fig.savefig(fp, bbox_inches="tight")
        plt.close(fig)
        log(f"[fig] {fp}")
    log(f"[done] {out}")


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("generate")
    g.add_argument("--run", required=True)
    g.add_argument("--n-prompts", type=int, default=50)
    g.add_argument("--prompts-file", default=None, help="jsonl with an 'instruction' field; default samples tatsu-lab/alpaca from the HF cache")
    g.add_argument("--max-new", type=int, default=900)
    g.add_argument("--seed", type=int, default=SEED)
    g.add_argument("--no-bos", dest="bos", action="store_false", help="drop <|startoftext|> (authors' probe.yaml test_prefix includes it)")
    e = sp.add_parser("extract")
    e.add_argument("--run", required=True)
    e.add_argument("--probe-run", default=str(REPL / "runs/full-249"))
    e.add_argument("--neutral-run", default=str(REPL / "runs/full-249"), help="prompts.parquet source for held-out neutral texts")
    e.add_argument("--layers", default="8,12,16")
    e.add_argument("--spaces", default="suca,sucat")
    e.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    e.add_argument("--mode", choices=["causal", "per-length"], default="causal")
    e.add_argument("--min-cot", type=int, default=25, help="drop prompts whose CoT is shorter than this")
    e.add_argument("--n-prompts", type=int, default=None)
    e.add_argument("--select", choices=["first", "longest"], default="longest",
                   help="which prompts to keep when --n-prompts is set; 'longest' favours CoTs that reach the 400 and 800 windows")
    e.add_argument("--allow-missing", action="store_true")
    e.add_argument("--no-bos", dest="bos", action="store_false")
    a = sp.add_parser("analyze")
    a.add_argument("--run", required=True)
    a.add_argument("--bucket", type=int, default=10)
    a.add_argument("--n-boot", type=int, default=2000)
    a.add_argument("--min-n", type=int, default=20, help="prompts needed in the late window for the decision rule to be evaluable")
    a.add_argument("--min-n-plot", type=int, default=1, help="hide curve buckets with fewer prompts than this")
    args = ap.parse_args()
    {"generate": cmd_generate, "extract": cmd_extract, "analyze": cmd_analyze}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
