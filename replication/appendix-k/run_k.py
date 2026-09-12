"""Appendix K position experiment (Figure 32) plus the added controls, on MLX gpt-oss-20b.

Spec: audits/2026-09-10-appendix-k-replication-spec.md, section 1 (authors' POS cells 6, 8, 9,
14 to 27 and POSR cells 3 to 9) and section 4 controls that need no extra data.

Subcommands
  build    make the prompt table (ids spliced directly, no decode round trip) for one input set:
             --conversations CSV  (e4 output: conv_id, user_query_ix, dataset, user_query, cot, assistant)
             --neutral PARQUET    (runs/full-249/prompts.parquet: one base text per question_ix)
  extract  MLX forward passes, Recorder on the post_attention_layernorm output (authors' hook site),
           project every token with the suca probes on the fly, write per-token probabilities.
  analyze  mean Systemness by token index per condition (raw and EWMA decay 0.75), bootstrap CI over
           prompts, n per index, Figure 32 pair plus control panels (SVG), block-vs-baseline table.

Conditions (block = 61-token canonical system prompt unless noted; index counts BOS as 0)
  none       no insert (POS cells 8 to 13, the authors' unreported control)
  sys_t1     block at index 1 (POS cell 23; Figure 32 left)
  sys_t100   block at index 100 (POS cell 15; Figure 32 right)
  sys_t25, sys_t50, sys_t150   block at other indices (spec control 3)
  user_t100  same text under <|start|>user<|message|>...<|end|> at 100 (spec control 4)
  bare_t100  same text without tags at 100, 57 tokens (spec control 4, optional)
  bos2_t100  a second <|startoftext|> at 100 (spec control 6, optional)
  nobos      no leading BOS, no insert (spec control 7, optional)
Rows shorter than or equal to the insert index are dropped for that condition and n is recorded.

Divergences from the authors, all written to build_meta.json / metadata.json:
  - shuffle of the three segments is seeded (authors: unseeded sample(frac=1), POS cell 14)
  - ids are spliced directly and fed as ids (authors: decode then re-encode, POS cell 15);
    the round trip is checked per prompt and recorded as roundtrip_exact
  - batch 1, no padding (authors: batch 8, left padding to 250)
  - MLX MXFP4-Q8 weights, sklearn-style L-BFGS probes from runs/full-249 (authors: transformers + cuML)
  - shaded region 100..160 / 1..61 (authors: 100..161 in POSR cell 7)

Usage
  python appendix-k/run_k.py build   --neutral runs/full-249/prompts.parquet --out appendix-k/runs/neutral
  python appendix-k/run_k.py build   --conversations e4/conversations.csv --out appendix-k/runs/conv
  python appendix-k/run_k.py extract --run appendix-k/runs/neutral --probe-run runs/full-249 --layers 8,12,16
  python appendix-k/run_k.py analyze --run appendix-k/runs/neutral
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "probes"))

REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
MLX_MODEL = "mlx-community/gpt-oss-20b-MXFP4-Q8"
CANON = ("You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n"
         "Current date: 2025-12-31\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. "
         "Channel must be included for every message.")
BLOCK_TEXT = {"system": f"<|start|>system<|message|>{CANON}<|end|>",
              "user": f"<|start|>user<|message|>{CANON}<|end|>",
              "bare": CANON,
              "bos": "<|startoftext|>"}
# name -> (block kind, insert index); None = no insert
CONDITIONS = {"none": (None, None), "sys_t1": ("system", 1), "sys_t25": ("system", 25), "sys_t50": ("system", 50),
              "sys_t100": ("system", 100), "sys_t150": ("system", 150), "user_t100": ("user", 100),
              "bare_t100": ("bare", 100), "bos2_t100": ("bos", 100), "nobos": ("nobos", None)}
DEFAULT_CONDITIONS = "none,sys_t1,sys_t25,sys_t50,sys_t100,sys_t150,user_t100"
ROLE_CHAR = {"s": "system", "u": "user", "c": "cot", "a": "assistant", "t": "tool"}
SEED = 123
SLATE = "#62748e"
BLOCK_COLOR = "#d97706"
BASE_COLOR = "#9ca3af"


def get_tokenizer():
    from transformers import AutoTokenizer
    snap = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots" / REVISION
    return AutoTokenizer.from_pretrained(str(snap), add_eos_token=False, add_bos_token=False)


def make_logger(path):
    f = open(path, "a")

    def log(msg):
        print(msg, flush=True)
        f.write(msg + "\n")
        f.flush()
    return log


# ----------------------------------------------------------------------------- build
def load_base_rows(args, tokenizer, log):
    """One row per base text: text (BOS + content), provenance, first-segment role."""
    if args.conversations:
        raw = pd.read_csv(args.conversations)
        need = ["user_query", "cot", "assistant"]
        missing = [c for c in need if c not in raw.columns]
        if missing:
            raise SystemExit(f"conversation CSV lacks columns {missing}; has {list(raw.columns)}")
        raw = raw.head(args.n).reset_index(drop=True)  # POS cell 8: raw_convs.head(200); prompt_ix = row index
        nan_rows = raw[need].isna().any(axis=1)
        log(f"[build] conversations: {len(raw)} rows read, {int(nan_rows.sum())} with a null segment (dropped; authors' join would raise)")
        rng = np.random.default_rng(args.seed)
        rows = []
        for ix, r in raw.iterrows():
            if nan_rows[ix]:
                continue
            segs = [("user", str(r.user_query)), ("cot", str(r.cot)), ("assistant", str(r.assistant))]
            order = rng.permutation(3)  # authors: x.sample(frac=1), unseeded (POS cell 14 L6)
            segs = [segs[i] for i in order]
            rows.append({"prompt_ix": int(ix), "input_set": "conv", "conv_id": r.get("conv_id"), "dataset": r.get("dataset"),
                         "user_query_ix": r.get("user_query_ix"), "segment_order": "-".join(s[0] for s in segs),
                         "first_segment_role": segs[0][0], "text": tokenizer.bos_token + " ".join(s[1] for s in segs)})
        return pd.DataFrame(rows), {"input": str(args.conversations), "shuffle_seed": args.seed, "n_null_dropped": int(nan_rows.sum())}
    p = pd.read_parquet(args.neutral)
    q = p.drop_duplicates("question_ix").sort_values("question_ix").head(args.n).reset_index(drop=True)
    log(f"[build] neutral: {len(q)} base texts from {args.neutral} (first {args.n} question_ix)")
    rows = [{"prompt_ix": int(i), "input_set": "neutral", "conv_id": None, "dataset": "probe-corpus", "user_query_ix": None,
             "segment_order": None, "first_segment_role": None, "question_ix": int(r.question_ix),
             "text": tokenizer.bos_token + str(r.question)} for i, r in q.iterrows()]
    return pd.DataFrame(rows), {"input": str(args.neutral), "shuffle_seed": None,
                                "note": "base texts also appear (in role tags) in the probe training corpus under the authors' prompt split"}


def cmd_build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = make_logger(out / "build.log")
    tok = get_tokenizer()
    bos_id = tok.bos_token_id
    base, prov = load_base_rows(args, tok, log)
    blocks = {k: tok.encode(v, add_special_tokens=False) for k, v in BLOCK_TEXT.items()}
    assert len(blocks["system"]) == 61, len(blocks["system"])  # spec 1.4
    conds = args.conditions.split(",")
    unknown = [c for c in conds if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}; known: {list(CONDITIONS)}")
    recs, dropped = [], {c: 0 for c in conds}
    for r in base.itertuples(index=False):
        base_ids = tok.encode(r.text, add_special_tokens=False)
        assert base_ids[0] == bos_id
        for c in conds:
            kind, pos = CONDITIONS[c]
            if kind == "nobos":
                ids, blk, tags = base_ids[1:], None, []
            elif kind is None:
                ids, blk, tags = list(base_ids), None, []
            else:
                if len(base_ids) <= pos:  # POS cell 15 L10: skip if len(tokens) <= INSERT_POS
                    dropped[c] += 1
                    continue
                b = blocks[kind]
                ids = base_ids[:pos] + b + base_ids[pos:]
                blk = (pos, pos + len(b) - 1)
                tags = ([pos, pos + 1, pos + 2, blk[1]] if kind in ("system", "user") else [pos] if kind == "bos" else [])
            full_len = len(ids)
            ids = ids[:args.window]
            rt = tok.encode(tok.decode(ids), add_special_tokens=False) == ids if blk is not None else None
            recs.append({"prompt_ix": r.prompt_ix, "input_set": r.input_set, "condition": c, "block_kind": kind,
                         "insert_pos": pos, "block_start": blk[0] if blk else None,
                         "block_end": min(blk[1], args.window - 1) if blk else None,
                         "block_len": len(blocks[kind]) if blk else 0, "block_truncated": bool(blk and blk[1] > args.window - 1),
                         "tag_ix": [t for t in tags if t < args.window], "base_len": len(base_ids), "full_len": full_len,
                         "n_tokens": len(ids), "has_bos": ids[0] == bos_id, "roundtrip_exact": rt,
                         "first_segment_role": r.first_segment_role, "segment_order": r.segment_order,
                         "conv_id": r.conv_id, "dataset": r.dataset, "user_query_ix": r.user_query_ix,
                         "question_ix": getattr(r, "question_ix", None), "ids": ids})
    df = pd.DataFrame(recs)
    df.to_parquet(out / "prompts.parquet", index=False)
    n_per = df.groupby("condition").size().to_dict()
    meta = {"input_set": base.input_set.iloc[0], **prov, "n_base": int(len(base)), "window": args.window,
            "conditions": conds, "n_per_condition": n_per, "dropped_short": dropped,
            "block_tokens": {k: len(v) for k, v in blocks.items()},
            "roundtrip_exact_all": bool(df.roundtrip_exact.dropna().all()) if df.roundtrip_exact.notna().any() else None,
            "mean_tokens_per_prompt": float(df.n_tokens.mean()), "total_tokens": int(df.n_tokens.sum()),
            "hf_revision": REVISION, "bos_id": int(bos_id),
            "divergences": ["seeded segment shuffle (authors unseeded, POS cell 14)",
                            "ids spliced and fed directly, no decode/encode round trip (POS cell 15); round trip checked and recorded",
                            "prompt_ix is the CSV row index (matches authors); rows with a null segment dropped"],
            "source": "POS cells 8, 9, 14, 15, 23; commit ec333c40fd43fe991e1ebf66765051b6d7e35784"}
    (out / "build_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"[build] n per condition {n_per}; dropped (len <= insert index) {dropped}")
    log(f"[build] mean tokens/prompt {df.n_tokens.mean():.1f}, total {df.n_tokens.sum()}, round trip exact on all: {meta['roundtrip_exact_all']}")
    log(f"[build] wrote {out / 'prompts.parquet'}")
    return 0


# ----------------------------------------------------------------------------- extract
def load_probes(probe_run, split, space, layers):
    """Probe npz from fit_probes(_mlx).py: keys '{space}_L{ll}__coef' / '__intercept'."""
    probe_run = Path(probe_run)
    files = sorted(f for f in probe_run.glob("probes*.npz") if ("basesplit" in f.name) == (split == "base"))
    found, src = {}, {}
    for f in files:
        z = np.load(f)
        for l in layers:
            k = f"{space}_L{l:02d}"
            if l not in found and f"{k}__coef" in z:
                found[l] = (z[f"{k}__coef"].astype(np.float32), z[f"{k}__intercept"].astype(np.float32))
                src[l] = f.name
    missing = [l for l in layers if l not in found]
    if missing:
        raise SystemExit(f"no {space} probe (split={split}) for layers {missing} in {probe_run}; files: {[f.name for f in files]}")
    acc = {}
    for f in files:
        j = probe_run / f.name.replace(".npz", ".json")
        if j.exists():
            for r in json.loads(j.read_text())["results"]:
                if r["role_space"] == space and r["layer_ix"] in layers and src[r["layer_ix"]] == f.name:
                    acc[r["layer_ix"]] = r["acc"]
    return found, src, acc


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def cmd_extract(args):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    run = Path(args.run)
    log = make_logger(run / "extract.log")
    layers = [int(x) for x in args.layers.split(",")]
    roles = [ROLE_CHAR[c] for c in args.space]
    probes, probe_src, probe_acc = load_probes(args.probe_run, args.probe_split, args.space, layers)
    log(f"[probe] {args.space} split={args.probe_split} layers {layers} from {probe_src}; held-out acc {probe_acc}")
    pdf = pd.read_parquet(run / "prompts.parquet")
    if args.conditions:
        pdf = pdf[pdf.condition.isin(args.conditions.split(","))]
    if args.limit:
        pdf = pdf[pdf.prompt_ix.isin(sorted(pdf.prompt_ix.unique())[: args.limit])]
    pdf = pdf.reset_index(drop=True)
    tok = get_tokenizer()

    model, _ = load(MLX_MODEL)
    mx.eval(model.parameters())
    inner = model.model if hasattr(model, "model") else model
    store = {}

    class Recorder(nn.Module):  # same hook as probes/extract_activations.py
        def __init__(self, wrapped, idx):
            super().__init__()
            self.wrapped, self.idx = wrapped, idx

        def __call__(self, x):
            y = self.wrapped(x)
            if self.idx in store:
                store[self.idx] = y
            return y

    for i, layer in enumerate(inner.layers):
        if i in layers:
            store[i] = None
            layer.post_attention_layernorm = Recorder(layer.post_attention_layernorm, i)

    parts, pos, t0 = [], 0, time.time()
    for k, r in enumerate(pdf.itertuples(index=False)):
        ids = [int(x) for x in r.ids]
        n = len(ids)
        logits = model(mx.array([ids]))
        mx.eval(logits, *[store[l] for l in layers])
        tag_set = set(int(t) for t in r.tag_ix)
        in_block = np.zeros(n, bool)
        if r.block_start is not None and not pd.isna(r.block_start):
            in_block[int(r.block_start): int(r.block_end) + 1] = True
        base = pd.DataFrame({"prompt_ix": r.prompt_ix, "condition": r.condition, "token_in_prompt_ix": np.arange(n),
                             "token_id": ids, "is_bos": np.array(ids) == tok.bos_token_id, "in_block": in_block,
                             "is_tag": np.array([i in tag_set for i in range(n)])})
        for l in layers:
            hs = np.array(store[l][0].astype(mx.float32))
            coef, b = probes[l]
            p = softmax(hs @ coef.T + b)
            d = base.copy()
            d["layer"] = l
            for j, role in enumerate(roles):
                d[f"p_{role}"] = p[:, j]
            parts.append(d)
        pos += n
        if k == 0:
            nxt = tok.decode([int(mx.argmax(logits[0, -1]).item())])
            log(f"[fwd] first prompt {n} tokens, next token {nxt!r}, peak {mx.get_peak_memory() / 1e9:.1f} GB")
        if (k + 1) % 20 == 0 or k + 1 == len(pdf):
            el = time.time() - t0
            log(f"[fwd] {k + 1}/{len(pdf)} prompts, {pos} tokens, {pos / el:.0f} tok/s, {el / 60:.1f} min")
    out = pd.concat(parts, ignore_index=True)
    out["token"] = tok.convert_ids_to_tokens(out.token_id.tolist())
    out.to_parquet(run / "tokens.parquet", index=False)
    el = time.time() - t0
    meta = {"backend": "mlx", "model": MLX_MODEL, "mlx": mx.__version__, "batch_size": 1, "padding": "none",
            "site": "post_attention_layernorm output (pre-MLP)", "layers": layers, "space": args.space, "roles": roles,
            "probe_run": str(args.probe_run), "probe_split": args.probe_split, "probe_files": probe_src, "probe_heldout_acc": probe_acc,
            "n_prompts": int(len(pdf)), "n_tokens": int(pos), "n_per_condition": pdf.groupby("condition").size().to_dict(),
            "seconds": round(el, 1), "tok_per_s": round(pos / el, 1),
            "divergences": ["mlx backend, 8-bit non-expert weights (authors transformers bf16 + Triton MXFP4)",
                            "batch 1 no padding (authors batch 8 left-padded to 250, POS cell 17)",
                            "sklearn-style L-BFGS probes (authors cuML); probabilities kept at full float32, not rounded to 8 dp"],
            "source": "POS cells 6, 17, 18, 24, 25; utils/probes.py run_and_export_states"}
    (run / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"[done] {len(out)} token rows, {pos} tokens in {el / 60:.1f} min ({pos / el:.0f} tok/s) -> {run / 'tokens.parquet'}")
    return 0


# ----------------------------------------------------------------------------- analyze
def curve_matrix(d, value_col, T):
    """prompts x T matrix of value_col (NaN where the prompt has no token at that index)."""
    ps = sorted(d.prompt_ix.unique())
    M = np.full((len(ps), T), np.nan, np.float32)
    row = {p: i for i, p in enumerate(ps)}
    M[d.prompt_ix.map(row).to_numpy(), d.token_in_prompt_ix.to_numpy()] = d[value_col].to_numpy()
    return M, ps


def boot_mean(M, n_boot, rng):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(M, 0)
        n = np.sum(~np.isnan(M), 0)
        idx = rng.integers(0, len(M), (n_boot, len(M)))
        bm = np.stack([np.nanmean(M[i], 0) for i in idx])  # POSR cell 7: resample prompts, mean per index
        lo, hi = np.nanpercentile(bm, 2.5, 0), np.nanpercentile(bm, 97.5, 0)
    return mean, lo, hi, n


def ewma_per_prompt(d, col, alpha):
    # POSR cell 4: right-aligned weights 0.75^k normalised over all prior tokens == ewm(alpha=0.25, adjust=True)
    d = d.sort_values(["prompt_ix", "token_in_prompt_ix"])
    return d.groupby("prompt_ix")[col].transform(lambda s: s.ewm(alpha=alpha, adjust=True, ignore_na=True).mean())


def svg_panel(x0, y0, w, h, curves, title, T, block=None, ylab=True, xlab=True, n=None):
    """One panel. curves: list of dict(mean, lo, hi, color, dash, block_color) arrays of length T."""
    def X(i):
        return x0 + w * i / T

    def Y(v):
        return y0 + h * (1 - v)
    s = [f'<text x="{x0 + w / 2:.1f}" y="{y0 - 6}" text-anchor="middle" font-size="10" font-weight="600">{title}</text>']
    if block is not None:
        s.append(f'<rect x="{X(block[0]):.1f}" y="{y0}" width="{X(block[1] + 1) - X(block[0]):.1f}" height="{h}" fill="#e5e7eb"/>')
    s.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#374151" stroke-width="0.8"/>')
    for v in (0.25, 0.5, 0.75):
        s.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e5e7eb" stroke-width="0.5"/>')
    for i in range(0, T + 1, 50):
        s.append(f'<line x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{y0 + h}" y2="{y0 + h + 3}" stroke="#374151" stroke-width="0.8"/>')
        if xlab:
            s.append(f'<text x="{X(i):.1f}" y="{y0 + h + 12}" text-anchor="middle" font-size="8">{i}</text>')
    for v in (0, 0.5, 1):
        s.append(f'<line x1="{x0 - 3}" x2="{x0}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#374151" stroke-width="0.8"/>')
        if ylab:
            s.append(f'<text x="{x0 - 5}" y="{Y(v) + 3:.1f}" text-anchor="end" font-size="8">{v:g}</text>')
    for c in curves:
        m, lo, hi = c["mean"], c.get("lo"), c.get("hi")
        ok = ~np.isnan(m)
        if lo is not None:
            pts = [f"{X(i):.1f},{Y(hi[i]):.1f}" for i in range(T) if ok[i]] + [f"{X(i):.1f},{Y(lo[i]):.1f}" for i in reversed(range(T)) if ok[i]]
            s.append(f'<polygon points="{" ".join(pts)}" fill="{c["color"]}" fill-opacity="0.25" stroke="none"/>')
        dash = ' stroke-dasharray="3,2"' if c.get("dash") else ""
        segs, cur = [], []
        for i in range(T):
            if ok[i]:
                cur.append(i)
            elif cur:
                segs.append(cur)
                cur = []
        if cur:
            segs.append(cur)
        for seg in segs:
            pts = " ".join(f"{X(i):.1f},{Y(m[i]):.1f}" for i in seg)
            s.append(f'<polyline points="{pts}" fill="none" stroke="{c["color"]}" stroke-width="1.1"{dash}/>')
            if block is not None and c.get("block_color"):
                bseg = [i for i in seg if block[0] <= i <= block[1]]
                if bseg:
                    pts = " ".join(f"{X(i):.1f},{Y(m[i]):.1f}" for i in bseg)
                    s.append(f'<polyline points="{pts}" fill="none" stroke="{c["block_color"]}" stroke-width="1.4"/>')
    if n is not None:
        s.append(f'<text x="{x0 + w - 3}" y="{y0 + 10}" text-anchor="end" font-size="7" fill="#6b7280">n={int(np.nanmax(n))} at 0, {int(n[T - 1])} at {T - 1}</text>')
    return "\n".join(s)


def svg_figure(panels, ncols, T, path, panel_w=220, panel_h=110, caption=""):
    """panels: list of (title, curves, block, n). Writes an SVG grid with shared axis labels."""
    nrows = (len(panels) + ncols - 1) // ncols
    ml, mt, gx, gy = 34, 34, 22, 30
    W = ml + ncols * (panel_w + gx)
    H = mt + nrows * (panel_h + gy) + 14
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif" fill="#111827">',
         f'<rect width="{W}" height="{H}" fill="white"/>']
    for k, (title, curves, block, n) in enumerate(panels):
        r, c = divmod(k, ncols)
        x0, y0 = ml + c * (panel_w + gx), mt + r * (panel_h + gy)
        s.append(svg_panel(x0, y0, panel_w, panel_h, curves, title, T, block, ylab=(c == 0), xlab=True, n=n))
    s.append(f'<text x="{ml + (ncols * (panel_w + gx)) / 2:.1f}" y="{H - 3}" text-anchor="middle" font-size="9">Token index</text>')
    s.append(f'<text transform="translate(9,{mt + (nrows * (panel_h + gy)) / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="9">Systemness</text>')
    if caption:
        s.append(f'<text x="{ml}" y="{mt - 12}" font-size="8" fill="#6b7280">{caption}</text>')
    s.append("</svg>")
    Path(path).write_text("\n".join(s))


def cmd_analyze(args):
    run = Path(args.run)
    log = make_logger(run / "analyze.log")
    rng = np.random.default_rng(args.seed)
    tok = pd.read_parquet(run / "tokens.parquet")
    pr = pd.read_parquet(run / "prompts.parquet")
    bmeta = json.loads((run / "build_meta.json").read_text())
    T = int(bmeta["window"])
    layers = sorted(int(l) for l in tok.layer.unique()) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    conds = [c for c in DEFAULT_CONDITIONS.split(",") + [c for c in CONDITIONS if c not in DEFAULT_CONDITIONS] if c in set(tok.condition)]
    tok["content_p_system"] = tok.p_system.where(~(tok.is_bos | tok.is_tag))  # spec control 2
    blocks = {c: (int(g.block_start.iloc[0]), int(g.block_end.iloc[0])) for c, g in pr.groupby("condition") if pd.notna(g.block_start.iloc[0])}
    rows, curves = [], {}
    for l in layers:
        tl = tok[tok.layer == l]
        for c in conds:
            d = tl[tl.condition == c].copy()
            d["ewma_all"] = ewma_per_prompt(d, "p_system", args.ewma_alpha)
            d["ewma_content"] = ewma_per_prompt(d, "content_p_system", args.ewma_alpha)
            for token_set, smoothing, col in [("all", "raw", "p_system"), ("content", "raw", "content_p_system"),
                                              ("all", "ewma", "ewma_all"), ("content", "ewma", "ewma_content")]:
                M, ps = curve_matrix(d, col, T)
                mean, lo, hi, n = boot_mean(M, args.n_boot, rng)
                curves[(l, c, token_set, smoothing)] = (mean, lo, hi, n)
                rows.append(pd.DataFrame({"layer": l, "condition": c, "token_set": token_set, "smoothing": smoothing,
                                          "token_in_prompt_ix": np.arange(T), "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": n}))
    cdf = pd.concat(rows, ignore_index=True)
    cdf["input_set"] = bmeta["input_set"]
    cdf.to_csv(run / "curves.csv", index=False)
    log(f"[analyze] curves.csv: {len(cdf)} rows; layers {layers}; conditions {conds}; n_boot {args.n_boot}; ewma alpha {args.ewma_alpha}")

    # block vs same-index baseline (spec control table)
    brows = []
    for l in layers:
        tl = tok[tok.layer == l]
        base = tl[tl.condition == "none"] if "none" in conds else None
        for c, (b0, b1) in blocks.items():
            d = tl[(tl.condition == c) & tl.in_block]
            for token_set in ("all", "content"):
                dd = d if token_set == "all" else d[~d.is_tag]
                per_p = dd.groupby("prompt_ix").p_system.mean()
                rec = {"layer": l, "condition": c, "token_set": token_set, "block_start": b0, "block_end": b1, "n_prompts": int(len(per_p)),
                       "block_mean": float(per_p.mean())}
                if base is not None:
                    bb = base[(base.token_in_prompt_ix >= b0) & (base.token_in_prompt_ix <= b1) & base.prompt_ix.isin(per_p.index)]
                    if token_set == "content":
                        bb = bb[~(bb.is_bos | bb.is_tag)]
                    base_p = bb.groupby("prompt_ix").p_system.mean()
                    both = pd.concat([per_p.rename("blk"), base_p.rename("base")], axis=1).dropna()
                    diff = (both.blk - both.base).to_numpy()
                    rec.update({"baseline_same_index_mean": float(both.base.mean()) if len(both) else np.nan, "n_paired": int(len(both)),
                                "diff": float(diff.mean()) if len(diff) else np.nan, "diff_ci_lo": np.nan, "diff_ci_hi": np.nan})
                    if len(diff):
                        bd = diff[rng.integers(0, len(diff), (args.n_boot, len(diff)))].mean(1)
                        rec.update({"diff_ci_lo": float(np.percentile(bd, 2.5)), "diff_ci_hi": float(np.percentile(bd, 97.5))})
                brows.append(rec)
    bt = pd.DataFrame(brows)
    bt.to_csv(run / "block_table.csv", index=False)
    log("[analyze] block Systemness vs same-index no-insert baseline (per-prompt mean, then across prompts; CI = bootstrap over prompts):\n"
        + bt.round(4).to_string(index=False))

    # early-window strata by first segment role (spec control 10), conversation input only
    if pr.first_segment_role.notna().any():
        early = tok[(tok.token_in_prompt_ix < 100) & ~tok.is_bos].merge(pr[["prompt_ix", "condition", "first_segment_role"]], on=["prompt_ix", "condition"])
        st = (early[early.condition.isin(["none", "sys_t100"])].groupby(["layer", "condition", "prompt_ix", "first_segment_role"]).p_system.mean()
              .groupby(["layer", "condition", "first_segment_role"]).agg(["mean", "std", "count"]).reset_index())
        st.to_csv(run / "strata_first_segment.csv", index=False)
        log("[analyze] mean Systemness over indices 1..99 by first segment role:\n" + st.round(4).to_string(index=False))

    # figures
    figs = run / "figures"
    figs.mkdir(exist_ok=True)
    ttl = {"none": "No insert", "sys_t1": "System prompt at start (index 1)", "sys_t100": "System prompt at token 100",
           "sys_t25": "System prompt at 25", "sys_t50": "System prompt at 50", "sys_t150": "System prompt at 150",
           "user_t100": "User-tagged block at 100", "bare_t100": "Untagged block at 100", "bos2_t100": "Second BOS at 100", "nobos": "No BOS, no insert"}
    for l in layers:
        for smoothing in ("raw", "ewma"):
            for token_set in ("all", "content"):
                sm = "EWMA decay 0.75 per prompt, then mean" if smoothing == "ewma" else "raw mean"
                cap = f"{bmeta['input_set']} text, layer {l}, suca probe, {sm}, tokens: {token_set}; ribbon = 95% bootstrap over prompts"
                pair = []
                for c in ("sys_t1", "sys_t100"):
                    if (l, c, token_set, smoothing) in curves:
                        m, lo, hi, n = curves[(l, c, token_set, smoothing)]
                        pair.append((ttl[c], [{"mean": m, "lo": lo, "hi": hi, "color": SLATE, "block_color": BLOCK_COLOR}], blocks.get(c), n))
                if pair:
                    svg_figure(pair, 2, T, figs / f"fig32-L{l:02d}-{smoothing}-{token_set}.svg", caption=cap)
                panels = []
                for c in conds:
                    m, lo, hi, n = curves[(l, c, token_set, smoothing)]
                    cs = [{"mean": m, "lo": lo, "hi": hi, "color": SLATE, "block_color": BLOCK_COLOR}]
                    if c != "none" and (l, "none", token_set, smoothing) in curves:
                        cs.insert(0, {"mean": curves[(l, "none", token_set, smoothing)][0], "color": BASE_COLOR, "dash": True})
                    panels.append((ttl[c], cs, blocks.get(c), n))
                svg_figure(panels, 3, T, figs / f"controls-L{l:02d}-{smoothing}-{token_set}.svg", caption=cap + "; dashed = no-insert baseline")
    # layer comparison for the two headline conditions
    for c in ("sys_t1", "sys_t100"):
        panels = [(f"{ttl[c]}, layer {l}", [{"mean": curves[(l, c, "all", "ewma")][0], "lo": curves[(l, c, "all", "ewma")][1],
                                            "hi": curves[(l, c, "all", "ewma")][2], "color": SLATE, "block_color": BLOCK_COLOR}], blocks.get(c),
                   curves[(l, c, "all", "ewma")][3]) for l in layers if (l, c, "all", "ewma") in curves]
        if panels:
            svg_figure(panels, len(panels), T, figs / f"layers-{c}-ewma-all.svg", caption=f"{bmeta['input_set']} text, EWMA, all tokens")
    log(f"[done] figures in {figs}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--conversations", help="e4 conversation CSV")
    g.add_argument("--neutral", help="prompts.parquet with question_ix/question columns")
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--window", type=int, default=250)
    b.add_argument("--conditions", default=DEFAULT_CONDITIONS, help=f"comma list from {list(CONDITIONS)}")
    b.set_defaults(fn=cmd_build)
    e = sub.add_parser("extract")
    e.add_argument("--run", required=True)
    e.add_argument("--probe-run", default=str(ROOT / "runs/full-249"))
    e.add_argument("--probe-split", choices=["prompt", "base"], default="prompt", help="prompt = authors' split")
    e.add_argument("--space", default="suca")
    e.add_argument("--layers", default="8,12,16")
    e.add_argument("--conditions", default=None)
    e.add_argument("--limit", type=int, default=None, help="first N prompt_ix only (smoke tests)")
    e.set_defaults(fn=cmd_extract)
    a = sub.add_parser("analyze")
    a.add_argument("--run", required=True)
    a.add_argument("--layers", default="all")
    a.add_argument("--n-boot", type=int, default=200)
    a.add_argument("--ewma-alpha", type=float, default=0.25)
    a.add_argument("--seed", type=int, default=SEED)
    a.set_defaults(fn=cmd_analyze)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
