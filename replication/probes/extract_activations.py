"""Extract probe-training activations for gpt-oss-20b at the authors' hook site.

Port of NB02 cells 12-16 (dataloader, run_and_export_states, label roles), with the
model run through MLX locally or transformers on a CUDA host.

Site: output of each block's post_attention_layernorm (the authors' `all_pre_mlp_hidden_states`,
utils/pretrained_models/gptoss.py lines 59-62). Stored as fp16, one file per layer,
row order = token table order (`sample_ix`).

Divergences from the authors, logged in metadata.json:
  - backend: MLX (mlx-community MXFP4-Q8 build; 8-bit non-expert weights) vs Triton MXFP4 + flash-attn3.
  - batch size 1, no padding. Authors: batch 32, left padding, position_ids = arange(N) over pads,
    so their content positions shift by pad count for shorter sequences. Batch 1 gives every
    sequence positions starting at 0, which is what the paper describes.
  - layers: all 24 by default (Hanan, 2026-09-10). Authors: every second layer.
  - subset runs (--n-base) are for local development only.

Usage (local, real corpus, small subset):
  python extract_activations.py --n-base 10 --out runs/dev-10 --layers all
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_probe_dataset import (AUTHORS, MODEL_PREFIX, REVISION, load_raw_ds, build_sample_seqs, token_table)  # noqa: E402
from utils.role_assignments import label_content_roles  # noqa: E402

MLX_MODEL = "mlx-community/gpt-oss-20b-MXFP4-Q8"


def get_tokenizer():
    from transformers import AutoTokenizer
    snap = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots" / REVISION
    return AutoTokenizer.from_pretrained(str(snap), add_eos_token=False, add_bos_token=False, padding_side="left")


def run_mlx(prompts, tokenizer, layers, out_dir, log):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    model, _ = load(MLX_MODEL)
    mx.eval(model.parameters())
    inner = model.model if hasattr(model, "model") else model
    n_layers = len(inner.layers)
    layers = list(range(n_layers)) if layers == "all" else layers
    store = {}

    class Recorder(nn.Module):
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

    ids_list = [tokenizer(p, add_special_tokens=False).input_ids for p in prompts]
    total = sum(len(x) for x in ids_list)
    d = inner.layers[layers[0]].post_attention_layernorm.wrapped.weight.shape[-1]  # first wrapped layer, not layer 0
    files = {l: np.lib.format.open_memmap(out_dir / f"layer{l:02d}.npy", mode="w+", dtype=np.float16, shape=(total, d)) for l in layers}
    pos, t0 = 0, time.time()
    for k, ids in enumerate(ids_list):
        logits = model(mx.array([ids]))
        mx.eval(logits, *[store[l] for l in layers])
        n = len(ids)
        for l in layers:
            files[l][pos:pos + n] = np.array(store[l][0].astype(mx.float16))
        pos += n
        if k == 0:
            nxt = tokenizer.decode([int(mx.argmax(logits[0, -1]).item())])
            log(f"[fwd] first prompt {n} tokens, next token {nxt!r}, peak {mx.get_peak_memory() / 1e9:.1f} GB")
        if (k + 1) % 10 == 0 or k + 1 == len(ids_list):
            el = time.time() - t0
            log(f"[fwd] {k + 1}/{len(ids_list)} prompts, {pos} tokens, {pos / el:.0f} tok/s, {el / 60:.1f} min")
    for f in files.values():
        f.flush()
    return ids_list, layers, {"backend": "mlx", "model": MLX_MODEL, "mlx": mx.__version__, "batch_size": 1, "padding": "none"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=250, help="base texts to sample (authors: 250 -> 249)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="all", help="'all' or comma list, e.g. 10,12,14")
    ap.add_argument("--backend", choices=["mlx"], default="mlx")
    ap.add_argument("--prompts", default=None, help="reuse a prompt table parquet instead of streaming the corpus")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "extract.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    tokenizer = get_tokenizer()
    if args.prompts:
        input_df = pd.read_parquet(args.prompts)
    else:
        t0 = time.time()
        raw = load_raw_ds(n_sample_size=args.n_base)
        log(f"[corpus] {len(raw)} base texts in {time.time() - t0:.0f}s: " + json.dumps(pd.Series([r['source'] for r in raw]).value_counts().to_dict()))
        input_df = build_sample_seqs(raw, tokenizer)
        input_df.to_parquet(out / "prompts.parquet", index=False)
    log(f"[prompts] {len(input_df)} prompts from {input_df.question_ix.nunique()} base texts")

    layers = "all" if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    ids_list, layers, backend_meta = run_mlx(input_df.prompt.tolist(), tokenizer, layers, out, log)

    tok_df = token_table(input_df, tokenizer)
    assert sum(len(x) for x in ids_list) == len(tok_df)
    lab = (label_content_roles(MODEL_PREFIX, tok_df)
           .assign(sample_ix=lambda df: range(len(df)))
           .merge(input_df[["prompt_ix", "role", "question_ix"]].rename(columns={"role": "target_role"}), on="prompt_ix")
           .assign(match_target_role=lambda df: df.role == df.target_role))
    lab.to_parquet(out / "tokens.parquet", index=False)
    content = lab[(lab.is_content == True) & lab.role.notna() & lab.match_target_role]  # noqa: E712
    log("[label] content tokens per role: " + json.dumps(content.groupby("role").size().to_dict()))
    meta = {"model_prefix": MODEL_PREFIX, "hf_revision": REVISION, "site": "post_attention_layernorm output (pre-MLP)",
            "layers": layers, "n_prompts": len(input_df), "n_base_texts": int(input_df.question_ix.nunique()),
            "n_tokens": len(lab), "dtype": "float16", **backend_meta,
            "divergences": ["mlx backend, 8-bit non-expert weights", "batch 1 no padding (authors batch 32 left-padded)",
                            f"layers {layers if layers != list(range(24)) else 'all 24'} (authors every 2nd)"],
            "source": "NB02 cells 12-16, commit ec333c40fd43fe991e1ebf66765051b6d7e35784"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    log("[done] " + json.dumps({k: v for k, v in meta.items() if k != "divergences"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
