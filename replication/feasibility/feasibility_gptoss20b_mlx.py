"""Feasibility test: gpt-oss-20b on Apple Silicon via MLX, with the authors' probe hook site.

Approved 2026-09-10 (see ../EXECUTION.md).

Steps:
  1. Load an MLX build of gpt-oss-20b (default: mlx-community/gpt-oss-20b-MXFP4-Q8,
     experts in native MXFP4, other weights 8-bit, ~12 GB on disk).
  2. Wrap every block's post_attention_layernorm so its output is recorded.
     That is the authors' "pre-MLP hidden state" site (utils/pretrained_models/gptoss.py
     lines 59-62 and demo/role-probe-demo.ipynb cell 4).
  3. Run one forward pass on a Harmony-formatted probe-style prompt; report
     latency, peak memory, hidden-state shapes, and the argmax next token.
  4. Run a short greedy generation with the standard system prompt as a sanity check.
  5. Optionally (--dump) save layer-12 hidden states and final logits to .npz so they can
     be compared against the transformers bf16 path on the same prompt.

Divergences from the authors' stack to record in any result:
  - MLX kernels instead of Triton MXFP4 kernels and flash-attn3.
  - Non-expert weights are 8-bit in the community build (authors: bf16).
  - Fidelity must be checked against a reference (authors compared ASR to hosted APIs).
"""
import argparse
import sys
import time

MODEL_ID = "mlx-community/gpt-oss-20b-MXFP4-Q8"
SYSTEM = ("You are ChatGPT, a large language model trained by OpenAI.\n"
          "Knowledge cutoff: 2024-06\nCurrent date: 2026-01-01\n\nReasoning: medium\n\n"
          "# Valid channels: analysis, commentary, final. Channel must be included for every message.")


def peak_gb(mx):
    for fn in (getattr(mx, "get_peak_memory", None), getattr(getattr(mx, "metal", None), "get_peak_memory", None)):
        if fn is not None:
            try:
                return fn() / 1e9
            except Exception:  # noqa: BLE001
                continue
    return float("nan")


def active_gb(mx):
    for fn in (getattr(mx, "get_active_memory", None), getattr(getattr(mx, "metal", None), "get_active_memory", None)):
        if fn is not None:
            try:
                return fn() / 1e9
            except Exception:  # noqa: BLE001
                continue
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--dump", default=None, help="path to .npz for hidden states + logits")
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    import mlx_lm
    from mlx_lm import load, generate

    print(f"[env] mlx={mx.__version__} mlx_lm={getattr(mlx_lm, '__version__', '?')}", flush=True)
    t0 = time.time()
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())
    print(f"[load] {args.model} in {time.time() - t0:.0f}s; active={active_gb(mx):.1f}GB peak={peak_gb(mx):.1f}GB", flush=True)

    inner = model.model if hasattr(model, "model") else model
    layers = inner.layers
    print(f"[arch] n_layers={len(layers)}; block attrs={[k for k in vars(layers[0]).keys() if not k.startswith('_')]}", flush=True)

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

    for i, layer in enumerate(layers):
        layer.post_attention_layernorm = Recorder(layer.post_attention_layernorm, i)

    # Probe-style prompt: bare user message, no system prompt, like the authors' training data.
    body = "The quick brown fox jumps over the lazy dog. " * 60
    prompt = "<|start|>user<|message|>" + body + "<|end|>"
    ids = tokenizer.encode(prompt, add_special_tokens=False)[: args.seq_len]
    x = mx.array([ids])
    t0 = time.time()
    logits = model(x)
    mx.eval(logits, *store.values())
    dt = time.time() - t0
    n = len(ids)
    hs12 = store.get(12)
    nxt = tokenizer.decode([int(mx.argmax(logits[0, -1]).item())])
    print(f"[forward] {n} tokens in {dt:.2f}s ({n / dt:.0f} tok/s); layers recorded={len(store)}; "
          f"hs[12] shape={tuple(hs12.shape) if hs12 is not None else None} dtype={hs12.dtype if hs12 is not None else None}; "
          f"next token={nxt!r}; peak={peak_gb(mx):.1f}GB", flush=True)
    est_h = 1245 * 1024 / (n / dt) / 3600
    print(f"[estimate] authors' probe extraction (1,245 seqs x 1,024 tok) at this rate: ~{est_h:.1f} h", flush=True)
    per_layer_gb = 1245 * 1024 * hs12.shape[-1] * 2 / 1e9 if hs12 is not None else float("nan")
    print(f"[estimate] fp16 activation storage: {per_layer_gb:.1f} GB per layer, {per_layer_gb * len(layers):.0f} GB for all {len(layers)} layers", flush=True)

    if args.dump:
        np.savez(args.dump, ids=np.array(ids), logits=np.array(logits[0, -1].astype(mx.float32)),
                 hs12=np.array(hs12[0].astype(mx.float32)))
        print(f"[dump] wrote {args.dump}", flush=True)

    if not args.skip_generate:
        store.clear()
        chat = ("<|start|>system<|message|>" + SYSTEM + "<|end|>"
                "<|start|>user<|message|>Write a haiku about GPUs<|end|><|start|>assistant")
        t0 = time.time()
        out = generate(model, tokenizer, prompt=chat, max_tokens=120, verbose=False)
        print(f"[generate] {time.time() - t0:.1f}s\n{out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
