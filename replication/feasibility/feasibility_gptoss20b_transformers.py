"""Feasibility test: can gpt-oss-20b run on this MacBook for probe-style forward passes?

DRAFT. Not authorized to run until Hanan approves the download and the run.

What it does, in order:
  1. Downloads openai/gpt-oss-20b at the pinned revision (about 13 GB) into the HF cache.
  2. Loads it with transformers. On a Mac there is no CUDA, so transformers
     dequantizes the MXFP4 experts to bf16 (see quantizers/quantizer_mxfp4.py).
     Expected weight footprint: about 42 GB. Machine has 48 GB unified memory.
  3. Runs ONE forward pass on a short Harmony-formatted prompt with the same
     hooks the authors use (post_attention_layernorm output per layer), on the
     device given by --device (cpu or mps).
  4. Reports load time, peak memory, forward latency, and next-token sanity.

Exit codes: 0 feasible, 2 load failed, 3 forward failed.
"""
import argparse
import gc
import os
import resource
import sys
import time

MODEL_ID = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"  # same pin as role-steering run and HF cache


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "mps"], default="mps")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    path = snapshot_download(MODEL_ID, revision=REVISION)
    print(f"[download] {path} in {time.time() - t0:.0f}s", flush=True)
    if args.download_only:
        return 0

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
                                        add_eos_token=False, add_bos_token=False, padding_side="left")
    t0 = time.time()
    try:
        # dtype auto + no CUDA => transformers sets dequantize=True and materializes bf16 experts.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, dtype="auto", device_map=None,
            attn_implementation="eager", low_cpu_mem_usage=True)
        model = model.to(args.device).eval()
    except Exception as e:  # noqa: BLE001
        print(f"[load] FAILED after {time.time() - t0:.0f}s: {type(e).__name__}: {e}", flush=True)
        return 2
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] ok in {time.time() - t0:.0f}s; params={n_params / 1e9:.2f}B; "
          f"expert dtype={model.model.layers[0].mlp.experts.down_proj.dtype}; "
          f"experts class={type(model.model.layers[0].mlp.experts).__name__}; rss={rss_gb():.1f}GB", flush=True)
    if args.device == "mps":
        print(f"[load] mps driver allocated={torch.mps.driver_allocated_memory() / 1e9:.1f}GB", flush=True)

    # Authors' probe site: output of each layer's post_attention_layernorm (pre-MLP).
    captured = []
    handles = [layer.post_attention_layernorm.register_forward_hook(
        lambda m, i, o: captured.append(o.detach().float().cpu())) for layer in model.model.layers]

    prompt = "<|start|>user<|message|>" + ("The quick brown fox jumps over the lazy dog. " * 40) + "<|end|>"
    enc = tok(prompt, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=args.seq_len)
    enc = {k: v.to(args.device) for k, v in enc.items()}
    t0 = time.time()
    try:
        with torch.inference_mode():
            out = model(**enc, use_cache=False)
        if args.device == "mps":
            torch.mps.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"[forward] FAILED: {type(e).__name__}: {e}", flush=True)
        return 3
    finally:
        for h in handles:
            h.remove()
    dt = time.time() - t0
    n_tok = enc["input_ids"].shape[1]
    nxt = tok.decode([int(out.logits[0, -1].argmax())])
    print(f"[forward] {n_tok} tokens in {dt:.2f}s ({n_tok / dt:.0f} tok/s); layers captured={len(captured)}; "
          f"hs shape={tuple(captured[0].shape)}; next token={nxt!r}; rss={rss_gb():.1f}GB", flush=True)
    if args.device == "mps":
        print(f"[forward] mps driver allocated={torch.mps.driver_allocated_memory() / 1e9:.1f}GB", flush=True)
    # Rough budget for the authors' probe run: ~1,245 sequences x 1,024 tokens.
    est_h = 1245 * 1024 / (n_tok / dt) / 3600
    print(f"[estimate] full probe-training extraction at this rate: ~{est_h:.1f} h (batch=1, no padding)", flush=True)
    del out, model
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
