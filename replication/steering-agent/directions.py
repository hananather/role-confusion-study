# TOY LAB (MATS 12.0 application experiment): dummy secrets, loopback receiver, isolated sandbox; runtime-only activation edits on a local cached gpt-oss-20b, weights never modified. See ../../TOY-LAB-NOTICE.md. Never copy this notice into model-facing prompts, fixtures, or trajectories.
"""Class-mean role directions at the block-11 output (residual stream) of gpt-oss-20b on MLX.

Data: the dev-10 probe corpus already on disk (replication/runs/dev-10): 45 prompts, 9 base
texts x 5 roles, with per-token role labels in tokens.parquet (authors' labeling, unchanged).
Site: output of model.model.layers[L] (zero-based), i.e. the residual stream entering block L+1.
This is the steering site used by Hanan's earlier role-steering report and by Zhang, Lee, Park.
The probes read a different site (post_attention_layernorm of layer 12); the manipulation check
uses those probes downstream of the edit.

Output: directions/block{L}.npz with class means, unit directions, gap norms, random unit vectors,
and a JSON with counts. Nothing here generates text.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPL = HERE.parent
DEV10 = REPL / "runs" / "dev-10"
MLX_MODEL = "mlx-community/gpt-oss-20b-MXFP4-Q8"
HF_REV = "6cee5e81ee83917806bbde320786a8fb61efebee"
ROLES = ["system", "user", "cot", "assistant", "tool"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="11")
    ap.add_argument("--out", type=Path, default=HERE / "directions")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    import mlx.core as mx
    from mlx_lm import load
    from transformers import AutoTokenizer

    prompts = pd.read_parquet(DEV10 / "prompts.parquet").sort_values("prompt_ix")
    tokens = pd.read_parquet(DEV10 / "tokens.parquet")
    snap = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots" / HF_REV
    tok = AutoTokenizer.from_pretrained(str(snap), add_eos_token=False, add_bos_token=False)
    model, _ = load(MLX_MODEL)
    mx.eval(model.parameters())
    inner = model.model
    store = {}

    class Rec:
        def __init__(self, block, idx):
            self.block, self.idx = block, idx
        def __call__(self, x, mask, cache=None):
            y = self.block(x, mask, cache)
            store[self.idx] = y
            return y

    for l in layers:
        inner.layers[l] = Rec(inner.layers[l], l)

    sums = {l: {r: np.zeros(2880, np.float64) for r in ROLES} for l in layers}
    counts = {r: 0 for r in ROLES}
    norms = {l: [] for l in layers}
    t0 = time.time()
    for _, row in prompts.iterrows():
        ids = tok(row["prompt"], add_special_tokens=False).input_ids
        tt = tokens[tokens.prompt_ix == row["prompt_ix"]].sort_values("token_ix")
        assert len(tt) == len(ids), (len(tt), len(ids))
        assert (tt.token_id.values == np.array(ids)).all(), "token ids differ from tokens.parquet"
        logits = model(mx.array([ids]))
        mx.eval(logits, *[store[l] for l in layers])
        content = tt.is_content.values.astype(bool)
        roles = tt.role.values
        for l in layers:
            h = np.array(store[l][0].astype(mx.float32))
            norms[l].append(np.linalg.norm(h[content], axis=1))
            for r in ROLES:
                m = content & (roles == r)
                sums[l][r] += h[m].sum(0)
                if l == layers[0]:
                    counts[r] += int(m.sum())
    print(f"forward done {time.time()-t0:.0f}s counts={counts}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260911)
    for l in layers:
        means = {r: sums[l][r] / counts[r] for r in ROLES}
        tu = means["tool"] - means["user"]
        tc = means["tool"] - means["cot"]
        cu = means["cot"] - means["user"]
        unit = lambda v: (v / np.linalg.norm(v)).astype(np.float32)
        rand = [unit(rng.normal(size=2880)) for _ in range(3)]
        allnorm = np.concatenate(norms[l])
        out = {f"mean_{r}": means[r].astype(np.float32) for r in ROLES}
        out.update(tool_minus_cot=unit(tc), tool_minus_user=unit(tu), cot_minus_user=unit(cu),
                   gap_tool_cot=np.float32(np.linalg.norm(tc)), gap_tool_user=np.float32(np.linalg.norm(tu)),
                   random_0=rand[0], random_1=rand[1], random_2=rand[2])
        np.savez(args.out / f"block{l:02d}.npz", **out)
        meta = {"layer_zero_based": l, "site": "TransformerBlock output (residual stream)", "model": MLX_MODEL,
                "source": str(DEV10), "n_prompts": len(prompts), "counts": counts,
                "gap_tool_cot": float(np.linalg.norm(tc)), "gap_tool_user": float(np.linalg.norm(tu)),
                "cos_tu_tc": float(unit(tu) @ unit(tc)), "cos_tu_cu": float(unit(tu) @ unit(cu)),
                "residual_norm_median": float(np.median(allnorm)), "residual_norm_mean": float(allnorm.mean())}
        (args.out / f"block{l:02d}.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
