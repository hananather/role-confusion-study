# TOY LAB (MATS 12.0 application experiment). Runtime-only activation statistics on gpt-oss-20b; weights never modified.
"""Role directions and their geometry (experiment 6), computed on the pod.

  python directions.py geometry --probes-dir /workspace/results/20260911T030050Z/probes-full \
      --acts-dir /workspace/acts/20260911T030050Z/probes-full --out directions
  python directions.py block --probes-dir ... --layers 5,8,11,14,17 --out directions [--n-prompts 300]

`geometry`: from the saved activation cube (post-attention-layernorm site, all 24 layers) and the token table,
compute class means for the five roles per layer and the full cosine matrix among the ten pairwise difference
directions, plus the trained-probe-row version. Writes geometry.json and geometry.png. No model needed.

`block`: forward the probe-corpus prompts (prompts.parquet) through the model with hooks on the requested
blocks, take class means at each block OUTPUT (the steering site), and write blockLL.npz with unit vectors
tool_minus_cot, tool_minus_user, cot_minus_user, user_minus_cot, gaps, and three random unit vectors.
"""
from __future__ import annotations
import argparse, json, sys
from itertools import combinations
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROLES5, save_json  # noqa: E402

PAIRS = list(combinations(ROLES5, 2))


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def geometry(args):
    import pandas as pd
    t = pd.read_parquet(Path(args.probes_dir) / "tokens.parquet")
    content = t.is_content.values.astype(bool); roles = t.role.values
    probes = np.load(Path(args.probes_dir) / "probes.npz")
    out = {"layers": {}, "pairs": [f"{a}-{b}" for a, b in PAIRS]}
    for L in range(24):
        f = Path(args.acts_dir) / f"layer{L:02d}.npy"
        if not f.exists():
            continue
        X = np.load(f, mmap_mode="r")
        means = {r: np.asarray(X[content & (roles == r)]).astype(np.float32).mean(0) for r in ROLES5}
        dirs = {f"{a}-{b}": unit(means[a] - means[b]) for a, b in PAIRS}
        names = list(dirs); M = np.stack([dirs[n] for n in names])
        cos = (M @ M.T).tolist()
        rec = {"cos_matrix_class_means": cos,
               "cos_tool_user__tool_cot": float(dirs["user-tool"] @ dirs["cot-tool"]),
               "cos_tool_user__cot_user": float(unit(means["tool"] - means["user"]) @ unit(means["cot"] - means["user"])),
               "gap_tool_cot": float(np.linalg.norm(means["tool"] - means["cot"])), "gap_tool_user": float(np.linalg.norm(means["tool"] - means["user"]))}
        key = f"sucat_L{L:02d}__coef"
        if key in probes:
            c = probes[key]; row = {r: c[i] for i, r in enumerate(ROLES5)}
            pd_ = {f"{a}-{b}": unit(row[a] - row[b]) for a, b in PAIRS}
            rec["cos_tool_user__tool_cot_probe_rows"] = float(unit(row["tool"] - row["user"]) @ unit(row["tool"] - row["cot"]))
            rec["cos_tool_user__cot_user_probe_rows"] = float(unit(row["tool"] - row["user"]) @ unit(row["cot"] - row["user"]))
        out["layers"][L] = rec
        print(json.dumps({"layer": L, "cos_tu_tc": round(rec["cos_tool_user__tool_cot"], 3), "cos_tu_cu": round(rec["cos_tool_user__cot_user"], 3)}), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_json(Path(args.out) / "geometry.json", out)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        Ls = sorted(out["layers"]); fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(Ls, [out["layers"][L]["cos_tool_user__cot_user"] for L in Ls], "o-", color="#fd9a00", label="cos(Tool−User, CoT−User)")
        ax.plot(Ls, [out["layers"][L]["cos_tool_user__tool_cot"] for L in Ls], "s-", color="#00a6f4", label="cos(Tool−User, Tool−CoT)")
        ax.axhline(0, color="#9ca3af", lw=0.8); ax.set_xlabel("layer"); ax.set_ylabel("cosine"); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(Path(args.out) / "geometry.png", dpi=200)
    except Exception as e:  # plotting is optional on the pod
        print("plot skipped:", e)


def block(args):
    import pandas as pd, torch
    from common import Model
    layers = [int(x) for x in args.layers.split(",")]
    prompts = pd.read_parquet(Path(args.probes_dir) / "prompts.parquet").sort_values("prompt_ix")
    tokens = pd.read_parquet(Path(args.probes_dir) / "tokens.parquet")
    if args.n_prompts:
        prompts = prompts.head(args.n_prompts)
    m = Model(loader=args.loader)
    store = {}
    def mk(L):
        def h(module, a, o):
            store[L] = (o[0] if isinstance(o, tuple) else o)[0].float().cpu().numpy()
        return h
    handles = [m.layers[L].register_forward_hook(mk(L)) for L in layers]
    sums = {L: {r: np.zeros(m.lm.config.hidden_size, np.float64) for r in ROLES5} for L in layers}
    counts = {r: 0 for r in ROLES5}; norms = {L: [] for L in layers}
    for _, row in prompts.iterrows():
        ids = m.tokenizer(row["prompt"], add_special_tokens=False).input_ids
        tt = tokens[tokens.prompt_ix == row["prompt_ix"]].sort_values("token_ix")
        if len(tt) != len(ids):
            print(json.dumps({"skip_prompt": int(row["prompt_ix"]), "table_tokens": len(tt), "tokenized": len(ids)}), flush=True)
            continue
        with torch.inference_mode():
            m.lm(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1)
        content = tt.is_content.values.astype(bool); roles = tt.role.values
        for L in layers:
            h = store[L]; norms[L].append(np.linalg.norm(h[content], axis=1))
            for r in ROLES5:
                sel = content & (roles == r); sums[L][r] += h[sel].sum(0)
                if L == layers[0]:
                    counts[r] += int(sel.sum())
    for hd in handles:
        hd.remove()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260911)
    for L in layers:
        mean = {r: sums[L][r] / counts[r] for r in ROLES5}
        d = {"tool_minus_cot": mean["tool"] - mean["cot"], "tool_minus_user": mean["tool"] - mean["user"],
             "cot_minus_user": mean["cot"] - mean["user"], "user_minus_cot": mean["user"] - mean["cot"]}
        arrays = {k: unit(v) for k, v in d.items()}
        arrays.update({f"gap_{k}": np.float32(np.linalg.norm(v)) for k, v in d.items()})
        for i in range(3):
            arrays[f"random_{i}"] = unit(rng.normal(size=mean["tool"].shape))
        arrays.update({f"mean_{r}": mean[r].astype(np.float32) for r in ROLES5})
        np.savez(Path(args.out) / f"block{L:02d}.npz", **arrays)
        meta = {"layer": L, "site": "block output", "counts": counts, "n_prompts": len(prompts), "residual_norm_median": float(np.median(np.concatenate(norms[L]))),
                **{f"gap_{k}": float(np.linalg.norm(v)) for k, v in d.items()},
                "cos_tu_tc": float(arrays["tool_minus_user"] @ arrays["tool_minus_cot"]), "cos_tu_cu": float(arrays["tool_minus_user"] @ arrays["cot_minus_user"])}
        save_json(Path(args.out) / f"block{L:02d}.json", meta); print(json.dumps(meta), flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geometry"); g.add_argument("--probes-dir", required=True); g.add_argument("--acts-dir", required=True); g.add_argument("--out", default=str(HERE / "directions"))
    b = sub.add_parser("block"); b.add_argument("--probes-dir", required=True); b.add_argument("--layers", default="5,8,11,14,17"); b.add_argument("--out", default=str(HERE / "directions"))
    b.add_argument("--n-prompts", type=int, default=0); b.add_argument("--loader", default="authors")
    args = ap.parse_args()
    {"geometry": geometry, "block": block}[args.cmd](args)


if __name__ == "__main__":
    main()
