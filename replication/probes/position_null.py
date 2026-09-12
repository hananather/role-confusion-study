"""E12a: position-only null curve. Mean probability of each role by token position on
held-out NEUTRAL text, from the full-249 activations and the fitted probes. No model run.

This is the curve Appendix H lacks for CoTness and Assistantness (Appendix K did Systemness
only). Any "climb with position" in Figure 25 must be read against it. Uses the base-text
split probes (probes-basesplit-L*.npz) so the held-out texts were never seen in training.

Usage: python position_null.py --run runs/full-249 --space suca --layers 10,12,14 --out runs/full-249/position-null-suca.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROLE_CHAR = {"s": "system", "u": "user", "c": "cot", "a": "assistant", "t": "tool"}
SEED = 123


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def load_probe(run, space, layer):
    for f in sorted(run.glob("probes-basesplit-L*.npz")):
        z = np.load(f)
        k = f"{space}_L{layer:02d}"
        if f"{k}__coef" in z:
            return z[f"{k}__coef"], z[f"{k}__intercept"]
    raise FileNotFoundError(f"no base-split probe for {space} layer {layer} in {run}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--space", default="suca")
    ap.add_argument("--layers", default="10,12,14")
    ap.add_argument("--bucket", type=int, default=25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    roles = [ROLE_CHAR[c] for c in args.space]
    lab = pd.read_parquet(run / "tokens.parquet")
    base = lab[(lab.is_content == True) & lab.role.notna() & lab.match_target_role & lab.role.isin(roles)]  # noqa: E712
    # same held-out base texts as fit_probes --split-by base
    _, te_q = train_test_split(base.question_ix.unique(), test_size=0.1, random_state=SEED)
    te = base[base.question_ix.isin(te_q)].copy()
    te["pos_bucket"] = (te.token_in_seg_ix.astype(int) // args.bucket) * args.bucket
    rows = []
    for layer in [int(x) for x in args.layers.split(",")]:
        f = run / f"layer{layer:02d}.npy"
        if not f.exists():
            print(f"[skip] layer {layer}: file not on disk")
            continue
        coef, intercept = load_probe(run, args.space, layer)
        hs = np.load(f, mmap_mode="r")
        probs = softmax(hs[te.sample_ix.to_numpy()].astype(np.float32) @ coef.T + intercept)
        pdf = pd.concat([te[["question_ix", "role", "pos_bucket"]].reset_index(drop=True), pd.DataFrame(probs, columns=roles)], axis=1)
        long = pdf.melt(id_vars=["question_ix", "role", "pos_bucket"], var_name="target_role", value_name="prob")
        # mean per text first, then across texts (intervals across texts, never tokens)
        per_text = long.groupby(["role", "target_role", "pos_bucket", "question_ix"]).prob.mean().reset_index()
        agg = per_text.groupby(["role", "target_role", "pos_bucket"]).prob.agg(["mean", "std", "count"]).reset_index()
        agg["layer_ix"] = layer
        rows.append(agg)
        diag = agg[agg.role == agg.target_role].pivot(index="pos_bucket", columns="role", values="mean").round(3)
        print(f"[layer {layer}] mean P(true role) by position bucket (held-out texts, n texts per bucket varies):\n{diag.to_string()}", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"[done] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
