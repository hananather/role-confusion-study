"""Fast probe fitting on the Mac GPU: same objective as sklearn/cuML L2 multinomial logistic
regression, L-BFGS via scipy with loss and gradient computed in MLX.

Objective (sklearn convention, C = 5e-3):  mean cross-entropy + ||W||^2 / (2 * C * n_train).
Same split rules as fit_probes.py (--split-by prompt | base). Output format identical.

Layout: one layer file is loaded into memory ONCE (sequential read), moved to the GPU as float16,
and every role space and both splits are fitted from it by GPU-side row gathers. This avoids the
random-access memmap reads that stalled the first version.

Divergence: solver (scipy L-BFGS on GPU gradients) instead of cuML's QN solver; same convex objective.

Usage: python fit_probes_mlx.py --run runs/full-249 --layers 0,1,20,21,22,23 [--role-spaces ...] [--splits prompt,base]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

SEED = 123
C = 5.0e-3
ROLE_CHAR = {"s": "system", "u": "user", "c": "cot", "a": "assistant", "t": "tool"}
AUTHORS_SPACES = ["ua", "uat", "uca", "ucat", "sua", "suat", "suca", "sucat"]


def fit_lbfgs(mx, x, y, k, max_iter=5000):
    """x: mx.array float32 (n, d) already on device; y: mx.array int32 (n,)."""
    n, d = x.shape
    lam = 1.0 / (C * n)

    def loss_fn(params):
        w = params[: k * d].reshape(k, d)
        b = params[k * d:]
        z = x @ w.T + b
        lse = mx.logsumexp(z, axis=1)
        picked = mx.take_along_axis(z, y[:, None], axis=1)[:, 0]
        return mx.mean(lse - picked) + 0.5 * lam * mx.sum(w * w)

    grad_fn = mx.value_and_grad(loss_fn)
    calls = {"n": 0}

    def f(p):
        val, g = grad_fn(mx.array(p.astype(np.float32)))
        mx.eval(val, g)
        calls["n"] += 1
        return float(val.item()), np.array(g, dtype=np.float64)

    res = minimize(f, np.zeros(k * d + k), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "maxcor": 30, "ftol": 1e-12, "gtol": 1e-6})
    w = res.x[: k * d].reshape(k, d).astype(np.float32)
    b = res.x[k * d:].astype(np.float32)
    return w, b, {"n_iter": int(res.nit), "n_calls": calls["n"], "loss": float(res.fun), "converged": bool(res.success)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--role-spaces", default=",".join(AUTHORS_SPACES))
    ap.add_argument("--layers", default="all")
    ap.add_argument("--splits", default="prompt,base")
    args = ap.parse_args()
    import mlx.core as mx
    run = Path(args.run)
    meta = json.loads((run / "metadata.json").read_text())
    layers = meta["layers"] if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    lab = pd.read_parquet(run / "tokens.parquet")
    base = lab[(lab.is_content == True) & lab.role.notna() & lab.match_target_role]  # noqa: E712
    splits = args.splits.split(",")
    spaces = args.role_spaces.split(",")

    # Precompute the index sets once (they do not depend on the layer).
    plan = {}
    for space in spaces:
        roles = [ROLE_CHAR[ch] for ch in space]
        roles_map = {r: i for i, r in enumerate(roles)}
        df = base[base.role.isin(roles)]
        for split in splits:
            if split == "prompt":
                tr_ix, te_ix = train_test_split(df.prompt_ix.unique(), test_size=0.1, random_state=SEED)
                tr, te = df[df.prompt_ix.isin(tr_ix)], df[df.prompt_ix.isin(te_ix)]
            else:
                tr_q, te_q = train_test_split(df.question_ix.unique(), test_size=0.1, random_state=SEED)
                tr, te = df[df.question_ix.isin(tr_q)], df[df.question_ix.isin(te_q)]
            plan[(space, split)] = (roles, roles_map, mx.array(tr.sample_ix.to_numpy().astype(np.int32)),
                                    mx.array(tr.role.map(roles_map).to_numpy().astype(np.int32)),
                                    mx.array(te.sample_ix.to_numpy().astype(np.int32)),
                                    te.role.map(roles_map).to_numpy())

    results = {s: [] for s in splits}
    probes = {s: {} for s in splits}
    for l in layers:
        f = run / f"layer{l:02d}.npy"
        if not f.exists():
            print(f"[skip] layer {l} not on disk", flush=True)
            continue
        t0 = time.time()
        X = mx.array(np.load(f))  # sequential read, float16 on device
        mx.eval(X)
        print(f"[load] layer {l}: {X.shape} in {time.time() - t0:.0f}s", flush=True)
        for (space, split), (roles, roles_map, tr_idx, y_tr, te_idx, y_te_np) in plan.items():
            t1 = time.time()
            x_tr = mx.take(X, tr_idx, axis=0).astype(mx.float32)
            mx.eval(x_tr)
            w, b, info = fit_lbfgs(mx, x_tr, y_tr, len(roles))
            del x_tr
            x_te = mx.take(X, te_idx, axis=0).astype(mx.float32)
            z = np.array(x_te @ mx.array(w).T + mx.array(b))
            del x_te
            pred = z.argmax(1)
            acc = float((pred == y_te_np).mean())
            by_role = {r: round(float((pred[y_te_np == i] == i).mean()), 4) if (y_te_np == i).any() else None for r, i in roles_map.items()}
            results[split].append({"role_space": space, "layer_ix": l, "acc": round(acc, 4), "acc_by_role": by_role,
                                   "n_train": int(tr_idx.shape[0]), "n_test": int(te_idx.shape[0]), **info, "sec": round(time.time() - t1, 1)})
            probes[split][f"{space}_L{l:02d}"] = {"coef": w, "intercept": b}
            print(f"[fit] {split:6s} {space:5s} layer {l:2d}: acc {acc:.3f} iters {info['n_iter']} ({time.time() - t1:.0f}s)", flush=True)
        del X
    suffix = f"-L{'-'.join(str(l) for l in layers)}" if args.layers != "all" else ""
    for split in splits:
        tag = "" if split == "prompt" else "-basesplit"
        np.savez(run / f"probes{tag}{suffix}.npz", **{f"{k}__{kk}": v for k, d in probes[split].items() for kk, v in d.items()})
        (run / f"probes{tag}{suffix}.json").write_text(json.dumps({
            "split_by": split, "C": C, "penalty": "l2", "fit_intercept": True, "objective": "mean CE + ||W||^2/(2 C n)",
            "solver": "scipy L-BFGS-B with MLX gradients (authors: cuml)", "role_char": ROLE_CHAR, "results": results[split],
            "source": "NB02 cells 18-20, commit ec333c40fd43fe991e1ebf66765051b6d7e35784"}, indent=2))
        if results[split]:
            print(f"--- {split} split ---")
            print(pd.DataFrame(results[split]).pivot(index="layer_ix", columns="role_space", values="acc").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
