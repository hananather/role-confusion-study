"""Fit role probes on extracted activations, following NB02 cells 18-20.

Authors: cuml LogisticRegression(penalty='l2', C=5e-3, fit_intercept=True, max_iter=5000,
linesearch_max_iter=100), no feature scaling for gptoss-20b, split 90/10 by prompt_ix with
seed 123, content tokens only, one probe per (role space, layer).

Divergences, logged in probes.json:
  - sklearn lbfgs instead of cuML on the Mac (same objective up to solver tolerance). On the
    H100 use cuML to match exactly.
  - sklearn train_test_split instead of cuml.train_test_split: same 90/10 rule, different RNG
    stream, so the held-out prompts differ from the authors' run.
  - role spaces: the authors' eight gptoss-20b combinations by default.

Usage:
  python fit_probes.py --run runs/dev-10 --role-spaces suca,sucat,uat
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 123
C = 5.0e-3
ROLE_CHAR = {"s": "system", "u": "user", "c": "cot", "a": "assistant", "t": "tool"}
AUTHORS_SPACES = ["ua", "uat", "uca", "ucat", "sua", "suat", "suca", "sucat"]  # NB02 cell 19 order


def fit_one(x_train, y_train, x_test, y_test):
    clf = LogisticRegression(penalty="l2", C=C, fit_intercept=True, max_iter=5000, solver="lbfgs")
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    pred = clf.predict(x_test)
    return clf, acc, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--role-spaces", default=",".join(AUTHORS_SPACES))
    ap.add_argument("--layers", default="all")
    ap.add_argument("--split-by", choices=["prompt", "base"], default="prompt",
                    help="prompt = authors (same base text can be in train and test); base = grouped by base text")
    args = ap.parse_args()
    run = Path(args.run)
    meta = json.loads((run / "metadata.json").read_text())
    layers = meta["layers"] if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    lab = pd.read_parquet(run / "tokens.parquet")
    SKIP_FIRST_N = 0  # NESTED_REASONING false for gptoss-20b (NB02 cell 18)
    base = lab[(lab.is_content == True) & lab.role.notna() & lab.match_target_role & (lab.token_in_seg_ix >= SKIP_FIRST_N)]  # noqa: E712
    results, probes = [], {}
    for space in args.role_spaces.split(","):
        roles = [ROLE_CHAR[ch] for ch in space]
        roles_map = {r: i for i, r in enumerate(roles)}
        df = base[base.role.isin(roles)].reset_index(drop=True)
        if args.split_by == "prompt":
            tr_ix, te_ix = train_test_split(df.prompt_ix.unique(), test_size=0.1, random_state=SEED)
            tr, te = df[df.prompt_ix.isin(tr_ix)], df[df.prompt_ix.isin(te_ix)]
        else:
            tr_q, te_q = train_test_split(df.question_ix.unique(), test_size=0.1, random_state=SEED)
            tr, te = df[df.question_ix.isin(tr_q)], df[df.question_ix.isin(te_q)]
        y_tr = tr.role.map(roles_map).to_numpy()
        y_te = te.role.map(roles_map).to_numpy()
        if len(np.unique(y_tr)) < len(roles):
            print(f"[skip] {space}: missing roles in train")
            continue
        for l in layers:
            hs = np.load(run / f"layer{l:02d}.npy", mmap_mode="r")
            t0 = time.time()
            clf, acc, pred = fit_one(hs[tr.sample_ix.to_numpy()].astype(np.float32), y_tr,
                                     hs[te.sample_ix.to_numpy()].astype(np.float32), y_te)
            by_role = {r: float((pred[y_te == i] == i).mean()) for r, i in roles_map.items()}
            results.append({"role_space": space, "layer_ix": l, "acc": round(float(acc), 4), "acc_by_role": {k: round(v, 4) for k, v in by_role.items()},
                            "n_train": int(len(tr)), "n_test": int(len(te)), "n_iter": int(clf.n_iter_[0]), "sec": round(time.time() - t0, 1)})
            probes[f"{space}_L{l:02d}"] = {"coef": clf.coef_.astype(np.float32), "intercept": clf.intercept_.astype(np.float32)}
            print(f"[fit] {space} layer {l:2d}: acc {acc:.3f} by role {by_role} iters {clf.n_iter_[0]} ({time.time() - t0:.0f}s)", flush=True)
    tag = "" if args.split_by == "prompt" else "-basesplit"
    np.savez(run / f"probes{tag}.npz", **{f"{k}__{kk}": v for k, d in probes.items() for kk, v in d.items()})
    (run / f"probes{tag}.json").write_text(json.dumps({
        "split_by": args.split_by, "C": C, "penalty": "l2", "fit_intercept": True, "max_iter": 5000, "scaling": False, "split": "90/10 by prompt_ix, seed 123",
        "solver": "sklearn lbfgs (authors: cuml)", "role_char": ROLE_CHAR, "results": results,
        "source": "NB02 cells 18-20, commit ec333c40fd43fe991e1ebf66765051b6d7e35784"}, indent=2))
    pd.DataFrame(results).pivot(index="layer_ix", columns="role_space", values="acc").pipe(lambda t: print(t.to_string()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
