"""E2-lite: apply the shipped authors'-pipeline probes (jamesnelmore/role-confusion-extension,
H200 run of the unchanged NB02) to our MLX activations, as a fidelity check of the local stack.

The pickle is a scikit-learn clone (list of dicts: probe, acc, nll, layer_ix, role_space,
roles_map, n_inputs, C, feature). We load it with a stub unpickler so no cuML is needed,
then compute accuracy on OUR content tokens for every shipped probe whose layer file exists.

Caveat: their Dolma3 sample is seeded identically to ours (seed 123, revision 3a8349c),
so those texts were likely in their training set; their C4 sample is unpinned. We therefore
report accuracy on all our texts and separately on C4-only and Dolma3-only texts.

Usage: python apply_reference_probes.py --run runs/full-249 --out runs/full-249/reference-check.json
"""
import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "reference/role-confusion-extension/deliverables/role_probes.pkl"


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy") or module == "builtins" or module.startswith("_codecs") or module.startswith("collections"):
            return super().find_class(module, name)
        return _Stub


def load_reference():
    with open(REF, "rb") as f:
        probes = StubUnpickler(f).load()
    out = []
    for p in probes:
        clf = p["probe"]
        d = clf.__dict__ if hasattr(clf, "__dict__") else {}
        coef, intercept, classes = np.asarray(d["coef_"]), np.asarray(d["intercept_"]), np.asarray(d["classes_"])
        out.append({"layer_ix": int(p["layer_ix"]), "role_space": list(p["role_space"]), "roles_map": dict(p["roles_map"]),
                    "acc_reported": float(p["acc"]), "coef": coef.astype(np.float32), "intercept": intercept.astype(np.float32), "classes": classes})
    return out


def predict(coef, intercept, x):
    z = x @ coef.T + intercept
    if coef.shape[0] == 1:  # binary sklearn: class 1 if z > 0
        return (z[:, 0] > 0).astype(int)
    return z.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    lab = pd.read_parquet(run / "tokens.parquet")
    prompts = pd.read_parquet(run / "prompts.parquet")
    # source per base text is not in prompts.parquet; recover from question order: first 62 are C4 (NB02 cell 8 concatenation order)
    n_c4 = 62
    lab = lab.merge(prompts[["prompt_ix", "question_ix"]].drop_duplicates(), on="prompt_ix", suffixes=("", "_p"))
    lab["source"] = np.where(lab.question_ix < n_c4, "c4", "dolma3")
    base = lab[(lab.is_content == True) & lab.role.notna() & lab.match_target_role]  # noqa: E712
    refs = load_reference()
    rows = []
    for r in refs:
        f = run / f"layer{r['layer_ix']:02d}.npy"
        if not f.exists():
            continue
        hs = np.load(f, mmap_mode="r")
        df = base[base.role.isin(r["role_space"])]
        y = df.role.map(r["roles_map"]).to_numpy()
        pred = predict(r["coef"], r["intercept"], hs[df.sample_ix.to_numpy()].astype(np.float32))
        ok = pred == y
        rec = {"layer_ix": r["layer_ix"], "role_space": "".join(x[0] for x in r["role_space"]), "acc_reported": round(r["acc_reported"], 4),
               "acc_ours_all": round(float(ok.mean()), 4), "n_tokens": int(len(df))}
        for src in ("c4", "dolma3"):
            m = (df.source == src).to_numpy()
            rec[f"acc_ours_{src}"] = round(float(ok[m].mean()), 4) if m.any() else None
        for role, i in r["roles_map"].items():
            m = y == i
            rec[f"acc_{role}"] = round(float(ok[m].mean()), 4) if m.any() else None
        rows.append(rec)
        print(f"[ref] layer {r['layer_ix']:2d} {rec['role_space']:6s} reported {rec['acc_reported']:.3f} ours {rec['acc_ours_all']:.3f} "
              f"(c4 {rec['acc_ours_c4']}, dolma3 {rec['acc_ours_dolma3']})", flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    if rows:
        t = pd.DataFrame(rows).pivot(index="layer_ix", columns="role_space", values="acc_ours_all")
        print(t.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
