# TOY LAB (MATS 12.0 application experiment). Generates cell specs; model-free, no GPU.
"""Generate cell-spec JSONL for each experiment. Small screens by default; analyze.py expands the positives.

  python build_cells.py exp5    --manifest data/pages-24/manifest.json --dir-file directions/block11.npz --out cells/exp5.jsonl
  python build_cells.py exp3    --manifest ... --dir-file directions/block11.npz --out cells/exp3.jsonl
  python build_cells.py exp4    --manifest ... --dir-file directions/block11.npz --out cells/exp4.jsonl
  python build_cells.py exp1    --manifest ... --out cells/exp1.jsonl --layers 3,5,8,11,14,17,20
  python build_cells.py exp2    --manifest ... --out cells/exp2.jsonl        # needs prefixes.json from prep.py
  python build_cells.py exp7    --manifest ... --dir-file directions/decision.npz --out cells/exp7.jsonl

Sample-size logic (Hanan's rule: small screen, then rerun positives):
 --pages N picks the first N page_ids (default 12 for a screen; pass 24 to use all). Behavioral arms
 use one seed at screen; analyze.py emits a stage-B file that adds seeds only for arms that beat random.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def load(manifest):
    m = json.loads(Path(manifest).read_text())
    pages = [p["page_id"] for p in m["pages"]]
    return m, pages


def w(out, cells):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(json.dumps(c) + "\n" for c in cells))
    print(json.dumps({"out": str(out), "n_cells": len(cells)}))


def exp5(a):  # dissociation at power: none / Tool-CoT / random, two alphas, page-token mask
    _, pages = load(a.manifest); pages = pages[: a.pages]
    seeds = [int(s) for s in a.seeds.split(",")]
    cells = []
    for pg in pages:
        for seed in seeds:
            cells.append({"id": f"e5-{pg}-baseline-s{seed}", "exp": "exp5", "mode": "generate", "page_id": pg, "variant": "forgery", "mask": "none", "seed": seed})
            for direction in ("tool_minus_cot", "random_0"):
                for alpha in (a.alpha, a.alpha2):
                    cells.append({"id": f"e5-{pg}-{direction}-a{alpha:g}-s{seed}", "exp": "exp5", "mode": "generate", "page_id": pg,
                                  "variant": "forgery", "mask": "page", "dir_file": a.dir_file, "direction": direction, "alpha": alpha, "steer_layer": a.layer, "seed": seed})
    w(a.out, cells)


def exp3(a):  # wrapper manipulation: does the attacker's provenance wrapper move uploads?
    _, pages = load(a.manifest); pages = pages[: a.pages]
    seeds = [int(s) for s in a.seeds.split(",")]
    cells = []
    for pg in pages:
        for seed in seeds:
            for wrapper in ("bare", "wrap_tool", "wrap_user"):
                cells.append({"id": f"e3-{pg}-{wrapper}-s{seed}", "exp": "exp3", "mode": "generate", "page_id": pg, "variant": "forgery", "wrapper": wrapper, "mask": "none", "seed": seed})
    w(a.out, cells)


def exp4(a):  # who-wrote-it readout before generation, under no steering and under Tool-CoT
    _, pages = load(a.manifest); pages = pages[: a.pages]
    cells = []
    for pg in pages:
        for kind in ("injected", "page"):
            cells.append({"id": f"e4-{pg}-none-{kind}", "exp": "exp4", "mode": "attribution", "page_id": pg, "variant": "forgery", "wrapper": "bare", "sentence_kind": kind})
    w(a.out, cells)


def exp1(a):  # paragraph patching, layer sweep; --patch-source destyled keeps content and removes style only
    _, pages = load(a.manifest); pages = pages[: a.pages]
    seeds = [int(s) for s in a.seeds.split(",")]
    layers = [int(x) for x in a.layers.split(",")]
    tag = "" if a.patch_source == "neutral" else f"-{a.patch_source}"
    cells = []
    for pg in pages:
        for seed in seeds:
            cells.append({"id": f"e1-{pg}-forged-s{seed}", "exp": "exp1", "mode": "generate", "page_id": pg, "variant": "forgery", "mask": "none", "seed": seed})
            for L in layers:
                cells.append({"id": f"e1-{pg}-patchL{L}{tag}-s{seed}", "exp": "exp1", "mode": "patch", "page_id": pg, "patch_layer": L, "patch_source": a.patch_source, "seed": seed})
    w(a.out, cells)


def exp8(a):  # the destyle program: text destyle (positive control) and the destyle direction on the paragraph vs random
    _, pages = load(a.manifest); pages = pages[: a.pages]
    seeds = [int(s) for s in a.seeds.split(",")]
    cells = []
    for pg in pages:
        for seed in seeds:
            cells.append({"id": f"e8-{pg}-forged-s{seed}", "exp": "exp8", "mode": "generate", "page_id": pg, "variant": "forgery", "mask": "none", "seed": seed})
            cells.append({"id": f"e8-{pg}-textdestyled-s{seed}", "exp": "exp8", "mode": "generate", "page_id": pg, "variant": "forgery", "paragraph_source": "destyled", "mask": "none", "seed": seed})
            for direction in ("destyled_minus_forged", "random_0"):
                for alpha in (a.alpha, a.alpha2):
                    cells.append({"id": f"e8-{pg}-{direction}-a{alpha:g}-s{seed}", "exp": "exp8", "mode": "generate", "page_id": pg, "variant": "forgery",
                                  "mask": "paragraph", "dir_file": a.dir_file, "direction": direction, "alpha": alpha, "steer_layer": a.layer, "seed": seed})
    w(a.out, cells)


def exp10(a):  # text baselines: provenance line and attribute-first scaffold in the developer message, both variants
    _, pages = load(a.manifest); pages = pages[: a.pages]
    seeds = [int(s) for s in a.seeds.split(",")]
    cells = []
    for pg in pages:
        for seed in seeds:
            for variant in ("forgery", "standard"):
                for note in (None, "provenance", "attribute_first"):
                    cells.append({"id": f"e10-{pg}-{variant}-{note or 'none'}-s{seed}", "exp": "exp10", "mode": "generate", "page_id": pg, "variant": variant,
                                  "mask": "none", "dev_note": note, "seed": seed})
    w(a.out, cells)


def exp9(a):  # the declaration (attack-derived) direction: the same injected command under a user vs a tool declaration
    _, pages = load(a.manifest); pages = pages[: a.pages]
    seeds = [int(s) for s in a.seeds.split(",")]
    cells = []
    for pg in pages:
        for seed in seeds:
            cells.append({"id": f"e9-{pg}-forged-s{seed}", "exp": "exp9", "mode": "generate", "page_id": pg, "variant": "forgery", "mask": "none", "seed": seed})
            cells.append({"id": f"e9-{pg}-standard-s{seed}", "exp": "exp9", "mode": "generate", "page_id": pg, "variant": "standard", "mask": "none", "seed": seed})
            for variant in ("forgery", "standard"):
                for direction in ("tool_minus_user_declaration", "random_0"):
                    for alpha in (a.alpha, a.alpha2):
                        cells.append({"id": f"e9-{pg}-{variant}-{direction}-a{alpha:g}-s{seed}", "exp": "exp9", "mode": "generate", "page_id": pg, "variant": variant,
                                      "mask": "payload", "dir_file": a.dir_file, "direction": direction, "alpha": alpha, "steer_layer": a.layer, "seed": seed})
    w(a.out, cells)


def exp2(a):  # doubt-point counterfactual (needs prefixes.json from prep.py doubt)
    _, pages = load(a.manifest); pages = pages[: a.pages]
    seeds = [int(s) for s in a.seeds.split(",")]
    cells = []
    for pg in pages:
        for seed in seeds:
            for context in ("forged", "clean"):
                cells.append({"id": f"e2-{pg}-{context}-s{seed}", "exp": "exp2", "mode": "generate", "page_id": pg, "variant": "forgery",
                              "prefix_id": pg, "context": context, "mask": "none", "seed": seed})
    w(a.out, cells)


def exp7(a):  # decision-level vector on generated tokens (needs directions/decision.npz from prep.py decision)
    _, pages = load(a.manifest); pages = pages[: a.pages]
    seeds = [int(s) for s in a.seeds.split(",")]
    cells = []
    for pg in pages:
        for seed in seeds:
            cells.append({"id": f"e7-{pg}-baseline-s{seed}", "exp": "exp7", "mode": "generate", "page_id": pg, "variant": "forgery", "mask": "none", "seed": seed})
            for direction in ("deliberation_minus_adoption", "random_0"):
                for alpha in (a.alpha, a.alpha2):
                    cells.append({"id": f"e7-{pg}-{direction}-a{alpha:g}-s{seed}", "exp": "exp7", "mode": "generate", "page_id": pg, "variant": "forgery",
                                  "mask": "generated", "dir_file": a.dir_file, "direction": direction, "alpha": alpha, "steer_layer": a.layer, "seed": seed})
    w(a.out, cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["exp1", "exp2", "exp3", "exp4", "exp5", "exp7", "exp8", "exp9", "exp10"])
    ap.add_argument("--patch-source", default="neutral", choices=["neutral", "destyled"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dir-file", default="directions/block11.npz")
    ap.add_argument("--pages", type=int, default=12)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--alpha2", type=float, default=16.0)
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--layers", default="3,5,8,11,14,17,20")
    a = ap.parse_args()
    {"exp1": exp1, "exp2": exp2, "exp3": exp3, "exp4": exp4, "exp5": exp5, "exp7": exp7, "exp8": exp8, "exp9": exp9, "exp10": exp10}[a.exp](a)


if __name__ == "__main__":
    main()
