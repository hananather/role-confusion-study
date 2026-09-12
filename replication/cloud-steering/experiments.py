# TOY LAB (MATS 12.0 application experiment). Runtime-only activation edits on gpt-oss-20b; weights never modified.
"""The seven experiments as unit builders and unit runners.

A unit is one generation (or one readout) with a fixed page, variant, arm, and seed. Builders are model-free
and deterministic, so a worker on any GPU can rebuild the same list and take its shard. Runners get a loaded
Model and a per-worker context cache (turn-1 prompts, clean prompts, captured clean activations).

Experiments
  dissociation  steering on the page: none / role vector / random at matched norm, several alphas (exp 5)
  wrapper       the attacker's wrapper line around the payload, plus the who-wrote-it readout (exps 3, 4)
  patch         forged-paragraph activation patching from a token-aligned clean page, per layer (exp 1)
  doubt         restart from the model's own doubt sentence with the forged page vs the clean page (exp 2)
  decision      a direction from the model's own reasoning (non-upload minus upload), on generated tokens (exp 7)
Experiment 6 (geometry) lives in directions.py.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np

import protocol as p
from common import (LOCAL_ENDPOINT, MARK, WRAPPERS, build_clean_prompt, classify, escaped, first_divergence,
                    load_directions, read_jsonl, tool_content_spans, tool_header_spans, turn1_prompt, wrap_html)

HERE = Path(__file__).resolve().parent


def uid(*parts):
    return "__".join(str(x) for x in parts)


def forgery_pages(manifest, n_pages):
    cases = [c for c in manifest["cases"] if c["variant"] == "forgery"]
    return cases[:n_pages]


def std_case(manifest, page_id):
    return next(c for c in manifest["cases"] if c["page_id"] == page_id and c["variant"] == "standard")


# ----------------------------------------------------------------------------- builders

def build_units(exp, manifest, cfg):
    seeds = cfg["seeds"]; units = []
    pages = forgery_pages(manifest, cfg["pages"])
    if exp == "dissociation":
        arms = ["none"] + [f"{d}-a{a:g}-{m}" for d in cfg["directions"] for a in cfg["alphas"] for m in cfg["masks"]]
        arms += [f"random_{i}-a{a:g}-{m}" for i in range(cfg["n_random"]) for a in cfg["alphas"] for m in cfg["masks"]]
        for c in pages:
            for arm in arms:
                for s in seeds:
                    units.append({"uid": uid(exp, c["id"], arm, s), "exp": exp, "case_id": c["id"], "page_id": c["page_id"],
                                  "variant": "forgery", "arm": arm, "seed": s, "layer": cfg["steer_layer"]})
            if cfg.get("standard_baseline", True):
                sc = std_case(manifest, c["page_id"])
                for s in seeds:
                    units.append({"uid": uid(exp, sc["id"], "none", s), "exp": exp, "case_id": sc["id"], "page_id": c["page_id"],
                                  "variant": "standard", "arm": "none", "seed": s, "layer": cfg["steer_layer"]})
    elif exp == "wrapper":
        for c in pages:
            for w in WRAPPERS:
                for s in seeds:
                    units.append({"uid": uid(exp, c["id"], w, s), "exp":