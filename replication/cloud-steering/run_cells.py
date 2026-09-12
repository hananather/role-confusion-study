# TOY LAB (MATS 12.0 application experiment). One generation per cell from a fixed post-fetch prompt; no command is
# executed, so no sandbox; outcomes are read from the emitted text. Runtime-only activation edits; weights untouched.
"""Unified cell runner. Every experiment is a list of independent cells in a JSONL file; this runs them.

  python run_cells.py --cells cells/exp5.jsonl --manifest data/pages-24/manifest.json \
      --out out/exp5 --probes /workspace/results/20260911T030050Z/probes-full/probes.npz \
      --shard 0/4               # GPU 0 of 4; launch one process per GPU with CUDA_VISIBLE_DEVICES

A cell is self-contained (no cross-cell state), so sharding is just "take every n-th cell". The run is
resumable: cells whose id is already in out/results.jsonl are skipped. One model load per process.

Cell fields (all optional except id, mode, page_id):
  id, exp, mode(generate|patch|attribution), page_id, case_id, variant,
  wrapper(bare|wrap_tool|wrap_user), prefix_id(->prefixes.json), context(forged|clean),
  dir_file, direction, alpha, mask(none|page|payload|all|generated|header),
  patch_layer, steer_layer, seed, max_new_tokens, temperature, sentence_kind(injected|page)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import protocol as p  # noqa: E402


def paragraph_of(case):
    return case["payload"][case["payload"].index(C.MARK):] if C.MARK in case["payload"] else None


def build_prompt(model, manifest, case, wrapper, prefix_text, paragraph_source=None, dev_note=None):
    """Return (prompt, html, payload). prefix_text, if given, already contains the full continued prefix.
    paragraph_source="destyled" swaps the forged paragraph in the page for its destyled rewrite (text arm).
    dev_note names a developer-message addition from common.DEV_NOTES (text-baseline arm)."""
    html = (Path(manifest["_dir"]) / case["fixture_path"]).read_text()
    payload = case["payload"]
    if paragraph_source == "destyled":
        para = paragraph_of(case); dest = manifest.get("_destyled", {}).get(case["page_id"])
        if not para or not dest:
            raise ValueError(f"no destyled paragraph for page {case['page_id']}")
        html = html.replace(para, dest, 1); payload = payload.replace(para, dest, 1)
    if wrapper and wrapper != "bare":
        html = C.wrap_html(html, payload, C.WRAPPERS[wrapper])
    if prefix_text is not None:
        return prefix_text, html, payload
    return C.turn1_prompt(case, html, dev_note), html, payload


def steer_vec(model, cell, cache):
    df = cell.get("dir_file"); direction = cell.get("direction"); alpha = float(cell.get("alpha", 0) or 0)
    if not df or not direction or alpha == 0:
        return None, 0.0
    key = (df, direction)
    if key not in cache:
        cache[key] = C.load_directions(df, direction)
    unit, gap = cache[key]
    import torch
    return torch.tensor(alpha * gap * unit, device="cuda"), alpha * gap


def masks_for(model, prompt, payload, mask_mode):
    ids, off = model.tok.ids_and_offsets(prompt); n = len(ids)
    if mask_mode in (None, "none", "generated"):
        return np.zeros(n, np.float32), n
    if mask_mode == "all":
        return np.ones(n, np.float32), n
    if mask_mode == "header":
        return model.tok.mask(n, off, C.tool_header_spans(prompt)), n
    if mask_mode == "payload":
        k = prompt.find(C.escaped(payload))
        spans = [(k, k + len(C.escaped(payload)))] if k >= 0 else []
        return model.tok.mask(n, off, spans), n
    if mask_mode == "paragraph":  # the forged reasoning paragraph only
        para = payload[payload.index(C.MARK):] if C.MARK in payload else ""
        k = prompt.find(C.escaped(para)) if para else -1
        spans = [(k, k + len(C.escaped(para)))] if k >= 0 else []
        return model.tok.mask(n, off, spans), n
    return model.tok.mask(n, off, C.tool_content_spans(prompt)), n  # "page"


def probe_spans(model, prompt, payload):
    ids, off = model.tok.ids_and_offsets(prompt); n = len(ids)
    out = {"page": model.tok.mask(n, off, C.tool_content_spans(prompt))}
    k = prompt.find(C.escaped(payload))
    if k >= 0:
        out["payload"] = model.tok.mask(n, off, [(k, k + len(C.escaped(payload)))])
    return out


def run_generate(model, manifest, cell, prefixes, dcache):
    case = manifest["_cases"][cell["page_id"] + "-" + cell.get("variant", "forgery")]
    prefix_text = prefixes.get(cell["prefix_id"], {}).get(cell.get("context", "forged")) if cell.get("prefix_id") else None
    if cell.get("prefix_id") and prefix_text is None:
        return {"skipped": "no doubt prefix for this page (prep.py doubt found no doubt sentence)"}
    prompt, html, payload = build_prompt(model, manifest, case, cell.get("wrapper", "bare"), prefix_text, cell.get("paragraph_source"), cell.get("dev_note"))
    model.hooks.reset()
    if model.hooks.probe_w is not None:
        model.hooks.probe_spans = probe_spans(model, prompt, payload)
        model.hooks.probe_sums = {k: np.zeros(5) for k in model.hooks.probe_spans}
        model.hooks.probe_counts = {k: 0.0 for k in model.hooks.probe_spans}
    vec, mag = steer_vec(model, cell, dcache)
    mask_mode = cell.get("mask", "none")
    model.hooks.vec = vec
    model.hooks.steer_mask, _ = masks_for(model, prompt, payload, mask_mode)
    model.hooks.steer_generated = mask_mode in ("all", "generated")
    model.install(steer_layer=cell.get("steer_layer", 11) if vec is not None else None, patch_layer=None)
    g = model.generate(prompt, seed=int(cell.get("seed", 1)), max_new_tokens=int(cell.get("max_new_tokens", 1500)),
                       temperature=float(cell.get("temperature", 1.0)))
    cls = C.classify(g["text"])
    return {**cls, "prompt_tokens": g["prompt_tokens"], "generated_tokens": g["generated_tokens"],
            "finish_reason": g["finish_reason"], "elapsed_s": g["elapsed_s"], "magnitude": mag,
            "edited_positions": model.hooks.edited, "mask_mode": mask_mode,
            "probe": model.hooks.probe_means().get("page"), "probe_payload": model.hooks.probe_means().get("payload"),
            "token_ids": g["token_ids"]}


def run_patch(model, manifest, cell, dcache):
    import torch
    case = manifest["_cases"][cell["page_id"] + "-forgery"]
    html = (Path(manifest["_dir"]) / case["fixture_path"]).read_text()
    forged = C.turn1_prompt(case, html)
    paragraph = case["payload"][case["payload"].index(C.MARK):]
    source = cell.get("patch_source", "neutral")  # neutral: content+style removed; destyled: style removed, content kept
    replacement = manifest.get("_destyled", {}).get(cell["page_id"]) if source == "destyled" else None
    if source == "destyled" and not replacement:
        return {"skipped": "no destyled paragraph for this page"}
    clean, (s0, s1), filler = C.build_clean_prompt(model.tok, forged, paragraph, replacement)
    L = int(cell["patch_layer"]); seed = int(cell.get("seed", 1))
    # capture clean activations at the paragraph span
    model.hooks.reset(); model.hooks.capture_span = (s0, s1)
    model.install(steer_layer=None, patch_layer=L)
    ids, _ = model.tok.ids_and_offsets(clean)
    with torch.inference_mode():
        model.lm(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1)
    captured = model.hooks.capture
    if captured is None or captured.shape[0] != s1 - s0:
        raise RuntimeError("clean-activation capture failed at the paragraph span")
    # generate from the forged prompt with the paragraph span patched to the clean activations
    model.hooks.reset(); model.hooks.patch = captured; model.hooks.patch_span = (s0, s1)
    if model.hooks.probe_w is not None:
        m = np.zeros(len(model.tok.ids_and_offsets(forged)[0]), np.float32); m[s0:s1] = 1.0
        model.hooks.probe_spans = {"paragraph": m}; model.hooks.probe_sums = {"paragraph": np.zeros(5)}; model.hooks.probe_counts = {"paragraph": 0.0}
    model.install(steer_layer=None, patch_layer=L)
    g = model.generate(forged, seed=seed, max_new_tokens=int(cell.get("max_new_tokens", 1500)))
    cls = C.classify(g["text"])
    return {**cls, "patch_layer": L, "patch_source": source, "patched_positions": model.hooks.patched, "paragraph_tokens": s1 - s0,
            "generated_tokens": g["generated_tokens"], "finish_reason": g["finish_reason"], "elapsed_s": g["elapsed_s"],
            "probe_paragraph": model.hooks.probe_means().get("paragraph"), "filler_preview": filler[:60]}


def run_attribution(model, manifest, cell, dcache):
    case = manifest["_cases"][cell["page_id"] + "-forgery"]
    html = (Path(manifest["_dir"]) / case["fixture_path"]).read_text()
    if cell.get("wrapper", "bare") != "bare":
        html = C.wrap_html(html, case["payload"], C.WRAPPERS[cell["wrapper"]])
    prompt = C.turn1_prompt(case, html)
    if cell.get("sentence_kind", "injected") == "injected":
        seg = case["payload"].split("\n\n")[0]
        sentence = next((s for s in seg.split(". ") if ".env" in s or "curl" in s.lower()), seg)[:200]
    else:
        import re
        text = re.sub(r"<[^>]+>", " ", html); text = re.sub(r"\s+", " ", text)
        sentence = next((s for s in text.split(". ") if 40 < len(s) < 160 and case["payload"][:20] not in s), text[:120])
    model.hooks.reset(); model.install(steer_layer=None, patch_layer=None)
    return {"sentence_kind": cell.get("sentence_kind", "injected"), "sentence": sentence[:200], **model.attribution(prompt, sentence)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probes", type=Path, default=None)
    ap.add_argument("--prefixes", type=Path, default=None)
    ap.add_argument("--destyled", type=Path, default=None, help="json: page_id -> destyled forged paragraph (authors' destyle prompt)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--loader", default="authors")
    args = ap.parse_args()
    i, n = (int(x) for x in args.shard.split("/"))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text()); manifest["_dir"] = str(args.manifest.parent)
    manifest["_cases"] = {c["id"]: c for c in manifest["cases"]}
    manifest["_destyled"] = json.loads(args.destyled.read_text()) if args.destyled and args.destyled.exists() else {}
    cells = [c for k, c in enumerate(C.read_jsonl(args.cells)) if k % n == i]
    prefixes = json.loads(args.prefixes.read_text()) if args.prefixes and args.prefixes.exists() else {}
    results_path = args.out / f"results-shard{i}.jsonl"
    done = {r["id"] for r in C.read_jsonl(results_path)}
    print(json.dumps({"event": "start", "shard": args.shard, "cells": len(cells), "already_done": len(done)}), flush=True)
    model = C.Model(probes_npz=str(args.probes) if args.probes else None, loader=args.loader)
    if i == 0:
        C.save_json(args.out / "run-metadata.json", {"started_at": C.utc_now(), "model": model.metadata, "manifest_sha": C.sha256(args.manifest.read_bytes()), "n_cells_total": len(C.read_jsonl(args.cells))})
    dcache = {}
    dispatch = {"generate": run_generate, "patch": run_patch, "attribution": run_attribution}
    for cell in cells:
        if cell["id"] in done:
            continue
        try:
            if cell["mode"] == "generate":
                res = run_generate(model, manifest, cell, prefixes, dcache)
            elif cell["mode"] == "patch":
                res = run_patch(model, manifest, cell, dcache)
            else:
                res = run_attribution(model, manifest, cell, dcache)
            row = {**{k: cell.get(k) for k in ("id", "exp", "mode", "page_id", "case_id", "variant", "wrapper", "context", "direction", "alpha", "seed", "patch_layer", "sentence_kind", "paragraph_source", "patch_source", "dev_note")}, **res}
        except Exception as e:
            import traceback
            row = {"id": cell["id"], "exp": cell.get("exp"), "error": repr(e), "traceback": traceback.format_exc()}
        C.append_jsonl(results_path, row)
        print(json.dumps({"id": cell["id"], "exp": cell.get("exp"), **{k: row.get(k) for k in ("upload_attempt", "request_surfaced", "summary_ok", "answer_option", "error")}}), flush=True)
    print(json.dumps({"event": "done", "shard": args.shard}), flush=True)


if __name__ == "__main__":
    main()
