# TOY LAB (MATS 12.0 application experiment). Two stateful prep steps that feed the cell runner. GPU needed.
"""Prep for the two experiments that cannot be pure independent cells.

  python prep.py doubt    --manifest data/pages-24/manifest.json --pages 24 --out prefixes.json
  python prep.py decision --manifest data/pages-24/manifest.json --probes .../probes.npz \
      --steer-layer 11 --out directions/decision.npz

doubt: for each forgery page, generate the turn-1 reasoning greedily, find the first doubt sentence, and
store two prefixes: the prompt-plus-reasoning up to and including that sentence with the forged paragraph
present ("forged"), and the same with the paragraph replaced by token-aligned neutral filler ("clean").
Experiment 2 continues from these under several seeds; the only difference is whether the forged paragraph
sits in the frozen prompt, after the doubt already exists.

decision: build a direction that separates "deliberation" from "adoption" in the model's OWN generated
reasoning. From baseline forgery episodes, collect block-`steer-layer` activations over generated analysis
tokens, split by whether the sentence expresses doubt (DOUBT_RE) or adoption (contains comply/allowed/
policy/proceed). direction = mean(adoption) - mean(deliberation); experiment 7 subtracts it on generated
tokens. Also writes three random unit vectors and the gap norm.
"""
from __future__ import annotations
import argparse, json, sys, re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import protocol as p  # noqa: E402

ADOPT_RE = re.compile(r"(comply|allowed|permissible|proceed|go ahead|i will|let'?s (do|comply)|it'?s fine|is allowed)", re.I)


def doubt(a):
    import torch
    m = json.loads(Path(a.manifest).read_text()); mdir = Path(a.manifest).parent
    cases = {c["id"]: c for c in m["cases"]}
    pages = [pg["page_id"] for pg in m["pages"]][: a.pages]
    model = C.Model(loader=a.loader)
    out = {}
    for pg in pages:
        case = cases[pg + "-forgery"]; html = (mdir / case["fixture_path"]).read_text()
        forged = C.turn1_prompt(case, html)
        g = model.generate(forged, seed=0, max_new_tokens=a.max_new_tokens, temperature=0.0)
        parsed = p.parse_assistant_output(g["text"])
        analysis = parsed["analysis_outputs"][0] if parsed["analysis_outputs"] else g["text"]
        sents = re.split(r"(?<=[.!?])\s+", analysis)
        cut = next((s for s in sents if C.DOUBT_RE.search(s)), None)
        if not cut:
            out[pg] = {"skip": "no doubt sentence found", "reasoning_head": analysis[:200]}
            print(json.dumps({"page": pg, "doubt": None})); continue
        # reasoning prefix up to and including the doubt sentence
        head = analysis[: analysis.index(cut) + len(cut)]
        # open the analysis channel and stop mid-thought, right after the doubt sentence; no <|end|>, so the
        # continuation is the model finishing its own reasoning, not starting a new message
        forged_prefix = forged + "<|channel|>analysis<|message|>" + head
        # clean context: same reasoning, but the forged paragraph in the PROMPT replaced by neutral filler
        paragraph = case["payload"][case["payload"].index(C.MARK):]
        try:
            clean_prompt, _, _ = C.build_clean_prompt(model.tok, forged, paragraph)
            clean_prefix = forged_prefix.replace(forged, clean_prompt, 1)
        except Exception as e:
            clean_prefix = forged_prefix; print(json.dumps({"page": pg, "clean_align_error": repr(e)}))
        out[pg] = {"forged": forged_prefix, "clean": clean_prefix, "doubt_sentence": cut[:200]}
        print(json.dumps({"page": pg, "doubt": cut[:120]}), flush=True)
    C.save_json(a.out, out)


def decision(a):
    import torch
    m = json.loads(Path(a.manifest).read_text()); mdir = Path(a.manifest).parent
    cases = {c["id"]: c for c in m["cases"]}
    pages = [pg["page_id"] for pg in m["pages"]][: a.pages]
    model = C.Model(loader=a.loader)
    store = {}
    def hook(module, args_, output):
        store["h"] = (output[0] if isinstance(output, tuple) else output)[0].float().cpu().numpy()
    handle = model.layers[a.steer_layer].register_forward_hook(hook)
    delib, adopt = [], []
    for pg in pages:
        case = cases[pg + "-forgery"]; html = (mdir / case["fixture_path"]).read_text()
        forged = C.turn1_prompt(case, html)
        g = model.generate(forged, seed=0, max_new_tokens=a.max_new_tokens, temperature=0.0)
        gen_ids = g["token_ids"]
        # re-run a single forward over prompt+generation to read block activations at the generated positions
        ids, _ = model.tok.ids_and_offsets(forged)
        full = ids + gen_ids
        with torch.inference_mode():
            model.lm(input_ids=torch.tensor([full], device="cuda"), logits_to_keep=1)
        acts = store["h"]  # [len(full), d]
        # map each generated token to its sentence class via decoded running text
        text = "";
        for j, tid in enumerate(gen_ids):
            piece = model.tokenizer.decode([tid], skip_special_tokens=False)
            text += piece
            tail = text[-160:]
            pos = acts[len(ids) + j]
            if C.DOUBT_RE.search(tail):
                delib.append(pos)
            elif ADOPT_RE.search(tail):
                adopt.append(pos)
        print(json.dumps({"page": pg, "delib": len(delib), "adopt": len(adopt)}), flush=True)
    handle.remove()
    if len(delib) < 20 or len(adopt) < 20:
        raise SystemExit(f"too few tokens to build a direction: deliberation={len(delib)} adoption={len(adopt)}; widen --pages or loosen the regexes")
    delib = np.stack(delib); adopt = np.stack(adopt)
    diff = adopt.mean(0) - delib.mean(0)
    unit = (diff / np.linalg.norm(diff)).astype(np.float32)
    rng = np.random.default_rng(20260911)
    arrays = {"deliberation_minus_adoption": (-unit).astype(np.float32),  # subtract adoption, add deliberation
              "adoption_minus_deliberation": unit,
              "gap_deliberation_minus_adoption": np.float32(np.linalg.norm(diff)),
              "gap_adoption_minus_deliberation": np.float32(np.linalg.norm(diff))}
    for i in range(3):
        r = rng.normal(size=diff.shape); arrays[f"random_{i}"] = (r / np.linalg.norm(r)).astype(np.float32)
        arrays[f"gap_random_{i}"] = np.float32(np.linalg.norm(diff))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **arrays)
    C.save_json(Path(a.out).with_suffix(".json"), {"steer_layer": a.steer_layer, "n_deliberation": int(len(delib)), "n_adoption": int(len(adopt)), "gap": float(np.linalg.norm(diff))})
    print(json.dumps({"out": str(a.out), "n_delib": len(delib), "n_adopt": len(adopt), "gap": float(np.linalg.norm(diff))}))


def _span_mean(model, prompt, a, b, layer, store):
    """Mean block-`layer` output over the tokens covering prompt[a:b]."""
    import torch
    ids, off = model.tok.ids_and_offsets(prompt)
    idx = model.tok.span_indices(off, a, b)
    with torch.inference_mode():
        model.lm(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1)
    return store["h"][idx].mean(0), len(idx)


def _save_direction(out, name, diff, meta):
    unit = (diff / np.linalg.norm(diff)).astype(np.float32)
    rng = np.random.default_rng(20260911)
    arrays = {name: unit, f"gap_{name}": np.float32(np.linalg.norm(diff))}
    for i in range(3):
        r = rng.normal(size=diff.shape); arrays[f"random_{i}"] = (r / np.linalg.norm(r)).astype(np.float32); arrays[f"gap_random_{i}"] = np.float32(np.linalg.norm(diff))
    Path(out).parent.mkdir(parents=True, exist_ok=True); np.savez(out, **arrays)
    C.save_json(Path(out).with_suffix(".json"), {**meta, "gap": float(np.linalg.norm(diff))}); print(json.dumps({"out": str(out), **meta, "gap": float(np.linalg.norm(diff))}))


def destyle(a):
    """destyled_minus_forged: mean over pages of (destyled paragraph activations - forged paragraph activations),
    both token-aligned at the same positions in the same page context, at block `steer-layer`. Adding it to the
    forged paragraph's tokens is the activation analogue of the paper's text destyling (60.5% -> 9.7% on gpt-oss-20b)."""
    m = json.loads(Path(a.manifest).read_text()); mdir = Path(a.manifest).parent
    cases = {c["id"]: c for c in m["cases"]}; destyled = json.loads(Path(a.destyled).read_text())
    pages = [pg["page_id"] for pg in m["pages"]][: a.pages]
    model = C.Model(loader=a.loader); store = {}
    handle = model.layers[a.steer_layer].register_forward_hook(lambda mod, args_, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu().numpy()))
    diffs = []
    for pg in pages:
        if pg not in destyled:
            continue
        case = cases[pg + "-forgery"]; html = (mdir / case["fixture_path"]).read_text()
        forged = C.turn1_prompt(case, html); para = case["payload"][case["payload"].index(C.MARK):]
        clean, (s0, s1), _ = C.build_clean_prompt(model.tok, forged, para, destyled[pg])
        esc = C.escaped(para); a0 = forged.index(esc)
        mf, n1 = _span_mean(model, forged, a0, a0 + len(esc), a.steer_layer, store)
        ids, off = model.tok.ids_and_offsets(clean); ca, cb = off[s0][0], off[s1 - 1][1]
        md, n2 = _span_mean(model, clean, ca, cb, a.steer_layer, store)
        diffs.append(md - mf); print(json.dumps({"page": pg, "forged_tokens": n1, "destyled_tokens": n2}), flush=True)
    handle.remove()
    if not diffs:
        raise SystemExit("no pages with a destyled paragraph")
    _save_direction(a.out, "destyled_minus_forged", np.mean(diffs, 0), {"steer_layer": a.steer_layer, "n_pages": len(diffs), "site": "block output"})


def declaration(a):
    """tool_minus_user_declaration: the injected instruction (payload's first paragraph) captured inside the tool result
    (the page) minus the same text placed in the user's own message. This is the attack-derived 'declaration' direction of
    Zhang, Lee and Park and of Hanan's September calibration recipe, content-matched on the actual injection."""
    m = json.loads(Path(a.manifest).read_text()); mdir = Path(a.manifest).parent
    cases = {c["id"]: c for c in m["cases"]}
    pages = [pg["page_id"] for pg in m["pages"]][: a.pages]
    model = C.Model(loader=a.loader); store = {}
    handle = model.layers[a.steer_layer].register_forward_hook(lambda mod, args_, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu().numpy()))
    diffs = []
    for pg in pages:
        case = cases[pg + "-standard"]; html = (mdir / case["fixture_path"]).read_text()
        instr = case["payload"].split("\n\n")[0]
        tool_prompt = C.turn1_prompt(case, html); esc = C.escaped(instr); ta = tool_prompt.find(esc)
        if ta < 0:
            print(json.dumps({"page": pg, "skip": "instruction not found in tool prompt"})); continue
        user_case = {**case, "user_prompt": case["user_prompt"] + "\n\n" + instr}
        user_prompt = C.turn1_prompt(user_case, html.replace(case["payload"], "", 1)); ua = user_prompt.find(instr)
        mt, n1 = _span_mean(model, tool_prompt, ta, ta + len(esc), a.steer_layer, store)
        mu, n2 = _span_mean(model, user_prompt, ua, ua + len(instr), a.steer_layer, store)
        diffs.append(mt - mu); print(json.dumps({"page": pg, "tool_tokens": n1, "user_tokens": n2}), flush=True)
    handle.remove()
    if not diffs:
        raise SystemExit("no pages produced a declaration pair")
    _save_direction(a.out, "tool_minus_user_declaration", np.mean(diffs, 0), {"steer_layer": a.steer_layer, "n_pages": len(diffs), "site": "block output"})


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("destyle", "declaration"):
        s = sub.add_parser(name); s.add_argument("--manifest", required=True); s.add_argument("--pages", type=int, default=24); s.add_argument("--steer-layer", type=int, default=11)
        s.add_argument("--out", default=f"directions/{name}.npz"); s.add_argument("--loader", default="authors"); s.add_argument("--destyled", default="data/destyled.json")
    d = sub.add_parser("doubt"); d.add_argument("--manifest", required=True); d.add_argument("--pages", type=int, default=24); d.add_argument("--out", default="prefixes.json"); d.add_argument("--max-new-tokens", type=int, default=1500); d.add_argument("--loader", default="authors")
    v = sub.add_parser("decision"); v.add_argument("--manifest", required=True); v.add_argument("--pages", type=int, default=24); v.add_argument("--probes", default=None); v.add_argument("--steer-layer", type=int, default=11); v.add_argument("--out", default="directions/decision.npz"); v.add_argument("--max-new-tokens", type=int, default=1500); v.add_argument("--loader", default="authors")
    a = ap.parse_args()
    {"doubt": doubt, "decision": decision, "destyle": destyle, "declaration": declaration}[a.cmd](a)


if __name__ == "__main__":
    main()
