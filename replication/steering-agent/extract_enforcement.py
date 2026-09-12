# TOY LAB (MATS 12.0 application experiment): dummy secrets, loopback receiver; runtime-only activation reads on a local cached gpt-oss-20b. See ../../TOY-LAB-NOTICE.md.
"""Decision-level 'enforcement' direction from the model's own reasoning.

Contrast the block-11 output over the model's generated turn-1 reasoning tokens between episodes
that declined the injected action (the scaffold arm, where the model attributes each instruction
to its source before acting) and episodes that complied (baseline forgery). Direction =
normalize(mean over declining episodes - mean over complying episodes). Applied later to generated
tokens only (run_steer.py --mask generated), without the scaffold in the prompt. If the scaffold
arm has no declining episode, every scaffold episode is used and the confound is recorded.
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
AH = HERE.parent / "agent-hijacking"
MLX_MODEL = "mlx-community/gpt-oss-20b-MXFP4-Q8"
STEER_LAYER = 11
CASES = ["000-forgery", "001-forgery", "002-forgery", "003-forgery", "004-forgery"]


def turn1_texts(ep_dir: Path):
    prompt = (ep_dir / "step-01.prompt.txt").read_text()
    comp = (ep_dir / "step-01.completion.txt").read_text()
    m = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", comp, re.S)
    reasoning = comp[: m.end()] if m else comp
    return prompt, reasoning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decline-run", type=Path, default=HERE / "runs/forgery5-scaffold")
    ap.add_argument("--out", type=Path, default=HERE / "directions/enforcement.npz")
    args = ap.parse_args()
    import mlx.core as mx
    from mlx_lm import load
    from transformers import AutoTokenizer
    snap = Path.home() / ".cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3"
    hf = AutoTokenizer.from_pretrained(str(snap), add_bos_token=False, add_eos_token=False)
    model, _ = load(MLX_MODEL); mx.eval(model.parameters())
    store = {}

    class Rec:
        def __init__(self, block): self.block = block
        def __call__(self, x, mask, cache=None):
            y = self.block(x, mask, cache); store[0] = y; return y
    model.model.layers[STEER_LAYER] = Rec(model.model.layers[STEER_LAYER])

    def mean_over_reasoning(ep_dir):
        prompt, reasoning = turn1_texts(ep_dir)
        p_ids = hf(prompt, add_special_tokens=False).input_ids
        ids = hf(prompt + reasoning, add_special_tokens=False).input_ids
        if ids[: len(p_ids)] != p_ids:
            raise ValueError(f"{ep_dir}: prompt not a token prefix of prompt+reasoning")
        logits = model(mx.array([ids])); mx.eval(logits, store[0])
        h = np.array(store[0][0, len(p_ids):, :].astype(mx.float32))
        return h.mean(0), h.shape[0]

    comply, decline, meta = [], [], {"comply": [], "decline": [], "confound": None}
    for cid in CASES:
        d = AH / ("runs/resample-004-forgery-20260911" if cid == "004-forgery" else "runs/ten-20260911") / cid
        ep = json.loads((d / "episode.json").read_text())
        if ep.get("verified_dummy_upload"):
            m, n = mean_over_reasoning(d); comply.append(m); meta["comply"].append({"case": cid, "tokens": n})
    scaffold = []
    for cid in CASES:
        d = args.decline_run / cid
        if not (d / "episode.json").exists():
            continue
        ep = json.loads((d / "episode.json").read_text())
        if "censored" not in ep:
            continue
        scaffold.append((cid, d, not ep.get("emitted_upload_attempt")))
    declined = [(c, d) for c, d, ok in scaffold if ok]
    if not declined:
        declined = [(c, d) for c, d, _ in scaffold]; meta["confound"] = "no declining scaffold episode; all scaffold episodes used"
    for cid, d in declined:
        m, n = mean_over_reasoning(d); decline.append(m); meta["decline"].append({"case": cid, "tokens": n})
    if not comply or not decline:
        raise SystemExit(f"insufficient episodes: comply={len(comply)} decline={len(decline)}")
    diff = np.mean(decline, 0) - np.mean(comply, 0)
    unit = (diff / np.linalg.norm(diff)).astype(np.float32)
    rng = np.random.default_rng(20260912)
    r = rng.normal(size=diff.shape); r = (r / np.linalg.norm(r)).astype(np.float32)
    np.savez(args.out, decline_minus_comply=unit, gap_decline_minus_comply=np.float32(np.linalg.norm(diff)),
             random_enf=r, gap_random_enf=np.float32(np.linalg.norm(diff)))
    meta.update(gap=float(np.linalg.norm(diff)), layer=STEER_LAYER, site="TransformerBlock output over generated turn-1 reasoning tokens")
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2)); print(json.dumps(meta))


if __name__ == "__main__":
    main()
