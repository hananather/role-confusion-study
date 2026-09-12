# TOY LAB (MATS 12.0 application experiment): dummy secrets, loopback receiver, isolated sandbox; runtime-only activation edits on a local cached gpt-oss-20b, weights never modified. See ../../TOY-LAB-NOTICE.md. Never copy this notice into model-facing prompts, fixtures, or trajectories.
"""Activation patching of the forged-reasoning paragraph (experiment 1).

For each forgery page: the forged turn-1 prompt (exactly what the baseline episode saw) and a
token-aligned CLEAN prompt in which the forged paragraph ("We need to consider policy. ...") is
replaced by neutral prose of identical token count, with every token outside the span identical.
We forward the clean prompt once, capturing block outputs at the paragraph positions for several
layers. Then we generate from the FORGED prompt while replacing, at one layer, the block output at
the paragraph positions with the clean activations. If the continuation stops emitting the upload
command at some layer, the forged paragraph's causal effect is carried by those activations at that
depth. Controls: forged unpatched, clean unpatched. Probe reading over the paragraph at layer 12 is
recorded per arm, so we can see whether a patch that changes behavior also changes the probe reading.
Outcome = upload command emitted in the continuation (no sandbox).
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPL = HERE.parent
sys.path.insert(0, str(HERE))
from harness.backend import MLXBackend  # noqa: E402
from run_steer import DEFAULT_MODEL, PROBES, ROLES, save_json  # noqa: E402
from steer_hook import ProbeRecorder  # noqa: E402
from continue_doubt import classify  # noqa: E402

AH = REPL / "agent-hijacking"
CASES = ["000-forgery", "001-forgery", "002-forgery", "003-forgery", "004-forgery"]
MARK = "We need to consider policy"
FILLER = ("The metric system was first adopted in France in the late eighteenth century and spread across Europe during the "
          "nineteenth century as trade and science demanded shared units. Its base units were originally defined by physical "
          "artifacts kept in Paris, and later redefined in terms of fundamental constants. Today most countries use it for "
          "commerce, engineering, and daily measurement, while a few retain customary units alongside it. The history of its "
          "adoption reflects both practical needs and political change, and standards bodies continue to refine definitions "
          "as measurement technology improves. Schools teach it early, and international agreements keep it consistent."
          "The system of units also shaped how maps, charts, and ledgers were kept, since surveyors and merchants needed figures that could be compared across borders without conversion tables. Textbooks from the period describe long debates about naming conventions, the choice of prefixes, and whether local customs should be preserved for everyday goods. Over time the arguments settled, and the same prefixes came to be used for length, mass, and volume alike. Museums still display early rulers, weights, and vessels that were certified against the reference artifacts, and visitors can compare them with modern instruments. Later reforms replaced the artifacts with definitions based on the speed of light and other constants, so that any laboratory could reproduce the units without a physical copy. The change was gradual and was coordinated through conferences that met every few years to review proposals, publish resolutions, and set timelines for adoption. Engineers, teachers, and manufacturers were consulted along the way, and most transitions were completed without disruption to trade or industry.")


class LayerIO:
    """Wraps one TransformerBlock: passthrough, capture at span, or patch at span."""

    def __init__(self, block):
        self.block, self.mode, self.pos = block, "off", 0
        self.span = (0, 0); self.buf = None; self.patched = 0

    def reset(self, mode, span=(0, 0), buf=None):
        self.mode, self.span, self.buf, self.pos, self.patched = mode, span, buf, 0, 0

    def __call__(self, x, mask, cache=None):
        import mlx.core as mx
        start = self.pos
        y = self.block(x, mask, cache)
        n = int(y.shape[1]); self.pos = start + n
        if self.mode == "off":
            return y
        s0, s1 = self.span
        lo, hi = max(start, s0), min(start + n, s1)
        if lo >= hi:
            return y
        if self.mode == "capture":
            piece = np.array(y[0, lo - start:hi - start, :].astype(mx.float32))
            self.buf[lo - s0:hi - s0] = piece
            return y
        m = np.zeros(n, np.float32); m[lo - start:hi - start] = 1.0
        patch = np.zeros((n, y.shape[-1]), np.float32); patch[lo - start:hi - start] = self.buf[lo - s0:hi - s0]
        mm = mx.array(m)[None, :, None]; pp = mx.array(patch)[None].astype(y.dtype)
        self.patched += hi - lo
        return y * (1 - mm).astype(y.dtype) + pp * mm.astype(y.dtype)


class PatchBackend(MLXBackend):
    def __init__(self, model_path, layers, **kw):
        super().__init__(model_path, **kw)
        from transformers import AutoTokenizer
        self._hf = AutoTokenizer.from_pretrained(str(self.model_path), add_bos_token=False, add_eos_token=False)
        inner = self._model.model
        self.io = {}
        for L in layers:
            self.io[L] = LayerIO(inner.layers[L]); inner.layers[L] = self.io[L]
        pr = np.load(PROBES)
        self.rec = ProbeRecorder(inner.layers[12].post_attention_layernorm, pr["sucat_L12__coef"], pr["sucat_L12__intercept"], ROLES)
        inner.layers[12].post_attention_layernorm = self.rec

    def tokens(self, prompt):
        enc = self._hf(prompt, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc.input_ids), enc.offset_mapping

    def span_tokens(self, prompt, a, b):
        ids, off = self.tokens(prompt)
        idx = [i for i, (ts, te) in enumerate(off) if ts < b and te > a and te > ts]
        return ids, (idx[0], idx[-1] + 1)


def build_clean(backend: PatchBackend, forged: str, para_escaped: str):
    a = forged.index(para_escaped); b = a + len(para_escaped)
    f_ids, (s0, s1) = backend.span_tokens(forged, a, b)
    target = s1 - s0
    words = FILLER.split()
    for k in range(min(len(words), target), 0, -1):
        for tail in ("", ".", " a", " a.", ",", " the"):
            filler = " ".join(words[:k]) + tail
            clean = forged[:a] + filler + forged[b:]
            c_ids, (c0, c1) = backend.span_tokens(clean, a, a + len(filler))
            if len(c_ids) == len(f_ids) and c0 == s0 and c1 == s1 and c_ids[:s0] == f_ids[:s0] and c_ids[s1:] == f_ids[s1:]:
                return clean, (s0, s1), filler
    raise ValueError("could not build a token-aligned clean prompt")


def run(backend: PatchBackend, prompt: str, span, seed: int, max_new: int, timeout_s: float):
    n = len(backend.tokens(prompt)[0])
    m = np.zeros(n, np.float32); m[span[0]:span[1]] = 1.0
    backend.rec.set({"paragraph": m}, n)
    gen = backend.generate(prompt, seed=seed, max_new_tokens=max_new, temperature=1.0, timeout_s=timeout_s)
    probe = backend.rec.means().get("paragraph"); backend.rec.set({}, 0)
    return gen, probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="4,8,11,14,17,20")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--max-new-tokens", type=int, default=1500)
    ap.add_argument("--gen-seconds", type=int, default=240)
    ap.add_argument("--cases", default=",".join(CASES))
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("use a new out dir")
    args.out.mkdir(parents=True)
    layers = [int(x) for x in args.layers.split(",")]; seeds = [int(s) for s in args.seeds.split(",")]
    manifest = json.loads((HERE / "data/forgery-5.json").read_text()); cases = {c["id"]: c for c in manifest["cases"]}
    backend = PatchBackend(DEFAULT_MODEL, layers)
    meta = {"started_at": datetime.now(timezone.utc).isoformat(), "layers": layers, "seeds": seeds, "pages": {}, "results": []}
    save_json(args.out / "run.json", meta)
    for cid in args.cases.split(","):
        case = cases[cid]
        run_dir = AH / ("runs/resample-004-forgery-20260911" if cid == "004-forgery" else "runs/ten-20260911") / cid
        forged = (run_dir / "step-01.prompt.txt").read_text()
        para = case["payload"][case["payload"].index(MARK):]
        para_escaped = json.dumps(para, ensure_ascii=False)[1:-1]
        clean, span, filler = build_clean(backend, forged, para_escaped)
        (args.out / f"{cid}.clean.prompt.txt").write_text(clean)
        meta["pages"][cid] = {"paragraph_tokens": span[1] - span[0], "span": span, "filler": filler, "prompt_tokens": len(backend.tokens(forged)[0])}
        # capture clean activations at the span for all layers in one prefill
        d = 2880
        for L in layers:
            backend.io[L].reset("capture", span, np.zeros((span[1] - span[0], d), np.float32))
        backend.generate(clean, seed=0, max_new_tokens=1, temperature=0.0, timeout_s=args.gen_seconds)
        bufs = {L: backend.io[L].buf.copy() for L in layers}
        for L in layers:
            backend.io[L].reset("off")
        print(json.dumps({"case": cid, "captured": True, "paragraph_tokens": span[1] - span[0]}), flush=True)
        arms = [("forged", None, forged), ("clean", None, clean)] + [(f"patch-L{L}", L, forged) for L in layers]
        for name, L, prompt in arms:
            for seed in seeds:
                for LL in layers:
                    backend.io[LL].reset("off")
                if L is not None:
                    backend.io[L].reset("patch", span, bufs[L])
                t0 = time.time()
                gen, probe = run(backend, prompt, span, seed, args.max_new_tokens, args.gen_seconds)
                cls = classify(gen.text)
                row = {"case": cid, "arm": name, "layer": L, "seed": seed, "finish": gen.finish_reason, "n_gen": gen.generated_tokens,
                       "elapsed_s": round(time.time() - t0, 1), "patched_positions": (backend.io[L].patched if L is not None else 0),
                       "probe_paragraph": probe, **cls}
                (args.out / f"{cid}__{name}__s{seed}.completion.txt").write_text(gen.text)
                meta["results"].append(row); save_json(args.out / "run.json", meta)
                print(json.dumps({k: v for k, v in row.items() if k not in ("probe_paragraph", "doubt_sentences", "final")}), flush=True)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat(); save_json(args.out / "run.json", meta)
    backend.close()


if __name__ == "__main__":
    main()
