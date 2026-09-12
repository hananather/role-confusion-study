# TOY LAB (MATS 12.0 application experiment): dummy secrets, loopback receiver, isolated sandbox; runtime-only activation edits on a local cached gpt-oss-20b, weights never modified. See ../../TOY-LAB-NOTICE.md. Never copy this notice into model-facing prompts, fixtures, or trajectories.
"""Source-attribution readout under intervention (experiment 5).

For each forgery page, take the exact turn-1 prompt the baseline episode saw (page fetched, before
any reasoning), drop the trailing "<|start|>assistant", append RH6's who-wrote-it question as a new
user turn (wording and options imported verbatim from rh/rh6_who_wrote_it.py), and read the model's
next-token distribution over the four options plus a short greedy answer. Two sentences per page: the
injected instruction sentence (expected confusion: "user") and a genuine page-prose sentence
(control: should read "tool"). Arms apply steering to the page span inside the prefix only; the
question turn is never steered. This measures the model's own attribution directly, not a probe.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPL = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPL / "rh"))
from run_steer import SteerBackend, DEFAULT_MODEL, save_json, tool_content_spans  # noqa: E402
from rh6_who_wrote_it import OPTIONS, QUESTION, ASSISTANT_START, h_user, option_token_ids  # noqa: E402

AH = REPL / "agent-hijacking"
CASES = ["000-forgery", "001-forgery", "002-forgery", "003-forgery", "004-forgery"]


def injected_sentence(payload: str) -> str:
    sents = re.split(r"(?<=[.!?])\s+", payload.split("\n\n")[0])
    for s in sents:
        if ".env" in s:
            return s.strip()
    return sents[0].strip()


def page_sentence(html: str, payload: str) -> str:
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    for s in re.split(r"(?<=[.!?])\s+", text):
        w = s.split()
        if 10 <= len(w) <= 25 and s not in payload and re.match(r"^[A-Z][A-Za-z ,'()\-]+[.]$", s):
            return s.strip()
    return ""


def answer(backend: SteerBackend, prefix: str, question: str, opt_ids: dict, max_new: int = 6):
    """Greedy answer plus the option distribution at the first answer token; steering on page span only."""
    import mlx.core as mx
    spans = {"page": tool_content_spans_from_prefix(prefix)}
    full = prefix + question
    ids, masks = backend.spans_to_masks(full, spans)
    n = len(ids)
    steer_mask = masks["page"] if backend.vec is not None else np.zeros(n, np.float32)
    backend.hook.set(backend.vec, steer_mask, n)
    backend.rec.set({"page": masks["page"]}, n)
    mx.random.seed(0)
    stream = backend._stream_generate(backend._model, backend._tokenizer, prompt=ids, max_tokens=max_new,
                                      sampler=backend._make_sampler(temp=0.0), prefill_step_size=backend.prefill_step_size)
    toks, first_lp = [], None
    try:
        for r in stream:
            if first_lp is None:
                first_lp = np.array(r.logprobs.astype(mx.float32))
            toks.append(int(r.token))
            if r.finish_reason is not None or int(r.token) in backend._stop_ids:
                break
    finally:
        stream.close()
    stats = {"edited_positions": backend.hook.edited, "probe_page": backend.rec.means().get("page")}
    backend.hook.set(None, None, 0); backend.rec.set({}, 0)
    p = np.exp(first_lp)
    popt = {o: float(p[ids_].sum()) for o, ids_ in opt_ids.items()}
    text = backend._decode(toks)
    first = backend._decode(toks[:1]).strip().lower()
    return {"answer": text, "answer_first": first, "answer_option": next((o for o in OPTIONS if first == o), "other"),
            **{f"p_{o}": popt[o] for o in OPTIONS}, "p_options_total": sum(popt.values()), **stats}


def tool_content_spans_from_prefix(prefix: str):
    spans = []
    for m in re.finditer(r"<\|start\|>functions\.[^<]*?<\|channel\|>commentary<\|message\|>(.*?)<\|end\|>", prefix, re.S):
        spans.append((m.start(1), m.end(1)))
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="none:0:block11,tool_minus_cot:8:block11,tool_minus_cot:16:block11,forgery_minus_standard:8:style")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("use a new out dir")
    args.out.mkdir(parents=True)
    manifest = json.loads((HERE / "data/forgery-5.json").read_text())
    cases = {c["id"]: c for c in manifest["cases"]}
    arms = [a.split(":") for a in args.arms.split(",")]
    backend = SteerBackend(DEFAULT_MODEL, "tool_minus_cot", 0.0, HERE / "directions/block11.npz")
    opt_ids = option_token_ids(backend._hf)
    dirs = {"block11": np.load(HERE / "directions/block11.npz")}
    if (HERE / "directions/style.npz").exists():
        dirs["style"] = np.load(HERE / "directions/style.npz")
    rows, meta = [], {"started_at": datetime.now(timezone.utc).isoformat(), "question": QUESTION, "options": OPTIONS,
                      "option_token_ids": opt_ids, "arms": arms, "results": rows}
    for cid in CASES:
        case = cases[cid]
        run_dir = AH / ("runs/resample-004-forgery-20260911" if cid == "004-forgery" else "runs/ten-20260911") / cid
        prompt = (run_dir / "step-01.prompt.txt").read_text()
        assert prompt.endswith("<|start|>assistant")
        prefix = prompt[: -len("<|start|>assistant")]
        html = Path(case["fixture_path"]).read_text()
        sentences = {"injected": injected_sentence(case["payload"]), "page": page_sentence(html, case["payload"])}
        for direction, alpha, dfile in arms:
            alpha = float(alpha)
            if dfile not in dirs:
                print(json.dumps({"skip": f"{direction} needs {dfile}.npz"}), flush=True); continue
            d = dirs[dfile]
            gap = float(d[f"gap_{direction}"]) if f"gap_{direction}" in d else float(d["gap_tool_cot"])
            backend.vec = None if alpha == 0 else (alpha * gap * d[direction].astype(np.float32)).astype(np.float32)
            arm = f"{direction}-a{alpha:g}"
            for kind, sent in sentences.items():
                if not sent:
                    continue
                q = h_user(QUESTION.format(sentence=sent)) + ASSISTANT_START
                t0 = time.time()
                res = answer(backend, prefix, q, opt_ids)
                row = {"case": cid, "arm": arm, "sentence_kind": kind, "sentence": sent, "elapsed_s": round(time.time() - t0, 1), **res}
                rows.append(row); save_json(args.out / "run.json", meta)
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k not in ("probe_page", "sentence")}), flush=True)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat(); save_json(args.out / "run.json", meta)
    backend.close()


if __name__ == "__main__":
    main()
