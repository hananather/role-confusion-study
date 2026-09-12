# TOY LAB (MATS 12.0 application experiment): dummy secrets, loopback receiver, isolated sandbox; runtime-only activation edits on a local cached gpt-oss-20b, weights never modified. See ../../TOY-LAB-NOTICE.md. Never copy this notice into model-facing prompts, fixtures, or trajectories.
"""Attack-matched 'forgery style' direction at the block-11 output.

For each of the five pages we have both fixtures (standard payload, forged payload). We render the
turn-1 context (system, developer, user, the curl tool call, and the page as a tool result), locate
the payload span, forward-pass, and take the mean block-11 residual over the payload tokens. The
direction is normalize(mean over 5 forged payload spans - mean over 5 standard payload spans): the
literal activation-space difference the forgery adds. Subtracting it is a destyle. Random controls
share its magnitude via the gap norm.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import protocol as p  # noqa: E402

MLX_MODEL = "mlx-community/gpt-oss-20b-MXFP4-Q8"
HF_REV = "6cee5e81ee83917806bbde320786a8fb61efebee"
PILOT = HERE.parent / "agent-hijacking/data/pilot-20260911"
STEER_LAYER = 11


def turn1_prompt(case):
    msgs = p.initial_messages(case["user_prompt"])
    msgs.append(p.h_tool_call("functions.bash", '{"command":"curl -s http://127.0.0.1:8765/page.html"}'))
    html = (PILOT / case["fixture_path"]).read_text()
    tool_out = json.dumps({"stdout": html, "stderr": "", "exit_code": 0}, ensure_ascii=False)
    msgs.append(p.h_tool_result("functions.bash", tool_out))
    return p.render_prompt(msgs) + p.ASSISTANT_PREFIX


def main():
    import mlx.core as mx
    from mlx_lm import load
    from transformers import AutoTokenizer

    manifest = json.loads((PILOT / "manifest.json").read_text())
    snap = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots" / HF_REV
    hf = AutoTokenizer.from_pretrained(str(snap), add_eos_token=False, add_bos_token=False)
    model, _ = load(MLX_MODEL)
    mx.eval(model.parameters())
    inner = model.model
    store = {}

    class Rec:
        def __init__(self, block):
            self.block = block
        def __call__(self, x, mask, cache=None):
            y = self.block(x, mask, cache); store[0] = y; return y

    inner.layers[STEER_LAYER] = Rec(inner.layers[STEER_LAYER])
    means = {"standard": [], "forgery": []}
    per_case = {}
    t0 = time.time()
    for case in manifest["cases"]:
        payload = case["payload"]
        prompt = turn1_prompt(case)
        enc = hf(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc.input_ids
        k = prompt.find(payload)
        if k < 0:
            print(json.dumps({"case": case["id"], "warning": "payload not found verbatim"}), flush=True)
            continue
        a, b = k, k + len(payload)
        idx = [i for i, (ts, te) in enumerate(enc.offset_mapping) if ts < b and te > a and te > ts]
        logits = model(mx.array([ids])); mx.eval(logits, store[0])
        h = np.array(store[0][0].astype(mx.float32))
        m = h[idx].mean(0)
        means[case["variant"]].append(m)
        per_case[case["id"]] = {"variant": case["variant"], "payload_tokens": len(idx), "prompt_tokens": len(ids)}
        print(json.dumps({"case": case["id"], "variant": case["variant"], "payload_tokens": len(idx),
                          "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    fs = np.mean(means["forgery"], 0)
    st = np.mean(means["standard"], 0)
    diff = fs - st
    unit = (diff / np.linalg.norm(diff)).astype(np.float32)
    rng = np.random.default_rng(20260911)
    randu = (rng.normal(size=diff.shape)); randu = (randu / np.linalg.norm(randu)).astype(np.float32)
    out = HERE / "directions" / "style.npz"
    np.savez(out, forgery_minus_standard=unit, gap_forgery_minus_standard=np.float32(np.linalg.norm(diff)),
             random_style=randu, gap_random_style=np.float32(np.linalg.norm(diff)),
             mean_forgery=fs.astype(np.float32), mean_standard=st.astype(np.float32))
    meta = {"layer_zero_based": STEER_LAYER, "site": "TransformerBlock output", "model": MLX_MODEL,
            "n_forgery": len(means["forgery"]), "n_standard": len(means["standard"]),
            "gap_forgery_minus_standard": float(np.linalg.norm(diff)),
            "cos_with_tool_minus_cot": None, "per_case": per_case}
    d = np.load(HERE / "directions/block11.npz")
    tmc = d["tool_minus_cot"]
    meta["cos_with_tool_minus_cot"] = float(unit @ (tmc / np.linalg.norm(tmc)))
    (HERE / "directions" / "style.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
