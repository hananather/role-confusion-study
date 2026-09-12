"""Appendix E replication: gardening ("tomato") conversation projected into role space.

Port of NB02 cells 48-55 (prompt construction, forward pass, labeling, projection) and the
plot spec of NB04 cell 7 (role space suca, layer 12, system messages excluded from the plot,
first 160 tokens per message segment). Figure 7 caption: "No transformations are applied",
so raw per-token probabilities are plotted (NB02 cell 54's EWMA is notebook-only).

Conditions built exactly as NB02 cell 49 for gpt-oss-20b:
  basic_no_format          BOS + messages joined by "\\n"                       (Experiment 2, no tags)
  everything_in_user_tags  BOS + custom template, one user message with all text (Experiment 3)
  proper_tags              BOS + custom template, system/user/assistant with <think> folded (Experiment 1)
NB04 also lists everything_in_assistant_tags; NB02 does not build it, so it is omitted here.
Both templated prompts use the authors' custom gptoss.j2 (no default system prompt, no BOS),
because NB02 cell 23 installs it before cell 49 runs.

Divergences: MLX backend (see extract_activations.py); probes from fit_probes.py (sklearn).

Usage:
  python tomato_projection.py --run ../probes/runs/dev-10 --space suca --layer 12 --out out/dev-10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "probes"))
from build_probe_dataset import AUTHORS, MODEL_PREFIX, REVISION  # noqa: E402
from extract_activations import MLX_MODEL, get_tokenizer  # noqa: E402
from utils.role_templates import load_chat_template  # noqa: E402
from utils.substring_assignments import flag_message_types  # noqa: E402

ROLE_CHAR = {"s": "system", "u": "user", "c": "cot", "a": "assistant", "t": "tool"}
BASE_MESSAGE_TYPES = ["system", "user", "cot", "assistant", "user", "cot", "assistant"]  # NB02 cell 49
LABELS = {"system": "Systemness", "user": "Userness", "cot": "CoTness", "assistant": "Asstness"}
COLORS = {"system": "#90a1b9", "user": "#00a6f4", "cot": "#fd9a00", "assistant": "#00d492"}  # NB04 point_colors


def build_prompts(tokenizer):
    base_messages = [m["content"] for m in yaml.safe_load(open(AUTHORS / "experiments/role-analysis/config/tomato.yaml"))["gptoss-20b"]]
    prefix = tokenizer.bos_token or ""
    tokenizer.chat_template = load_chat_template(str(AUTHORS / "utils/chat_templates"), MODEL_PREFIX)
    prompts = {
        "basic_no_format": prefix + "\n".join(base_messages),
        "everything_in_user_tags": prefix + tokenizer.apply_chat_template(
            [{"role": "user", "content": "\n".join(base_messages)}], tokenize=False, add_generation_prompt=False),
        "proper_tags": prefix + tokenizer.apply_chat_template([
            {"role": "system", "content": base_messages[0]},
            {"role": "user", "content": base_messages[1]},
            {"role": "assistant", "content": f"<think>{base_messages[2]}</think>{base_messages[3]}"},
            {"role": "user", "content": base_messages[4]},
            {"role": "assistant", "content": f"<think>{base_messages[5]}</think>{base_messages[6]}"},
        ], tokenize=False, add_generation_prompt=False),
    }
    return base_messages, prompts


def reconstructable_tokens(tokenizer, text):
    """Token substrings from offsets, as utils/dataset.py ReconstructableTextDataset does."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    toks, last_end = [], 0
    for (s, e) in enc.offset_mapping:
        emit = max(s, last_end)
        toks.append(text[emit:e] if emit < e else "")
        last_end = max(last_end, e)
    return enc.input_ids, toks


def forward_mlx(ids_list, layer):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    model, _ = load(MLX_MODEL)
    mx.eval(model.parameters())
    inner = model.model if hasattr(model, "model") else model
    store = {}

    class Recorder(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def __call__(self, x):
            y = self.wrapped(x)
            store["h"] = y
            return y

    inner.layers[layer].post_attention_layernorm = Recorder(inner.layers[layer].post_attention_layernorm)
    out = []
    for ids in ids_list:
        logits = model(mx.array([ids]))
        mx.eval(logits, store["h"])
        out.append(np.array(store["h"][0].astype(mx.float32)))
    return out


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="probe run dir containing probes.npz")
    ap.add_argument("--space", default="suca")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    roles = [ROLE_CHAR[c] for c in args.space]
    pz = np.load(Path(args.run) / "probes.npz")
    key = f"{args.space}_L{args.layer:02d}"
    coef, intercept = pz[f"{key}__coef"], pz[f"{key}__intercept"]

    tokenizer = get_tokenizer()
    base_messages, prompts = build_prompts(tokenizer)
    (out / "prompts.json").write_text(json.dumps(prompts, indent=2, ensure_ascii=False))
    rows, ids_list = [], []
    for p_ix, (pk, text) in enumerate(prompts.items()):
        ids, toks = reconstructable_tokens(tokenizer, text)
        ids_list.append(ids)
        rows += [(p_ix, pk, i, t, tid) for i, (t, tid) in enumerate(zip(toks, ids))]
    sample_df = pd.DataFrame(rows, columns=["prompt_ix", "prompt_key", "token_ix", "token", "token_id"])
    print(f"[prompts] tokens per condition: {sample_df.groupby('prompt_key').size().to_dict()}", flush=True)

    hs = np.concatenate(forward_mlx(ids_list, args.layer))
    assert len(hs) == len(sample_df)
    labeled = (flag_message_types(sample_df, base_messages)  # NB02 cell 51
               .assign(sample_ix=lambda df: range(len(df)))
               .assign(token_in_prompt_ix=lambda df: df.groupby("prompt_ix").cumcount())
               .merge(pd.DataFrame({"base_message_ix": range(len(BASE_MESSAGE_TYPES)), "base_message_type": BASE_MESSAGE_TYPES}), on="base_message_ix", how="inner"))
    print("[label] tokens per (condition, message):\n" + labeled.groupby(["prompt_key", "base_message_ix", "base_message_type"]).size().to_string(), flush=True)

    valid = labeled[labeled.base_message_type.isin(roles)]
    probs = softmax(hs[valid.sample_ix.to_numpy()] @ coef.T + intercept)
    proj = pd.concat([valid.reset_index(drop=True), pd.DataFrame(probs, columns=roles)], axis=1)
    long = proj.melt(id_vars=[c for c in proj.columns if c not in roles], value_vars=roles, var_name="target_role", value_name="prob")
    long.assign(layer_ix=args.layer, role_space=args.space).to_csv(out / f"tomato-role-projections-{MODEL_PREFIX}.csv", index=False)

    # Paper numbers for comparison (Appendix E, correct tags): CoT 85%, user 74%, assistant 96%.
    summary = (long[long.target_role == long.base_message_type]
               .groupby(["prompt_key", "base_message_type"]).prob.mean().unstack().round(3))
    print("[summary] mean probability of the true role, by condition:\n" + summary.to_string(), flush=True)
    summary.to_csv(out / "summary-true-role-mean.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_df = long[long.base_message_type != "system"].copy()  # NB04 cell 7
    plot_df["token_in_seg_ix"] = plot_df.groupby(["prompt_key", "target_role", "base_message_ix"]).cumcount() + 1
    plot_df = plot_df[plot_df.token_in_seg_ix <= 160]
    titles = {"basic_no_format": "No tags", "everything_in_user_tags": "Everything in user tags", "proper_tags": "Correct tags"}
    for pk, g in plot_df.groupby("prompt_key"):
        fig, axes = plt.subplots(len(roles), 1, figsize=(10, 6), sharex=True)
        for ax, r in zip(axes, roles):
            gg = g[g.target_role == r].sort_values("token_in_prompt_ix").reset_index(drop=True)
            for bmt, ggg in gg.groupby("base_message_type"):
                ax.scatter(ggg.index, ggg.prob, s=6, color=COLORS[bmt], label=bmt)
            ax.set_ylim(0, 1)
            ax.set_ylabel(LABELS[r])
        axes[0].set_title(f"{titles[pk]} (layer {args.layer}, {args.space})")
        axes[0].legend(loc="upper right", fontsize=7, markerscale=2)
        fig.tight_layout()
        fig.savefig(out / f"role-space-{pk}.png", dpi=150)
        plt.close(fig)
    print(f"[done] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
