"""E4 step 3: run the four-condition prompts through gpt-oss-20b, project every content token onto the
role probes, aggregate as the authors do, and draw the gpt-oss-20b panel of Figure 23.

Port of NB02 cells 29-30 and 36-38 (experiments/role-analysis/02-train-role-probes.ipynb) and the plot
spec of NB03 cell 8, commit ec333c40fd43fe991e1ebf66765051b6d7e35784:
  cell 29  forward pass on every prompt, hook = post_attention_layernorm output (same site as the probes)
  cell 30  flag_message_types(sample_df, stripped message contents, allow_ambiguous=False) -> role per token;
           role tags and separators stay unlabeled (NaN) and are excluded
  cell 36  softmax(h @ W.T + b) for every (role_space, layer) probe, over tokens whose role is in the space
  cell 37  mean true-class probability: per (conv_type, role_space, layer, role, prompt) mean, then mean over prompts
  cell 38  argmax accuracy, same two-level mean
  NB03 c8  Figure 23: y = argmax accuracy from all_conv_acc (cell 38), role_space uat, conv types
           tagged/untagged/tool_tagged, rows user/assistant, x = layer

Additions (not in the authors' code): cluster bootstrap 95% CI over conversations for every table cell
(--n-boot resamples of conversation ids, percentile interval); a Table 3 style summary at --table-layer;
per-token probabilities saved for --token-spaces (default uat) in tokens-projections.parquet.

Memory: activations are projected on the fly, prompt by prompt, so nothing is written to disk per layer.
--layer-chunk N runs the forward passes N recorded layers at a time (more passes, less resident memory);
the default 0 records all requested layers in one pass (about 24 x tokens x 2880 x 2 bytes per prompt).

Deviation from cell 30: a conversation whose labeling fails (ambiguous match) is dropped in ALL four
conditions, so every condition covers the same conversation set. The authors drop only the failing prompt.

Usage:
  python e4/project_conversations.py --prompts e4/data/prompts.parquet --run runs/full-249 --out e4/out/full-249
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "probes"))
from build_probe_dataset import MODEL_PREFIX  # noqa: E402
from extract_activations import MLX_MODEL, get_tokenizer  # noqa: E402
from fit_probes_mlx import ROLE_CHAR  # noqa: E402
from utils.substring_assignments import flag_message_types  # noqa: E402

CONV_TYPES = ["untagged", "tagged", "user_tagged", "tool_tagged"]
FIG_TYPES = {"tagged": ("Baseline", "#62748e"), "untagged": ("No tags", "#00a6f4"), "tool_tagged": ("Injection (tool tagged)", "#ff6467")}
FIG_ROWS = [("user", "Userness", "(of user-style text)"), ("assistant", "Assistantness", "(of assistant-style text)")]


def reconstructable_tokens(tokenizer, text):
    """Token substrings from offsets (utils/dataset.py ReconstructableTextDataset); same as appendix-e."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    toks, last_end = [], 0
    for (s, e) in enc.offset_mapping:
        emit = max(s, last_end)
        toks.append(text[emit:e] if emit < e else "")
        last_end = max(last_end, e)
    return enc.input_ids, toks


def load_probes(path, spaces, layers):
    z = np.load(path)
    probes = {}
    for k in z.files:
        m = re.fullmatch(r"([a-z]+)_L(\d+)__coef", k)
        if not m:
            continue
        space, layer = m.group(1), int(m.group(2))
        if (spaces is None or space in spaces) and (layers is None or layer in layers):
            probes[(space, layer)] = (z[k].astype(np.float32), z[f"{space}_L{layer:02d}__intercept"].astype(np.float32))
    return probes


def label_tokens(input_df, tokenizer, log):
    """NB02 cell 30. Returns token table with role per token, and the set of dropped conv_ids."""
    rows, bad = [], set()
    for r in input_df.itertuples():
        ids, toks = reconstructable_tokens(tokenizer, r.prompt)
        df = pd.DataFrame({"prompt_ix": r.prompt_ix, "token_ix": range(len(ids)), "token": toks, "token_id": ids})
        msgs = json.loads(r.messages)
        spans = [m["content"].strip() for m in msgs]
        roles = [m["role"] for m in msgs]
        try:
            lab = (flag_message_types(df, spans, False)
                   .merge(pd.DataFrame({"role": roles, "base_message_ix": range(len(spans))}), on="base_message_ix", how="left"))
            assert len(lab) == len(df)
        except Exception as e:  # ambiguous match
            log(f"[label] drop conv {r.conv_id} ({r.conv_type}): {str(e)[:120]}")
            bad.add(r.conv_id)
            continue
        rows.append(lab.assign(conv_type=r.conv_type, conv_id=r.conv_id))
    tok = pd.concat(rows, ignore_index=True)
    tok = tok[~tok.conv_id.isin(bad)].reset_index(drop=True)
    tok["token_in_prompt_ix"] = tok.groupby("prompt_ix").cumcount()
    return tok, bad


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def run_passes(input_df, tok, probes, layer_chunk, token_spaces, out, log):
    """Forward passes with a Recorder on the hook site; project on the fly. Returns per-conv tables."""
    import mlx.core as mx
    import mlx.nn as nn
    import pyarrow as pa
    import pyarrow.parquet as pq
    from mlx_lm import load

    model, _ = load(MLX_MODEL)
    mx.eval(model.parameters())
    inner = model.model if hasattr(model, "model") else model
    store, active = {}, set()

    class Recorder(nn.Module):
        def __init__(self, wrapped, idx):
            super().__init__()
            self.wrapped, self.idx = wrapped, idx

        def __call__(self, x):
            y = self.wrapped(x)
            if self.idx in active:
                store[self.idx] = y
            return y

    layers = sorted({l for _, l in probes})
    for i, layer in enumerate(inner.layers):
        if i in layers:
            layer.post_attention_layernorm = Recorder(layer.post_attention_layernorm, i)
    chunks = [layers] if layer_chunk <= 0 else [layers[i:i + layer_chunk] for i in range(0, len(layers), layer_chunk)]
    spaces = sorted({s for s, _ in probes})
    space_roles = {s: [ROLE_CHAR[c] for c in s] for s in spaces}
    ids_by_prompt = tok.groupby("prompt_ix").token_id.apply(list).to_dict()
    meta = input_df.set_index("prompt_ix")
    prob_rows, acc_rows = [], []
    writer, tok_buf = None, []
    n_tokens_total = sum(len(v) for v in ids_by_prompt.values())

    for c_ix, chunk in enumerate(chunks):
        active.clear()
        active.update(chunk)
        store.clear()
        pos, t0 = 0, time.time()
        for k, (p_ix, ids) in enumerate(ids_by_prompt.items()):
            logits = model(mx.array([ids]))
            mx.eval(logits, *[store[l] for l in chunk])
            hs = {l: np.array(store[l][0].astype(mx.float32)) for l in chunk}
            g = tok[tok.prompt_ix == p_ix]
            assert len(g) == len(ids)
            conv_type, conv_id = meta.conv_type[p_ix], int(meta.conv_id[p_ix])
            for s in spaces:
                roles = space_roles[s]
                valid = g[g.role.isin(roles)]
                if valid.empty:
                    continue
                vix = valid.token_ix.to_numpy()
                y_true = valid.role.map({r: i for i, r in enumerate(roles)}).to_numpy()
                for l in chunk:
                    if (s, l) not in probes:
                        continue
                    W, b = probes[(s, l)]
                    probs = softmax(hs[l][vix] @ W.T + b)
                    pred = probs.argmax(1)
                    p_true = probs[np.arange(len(vix)), y_true]
                    for r_i, r in enumerate(roles):
                        m = y_true == r_i
                        if not m.any():
                            continue
                        n = int(m.sum())
                        acc_rows.append((conv_type, s, l, r, conv_id, n, float((pred[m] == r_i).mean())))
                        mp = probs[m].mean(0)
                        for t_i, t in enumerate(roles):
                            prob_rows.append((conv_type, s, l, r, t, conv_id, n, float(mp[t_i])))
                    if s in token_spaces:
                        tok_buf.append(pd.DataFrame({"prompt_ix": p_ix, "conv_type": conv_type, "conv_id": conv_id, "role_space": s, "layer_ix": l,
                                                     "token_in_prompt_ix": valid.token_in_prompt_ix.to_numpy(), "role": valid.role.to_numpy(),
                                                     "pred_role": np.array(roles)[pred], "p_true": p_true.astype(np.float32),
                                                     **{f"p_{t}": probs[:, t_i].astype(np.float32) for t_i, t in enumerate(roles)}}))
            pos += len(ids)
            if tok_buf and (len(tok_buf) >= 200 or k + 1 == len(ids_by_prompt)):
                tdf = pd.concat(tok_buf, ignore_index=True)
                tok_buf = []
                table = pa.Table.from_pandas(tdf, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out / "tokens-projections.parquet", table.schema)
                writer.write_table(table)
            if k == 0 and c_ix == 0:
                nxt = int(mx.argmax(logits[0, -1]).item())
                log(f"[fwd] first prompt {len(ids)} tokens, next token id {nxt}, peak {mx.get_peak_memory() / 1e9:.1f} GB")
            if (k + 1) % 10 == 0 or k + 1 == len(ids_by_prompt):
                el = time.time() - t0
                log(f"[fwd chunk {c_ix + 1}/{len(chunks)} layers {chunk[0]}-{chunk[-1]}] {k + 1}/{len(ids_by_prompt)} prompts, "
                    f"{pos}/{n_tokens_total} tokens, {pos / el:.0f} tok/s, {el / 60:.1f} min, peak {mx.get_peak_memory() / 1e9:.1f} GB")
    if writer is not None:
        writer.close()
    prob = pd.DataFrame(prob_rows, columns=["conv_type", "role_space", "layer_ix", "role", "target_role", "conv_id", "n_tokens", "mean_prob"])
    acc = pd.DataFrame(acc_rows, columns=["conv_type", "role_space", "layer_ix", "role", "conv_id", "n_tokens", "mean_acc"])
    return prob, acc


def cluster_bootstrap(per_conv, value_col, group_cols, n_boot, seed, cluster_col="conv_id"):
    """Mean of per-conversation means per group, with a percentile bootstrap over conversations.
    The same conversation resample is applied to every group (cluster bootstrap)."""
    wide = per_conv.pivot_table(index=cluster_col, columns=group_cols, values=value_col, aggfunc="mean")
    x = wide.to_numpy(dtype=np.float64)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.stack([np.nanmean(x[ix], axis=0) for ix in idx])  # (n_boot, n_groups)
    res = pd.DataFrame({"mean_acc": np.nanmean(x, axis=0), "ci_lo": np.nanpercentile(boots, 2.5, axis=0),
                        "ci_hi": np.nanpercentile(boots, 97.5, axis=0), "n_convs": np.sum(~np.isnan(x), axis=0)},
                       index=wide.columns).reset_index()
    return res


def draw_figure23(table, out, title, n_convs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, PercentFormatter

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 9,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4})
    d = table[table.role_space == "uat"]
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.9), sharex=True)
    for ax, (role, name, sub) in zip(axes, FIG_ROWS):
        for ct, (label, color) in FIG_TYPES.items():
            g = d[(d.role == role) & (d.conv_type == ct)].sort_values("layer_ix")
            if g.empty:
                continue
            ax.fill_between(g.layer_ix, g.ci_lo, g.ci_hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(g.layer_ix, g.mean_acc, color=color, linewidth=0.6, alpha=0.7, label=label)
            ax.scatter(g.layer_ix, g.mean_acc, color=color, s=9, zorder=3)
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 1]))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(True, color="#e5e7eb", linewidth=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2, labelsize=8)
        ax.text(-0.26, 0.5, name, transform=ax.transAxes, rotation=90, va="center", ha="center", fontweight="bold", fontsize=9)
        ax.text(-0.17, 0.5, sub, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=6.5)
    axes[0].set_title(f"{title} (n = {n_convs} conversations)", fontsize=9, fontweight="bold")
    axes[1].set_xlabel("Layer index", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.55, -0.01),
               handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(rect=(0.04, 0.06, 1, 1))
    fig.savefig(out / f"figure23-{MODEL_PREFIX}.png", dpi=300)
    fig.savefig(out / f"figure23-{MODEL_PREFIX}.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(HERE / "data/prompts.parquet"))
    ap.add_argument("--run", default=str(HERE.parent / "runs/full-249"), help="probe run dir")
    ap.add_argument("--probes", default="probes.npz", help="probe file inside --run (authors' split: probes.npz; base split: probes-basesplit.npz)")
    ap.add_argument("--spaces", default="all", help="comma list of role spaces, or 'all' present in the probe file")
    ap.add_argument("--layers", default="all", help="'all' present in the probe file, or comma list")
    ap.add_argument("--layer-chunk", type=int, default=0, help="recorded layers per forward pass (0 = all in one pass)")
    ap.add_argument("--token-spaces", default="uat", help="spaces whose per-token probabilities are saved")
    ap.add_argument("--max-convs", type=int, default=0, help="keep only conv_id < N (smoke tests)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--table-layer", type=int, default=12, help="layer for the Table 3 style summary (NB02 cell 33: middle probed layer)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "project.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    spaces = None if args.spaces == "all" else args.spaces.split(",")
    layers = None if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    probe_path = Path(args.run) / args.probes
    probes = load_probes(probe_path, spaces, layers)
    if not probes:
        raise SystemExit(f"no probes matched in {probe_path}")
    log(f"[probes] {probe_path}: {len(probes)} probes, spaces {sorted({s for s, _ in probes})}, layers {sorted({l for _, l in probes})}")

    input_df = pd.read_parquet(args.prompts)
    if args.max_convs:
        input_df = input_df[input_df.conv_id < args.max_convs].reset_index(drop=True)
    tokenizer = get_tokenizer()
    t0 = time.time()
    tok, bad = label_tokens(input_df, tokenizer, log)
    input_df = input_df[~input_df.conv_id.isin(bad)].reset_index(drop=True)
    counts = (tok[tok.role.notna()].groupby(["conv_type", "role"]).size().unstack(fill_value=0))
    log(f"[label] {input_df.conv_id.nunique()} convs x {input_df.conv_type.nunique()} conditions, {len(tok)} tokens, "
        f"{int(tok.role.notna().sum())} labeled ({time.time() - t0:.0f}s); dropped convs {sorted(bad)}\n"
        f"[label] labeled tokens by (conv_type, role):\n{counts.to_string()}")
    tok.drop(columns=["base_message"]).to_parquet(out / "tokens.parquet", index=False)

    prob, acc = run_passes(input_df, tok, probes, args.layer_chunk, set(args.token_spaces.split(",")), out, log)
    prob.to_csv(out / "per-conv-prob.csv", index=False)
    acc.to_csv(out / "per-conv-acc.csv", index=False)

    # NB02 cell 37: mean true-class probability, per-conversation means first, then across conversations (+ bootstrap CI)
    projs = cluster_bootstrap(prob[prob.role == prob.target_role], "mean_prob", ["conv_type", "role_space", "layer_ix", "role"], args.n_boot, args.seed)
    projs.to_csv(out / f"all_conv_projs_{MODEL_PREFIX}.csv", index=False)
    # same, keeping every target role (needed for Toolness of user text etc.)
    projs_all = cluster_bootstrap(prob, "mean_prob", ["conv_type", "role_space", "layer_ix", "role", "target_role"], args.n_boot, args.seed)
    projs_all.to_csv(out / f"all_conv_projs_alltargets_{MODEL_PREFIX}.csv", index=False)
    # NB02 cell 38: argmax accuracy
    accs = cluster_bootstrap(acc, "mean_acc", ["conv_type", "role_space", "layer_ix", "role"], args.n_boot, args.seed)
    accs.to_csv(out / f"all_conv_acc_{MODEL_PREFIX}.csv", index=False)

    # Table 3 style summary at the middle layer, uat space
    tl = args.table_layer if args.table_layer in set(projs_all.layer_ix) else sorted(set(projs_all.layer_ix))[len(set(projs_all.layer_ix)) // 2]
    t3 = projs_all[(projs_all.role_space == "uat") & (projs_all.layer_ix == tl) & projs_all.role.isin(["user", "assistant"])
                   & projs_all.conv_type.isin(["tagged", "untagged", "tool_tagged"])]
    t3 = t3.assign(cell=lambda d: d.mean_acc.map("{:.1%}".format) + " [" + d.ci_lo.map("{:.0%}".format) + ", " + d.ci_hi.map("{:.0%}".format) + "]")
    t3w = t3.pivot(index=["conv_type", "role"], columns="target_role", values="cell").reindex(
        [(c, r) for c in ["tagged", "untagged", "tool_tagged"] for r in ["user", "assistant"]])
    t3.to_csv(out / f"table3_{MODEL_PREFIX}.csv", index=False)
    log(f"[table3] layer {tl}, role space uat, mean prob of target role over tokens of each style (paper gpt-oss-20b mid-layer: "
        f"tagged user 99.7% user / 0.0% tool; tool_tagged user 87.6% user / 9.3% tool):\n{t3w.to_string()}")
    fig_rows = accs[(accs.role_space == "uat") & accs.role.isin(["user", "assistant"]) & accs.conv_type.isin(FIG_TYPES)]
    log("[figure23] uat argmax accuracy of the true role by layer (NB03 cell 8 plots all_conv_acc):\n"
        + fig_rows.pivot(index="layer_ix", columns=["role", "conv_type"], values="mean_acc").round(3).to_string())
    if "uat" in set(accs.role_space):
        draw_figure23(accs, out, "gpt-oss-20b", int(acc.conv_id.nunique()))
        log(f"[figure23] wrote {out / f'figure23-{MODEL_PREFIX}.png'}")
    meta = {"prompts": args.prompts, "probes": str(probe_path), "spaces": sorted({s for s, _ in probes}), "layers": sorted({l for _, l in probes}),
            "n_convs": int(prob.conv_id.nunique()), "dropped_convs": sorted(int(b) for b in bad), "n_boot": args.n_boot, "seed": args.seed,
            "table_layer": int(tl), "model": MLX_MODEL, "site": "post_attention_layernorm output (pre-MLP)", "batch_size": 1, "padding": "none",
            "source": "NB02 cells 29-30, 36-38; NB03 cell 8; commit ec333c40fd43fe991e1ebf66765051b6d7e35784"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    log(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
