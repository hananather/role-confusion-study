#!/usr/bin/env python3
"""I port NB04 cells 3, 7 and 14 into inspectable tables and Matplotlib figures.

I retain the authors' raw probability values and token selection exactly. I record
the rendering backend/font change separately; I do not claim pixel equivalence
to the authors' R/ggplot/Cairo output. This module never loads a model.
"""
from pathlib import Path
import re

CONDITIONS = ("proper_tags", "basic_no_format", "everything_in_user_tags")
ROLES = ("system", "user", "cot", "assistant")
ROLE_TITLES = ("Systemness", "Userness", "CoTness", "Asstness")
CONDITION_TITLES = ("Correct tags", "No tags", "Everything in user tags")
POINT_COLORS = dict(zip(ROLES, ("#90a1b9", "#00a6f4", "#fd9a00", "#00d492")))
TEXT_COLORS = dict(zip(ROLES, ("#62748e", "#0084d1", "#e17100", "#009966")))


def prepare_subsets(raw):
    """I reproduce R's consecutive_id and one-based row_number before filtering."""
    import pandas as pd

    keys = ["role_space", "layer_ix", "prompt_key", "prompt_ix", "target_role"]
    required = keys + ["sample_ix", "token_in_prompt_ix", "base_message_type", "token", "prob"]
    missing = set(required).difference(raw.columns)
    if missing:
        raise ValueError(f"Missing projection columns: {sorted(missing)}")
    raw = raw[(raw.role_space == "suca") & (raw.layer_ix == 12)].copy()
    if set(raw.prompt_key) != set(CONDITIONS) or set(raw.target_role) != set(ROLES):
        raise ValueError("I require all three conditions and four role readings.")
    raw = raw.sort_values(keys + ["token_in_prompt_ix"], kind="stable")
    raw["seg_ix"] = raw.groupby(keys, sort=False)["base_message_type"].transform(
        lambda series: series.ne(series.shift()).cumsum())
    raw = raw.sort_values(keys + ["seg_ix", "sample_ix"], kind="stable")
    raw["token_in_seg_ix"] = raw.groupby(keys + ["seg_ix"], sort=False).cumcount() + 1
    raw["original_token_in_prompt_ix"] = raw.token_in_prompt_ix
    four = raw[(raw.base_message_type != "system") & (raw.token_in_seg_ix <= 160)].copy()
    four = four.sort_values(["prompt_key", "target_role", "token_in_prompt_ix"], kind="stable")
    four["display_token_ix"] = four.groupby(["prompt_key", "target_role"]).cumcount() + 1
    candidates = raw[(raw.base_message_type != "system") & (raw.target_role == "cot")
                     & (raw.token_in_seg_ix <= 120)].copy()
    matching = candidates.groupby(["seg_ix", "token_in_seg_ix"], as_index=False).agg(
        n_distinct_toks=("token", "nunique"), n_distinct_prompts=("prompt_key", "nunique"))
    matching = matching[(matching.n_distinct_toks == 1)
                        & (matching.n_distinct_prompts == len(CONDITIONS))]
    overview = candidates.merge(matching, on=["seg_ix", "token_in_seg_ix"], how="inner")
    overview = overview.sort_values(["prompt_key", "seg_ix", "token_in_seg_ix"], kind="stable")
    overview["display_token_ix"] = overview.groupby("prompt_key").cumcount() + 1
    if overview.empty:
        raise ValueError("No common tokens survive the Figure 7 source filter.")
    # I retain every role reading on Figure 7's token subset for denominator audits.
    overview_all = raw.merge(overview[["sample_ix", "display_token_ix"]], on="sample_ix", how="inner")
    for condition in CONDITIONS:
        n_overview = overview.loc[overview.prompt_key == condition, "sample_ix"].nunique()
        if n_overview == 0 or n_overview != len(overview) // 3:
            raise ValueError("Figure 7 has inconsistent condition denominators.")
    return raw, four, overview, overview_all, matching


def wrap_tokens(tokens, line_width, max_lines=2, ellipsis=".."):
    """I mirror the authors' token-preserving wrap, with literal plot text."""
    lines, current, used = [], "", 0
    for token in tokens:
        candidate = current + token
        if len(candidate) <= line_width or current == "":
            current = candidate
            used += 1
        else:
            lines.append(current)
            if len(lines) >= max_lines:
                break
            current = token
            used += 1
    if len(lines) < max_lines and current:
        lines.append(current)
    # I disable math parsing in Matplotlib instead of adding Markdown escapes.
    lines = [re.sub(r"<br\s*/?>", "\n", line, flags=re.I).replace("\r\n", "\n")
             .replace("\r", "\n").replace("\n", r"\n") for line in lines]
    return "\n".join(lines) + (ellipsis if used < len(tokens) else "")


def segment_annotations(rows, overview=False):
    out = []
    rows = rows.sort_values("display_token_ix").drop_duplicates("sample_ix")
    for seg_ix, group in rows.groupby("seg_ix", sort=False):
        role = group.base_message_type.iloc[0]
        if overview:
            width = 6 if role == "user" else (8 if seg_ix == 3 else 18) if role == "cot" else 22
        else:
            width = 4 if role == "user" else (10 if seg_ix == 3 else 14) if role == "cot" else 24
        out.append({"seg_ix": int(seg_ix), "role": role,
                    "start_ix": int(group.display_token_ix.iloc[0]),
                    "label": wrap_tokens(group.token.tolist(), width, ellipsis="…" if overview else "..")})
    return out


def render_figures(four, overview, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    out_dir = Path(out_dir)
    families = {font.name for font in font_manager.fontManager.ttflist}
    font = "TeX Gyre Termes" if "TeX Gyre Termes" in families else "STIXGeneral"
    plt.rcParams.update({"font.family": font, "font.size": 9, "pdf.fonttype": 42,
                         "text.usetex": False, "text.parse_math": False,
                         "axes.linewidth": 0.5, "savefig.facecolor": "white"})
    artifacts, labels = [], {}

    def draw(ax, rows, annotations, show_labels):
        for role in ROLES:
            group = rows[rows.base_message_type == role]
            ax.scatter(group.display_token_ix, group.prob, s=1.1, linewidths=0,
                       color=POINT_COLORS[role], alpha=0.9 if len(axes) == 3 else 1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0, 1], labels=["0%", "100%"])
        ax.tick_params(axis="y", labelsize=8, width=0.4, length=2)
        ax.set_xlim(1, int(rows.display_token_ix.max()))
        ax.set_xticks([item["start_ix"] for item in annotations])
        ax.set_xticklabels([item["label"] if show_labels else "" for item in annotations],
                           ha="left", fontsize=7 if len(axes) == 3 else 8)
        for tick, item in zip(ax.get_xticklabels(), annotations):
            tick.set_color(TEXT_COLORS[item["role"]])
        ax.tick_params(axis="x", length=2, width=0.4, pad=3)
        ax.set_axisbelow(True)
        ax.grid(axis="both", color="#d9d9d9", linewidth=0.3)
        for spine in ax.spines.values():
            spine.set_color("#333333")

    def save(fig, stem):
        # I preserve the specified canvas dimensions; tight cropping would change them.
        for extension in ("png", "pdf"):
            path = out_dir / f"{stem}.{extension}"
            fig.savefig(path, dpi=300)
            artifacts.append(path.name)
        plt.close(fig)

    for index, condition in enumerate(CONDITIONS):
        rows = four[four.prompt_key == condition]
        annotations = segment_annotations(rows)
        labels[f"figure-{20 + index}"] = annotations
        fig, axes = plt.subplots(4, 1, figsize=(6.75, 3.4), sharex=True)
        for row_index, (ax, role, title) in enumerate(zip(axes, ROLES, ROLE_TITLES)):
            draw(ax, rows[rows.target_role == role], annotations, row_index == 3)
            ax.set_ylabel(title, fontsize=8, fontweight="bold", labelpad=9)
        # I follow theme_iclr, whose plot.title is blank despite the notebook's labs(title=...).
        fig.subplots_adjust(left=0.105, right=0.9, top=0.97, bottom=0.15, hspace=0.29)
        save(fig, f"figure-{20 + index}-{condition}")

    fig, axes = plt.subplots(3, 1, figsize=(6.75, 2.8), sharex=True)
    for index, (ax, condition) in enumerate(zip(axes, CONDITIONS)):
        rows = overview[overview.prompt_key == condition]
        annotations = segment_annotations(rows, overview=True)
        labels[f"figure-7-{condition}"] = annotations
        draw(ax, rows, annotations, index == 2)
        title = ("Experiment 1: Correct tags", "Experiment 2: No tags", "Experiment 3: All in user tags")[index]
        ax.set_title(title, fontsize=7.5, fontweight="bold", pad=3)
    fig.text(0.016, 0.55, "CoTness", rotation=90, va="center", fontsize=8)
    fig.subplots_adjust(left=0.105, right=0.9, top=0.92, bottom=0.18, hspace=0.5)
    save(fig, "figure-7-cotness-overview")
    return {"artifacts": artifacts, "font_family": font, "segment_annotations": labels,
            "renderer": "Matplotlib", "source_renderer": "R ggplot2 and Cairo PDF",
            "divergence": "I preserve probability rows, filters, role colors, segment labels, axes and canvas sizes; font metrics and facet spacing differ from the authors' R renderer."}


def summarize_and_compare(subsets):
    import pandas as pd
    frames = []
    for name, rows in subsets.items():
        summary = rows.groupby(["prompt_key", "base_message_type", "target_role"], as_index=False).agg(
            n_tokens=("sample_ix", "nunique"), mean_probability=("prob", "mean"))
        summary["subset"] = name
        summary["observed_percent"] = 100 * summary.mean_probability
        frames.append(summary)
    means = pd.concat(frames, ignore_index=True)
    references = []
    values = ((74, 85, 96), (75, 82, 92), (82, 85, 92))
    for condition, triple in zip(CONDITIONS, values):
        for role, value in zip(ROLES[1:], triple):
            references.append(dict(prompt_key=condition, base_message_type=role, target_role=role,
                                   source_percent=value, source="Appendix E, pp.21-23"))
    references.extend([
        dict(prompt_key="basic_no_format", base_message_type="cot", target_role="cot",
             source_percent=83, source="Figure 7 caption/main text; Appendix E instead reports 82%"),
        dict(prompt_key="everything_in_user_tags", base_message_type="cot", target_role="user",
             source_percent=2, source="Appendix E, Experiment 3"),
        dict(prompt_key="everything_in_user_tags", base_message_type="assistant", target_role="user",
             source_percent=1, source="Appendix E, Experiment 3"),
    ])
    comparison = means.merge(pd.DataFrame(references),
                             on=["prompt_key", "base_message_type", "target_role"], how="inner")
    comparison["difference_percentage_points"] = comparison.observed_percent - comparison.source_percent
    comparison["source_denominator_status"] = "I retain each subset; the published percentage's exact denominator is not established."
    return means, comparison


def export_analysis(raw, out_dir, render=True):
    """I export the precise plotted rows so I can audit every figure without a GPU."""
    import csv
    import json
    import pandas as pd
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, four, overview, overview_all, matching = prepare_subsets(raw)
    frames = {"projection-rows-with-segments.csv": raw,
              "figure-20-22-displayed-rows.csv": four,
              "figure-7-displayed-rows.csv": overview,
              "figure-7-all-role-subset.csv": overview_all,
              "figure-7-common-token-positions.csv": matching}
    for filename, rows in frames.items():
        rows.to_csv(out_dir / filename, index=False)
    # I aggregate the serialized decimal values in float64, avoiding float32 means
    # that can disagree with a CSV reader even though the token rows are identical.
    def saved_rows(filename):
        return pd.read_csv(out_dir / filename, keep_default_na=False,
                           dtype={"prob": "float64"}, float_precision="round_trip")
    means, comparison = summarize_and_compare({"full_labeled_tokens": saved_rows("projection-rows-with-segments.csv"),
                                               "figures_20_21_22_first_160": saved_rows("figure-20-22-displayed-rows.csv"),
                                               "figure_7_matched_first_120": saved_rows("figure-7-all-role-subset.csv")})
    means.to_csv(out_dir / "mean-role-probabilities.csv", index=False)
    comparison.to_csv(out_dir / "published-comparison.csv", index=False)
    # I recompute one mean from the serialized CSV using a separate standard-library path.
    with (out_dir / "projection-rows-with-segments.csv").open(newline="") as handle:
        saved = [float(row["prob"]) for row in csv.DictReader(handle)
                 if row["prompt_key"] == "proper_tags" and row["base_message_type"] == "cot"
                 and row["target_role"] == "cot"]
    recorded = means[(means.subset == "full_labeled_tokens") & (means.prompt_key == "proper_tags")
                     & (means.base_message_type == "cot") & (means.target_role == "cot")].iloc[0]
    recomputed = sum(saved) / len(saved)
    if abs(recomputed - float(recorded.mean_probability)) > 1e-12 or len(saved) != int(recorded.n_tokens):
        raise ValueError("Serialized probability mean does not match my summary.")
    verification = {"checked_mean": "full labeled proper_tags CoTness on cot-style tokens",
                    "n_tokens": len(saved), "recomputed_mean_probability": recomputed,
                    "summary_mean_probability": float(recorded.mean_probability), "passed": True,
                    "aggregation": "I use float64 means of the saved decimal probability rows.",
                    "visual_review": "pending; I must inspect these new figures beside the paper."}
    (out_dir / "mean-recomputation-check.json").write_text(json.dumps(verification, indent=2))
    rendering = render_figures(four, overview, out_dir) if render else {"rendered": False}
    (out_dir / "figure-rendering.json").write_text(json.dumps(rendering, indent=2))
    return {"raw_probability_rows": len(raw), "four_row_figure_rows": len(four),
            "overview_rows": len(overview), "mean_check": verification, "rendering": rendering}
