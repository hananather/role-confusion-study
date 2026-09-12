"""I summarize paired, fixed-text E9 readouts without loading or running a model.

I retain Figure 23's uat argmax accuracy as one measurement and plot mean
probabilities separately. I resample whole conversations with the same draws
for every cell. My plots describe the supplied data; rendering does not establish
that its vectors were propagated through the model or that behavior changed.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


LAYERS = tuple(range(24))
ROLES = ("user", "assistant")
CONDITIONS = ("tagged", "untagged", "tool_tagged")
FAMILIES = ("historical", "probe_user", "probe_tool")
ARMS = ("positive", "negative", "random_0", "random_1", "random_2")
METRICS = ("correct_fraction", "p_user", "p_assistant", "p_tool")
KEY_FIELDS = ("condition", "family", "arm", "layer_ix", "original_role")
REQUIRED = ("conv_id", *KEY_FIELDS, "n_tokens", *METRICS)
PROBABILITY_TOLERANCE = 1e-6
DEFAULT_FONT = Path("/usr/local/texlive/2019/texmf-dist/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf")
COLORS = {"tagged": "#62748e", "untagged": "#00a6f4", "tool_tagged": "#ff6467"}
CONDITION_LABELS = {"tagged": "Baseline", "untagged": "No tags", "tool_tagged": "Injection (tool tagged)"}
FAMILY_LABELS = {"historical": "Historical Tool−User vector", "probe_user": "Saved User probe vector", "probe_tool": "Saved Tool probe vector"}
LINESTYLES = {"zero": "-", "positive": "--", "negative": ":", "random_0": "-.",
              "random_1": (0, (5, 1, 1, 1)), "random_2": (0, (1, 1, 3, 1))}
ARM_LABELS = {"zero": "Zero offset", "positive": "+ vector", "negative": "− vector",
              "random_0": "Random 0", "random_1": "Random 1", "random_2": "Random 2"}


@dataclass(frozen=True)
class Dataset:
    conversations: tuple[str, ...]
    cells: tuple[tuple, ...]
    values: np.ndarray  # conversation × cell × metric
    token_counts: np.ndarray  # conversation × cell; never bootstrap weights
    families: tuple[str, ...]
    optional_zero_families: tuple[str, ...]


def fail(message):
    raise ValueError(f"I cannot analyze this input: {message}")


def integer(value, field, row_number):
    try:
        number = int(str(value))
    except (ValueError, TypeError):
        fail(f"row {row_number} has a non-integer {field}")
    return number


def validate_rows(rows, families=FAMILIES):
    """I require every expected cell for the same conversation cohort.

    My source cells are none/zero at all three conditions. Each selected vector
    family has all five nonzero/control arms at tool_tagged. A family-specific
    zero is optional, but if present I require full coverage and baseline equality.
    """
    families = tuple(families)
    if len(set(families)) != len(families) or any(f not in FAMILIES for f in families):
        fail("the selected vector families are unknown or duplicated")
    parsed = {}
    conversations = set()
    optional_zero = set()
    for row_number, row in enumerate(rows, start=2):
        missing = [field for field in REQUIRED if field not in row or row[field] is None]
        if missing:
            fail(f"row {row_number} lacks required fields: {', '.join(missing)}")
        conv = str(row["conv_id"]).strip()
        if not conv:
            fail(f"row {row_number} has an empty conv_id")
        condition, family, arm = (str(row[field]) for field in KEY_FIELDS[:3])
        layer = integer(row["layer_ix"], "layer_ix", row_number)
        role = str(row["original_role"])
        if condition not in CONDITIONS or role not in ROLES or layer not in LAYERS:
            fail(f"row {row_number} has an unknown condition, original role, or layer")
        if family == "none":
            if arm != "zero":
                fail(f"row {row_number}: family none must use arm zero")
        elif family in families:
            if condition != "tool_tagged" or arm not in (*ARMS, "zero"):
                fail(f"row {row_number}: vector arms must use tool_tagged and a known arm")
            if arm == "zero":
                optional_zero.add(family)
        else:
            fail(f"row {row_number} uses an unselected or unknown family: {family}")
        n_tokens = integer(row["n_tokens"], "n_tokens", row_number)
        if n_tokens <= 0:
            fail(f"row {row_number} has no scored content tokens")
        try:
            values = np.array([float(row[field]) for field in METRICS], dtype=np.float64)
        except (ValueError, TypeError):
            fail(f"row {row_number} has a nonnumeric metric")
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            fail(f"row {row_number} has a nonfinite or out-of-range metric")
        if abs(float(values[1:].sum()) - 1.0) > PROBABILITY_TOLERANCE:
            fail(f"row {row_number}: p_user + p_assistant + p_tool must equal one")
        cell = (condition, family, arm, layer, role)
        key = (conv, cell)
        if key in parsed:
            fail(f"row {row_number} duplicates a conversation/cell key")
        parsed[key] = (values, n_tokens)
        conversations.add(conv)
    if not parsed:
        fail("there are no observations")

    cells = [(condition, "none", "zero", layer, role)
             for condition in CONDITIONS for layer in LAYERS for role in ROLES]
    for family in families:
        arms = (("zero",) if family in optional_zero else ()) + ARMS
        cells.extend(("tool_tagged", family, arm, layer, role)
                     for arm in arms for layer in LAYERS for role in ROLES)
    cells = tuple(cells)
    cohort = tuple(sorted(conversations))
    expected_count = len(cohort) * len(cells)
    if len(parsed) != expected_count:
        for conv in cohort:
            for cell in cells:
                if (conv, cell) not in parsed:
                    fail(f"missing cell for conversation {conv}: {cell}; every arm/layer/role needs the same cohort")
        fail("the observation count differs from the expected design")
    values = np.empty((len(cohort), len(cells), len(METRICS)), dtype=np.float64)
    token_counts = np.empty((len(cohort), len(cells)), dtype=np.int64)
    for i, conv in enumerate(cohort):
        for j, cell in enumerate(cells):
            if (conv, cell) not in parsed:
                fail(f"missing cell for conversation {conv}: {cell}")
            values[i, j], token_counts[i, j] = parsed[(conv, cell)]
    lookup = {cell: i for i, cell in enumerate(cells)}
    for j, (condition, family, arm, layer, role) in enumerate(cells):
        ref = lookup[(condition, "none", "zero", 0, role)]
        if not np.array_equal(token_counts[:, j], token_counts[:, ref]):
            fail(f"the fixed scored token count changes across layers/arms: {cells[j]}")
        if family != "none" and arm == "zero":
            ref = lookup[("tool_tagged", "none", "zero", layer, role)]
            if not np.allclose(values[:, j], values[:, ref], atol=PROBABILITY_TOLERANCE, rtol=0):
                fail(f"family-specific zero differs from the unsteered baseline: {cells[j]}")
    return Dataset(cohort, cells, values, token_counts, families,
                   tuple(f for f in families if f in optional_zero))


def load_csv(path, families=FAMILIES):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if len(set(header)) != len(header):
            fail("the CSV has duplicated column names")
        if not set(REQUIRED).issubset(header):
            fail(f"the CSV header must include {', '.join(REQUIRED)}")
        return validate_rows(reader, families)


def paired_statistics(data, n_boot=2000, seed=123):
    """I use equal conversation weights and one shared bootstrap draw matrix."""
    if not isinstance(n_boot, int) or n_boot <= 0:
        fail("n_boot must be a positive integer")
    n = len(data.conversations)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    weights = np.stack([np.bincount(draw, minlength=n) for draw in indices]).astype(np.float64) / n
    draws = (weights @ data.values.reshape(n, -1)).reshape(n_boot, len(data.cells), len(METRICS))
    mean = data.values.mean(axis=0)
    bounds = np.percentile(draws, [2.5, 97.5], axis=0)
    lookup = {cell: i for i, cell in enumerate(data.cells)}
    refs = np.array([lookup[("tool_tagged", "none", "zero", cell[3], cell[4])]
                     for cell in data.cells], dtype=np.int64)
    delta_mean = (data.values - data.values[:, refs]).mean(axis=0)
    delta_bounds = np.percentile(draws - draws[:, refs], [2.5, 97.5], axis=0)
    aggregate, deltas = [], []
    for j, cell in enumerate(data.cells):
        common = dict(zip(KEY_FIELDS, cell))
        common.update(n_conversations=n, n_tokens_total=int(data.token_counts[:, j].sum()))
        for k, metric in enumerate(METRICS):
            aggregate.append({**common, "metric": metric, "mean": float(mean[j, k]),
                              "ci_lo": float(bounds[0, j, k]), "ci_hi": float(bounds[1, j, k])})
            deltas.append({**common, "metric": metric,
                           "reference_condition": "tool_tagged", "reference_family": "none", "reference_arm": "zero",
                           "mean_delta": float(delta_mean[j, k]), "ci_lo": float(delta_bounds[0, j, k]),
                           "ci_hi": float(delta_bounds[1, j, k])})
    return aggregate, deltas, indices


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plotting(font_path=DEFAULT_FONT):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    path = Path(font_path)
    if not path.is_file():
        fail(f"TeX Gyre Termes font is absent at {path}; supply --font with the intended font file")
    for sibling in sorted(path.parent.glob("texgyretermes-*.otf")):
        font_manager.fontManager.addfont(str(sibling))
    font_manager.fontManager.addfont(str(path))
    family = font_manager.FontProperties(fname=str(path)).get_name()
    matplotlib.rcParams.update({"font.family": "serif", "font.serif": [family], "font.size": 9,
                               "axes.linewidth": 0.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
                               "text.usetex": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    return family


def render_figures(aggregate, out, families=FAMILIES, font_path=DEFAULT_FONT,
                   title="gpt-oss-20b", n_conversations=None):
    """I reserve color for input condition and line style for intervention arm."""
    font_family = configure_plotting(font_path)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, PercentFormatter
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lookup = {(row["condition"], row["family"], row["arm"], row["layer_ix"], row["original_role"], row["metric"]): row["mean"]
              for row in aggregate}
    if n_conversations is None:
        n_conversations = aggregate[0]["n_conversations"]
    written = []

    def panel_plot(name, metric_kind, panel_families, random_controls=False, baseline_only=False):
        figure, axes = plt.subplots(2, len(panel_families), figsize=(3.55 * len(panel_families), 4.25),
                                   sharex=True, sharey=True, squeeze=False)
        for column, family in enumerate(panel_families):
            for row_index, role in enumerate(ROLES):
                axis = axes[row_index, column]
                metric = "correct_fraction" if metric_kind == "accuracy" else ("p_tool" if metric_kind == "tool" else f"p_{role}")
                conditions = ("tool_tagged",) if random_controls else CONDITIONS
                for condition in conditions:
                    y = [lookup[(condition, "none", "zero", layer, role, metric)] for layer in LAYERS]
                    label = ARM_LABELS["zero"] if random_controls else CONDITION_LABELS[condition]
                    axis.plot(LAYERS, y, color=COLORS[condition], linestyle="-", marker="o", markersize=2.8,
                              linewidth=0.65, label=label)
                if not baseline_only:
                    for arm in (ARMS if random_controls else ("positive", "negative")):
                        y = [lookup[("tool_tagged", family, arm, layer, role, metric)] for layer in LAYERS]
                        axis.plot(LAYERS, y, color=COLORS["tool_tagged"], linestyle=LINESTYLES[arm],
                                  linewidth=0.95, label=ARM_LABELS[arm])
                axis.set_ylim(-0.02, 1.02)
                axis.set_xlim(-0.5, 23.5)
                axis.yaxis.set_major_locator(FixedLocator([0, 0.5, 1]))
                axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
                axis.set_xticks([0, 5, 10, 15, 20, 23])
                axis.grid(True, color="#e5e7eb", linewidth=0.4)
                for side in ("top", "right"):
                    axis.spines[side].set_visible(False)
                axis.tick_params(length=2, labelsize=8)
                if column == 0:
                    if metric_kind == "accuracy":
                        ylabel = f"{role.title()} classification accuracy"
                    elif metric_kind == "tool":
                        ylabel = "Mean Tool probability"
                    else:
                        ylabel = f"Mean {role.title()} probability"
                    axis.set_ylabel(ylabel + f"\n(on {role}-style content)", fontsize=8)
                if row_index == 0:
                    axis.set_title(title if baseline_only else FAMILY_LABELS[family], fontsize=9)
                else:
                    axis.set_xlabel("Layer index")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        legend_columns = min(len(labels), 3 if len(panel_families) == 1 else 6)
        figure.legend(handles, labels, loc="lower center", ncol=legend_columns, frameon=False, fontsize=7,
                      bbox_to_anchor=(0.5, 0.005), handlelength=2.5, columnspacing=1.2)
        if not baseline_only:
            figure.suptitle(f"{title} · {n_conversations} paired conversations" + (" · random controls" if random_controls else ""), fontsize=10)
        else:
            axes[0, 0].set_title(f"{title} (n = {n_conversations} conversations)", fontsize=9)
        figure.tight_layout(rect=(0, 0.12 if len(panel_families) == 1 else 0.08, 1, 0.95 if not baseline_only else 1))
        for extension in ("png", "pdf"):
            destination = out / f"{name}.{extension}"
            figure.savefig(destination, dpi=300)
            written.append(destination)
        plt.close(figure)

    panel_plot("figure23-baseline-accuracy", "accuracy", ("none",), baseline_only=True)
    if families:
        for metric_kind, name in (("accuracy", "accuracy"), ("role", "role-probability"), ("tool", "tool-probability")):
            panel_plot(f"figure23-steering-{name}", metric_kind, tuple(families))
            panel_plot(f"figure23-random-controls-{name}", metric_kind, tuple(families), random_controls=True)
    return written, font_family


def run_analysis(input_path, out, families=FAMILIES, no_plots=False, font_path=DEFAULT_FONT,
                 title="gpt-oss-20b"):
    input_path, out = Path(input_path).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        fail("the output directory is nonempty; I require a new directory to preserve previous analyses")
    data = load_csv(input_path, families)
    aggregate, deltas, indices = paired_statistics(data)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "aggregate-metrics.csv", aggregate)
    write_csv(out / "paired-deltas.csv", deltas)
    np.save(out / "bootstrap-conversation-indices.npy", indices, allow_pickle=False)
    plots, font_family = ([], None) if no_plots else render_figures(aggregate, out / "figures", data.families, font_path, title, len(data.conversations))
    metadata = {
        "input_path": str(input_path), "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "n_conversations": len(data.conversations),
        "conversation_order": list(data.conversations), "n_cells": len(data.cells),
        "families": list(data.families), "optional_zero_families": list(data.optional_zero_families),
        "layers": list(LAYERS), "original_roles": list(ROLES), "metrics": list(METRICS),
        "bootstrap": {"unit": "conversation", "n_boot": 2000, "seed": 123, "interval_percentiles": [2.5, 97.5],
                      "same_draws_for_all_cells": True, "weighting": "equal conversation weights, never token weights",
                      "indices_file": "bootstrap-conversation-indices.npy",
                      "indices_sha256": hashlib.sha256((out / "bootstrap-conversation-indices.npy").read_bytes()).hexdigest()},
        "paired_reference": {"condition": "tool_tagged", "family": "none", "arm": "zero", "match": ["conv_id", "layer_ix", "original_role"]},
        "rendering": {"font_file": str(Path(font_path).resolve()) if not no_plots else None, "font_family": font_family,
                      "condition_colors": COLORS, "main_plot_CI_shading": False,
                      "color_encodes": "input condition", "line_style_encodes": "intervention arm in extension panels only",
                      "figures": [str(path.relative_to(out)) for path in plots]},
        "evidence_boundary": "I summarize the supplied per-conversation readouts. I do not infer propagated steering, behavioral effects, or hypothesis confirmation from a CSV or rendered graph.",
        "uncertainty_boundary": "My intervals are pointwise paired conversation bootstrap intervals, not simultaneous bands or independent token-level uncertainty.",
        "validation_boundary": "I verify design coverage, shared cohorts, token counts, and metric consistency. Matching counts alone cannot establish identical underlying text or the runner's intervention site; those require its separate provenance.",
    }
    (out / "analysis-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Per-conversation long CSV")
    parser.add_argument("--out", required=True, type=Path, help="New, empty analysis directory")
    parser.add_argument("--families", default=",".join(FAMILIES), help="Explicit comma-separated subset of historical,probe_user,probe_tool; none for source-only analysis")
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--title", default="gpt-oss-20b")
    parser.add_argument("--no-plots", action="store_true", help="Write tables and bootstrap provenance only")
    args = parser.parse_args(argv)
    families = () if args.families == "none" else tuple(args.families.split(","))
    try:
        metadata = run_analysis(args.input, args.out, families, args.no_plots, args.font, args.title)
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(f"I analyzed {metadata['n_conversations']} paired conversations across {metadata['n_cells']} cells; tables and provenance are in {args.out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
