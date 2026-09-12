"""I plot the audited RH6 permission contrasts from my completed saved run."""

from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SESSION = HERE.parent / "cloud/persistent/sessions/20260911T030050Z"
SOURCE = SESSION / "outputs/20260911T030050Z/rh6"
AUDIT = SESSION / "rh6-integrity-audit-20260911T045805Z/audit.json"
FONT = Path("/usr/local/texlive/2019/texmf-dist/fonts/opentype/public/tex-gyre")
GRAY = "#62748e"
METRICS = ("p_user", "log_user_tool")
SPACES = ("sucat", "uat")
LAYERS = (8, 12, 16)
ORDERS = {"B": "Marker first", "A": "Marker second"}
CAPTION = (
    "Permission raises relative User/Tool in every tested setting; Userness falls at layer 12 with SUCAT. "
    "Bars are pointwise 95% paired-bootstrap intervals; these readouts do not measure behavior."
)


def fingerprint(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_contrasts():
    """I require the completed source files to match the frozen integrity audit."""
    audit = json.loads(AUDIT.read_text())
    assert audit["passed"] and len(audit["checks"]) == 44
    assert all(value is True for value in audit["checks"].values())
    sources = [fingerprint(AUDIT)]
    for name in ("user_turn_effects.csv", "metadata.json", "input-contract.json"):
        current = fingerprint(SOURCE / name)
        assert current["sha256"] == audit["source_files"][name]["sha256"], name
        sources.append(current)
    metadata = json.loads((SOURCE / "metadata.json").read_text())
    assert metadata["status"] == "complete" and metadata["analysis_complete"]
    assert metadata["n_items"] == metadata["n_completed"] == 1200
    source = pd.read_csv(SOURCE / "user_turn_effects.csv", float_precision="round_trip")
    assert len(source) == 432
    selected = source[
        source.construction.eq("command-minus-null")
        & source.command.eq("marker")
        & source.contrast.eq("permission-neutral")
        & source.metric.isin(METRICS)
    ].copy()
    keys = ["metric", "space", "order", "layer"]
    assert len(selected) == 24 and not selected.duplicated(keys).any()
    assert set(selected[keys].itertuples(index=False, name=None)) == set(
        product(METRICS, SPACES, ORDERS, LAYERS)
    )
    assert selected.n.eq(100).all()
    assert selected.span.eq(selected.order.map({"A": "slot2", "B": "slot1"})).all()
    assert np.isfinite(selected[["mean", "lo", "hi"]].to_numpy()).all()
    assert selected.lo.le(selected["mean"]).all() and selected["mean"].le(selected.hi).all()
    selected["order_label"] = selected.order.map(ORDERS)
    selected["plot_units"] = np.where(selected.metric.eq("p_user"), "percentage points", "natural-log units")
    selected["plot_scale"] = np.where(selected.metric.eq("p_user"), 100.0, 1.0)
    for column in ("mean", "lo", "hi"):
        selected[f"plot_{column}"] = selected[column] * selected.plot_scale
    selected = selected.sort_values(keys).reset_index(drop=True)
    return selected, audit, metadata, sources


def panel(ax):
    ax.grid(color="#d9d9d9", linewidth=.3)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.tick_params(width=.4, length=2, pad=3)
    ax.axhline(0, color="#999999", linewidth=.5, linestyle="--", zorder=1)


def main():
    data, audit, metadata, sources = load_contrasts()
    for variant in ("regular", "bold", "italic", "bolditalic"):
        font_manager.fontManager.addfont(FONT / f"texgyretermes-{variant}.otf")
    plt.rcParams.update({
        "font.family": "TeX Gyre Termes", "font.size": 9, "text.usetex": False,
        "text.parse_math": False, "pdf.fonttype": 42, "axes.linewidth": .5,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True, sharey="row")
    for row, metric in enumerate(METRICS):
        for column, space in enumerate(SPACES):
            ax = axes[row, column]
            panel(ax)
            for order, label in ORDERS.items():
                q = data[data.metric.eq(metric) & data.space.eq(space) & data.order.eq(order)].sort_values("layer")
                yerr = np.vstack((q.plot_mean - q.plot_lo, q.plot_hi - q.plot_mean))
                ax.errorbar(
                    q.layer, q.plot_mean, yerr=yerr, label=label, color=GRAY,
                    linewidth=.85, linestyle="-" if order == "B" else "--",
                    marker="o", markersize=3, markeredgewidth=.65,
                    markerfacecolor=GRAY if order == "B" else "white",
                    elinewidth=.65, capsize=2, capthick=.65, zorder=3,
                )
            ax.set_xlim(7.4, 16.6)
            ax.set_xticks(LAYERS)
            if row == 0:
                ax.set_ylim(-5, 25)
                ax.set_yticks([-5, 0, 10, 20])
                ax.set_title(
                    "SUCAT · five-role probe" if space == "sucat" else "UAT · three-role probe",
                    fontweight="bold",
                )
            else:
                ax.set_ylim(-.08, 1.65)
                ax.set_yticks([0, .5, 1, 1.5], labels=["0", "0.5", "1.0", "1.5"])
                ax.set_xlabel("Layer index")
    axes[0, 0].set_ylabel("Change in Userness\n(percentage points)", fontweight="bold")
    axes[1, 0].set_ylabel("Change in log(User/Tool)", fontweight="bold")
    fig.text(.11, .975, "Relative role shifts agree; absolute Userness depends on the probe", fontweight="bold", fontsize=11)
    fig.text(.11, .935, "Marker span · (permission − neutral) − paired null-span effect · n = 100 template–page pairs", fontsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.54, .066), ncol=2, frameon=False, handlelength=2.5)
    fig.text(.11, .034, "Permission raises relative User/Tool in every tested setting; Userness falls at layer 12 with SUCAT.", fontsize=7.2)
    fig.text(.11, .009, "Bars are pointwise 95% paired-bootstrap intervals; these readouts do not measure behavior.", fontsize=7.2)
    fig.subplots_adjust(left=.11, right=.96, top=.855, bottom=.20, hspace=.24, wspace=.18)
    for subdirectory in ("figures", "data"):
        (HERE / subdirectory).mkdir(exist_ok=True)
    data_path = HERE / "data/rh6-plotted.csv"
    data.to_csv(data_path, index=False)
    reread = pd.read_csv(data_path, float_precision="round_trip")
    assert np.array_equal(reread[["plot_mean", "plot_lo", "plot_hi"]].values, data[["plot_mean", "plot_lo", "plot_hi"]].values)
    for extension in ("png", "pdf"):
        fig.savefig(HERE / f"figures/04-rh6-readout.{extension}", dpi=300)
    plt.close(fig)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "I distill my completed RH6 readouts without a new model run or a new estimator.",
        "sources": sources,
        "plotted_data": fingerprint(data_path),
        "renderer": fingerprint(Path(__file__).resolve()),
        "selection": {"construction": "command-minus-null", "command": "marker", "contrast": "permission-neutral", "metrics": list(METRICS), "spaces": list(SPACES), "layers": list(LAYERS)},
        "n_plotted_cells": 24, "n_paired_template_page_units_per_cell": 100,
        "estimand": "I average, equally over paired template/page units, (marker permission minus marker neutral) minus (matched null-span permission minus matched null-span neutral).",
        "metric_definitions": {"p_user": "Mean span P_user; I scale the saved contrast and its bounds by 100 to percentage points.", "log_user_tool": "ln(mean span P_user) minus ln(mean span P_tool), before computing paired contrasts; this is not the mean of tokenwise log ratios."},
        "uncertainty": {"type": "Existing pointwise 95% paired-bootstrap percentile intervals", "resamples": 2000, "seed": 123, "recomputed_for_figure": False, "simultaneous_intervals": False},
        "order_mapping": {"A": "Exfil first, marker second; marker is slot2", "B": "Marker first, exfil second; marker is slot1"},
        "style": {"font": "TeX Gyre Termes", "font_directory": str(FONT), "color": GRAY, "color_meaning": "Neutral readouts of tool-origin spans; I do not assign source-role passage colors.", "order_B": "solid line, filled circle", "order_A": "dashed line, open circle", "shared_y_axis_within_each_metric": True},
        "validation": {"source_hashes_match_frozen_audit": True, "audit_passed": audit["passed"], "audit_checks_passed": len(audit["checks"]), "source_recomputation": audit["summary_recomputation"]["user_turn_effects"], "complete_24_cell_grid": True, "plotted_csv_float_roundtrip_exact": True, "completed_items": metadata["n_completed"]},
        "interpretation_boundary": "I report role-probe readouts on the prepared synthetic pages and fixed prefills. These contrasts do not establish behavioral compliance or a causal role mechanism; intervals are pointwise over the 100 paired template/page units.",
        "new_model_runs": 0, "new_gpu_jobs": 0, "caption": CAPTION,
    }
    (HERE / "data/rh6-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"plotted_cells": len(data), "source_matches_44_check_audit": True, "new_model_runs": 0}))


if __name__ == "__main__":
    main()
