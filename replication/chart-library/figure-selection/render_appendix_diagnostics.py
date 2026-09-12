"""I refine two measurement diagnostics while preserving every saved estimate."""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LIBRARY = HERE.parent
DATA = HERE / "data"
OUT = HERE / "figures"
RUN = LIBRARY.parent / "cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z"
for d in (DATA, OUT):
    d.mkdir(parents=True, exist_ok=True)
for v in ("regular", "bold", "italic", "bolditalic"):
    font_manager.fontManager.addfont(LIBRARY / f"fonts/termes-ttf/texgyretermes-{v}.ttf")
plt.rcParams.update({"font.family": "TeX Gyre Termes", "font.size": 10,
    "axes.linewidth": .5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "path", "text.parse_math": False,
    "savefig.facecolor": "white", "axes.edgecolor": "#444444"})
GRAY = "#62748e"
INK = "#202020"
COLORS = {"system": "#90a1b9", "user": "#00a6f4", "cot": "#fd9a00",
          "assistant": "#00d492", "tool": "#7e6cff"}
LABELS = {"system": "System", "user": "User", "cot": "CoT",
          "assistant": "Assistant", "tool": "Tool"}
MARKERS = {"system": "D", "user": "o", "cot": "^", "assistant": "s", "tool": "v"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def panel(ax):
    ax.grid(color="#dedede", linewidth=.4)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.5, labelsize=9.5)


def save(fig, name, metadata, source_files):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():
            continue
        b = t.get_window_extent(renderer)
        if b.width and b.height and (b.x0 < -1 or b.y0 < -1 or
            b.x1 > fig.bbox.width + 1 or b.y1 > fig.bbox.height + 1):
            clipped.append(t.get_text())
    assert not clipped, (name, clipped)
    files = {}
    for ext in ("png", "pdf", "svg"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, dpi=400)
        files[p.name] = digest(p)
    metadata.update({"figure_id": name, "source_sha256": {str(p): digest(p) for p in source_files},
        "renderer_sha256": digest(Path(__file__)), "artifact_sha256": files,
        "font": "TeX Gyre Termes", "size_inches": list(fig.get_size_inches()),
        "visible_text_within_canvas": True, "new_model_runs": 0, "new_estimators": 0})
    (DATA / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (DATA / f"{name}.md").write_text("# " + metadata["title"] + "\n\n*" +
        metadata["caption"] + "*\n\nSuggested placement: " + metadata["placement"] +
        "\n\nScope: " + metadata["scope"] + "\n\nMethods: " + metadata["methods"] + "\n")
    plt.close(fig)
    print(name, metadata["verification"])


# I preserve the existing RH6 estimand, all 24 points, and all 48 interval endpoints.
rh6_path = LIBRARY / "data/rh6-plotted.csv"
rh6_provenance_path = LIBRARY / "data/rh6-provenance.json"
rh6_raw_path = RUN / "rh6/user_turn_effects.csv"
r = read(rh6_path)
assert len(r) == 24
assert set(r.n) == {100}
assert not r.duplicated(["space", "metric", "order", "layer"]).any()
raw = read(rh6_raw_path)
keys = ["construction", "span", "command", "order", "layer", "space", "contrast", "metric"]
matched = r.merge(raw, on=keys, suffixes=("_plot", "_raw"), validate="one_to_one")
assert len(matched) == 24
source_max_error = 0.
for col in ("mean", "lo", "hi"):
    err = np.abs(matched[col + "_plot"] - matched[col + "_raw"])
    source_max_error = max(source_max_error, float(err.max()))
    assert err.max() < 1e-15
    assert np.allclose(r["plot_" + col], r[col] * r.plot_scale, rtol=0, atol=1e-14)
rp = json.loads(rh6_provenance_path.read_text())
assert digest(rh6_path) == rp["plotted_data"]["sha256"]
assert digest(rh6_raw_path) == next(p["sha256"] for p in rp["sources"] if p["path"] == str(rh6_raw_path))
shutil.copy2(rh6_path, DATA / "rh6-readout.csv")

fig, axs = plt.subplots(2, 2, figsize=(8.3, 5.9), sharex=True, sharey="row")
fig.text(.045, .975, "Permission changes Userness in opposite directions at layer 12",
         fontsize=15.3, weight="bold", color=INK, va="top")
fig.text(.046, .921, "Identical inputs · separately fitted probes · 100 paired template–page units · readouts only",
         fontsize=10, color=INK, va="top")
for row, metric in enumerate(("p_user", "log_user_tool")):
    for col, space in enumerate(("sucat", "uat")):
        ax = axs[row, col]
        panel(ax)
        ax.axhline(0, color="#888888", lw=.7, ls=(0, (3, 3)), zorder=2)
        if row == 0:
            ax.axvspan(11.40, 12.60, color=GRAY, alpha=.10, lw=0)
        for order, label in (("B", "Marker first"), ("A", "Marker second")):
            q = r[(r.metric == metric) & (r.space == space) & (r.order == order)].sort_values("layer")
            assert len(q) == 3 and q.layer.tolist() == [8, 12, 16]
            ax.errorbar(q.layer, q.plot_mean,
                yerr=np.vstack([q.plot_mean - q.plot_lo, q.plot_hi - q.plot_mean]),
                color=GRAY, ls="-" if order == "B" else "--", lw=1.05,
                marker="o", ms=4.3, mew=.9, mfc=GRAY if order == "B" else "white",
                elinewidth=.85, capsize=2.4, capthick=.85, label=label, zorder=4)
        ax.set(xlim=(7.4, 16.6), xticks=[8, 12, 16])
        if row == 0:
            ax.set_ylim(-4, 25)
            ax.set_yticks([0, 10, 20])
            ax.set_title(("A · Five-role probe" if space == "sucat" else "B · Three-role probe") +
                ("\nSystem · User · CoT · Assistant · Tool" if space == "sucat" else "\nUser · Assistant · Tool"),
                fontsize=10.5, weight="bold", linespacing=1.6, pad=9)
            ax.annotate("Lower Userness" if space == "sucat" else "Higher Userness",
                xy=(12, -1.13 if space == "sucat" else 6.1),
                xytext=(12, 8.5 if space == "sucat" else 15.5),
                ha="center", va="center", fontsize=9.5, weight="bold", color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
                arrowprops={"arrowstyle": "-", "color": GRAY, "lw": .6,
                            "shrinkA": 4, "shrinkB": 8}, zorder=5)
        else:
            ax.set(ylim=(-.08, 1.65), yticks=[0, .5, 1, 1.5], xlabel="Layer index")
            ax.set_yticklabels(["0", "0.5", "1.0", "1.5"])
axs[0, 0].set_ylabel("Adjusted Userness effect\n(percentage points)", fontsize=10.5, labelpad=9)
axs[1, 0].set_ylabel("Adjusted ln(User/Tool) effect", fontsize=10.5, labelpad=9)
handles = [Line2D([], [], color=GRAY, marker="o", ms=4.3, mfc=GRAY if order == "B" else "white",
                  ls="-" if order == "B" else "--", lw=1.05, label=label)
           for order, label in (("B", "Marker first"), ("A", "Marker second"))]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.55, .035), ncol=2,
           frameon=False, handlelength=2.5, columnspacing=2, fontsize=10.5)
fig.subplots_adjust(left=.125, right=.970, top=.780, bottom=.160, wspace=.18, hspace=.24)

save(fig, "rh6-readout", {
    "title": "Permission changes Userness in opposite directions at layer 12",
    "caption": "At layer 12, the same permission contrast lowers Userness under the five-role probe and raises it under the three-role probe; bars show pointwise 95% paired-bootstrap intervals.",
    "alt": "Four panels compare two probes over layers 8, 12 and 16, retaining marker-first and marker-second command orders. The top panels highlight layer 12: both five-role Userness effects and confidence bounds are negative, while both three-role effects and bounds are positive. Every relative log User/Tool effect in the bottom panels is positive.",
    "placement": "Measurement appendix, after interpreting role-score changes and before five-role validation.",
    "scope": "Controlled prefill/readout experiment with 100 paired template–page units per estimate; no generated behavior. Two fitted role spaces, two metrics, two command positions, three layers: all 24 estimates and 48 interval endpoints are retained.",
    "methods": rp["estimand"] + " " + rp["metric_definitions"]["log_user_tool"] +
        " Uncertainty is the saved pointwise 95% paired-bootstrap percentile interval: 2,000 resamples, seed 123; no recomputation and no simultaneous coverage claim. Marker-first versus marker-second is encoded by filled/solid versus open/dashed neutral gray lines. Lines connect only the three measured layers; the highlighted band marks layer 12.",
    "metric_definitions": rp["metric_definitions"],
    "uncertainty": rp["uncertainty"],
    "palette": {"command_position": GRAY},
    "verification": {"estimates": 24, "interval_bounds": 48, "source_max_error": source_max_error,
        "display_csv_byte_identical": digest(DATA / "rh6-readout.csv") == digest(rh6_path),
        "frozen_source_hashes_match": True},
}, [rh6_path, rh6_provenance_path, rh6_raw_path, RUN / "rh6/input-contract.json"])


# I independently recover all 240 validation values from the source confusion counts.
validation_path = LIBRARY / "data/probe-recall-by-layer.csv"
v = read(validation_path)
assert len(v) == 240 and not v.duplicated(["split", "role", "layer_ix"]).any()
validation_sources = [validation_path, RUN / "probes-full/split-audit.json"]
validation_max_error = 0.
split_counts = {}
audit = json.loads((RUN / "probes-full/split-audit.json").read_text())
for split, filename in (("prompt", "acc_by_role_gptoss-20b.csv"),
                        ("base", "acc_by_role_gptoss-20b-basesplit.csv")):
    path = RUN / "probes-full" / filename
    validation_sources.append(path)
    cm = read(path)
    cm = cm[cm.role_space == "system,user,cot,assistant,tool"]
    total = cm.groupby(["layer_ix", "role"])["count"].sum().rename("count_raw")
    correct = cm[cm.role == cm.pred].groupby(["layer_ix", "role"])["count"].sum().rename("correct_raw")
    calc = pd.concat([total, correct], axis=1).fillna(0).reset_index()
    q = v[v.split == split].merge(calc, on=["layer_ix", "role"], validate="one_to_one")
    assert len(q) == 120
    assert (q["count"] == q.count_raw).all() and (q.correct == q.correct_raw).all()
    error = np.abs(q.recall - q.correct_raw / q.count_raw)
    validation_max_error = max(validation_max_error, float(error.max()))
    assert error.max() < 1e-15
    a = audit["splits"][split + ":sucat"]
    split_counts[split] = {k: a[k] for k in ("group_column", "n_original_groups", "n_train_groups", "n_test_groups", "n_test_rows")}
assert split_counts["prompt"]["n_test_groups"] == 124
assert split_counts["base"]["n_test_groups"] == 24
for _, group in v[v.layer_ix >= 15].groupby(["split", "layer_ix"]):
    assert set(group.nlargest(2, "recall").role) == {"user", "assistant"}
shutil.copy2(validation_path, DATA / "probe-validation.csv")

fig, axs = plt.subplots(1, 2, figsize=(8.3, 4.25), sharey=True)
fig.text(.05, .965, "User and Assistant have the highest recall in later layers",
         fontsize=15.3, weight="bold", color=INK, va="top")
fig.text(.051, .891, "GPT-OSS-20B · five-role probes · neutral role-labeled text · all 24 layers",
         fontsize=10.5, color=INK, va="top")
for ax, split, title in zip(axs, ("prompt", "base"),
    ("A · 124 held-out prompt variants", "B · 24 held-out base texts")):
    panel(ax)
    for role in COLORS:
        q = v[(v.split == split) & (v.role == role)].sort_values("layer_ix")
        assert q.layer_ix.tolist() == list(range(24))
        ax.plot(q.layer_ix, q.recall * 100, color=COLORS[role], lw=1,
                marker=MARKERS[role], ms=3.4, mew=.25, label=LABELS[role], zorder=3)
    ax.set(xlim=(-.35, 23.35), ylim=(-2, 102), xticks=[0, 8, 12, 16, 23],
           yticks=[0, 50, 100], xlabel="Layer index")
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_title(title, fontsize=11.2, weight="bold", pad=11)
axs[0].set_ylabel("Recall within each true role", fontsize=11.2, labelpad=10)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.55, .035),
           ncol=5, frameon=False, handlelength=1.5, columnspacing=1.45, fontsize=10.5)
fig.subplots_adjust(left=.10, right=.979, top=.755, bottom=.255, wspace=.20)
save(fig, "probe-validation", {
    "title": "User and Assistant have the highest recall in later layers",
    "caption": "The two saved evaluation splits show higher late-layer recall for User and Assistant than for CoT and Tool, while testing different held-out examples.",
    "alt": "Two line charts show all five role recalls at every layer. User blue and Assistant green rise above the System gray, CoT orange and Tool purple curves in later layers, in both prompt-variant and base-text evaluations. Colors and distinct point markers identify roles.",
    "placement": "Last measurement-appendix figure, linked from the initial explanation of role probes.",
    "scope": "Held-out validation of locally fitted five-role probes under a neutral-text construction, across all 24 layers. The two panels use separately fitted probes and different test sets. Their difference is descriptive and is not a causal estimate of changing the split strategy. This does not validate authority tracking or the separately fitted three- and four-role probes.",
    "methods": "Recall is the number of correctly predicted tokens divided by all tokens with that true role, separately for each layer and split. Every one of the 240 saved recalls is shown; no smoothing or population interval is added. The prompt split holds out 124 of 1,245 prompt variants; the base split holds out 24 of 249 base texts. The split audit uses seed 123 and confirms each split partitions groups and rows exactly once.",
    "split_counts": split_counts,
    "palette": COLORS, "markers": MARKERS,
    "verification": {"recall_values": 240, "count_and_correct_exact_match": True,
        "max_recall_error_from_raw_counts": validation_max_error,
        "user_and_assistant_top_two_at_all_layers_15_through_23_in_both_splits": True,
        "display_csv_byte_identical": digest(DATA / "probe-validation.csv") == digest(validation_path)},
}, validation_sources)
