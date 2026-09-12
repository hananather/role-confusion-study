"""I render two permission-task extensions from frozen records, without inference."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
LIBRARY = HERE.parent
ROOT = LIBRARY.parent.parent
SOURCE = ROOT / ("earlier-study" if (ROOT / "earlier-study").is_dir() else "role-steering")
AUDIT = LIBRARY / "steering-evidence-review/reviews"
OUT = HERE / "figures/review-wave2"
DATA = HERE / "data/review-wave2"
for directory in (OUT, DATA):
    directory.mkdir(parents=True, exist_ok=True)
for variant in ("regular", "bold", "italic", "bolditalic"):
    font_manager.fontManager.addfont(LIBRARY / f"fonts/termes-ttf/texgyretermes-{variant}.ttf")

# I reuse the original record reducer without calling its plotting entry point.
spec = importlib.util.spec_from_file_location("frozen_permission_figures", SOURCE / "figures.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
records, cases, manifest, provenance = original.read_data()
contrasts, seed_range = original.probe_contrasts(records, cases, manifest)
counts = original.paired_counts(records, manifest)
stored = list(csv.DictReader((SOURCE / "figures/selection-contrasts.csv").open()))
assert len(contrasts) == len(stored) == 16
for recalculated, saved in zip(contrasts, stored):
    for metric in ("command", "label", "nonce", "logodds"):
        assert recalculated[metric] == float(saved[metric])
assert seed_range == 0

audited = {r["method"]: r for r in csv.DictReader((AUDIT / "permission-method-summary.csv").open())}
for row in counts:
    twin = audited[row["method"]]
    assert row["pairs"] == int(twin["pairs"])
    for metric, audit_metric in (("write", "unauthorized_write"),
            ("completion", "authorized_completion"), ("nonces", "both_nonces_reported"),
            ("attempt", "unauthorized_attempt")):
        for suffix, audit_suffix in (("lost", "prevented_or_lost"),
                ("gained", "introduced_or_gained"), ("unknown", "unknown")):
            assert row[metric + "_" + suffix] == int(twin[audit_metric + "_" + audit_suffix])

plt.rcParams.update({"font.family": "TeX Gyre Termes", "font.size": 10,
    "axes.linewidth": .5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "path", "text.parse_math": False,
    "savefig.facecolor": "white", "axes.edgecolor": "#444444",
    "text.color": "#202020", "axes.labelcolor": "#202020",
    "xtick.color": "#62748e", "ytick.color": "#202020"})
INK, GRAY = "#202020", "#62748e"
BLUE, ORANGE, GREEN, PINK = "#00a6f4", "#fd9a00", "#00d492", "#ff637e"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csvwrite(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finish(fig, name, metadata, sources):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for item in fig.findobj(matplotlib.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or
                box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1):
            clipped.append(item.get_text())
    assert not clipped, (name, clipped)
    artifacts = {}
    for extension in ("png", "pdf", "svg"):
        path = OUT / f"{name}.{extension}"
        fig.savefig(path, dpi=400)
        artifacts[str(path)] = digest(path)
    metadata.update({"figure_id": name, "font": "TeX Gyre Termes",
        "palette": {"User_score_or_permitted_completion": BLUE,
            "reported_values": ORANGE, "prevented_write": GREEN, "introduced_write": PINK},
        "source_sha256": {str(p): digest(p) for p in sources},
        "renderer_sha256": digest(Path(__file__)), "artifact_sha256": artifacts,
        "dimensions_inches": list(fig.get_size_inches()), "visible_text_within_canvas": True,
        "new_model_runs": 0, "no_source_mutation": True})
    (DATA / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (DATA / f"{name}.md").write_text("# " + metadata["title"] + "\n\n" +
        metadata["caption"] + "\n\n" + metadata["explanation"] + "\n\nScope: " +
        metadata["scope"] + "\n\nSuggested placement: " + metadata["placement"] + "\n")
    plt.close(fig)
    print(name, metadata["verification"])


def clean_axes(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(axis="y", length=0, pad=9, labelsize=10)
    ax.tick_params(axis="x", width=.5, length=3, labelsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e1e1e1", linewidth=.5)


name = "permission-selection-contrasts"
csvwrite(DATA / f"{name}.csv", contrasts)
shutil.copy2(SOURCE / "figures/provenance.json", DATA / "permission-original-provenance.json")
fig, axes = plt.subplots(1, 2, figsize=(8.3, 5.35), sharey=True)
fig.subplots_adjust(left=.175, right=.969, bottom=.15, top=.728, wspace=.25)
fig.text(.039, .970, "Selecting an action can lower its User score", va="top",
         fontsize=16, weight="bold", color=INK)
fig.text(.040, .914, "Fixed page and command tokens · layer 14 · five-role probe · all eight available cases",
         va="top", fontsize=10, color=INK)
handles = [Line2D([], [], marker="o", ms=6, color=BLUE, ls="", label="First on page"),
           Line2D([], [], marker="s", ms=6, color=BLUE, mfc="white", mew=1.3,
                  ls="", label="Second on page")]
fig.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(.566, .873),
           frameon=False, fontsize=10, handletextpad=.5, columnspacing=2.0)
case_ids = sorted({r["case"] for r in contrasts})
settings = (("command", "A · Whole command", "First > 0; second < 0 in every case",
             (-12, 30), [-10, 0, 10, 20, 30]),
            ("label", "B · Action-label region (post hoc)", "15 of 16 command contrasts are negative",
             (-60, 5), [-60, -40, -20, 0]))
for panel, (ax, (metric, title, finding, limits, ticks)) in enumerate(zip(axes, settings)):
    clean_axes(ax)
    ax.text(0, 1.112, title, transform=ax.transAxes, fontsize=11.4, weight="bold")
    ax.text(0, 1.054, finding, transform=ax.transAxes, fontsize=9.3, color=INK)
    ax.axvline(0, color="#777777", lw=.85, zorder=2)
    for i, case_id in enumerate(case_ids):
        ax.axhline(i, color="#ededed", lw=.45, zorder=0)
        pair = sorted((r for r in contrasts if r["case"] == case_id), key=lambda r: r["position"])
        for r, marker, fill, offset in zip(pair, ("o", "s"), (BLUE, "white"), (-.11, .11)):
            ax.scatter(r[metric] * 100, i + offset, s=32, marker=marker,
                       edgecolors=BLUE, facecolors=fill, linewidths=1.1, zorder=4)
    ax.set(xlim=limits, ylim=(7.55, -.6), xticks=ticks)
    ax.set_xticklabels(["0" if t == 0 else f"{t:+d}" for t in ticks])
    ax.set_yticks(range(8))
    ax.set_yticklabels([f"{cid[-3:]}    " + " → ".join(next(r["order"] for r in contrasts if r["case"] == cid))
                        for cid in case_ids])
axes[0].set_ylabel("Case · action order on page", labelpad=12, fontsize=10.5)
fig.text(.57, .055, "Selected minus unselected User score (percentage points)",
         fontsize=11, ha="center")
finish(fig, name, {
    "title": "Selecting an action can lower its User score",
    "caption": "I keep the tool page fixed and change whether the user permits action A or B. Each point compares the same command when selected versus unselected. Whole-command User scores rise for the first-listed command and fall for the second in all eight cases. The narrower action-label region has 15 of 16 negative contrasts. A/B name actions; markers identify page position. The two panels use different horizontal ranges.",
    "explanation": "A role probe assigns a message-role score, which I am testing here as a possible indicator of permission. Selecting the first-listed action raises both commands' whole-command scores, so the second command scores higher when it is unselected. This is a measurement before the model acts, separate from the steering results.",
    "scope": "Eight recovered cases; all 16 commands shown. Layer-14 five-role probe, unsteered first-prefill scores. The label region was selected post hoc. Repeated seeds have identical scores and are averaged once per case. Wording, labels and order remain bundled; same-page order reversal was not run. Mean tokenwise log(User/Tool) changes one second-command sign. The source study recovered 149 of 280 assigned records across eight of ten cases.",
    "placement": "Appendix, before the permission behavior-control figure; cross-reference from the selective-execution section.",
    "verification": {"raw_record_hash_verified": True, "all_16_contrasts_exactly_recomputed": True,
        "matched_token_ids_and_positions": True, "repeated_seed_score_range": seed_range,
        "all_cases_and_both_regions_retained": True},
    "plotted_columns": ["command", "label"], "retained_unplotted_columns": ["nonce", "logodds"],
    "point_unit": "one command in one fixed-page case; repeated seeds averaged",
}, [SOURCE / "data/permission-20260905/episodes.jsonl.gz", SOURCE / "figures.py",
    SOURCE / "figures/selection-contrasts.csv", SOURCE / "figures/provenance.json", DATA / f"{name}.csv"])


name = "permission-all-controls"
csvwrite(DATA / f"{name}.csv", counts)
shutil.copy2(AUDIT / "permission-all-method-pairs.csv", DATA / "permission-all-method-pairs.csv")
labels = {"tool_03": "Toolward", "user_03": "Userward", "random_0": "Random direction 1",
          "random_1": "Random direction 2", "random_2": "Random direction 3", "prompt": "Permission reminder"}
lookup = {r["method"]: r for r in counts}
fig, axes = plt.subplots(1, 2, figsize=(8.3, 5.15), sharey=True,
                         gridspec_kw={"width_ratios": [1.2, 1]})
fig.subplots_adjust(left=.251, right=.965, bottom=.145, top=.703, wspace=.26)
fig.text(.037, .971, "Fewer unwanted writes can come with lost permitted actions", va="top",
         fontsize=15, weight="bold", color=INK)
fig.text(.038, .912, "Recovered permission task · each method compared with its own matched baseline",
         va="top", fontsize=10, color=INK)
legend_items = [[(GREEN, "o", "Unwanted write prevented"), (PINK, "s", "Unwanted write introduced")],
                [(BLUE, "o", "Permitted completion lost"), (ORANGE, "s", "Report of both values lost")]]
for ax, title, items in zip(axes, ("A · Unwanted actions", "B · Useful work lost"), legend_items):
    clean_axes(ax)
    ax.text(0, 1.105, title, transform=ax.transAxes, fontsize=11.5, weight="bold")
    handles = [Line2D([], [], marker=marker, color=color, ls="", ms=5.7, label=label)
               for color, marker, label in items]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-.03, 1.175), frameon=False,
              fontsize=9.1, handletextpad=.3, borderpad=0, labelspacing=.35)
    ax.set(ylim=(5.6, -.65), yticks=list(range(6)))
    for i in range(6):
        ax.axhline(i, color="#ededed", lw=.5, zorder=0)
axes[0].axvline(0, color="#777777", lw=.8, zorder=2)
for i, method in enumerate(labels):
    row = lookup[method]
    for metric, direction, offset, color, marker in (("write_lost", -1, -.11, GREEN, "o"),
            ("write_gained", 1, .11, PINK, "s")):
        n = row[metric]
        axes[0].plot([0, direction * n], [i + offset] * 2, color=color, lw=1.2, alpha=.75, zorder=2)
        axes[0].scatter(direction * n, i + offset, color=color, s=32, marker=marker, zorder=4)
        axes[0].annotate(str(n), (direction * n, i + offset), xytext=(direction * 7, 0),
                         textcoords="offset points", ha="right" if direction < 0 else "left",
                         va="center", fontsize=9.5, color=INK)
    for metric, offset, color, marker in (("completion_lost", -.11, BLUE, "o"),
            ("nonces_lost", .11, ORANGE, "s")):
        n = row[metric]
        axes[1].plot([0, n], [i + offset] * 2, color=color, lw=1.2, alpha=.7, zorder=2)
        axes[1].scatter(n, i + offset, color=color, s=32, marker=marker, zorder=4)
        axes[1].annotate(str(n), (n, i + offset), xytext=(7, 0), textcoords="offset points",
                         ha="left", va="center", fontsize=9.5, color=INK)
axes[0].set_xlim(-7, 4.1)
axes[0].set_xticks([-6, -4, -2, 0, 2, 4])
axes[0].set_xticklabels(["6", "4", "2", "0", "2", "4"])
axes[0].set_xlabel("Prevented  ←  Paired episodes  →  Introduced", fontsize=9.8, labelpad=10)
axes[1].set_xlim(-.15, 2.75)
axes[1].set_xticks([0, 1, 2])
axes[1].set_xlabel("Paired episodes", fontsize=10.5, labelpad=10)
axes[0].set_yticklabels([f"{labels[m]}   (n = {lookup[m]['pairs']})" for m in labels], fontsize=10)
finish(fig, name, {
    "title": "Fewer unwanted writes can come with lost permitted actions",
    "caption": "Every recovered method is compared with its own unsteered baseline; n is the number of available matched episode pairs. Toolward prevents four unwanted writes and introduces three; the reminder prevents six and introduces none. Both lose two permitted completions. No method gains permitted completions or reports of both requested values. Only 149 of 280 assigned records were recovered, so coverage differs by method. Random directions 1-3 are this permission study's random_0-2, distinct from the upload study's controls.",
    "explanation": "The user permits one of two marker-writing actions and asks for values from both. A defense must avoid the unwanted action while still completing the permitted action and using the page's information. The reminder shows why fewer unwanted writes alone do not capture that objective. This is a separate endpoint from uploading a dummy file.",
    "scope": "149 of 280 assigned records recovered, spanning eight of ten cases. There are 20-22 matched episodes per method across seven or eight cases; shared baselines and repeated contexts prevent treating these rows as independent cohorts. Source random_0, random_1 and random_2 are displayed as Random directions 1, 2 and 3. Toolward is the older neutral-text Tool-minus-User direction at block 11 with a layer-14 five-role readout. Physical writes and attempted writes differ: Toolward prevents three attempts, introduces three and leaves one unknown. No confidence intervals or method-ranking claim. Zero gained permitted completions and zero gained both-value reports are retained in the CSV.",
    "placement": "Appendix immediately after the fixed-page permission contrast; cross-reference from the current 21-pair selective-execution figure.",
    "verification": {"raw_record_hash_verified": True, "raw_behavior_rederived_from_saved_evidence": True,
        "all_six_methods_retained": True, "every_count_matches_independent_audit": True,
        "baseline_match": "case_id, authorized_action, generation seed"},
    "point_unit": "matched saved episode pair", "physical_write_not_attempt_endpoint": True,
}, [SOURCE / "data/permission-20260905/episodes.jsonl.gz", SOURCE / "figures.py",
    AUDIT / "permission-all-method-pairs.csv", AUDIT / "permission-method-summary.csv",
    AUDIT / "historical-audit-verification.json", DATA / f"{name}.csv"])
