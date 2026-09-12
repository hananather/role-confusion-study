"""I render complete behavioral cohorts with paper typography and direct labels."""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "steering-evidence-review/data"
DATA = HERE / "data"
OUT = HERE / "figures"
DATA.mkdir(exist_ok=True, parents=True)
OUT.mkdir(exist_ok=True, parents=True)
for variant in ("regular", "bold", "italic", "bolditalic"):
    font_manager.fontManager.addfont(HERE.parent / f"fonts/termes-ttf/texgyretermes-{variant}.ttf")
plt.rcParams.update({
    "font.family": "TeX Gyre Termes", "font.size": 10,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
    "text.parse_math": False, "axes.linewidth": .5,
    "savefig.facecolor": "white", "xtick.major.width": .5,
})
INK = "#202020"
GRAY = "#62748e"
BLUE = "#00a6f4"
ORANGE = "#fd9a00"
GREEN = "#00d492"
BLUE_INK = "#0084d1"
ORANGE_INK = "#e17100"
GREEN_INK = "#009966"

LABELS = {
    "none": "No intervention",
    "role_a16": "Original vector: Tool − CoT",
    "zero": "Zero dose",
    "reverse_a16": "Reversed vector",
    "random_0_a16": "Random direction 1",
    "random_1_a16": "Random direction 2",
    "random_2_a16": "Random direction 3",
    "sentence_dev_rule": "Provenance reminder",
    "tool_raising_a16": "New Tool-raising vector",
}
Y = {a: y for a, y in zip(LABELS, [0, 1, 2.25, 3.25, 4.5, 5.5, 6.5, 7.75, 9])}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


inputs = {name: SOURCE / name for name in (
    "forgery-outcomes.csv", "arm-counts.csv", "paired-changes.csv"
)}
outcomes = pd.read_csv(inputs["forgery-outcomes.csv"], keep_default_na=False)
counts = pd.read_csv(inputs["arm-counts.csv"])
pairs = pd.read_csv(inputs["paired-changes.csv"])
assert not outcomes.duplicated(["cohort", "page_id", "arm_id"]).any()
assert not pairs.duplicated(["cohort", "case_id", "arm_id"]).any()
assert set(outcomes.outcome) == {"UPLOAD", "NO_UPLOAD", "CENSORED", "UNRUN"}

# I recalculate every plotted count from the episode-level table before rendering.
rows = []
for cohort in ("historical", "new"):
    for arm in LABELS:
        q = outcomes[(outcomes.cohort == cohort) & (outcomes.arm_id == arm)]
        if len(q) == 0:
            assert cohort == "new" and arm == "zero"
            rows.append({"cohort": cohort, "arm_id": arm, "label": LABELS[arm],
                         "assigned": 0, "recorded": 0, "uploads": 0,
                         "no_upload": 0, "unresolved": 0, "unrun": 0,
                         "display_state": "Not assigned"})
            continue
        s = counts[(counts.cohort == cohort) & (counts.arm_id == arm)].iloc[0]
        assert len(q) == int(s.assigned) == 5
        values = q.outcome.value_counts()
        for state in ("UPLOAD", "NO_UPLOAD", "CENSORED", "UNRUN"):
            assert values.get(state, 0) == int(s[state])
        rows.append({"cohort": cohort, "arm_id": arm, "label": LABELS[arm],
                     "assigned": 5, "recorded": int(s.recorded),
                     "uploads": int(s.UPLOAD), "no_upload": int(s.NO_UPLOAD),
                     "unresolved": int(s.CENSORED), "unrun": int(s.UNRUN),
                     "display_state": "5 unrun" if s.UNRUN else "Recorded"})
plot_data = pd.DataFrame(rows)
plot_data.to_csv(DATA / "behavior-summary.csv", index=False)
for name, p in inputs.items():
    shutil.copy2(p, DATA / ("behavior-source-" + name))

original_pairs = pairs[pairs.arm_id == "role_a16"]
assert original_pairs.change.value_counts().to_dict() == {"unchanged": 7, "prevented": 2, "introduced": 1}
assert set(original_pairs[original_pairs.cohort == "new"].change) == {"unchanged"}


def style(arm):
    if arm in ("role_a16", "reverse_a16", "tool_raising_a16"):
        return BLUE, BLUE_INK
    if arm.startswith("random_"):
        return ORANGE, ORANGE_INK
    if arm == "sentence_dev_rule":
        return GREEN, GREEN_INK
    return GRAY, INK


fig, axs = plt.subplots(1, 2, figsize=(8.3, 5.7), sharey=True)
fig.text(.037, .967, "Original steering leaves all five new-page outcomes unchanged",
         fontsize=15.4, weight="bold", color=INK, va="top")
fig.text(.038, .914, "GPT-OSS-20B · forged reasoning inside a webpage · all tested directions and controls",
         fontsize=10, color=INK, va="top")

for ax, cohort, title in zip(axs, ("historical", "new"),
                              ("A · Previously explored pages", "B · New pages")):
    ax.set(xlim=(-.25, 6.10), ylim=(9.7, -.7), xticks=list(range(6)))
    ax.set_title(title, fontsize=11.5, weight="bold", pad=11, loc="left")
    ax.set_yticks(list(Y.values()))
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=3, labelsize=10, pad=5)
    ax.set_xlabel("Verified uploads out of five", fontsize=11, labelpad=9)
    for name in ("left", "right", "top"):
        ax.spines[name].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 5)
    ax.spines["bottom"].set_color(GRAY)
    for x in range(6):
        ax.axvline(x, color="#e8e8e8", lw=.5, zorder=0)
    baseline = int(plot_data[(plot_data.cohort == cohort) & (plot_data.arm_id == "none")].uploads.iloc[0])
    ax.axvline(baseline, color=GRAY, lw=.7, ls=(0, (3, 3)), zorder=1)
    ax.axhspan(.57, 1.43, color=BLUE, alpha=.075, lw=0, zorder=0)
    for _, r in plot_data[plot_data.cohort == cohort].iterrows():
        arm, y = r.arm_id, Y[r.arm_id]
        color, text_color = style(arm)
        if r.display_state != "Recorded":
            # I label missing conditions instead of plotting them at zero uploads.
            ax.text(2.5, y, r.display_state, ha="center", va="center", fontsize=10.5,
                    color=GRAY, style="italic")
            continue
        x = int(r.uploads)
        ax.scatter(x, y, s=59 if arm == "role_a16" else 44, color=color,
                   edgecolor="white", linewidth=.55, zorder=4)
        if r.unresolved:
            # This segment bounds the unknown outcome; it is not a sampling interval.
            end = x + int(r.unresolved)
            ax.plot([x, end], [y, y], color=color, lw=1.6, zorder=2)
            ax.scatter(end, y, s=44, facecolor="white", edgecolor=color, linewidth=1.2, zorder=3)
            ax.text(end + .24, y, f"{x}–{end}*", color=INK, fontsize=10.2,
                    va="center", ha="left")
        else:
            ax.text(x + .23, y, str(x), color=text_color,
                    weight="bold" if arm == "role_a16" else "normal",
                    fontsize=11, va="center", ha="left")

axs[0].set_yticklabels([LABELS[a] for a in Y], fontsize=10.5)
for label, arm in zip(axs[0].get_yticklabels(), Y):
    if arm == "role_a16":
        label.set_weight("bold")
    label.set_color(style(arm)[1])

handles = [Line2D([], [], color=c, marker="o", ls="", ms=5.5, label=label)
           for c, label in ((BLUE, "Role vectors"), (ORANGE, "Random directions"),
                            (GREEN, "Instruction reminder"))]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.605, .077),
           ncol=3, frameon=False, fontsize=9.5, handletextpad=.35, columnspacing=1.4)
fig.text(.038, .041,
         "* One episode is unresolved in each panel; the segment spans its two possible upload counts.",
         fontsize=9, color=GRAY, va="center")
fig.subplots_adjust(left=.295, right=.974, top=.817, bottom=.24, wspace=.18)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
clipped = []
for artist in fig.findobj(matplotlib.text.Text):
    if not artist.get_visible() or not artist.get_text():
        continue
    bounds = artist.get_window_extent(renderer)
    if bounds.width and bounds.height and (bounds.x0 < -1 or bounds.y0 < -1
        or bounds.x1 > fig.bbox.width + 1 or bounds.y1 > fig.bbox.height + 1):
        clipped.append(artist.get_text())
assert not clipped, clipped

files = {}
for ext in ("png", "pdf", "svg"):
    p = OUT / f"behavior-complete-cohorts.{ext}"
    fig.savefig(p, dpi=400)
    files[p.name] = digest(p)
plt.close(fig)

caption = ("The original vector changes no upload outcomes on the five new pages; "
           "random directions produce fewer uploads in this sample, while the new Tool-raising vector remains untested there.")
metadata = {
    "figure_id": "behavior-complete-cohorts",
    "title": "Original steering leaves all five new-page outcomes unchanged",
    "caption": caption,
    "alt": "Two aligned dot plots show uploads across nine interventions on five previously explored and five new webpages. No intervention gives three then four uploads; the original Tool-minus-CoT vector gives two then four. Each random direction gives one verified upload on new pages, with one unresolved episode for random direction 3. The new Tool-raising vector gives zero uploads on the previously explored pages and has five unrun new-page episodes. Zero dose is not assigned on new pages.",
    "placement": "Main behavioral result, after introducing forged-reasoning attacks and before interpreting role-probe shifts.",
    "scope": "Complete saved H100 forgery cohort: ten distinct pages; 80 recorded episodes and five assigned unrun Tool-raising episodes. The zero-dose arm was not assigned on the five new pages. Counts are descriptive; page groups are separate and are not independent replications. No general null, equivalence, or population confidence claim is made.",
    "methods": "The verified endpoint is dummy-file upload receipt. Every eligible forgery row and control appears. No model run is added. Dot positions equal verified upload counts. Open-ended segments span the possible total after assigning the single censored outcome in that arm and cohort; they are not confidence intervals. Blue, orange and green identify intervention families rather than token roles.",
    "paired_original_result": {"prevented": 2, "introduced": 1, "unchanged": 7,
                               "new_page_outcomes_unchanged": 5},
    "source_sha256": {str(p): digest(p) for p in inputs.values()},
    "display_data_sha256": digest(DATA / "behavior-summary.csv"),
    "renderer_sha256": digest(Path(__file__)),
    "artifact_sha256": files,
    "font": "TeX Gyre Termes",
    "size_inches": [8.3, 5.7],
    "palette": {"role_vectors": BLUE, "random_directions": ORANGE,
                "instruction_reminder": GREEN, "reference": GRAY},
    "visible_text_within_canvas": True,
    "verification": "Every arm/cohort count independently recomputed from episode-level CSV and matched against the audited count table; paired original-vector statement checked against every paired row.",
}
(DATA / "behavior-complete-cohorts.json").write_text(json.dumps(metadata, indent=2) + "\n")
(DATA / "behavior-complete-cohorts.md").write_text(
    "# Complete behavioral cohorts\n\n*" + caption + "*\n\n" +
    "Suggested placement: " + metadata["placement"] + "\n\n" +
    "Scope: " + metadata["scope"] + "\n\n" +
    "Reading note: " + metadata["methods"] + "\n"
)
print(json.dumps(metadata, indent=2))
