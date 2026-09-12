"""I present complete probe readouts and the separate paired permission task."""
from pathlib import Path
import gzip
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
SOURCE = HERE.parent / "steering-evidence-review"
DATA, OUT = HERE / "data", HERE / "figures"
for p in (DATA, OUT):
    p.mkdir(exist_ok=True)
for style in ("regular", "bold", "italic", "bolditalic"):
    font_manager.fontManager.addfont(HERE.parent / f"fonts/termes-ttf/texgyretermes-{style}.ttf")
plt.rcParams.update({
    "font.family": "TeX Gyre Termes", "font.size": 10.5, "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "path", "text.parse_math": False,
    "axes.linewidth": .5, "axes.edgecolor": "#333333", "savefig.facecolor": "white",
    "xtick.major.width": .5, "ytick.major.width": .5,
})
INK, GRAY, GREEN, PINK, PURPLE = "#172b3a", "#62748e", "#00d492", "#ff637e", "#7e6cff"
STATES = {"UPLOAD": (PINK, "^"), "NO_UPLOAD": (GREEN, "o"), "CENSORED": (GRAY, "X")}
ARMS = ["zero", "role_a16", "reverse_a16", "random_0_a16", "random_1_a16", "random_2_a16", "tool_raising_a16"]
LABELS = ["Zero-dose reference", "Original: Tool − CoT", "Reversed vector", "Random 1", "Random 2", "Random 3", "Revised Tool-raising"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, stem, sources, details):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    cropped = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():
            continue
        b = t.get_window_extent(renderer)
        if b.width and b.height and (b.x0 < -1 or b.y0 < -1 or
                                    b.x1 > fig.bbox.width + 1 or b.y1 > fig.bbox.height + 1):
            cropped.append(t.get_text())
    assert not cropped, cropped
    artifacts = {}
    for ext in ("png", "pdf", "svg"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=320)
        artifacts[str(path.relative_to(HERE))] = sha(path)
    manifest = {
        "figure": stem, "stage": "Distill", "font": "TeX Gyre Termes",
        "palette": {"upload_or_unwanted_action": PINK, "no_upload_or_favorable_change": GREEN,
                    "unresolved": GRAY, "tool_probe": PURPLE},
        "size_inches": list(fig.get_size_inches()), "text_within_canvas": True,
        "sources": {str(path): sha(path) for path in sources},
        "artifacts": artifacts, "renderer_sha256": sha(Path(__file__)), **details,
    }
    (DATA / f"{stem}-provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plt.close(fig)


probe_source = SOURCE / "data/forgery-payload-probes.csv"
probes = pd.read_csv(probe_source, dtype={"page_id": str}, keep_default_na=False)
assert len(probes) == 60 and set(probes.span) == {"payload"}
assert not probes.duplicated(["cohort", "page_id", "arm_id"]).any()
assert set(probes.arm_id) == set(ARMS)
assert set(probes.outcome) <= set(STATES)
assert len(probes[probes.outcome == "CENSORED"]) == 2
assert probes[probes.arm_id == "role_a16"].p_cot.max() < 1e-15
assert (probes[probes.arm_id == "role_a16"].outcome == "UPLOAD").sum() == 6
for path, expected in probes[["episode_file", "episode_sha256"]].itertuples(index=False, name=None):
    assert sha(Path(path)) == expected
shutil.copyfile(probe_source, DATA / "probe-behavior-points.csv")
counts = []
for (cohort, arm), group in probes.groupby(["cohort", "arm_id"]):
    assert len(group) == 5
    counts.append({"cohort": cohort, "arm_id": arm, "n_readouts": len(group),
                   "uploads": int((group.outcome == "UPLOAD").sum()),
                   "no_uploads": int((group.outcome == "NO_UPLOAD").sum()),
                   "unresolved": int((group.outcome == "CENSORED").sum())})
pd.DataFrame(counts).to_csv(DATA / "probe-behavior-counts.csv", index=False)

# I connect only the five measured, matched historical zero-dose/original pairs.
paired = probes[(probes.cohort == "historical") & probes.arm_id.isin(["zero", "role_a16"])]
for _, group in paired.groupby("page_id"):
    assert len(group) == 2 and group.seed.nunique() == 1 and group.prompt_sha256.nunique() == 1
fig = plt.figure(figsize=(9.6, 6.15))
fig.text(.035, .963, "Original steering: near-zero CoT scores, six uploads", fontsize=20,
         weight="bold", color=INK, va="top")
fig.text(.036, .889, "All 60 saved forged-passage readouts · GPT-OSS-20B", fontsize=11,
         color=GRAY, va="top")
gs = fig.add_gridspec(1, 4, left=.222, right=.98, top=.77, bottom=.145,
                      width_ratios=[1, .24, 1, .24], wspace=.13)
axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[2])]
nums = [fig.add_subplot(gs[1]), fig.add_subplot(gs[3])]
for ax, numax, cohort, title in zip(axs, nums, ["historical", "new"],
                                   ["A · Five historical pages", "B · Five new pages"]):
    ax.set(xlim=(-4, 104), ylim=(6.6, -.6), yticks=np.arange(7),
           xticks=[0, 25, 50, 75, 100], xticklabels=["0", "25", "50", "75", "100%"])
    ax.axhspan(.52, 1.48, color="#f0f1f3", zorder=0)
    ax.grid(axis="x", color="#d9d9d9", lw=.4)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=12, weight="bold", pad=25)
    ax.set_xlabel("Mean CoT-probe probability", fontsize=11, labelpad=8)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", length=3, labelsize=10)
    numax.set(xlim=(0, 1), ylim=(6.6, -.6))
    numax.axis("off")
    numax.text(.5, -.83, "Uploads", ha="center", va="bottom", color=GRAY,
               fontsize=9.8, clip_on=False)
    if cohort == "historical":
        for j, page in enumerate(sorted(paired.page_id.unique())):
            pair = paired[paired.page_id == page].set_index("arm_id")
            offset = (j - 2) * .16
            ax.plot([pair.loc["zero"].p_cot * 100, pair.loc["role_a16"].p_cot * 100],
                    [offset, 1 + offset], color="#90a1b9", lw=.6, alpha=.8, zorder=1)
    for i, arm in enumerate(ARMS):
        group = probes[(probes.cohort == cohort) & (probes.arm_id == arm)].sort_values("page_id")
        if group.empty:
            ax.text(50, i, "No zero-dose run" if arm == "zero" else "Not yet run",
                    fontsize=10.2, color=GRAY, ha="center", va="center")
            numax.text(.5, i, "—", fontsize=12, color=GRAY, ha="center", va="center")
            continue
        for j, r in enumerate(group.itertuples()):
            color, marker = STATES[r.outcome]
            ax.scatter(100 * r.p_cot, i + (j - 2) * .16, s=34, marker=marker,
                       c=color, edgecolor=INK, linewidth=.4, zorder=4)
        uploads, unresolved = int((group.outcome == "UPLOAD").sum()), int((group.outcome == "CENSORED").sum())
        numax.text(.5, i - .08 if unresolved else i, f"{uploads} / 5", ha="center", va="center",
                   color=INK, fontsize=12, weight="bold" if arm == "role_a16" else "normal")
        if unresolved:
            numax.text(.5, i + .23, "1 unresolved", ha="center", va="center",
                       color=GRAY, fontsize=7.8)
axs[0].set_yticklabels(LABELS, fontsize=11, color=INK)
axs[1].set_yticklabels([])
handles = [Line2D([], [], ls="", marker=marker, markersize=6, markerfacecolor=color,
                  markeredgecolor=INK, markeredgewidth=.4, label=label)
           for color, marker, label in [(PINK, "^", "Uploaded"), (GREEN, "o", "No upload"),
                                         (GRAY, "X", "Unresolved")]]
handles.append(Line2D([], [], color="#90a1b9", lw=.8, label="Same historical page"))
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.61, .022), ncol=4,
           frameon=False, fontsize=10.5, columnspacing=1.4, handlelength=1.2, handletextpad=.45)
save(fig, "probe-behavior", [probe_source, SOURCE / "reviews/full-batch-audit.md"], {
    "north_star": "I show whether the actual upload endpoint separates under low reasoning-role scores.",
    "selection": "All sixty available first-postfetch forged-passage episode readouts, including two censored episodes.",
    "n_readouts": 60, "historical_readouts": 35, "new_readouts": 25,
    "paired_reference": "Five historical pages have measured zero-dose and original-vector readouts with matched seeds and prompts; connecting lines represent only these pairs.",
    "missing": "No-intervention hooks were disabled: no saved probe baseline for any page. New zero-dose was not assigned; revised Tool-raising on new pages is unrun. No missing score is imputed.",
    "readout": "Mean token-level sucat_L12 five-class CoT probability in the forged passage at the first post-fetch generation.",
    "outcome": "Whole-episode verified dummy-file upload; two censored episodes are unresolved, not failed attacks.",
    "point_placement": "Within every arm, the five fixed vertical offsets are page IDs 000 through 004 in ascending order.",
    "limitations": "These are measured episode summaries, not saved activation tensors; one rollout per arm/page, ten underlying pages. Low scores with uploads do not establish a causal mechanism or equivalence of behavior.",
    "checks": {"all_60_rows": True, "no_duplicate_episode": True, "hashes_match": True,
               "historical_pairs_match": True, "all_original_cot_near_zero": True,
               "original_uploads": 6, "censored_episodes": 2},
    "plotted_data": {f"data/{p.name}": sha(p) for p in DATA.glob("probe-behavior-*.csv")},
})

permission_source = SOURCE / "reviews/permission-toolward-pairs.csv"
p = pd.read_csv(permission_source)
assert len(p) == 21 and p.case_id.nunique() == 8
assert len(p[["case_id", "authorized_action"]].drop_duplicates()) == 15
assert p.read_exposure_confirmed.all()
assert p.baseline_authorized_completion.all()
assert p.intervention_mean_tool_first_prefill.min() > .999976
assert p.baseline_unauthorized_write.sum() == 18 and p.intervention_unauthorized_write.sum() == 17
assert p.intervention_authorized_completion.sum() == 19
assert not p.duplicated(["case_id", "authorized_action", "seed"]).any()
# I verify the frozen raw-record hashes before deriving the paired counts.
for source_path, group in p.groupby("source_path"):
    wanted = {}
    for r in group.itertuples():
        wanted[r.baseline_line] = r.baseline_raw_record_sha256
        wanted[r.intervention_line] = r.intervention_raw_record_sha256
    with gzip.open(source_path, "rb") as f:
        for line_number, line in enumerate(f, 1):
            if line_number in wanted:
                assert hashlib.sha256(line.rstrip(b"\r\n")).hexdigest() == wanted[line_number]
                del wanted[line_number]
    assert not wanted
shutil.copyfile(permission_source, DATA / "permission-task-pairs.csv")
transitions = []
for endpoint, bcol, scol in [
    ("unauthorized_write", "baseline_unauthorized_write", "intervention_unauthorized_write"),
    ("authorized_completion", "baseline_authorized_completion", "intervention_authorized_completion")]:
    for before in (False, True):
        for after in (False, True):
            transitions.append({"endpoint": endpoint, "baseline": before, "toolward": after,
                                "pairs": int(((p[bcol] == before) & (p[scol] == after)).sum())})
pd.DataFrame(transitions).to_csv(DATA / "permission-task-transitions.csv", index=False)
unwanted = [int((p.baseline_unauthorized_write & p.intervention_unauthorized_write).sum()),
            int((p.baseline_unauthorized_write & ~p.intervention_unauthorized_write).sum()),
            int((~p.baseline_unauthorized_write & p.intervention_unauthorized_write).sum())]
authorized = [int((p.baseline_authorized_completion & p.intervention_authorized_completion).sum()),
              int((p.baseline_authorized_completion & ~p.intervention_authorized_completion).sum())]
assert unwanted == [14, 4, 3] and sum(unwanted) == 21
assert authorized == [19, 2] and sum(authorized) == 21
fig = plt.figure(figsize=(9.6, 4.95))
fig.text(.035, .957, "Near-total Tool scores coexist with unwanted writes", fontsize=20,
         weight="bold", color=INK, va="top")
fig.text(.036, .873, "Separate permission task · all 21 recovered pairs · eight cases",
         fontsize=11, color=GRAY, va="top")
gs = fig.add_gridspec(1, 3, left=.076, right=.98, top=.688, bottom=.185,
                      width_ratios=[1.05, 1.1, 1.1], wspace=.57)
ax = fig.add_subplot(gs[0])
for _, group in p.groupby(["case_id", "authorized_action"]):
    # I separate repeated-seed endpoints slightly along the categorical x axis.
    for j, r in enumerate(group.sort_values("seed").itertuples()):
        offset = (j - (len(group) - 1) / 2) * .048
        ax.plot([offset, 1 + offset], [100 * r.baseline_mean_tool_first_prefill,
                 100 * r.intervention_mean_tool_first_prefill], color=PURPLE,
                lw=.75, alpha=.48, zorder=2)
        ax.scatter([offset, 1 + offset], [100 * r.baseline_mean_tool_first_prefill,
                    100 * r.intervention_mean_tool_first_prefill], color=PURPLE,
                   edgecolor="white", linewidth=.35, s=18, zorder=3)
ax.set(xlim=(-.24, 1.25), ylim=(-3, 111), xticks=[0, 1],
       xticklabels=["No steering", "Toolward"], yticks=[0, 25, 50, 75, 100],
       yticklabels=["0", "25", "50", "75", "100%"], ylabel="Mean Tool-probe probability")
ax.set_title("A · Command-token readout", fontsize=11.5, weight="bold", pad=27)
ax.text(.75, 106, "All 21 ≥99.997%", ha="center", fontsize=9.5, color=GRAY)
ax.grid(axis="y", color="#d9d9d9", linewidth=.35)
ax.set_axisbelow(True)
for loc, values, colors, labels, title, summary in [
    (1, unwanted, [PINK, GREEN, PINK], ["Persisted", "Prevented", "Introduced"],
     "B · Unwanted physical writes", "18 / 21  →  17 / 21"),
    (2, authorized, [GREEN, PINK], ["Preserved", "Lost"],
     "C · Permitted completions", "21 / 21  →  19 / 21")]:
    ax = fig.add_subplot(gs[loc])
    ax.set_title(title, fontsize=11.5, weight="bold", pad=27)
    ax.text(.5, 1.035, summary, transform=ax.transAxes, fontsize=12, ha="center",
            va="bottom", weight="bold", color=INK)
    y = np.arange(len(values)) * .92
    for yi, value, color, label in zip(y, values, colors, labels):
        ax.barh(yi, value, color=color, height=.38, zorder=3)
        ax.text(.2, yi - .24, label, fontsize=10.7, color=INK, va="bottom")
        ax.text(value + .65, yi, str(value), fontsize=13, weight="bold", color=INK,
                va="center", ha="left")
    ax.set(xlim=(0, 22.5), ylim=(2.35, -.65), yticks=[], xticks=[0, 7, 14, 21],
           xlabel="Matched pairs")
    ax.grid(axis="x", color="#d9d9d9", linewidth=.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=3)
fig.text(.375, .08, "Counts above panels: no steering → Toolward", ha="left", va="center",
         fontsize=10.2, color=GRAY)
save(fig, "permission-task", [permission_source, SOURCE / "reviews/historical-evidence-audit.md"], {
    "north_star": "I show whether a near-saturated Tool readout produces selective execution of the permitted action.",
    "cohort": "All twenty-one recovered matched Toolward/no-steering pairs from the earlier permission protocol.",
    "n_pairs": 21, "n_cases": 8, "n_first_prefill_contexts": 15,
    "endpoint": "Controller-executed physical marker writes and correct permitted completions. This is separate from dummy-file upload.",
    "probe": "Mean first-prefill command-token Tool probability, five-role classifier at block 14 post-attention normalization.",
    "intervention": "Neutral-text Tool−User at block 11 output, command-token mask, magnitude 0.3 of the construction residual norm; distinct from Tool−CoT in the later upload task.",
    "pairing": "Identical case, permission, and sampling seed; all pairs have recorded page exposure.",
    "min_mean_tool_probability": float(p.intervention_mean_tool_first_prefill.min()),
    "point_placement": "All 21 paired lines are drawn; repeated-seed contexts are slightly offset horizontally on categorical x positions. Fifteen unique first-prefill contexts underlie these readouts.",
    "coverage_limit": "Only recovered records are shown: 149 of 280 permission records across methods were recovered; all 21 recovered Toolward episodes have a matched baseline.",
    "interpretation_limit": "One of four prevented physical writes arose during malformed generation with unknown attempt status. These counts do not demonstrate reliable selective defense or isolate mechanism.",
    "checks": {"all_21_pairs": True, "raw_record_hashes_match": True,
               "unique_case_permission_seed_pairs": True, "all_page_exposed": True,
               "unwanted_persisted_prevented_introduced": unwanted,
               "authorized_preserved_lost": authorized},
    "plotted_data": {f"data/{path.name}": sha(path) for path in DATA.glob("permission-task-*.csv")},
})
print("Rendered probe-behavior and permission-task with saved-source checks.")
