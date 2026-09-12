"""I compare complete, matched role compositions without rerunning the model."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "steering-evidence-review/data/first-exposure-probes.csv"
AUDIT = HERE.parent / "steering-evidence-review/reviews/full-batch-audit.md"
DATA, OUT = HERE / "data", HERE / "figures"
for directory in (DATA, OUT, HERE / "private-layouts"):
    directory.mkdir(exist_ok=True)
for style in ("regular", "bold", "italic", "bolditalic"):
    font_manager.fontManager.addfont(
        HERE.parent / f"fonts/termes-ttf/texgyretermes-{style}.ttf")
plt.rcParams.update({
    "font.family": "TeX Gyre Termes", "font.size": 10,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
    "text.parse_math": False, "axes.linewidth": .5,
    "axes.edgecolor": "#333333", "savefig.facecolor": "white",
    "xtick.major.width": .5, "ytick.major.width": .5,
})
INK, GRAY = "#172b3a", "#62748e"
ROLES = ["system", "user", "cot", "assistant", "tool"]
COLORS = dict(zip(ROLES, ["#90a1b9", "#00a6f4", "#fd9a00", "#00d492", "#7e6cff"]))
ROLE_LABELS = dict(zip(ROLES, ["System", "User", "CoT", "Assistant", "Tool"]))
ARMS = ["zero", "role_a16", "tool_raising_a16"]
ARM_LABELS = ["Zero-dose control", "Original steering\nTool − CoT",
              "Revised steering\nTool − mean(User, CoT)"]
SPANS = ["page", "payload"]
SPAN_LABELS = {"page": "Entire tool response", "payload": "Forged reasoning passage"}
PROBS = [f"p_{role}" for role in ROLES]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


all_rows = pd.read_csv(SOURCE, dtype={"page_id": str}, keep_default_na=False)
rows = all_rows[(all_rows.cohort == "historical") & all_rows.arm_id.isin(ARMS)
                & all_rows.span.isin(SPANS)].copy()
assert len(rows) == 30
assert not rows.duplicated(["arm_id", "page_id", "span"]).any()
assert rows.groupby(["arm_id", "span"]).size().eq(5).all()
assert set(rows.page_id) == {"000", "001", "002", "003", "004"}
assert np.allclose(rows[PROBS].sum(axis=1), 1, atol=3e-7)
assert np.isfinite(rows[PROBS]).all().all()
assert ((rows[PROBS] >= 0) & (rows[PROBS] <= 1)).all().all()
# I verify page, seed, first-response prompt and token masks match across arms.
for (_, _), group in rows.groupby(["page_id", "span"]):
    assert group.seed.nunique() == 1
    assert group.prompt_sha256.nunique() == 1
    assert group.n_tokens.nunique() == 1
for path, expected_hash in rows[["episode_file", "episode_sha256"]].drop_duplicates().itertuples(index=False, name=None):
    assert sha(Path(path)) == expected_hash
rows = rows.sort_values(["page_id", "arm_id", "span"])
rows.to_csv(DATA / "role-composition-pages.csv", index=False)
means = rows.groupby(["span", "arm_id"])[PROBS].mean()
mean_records = []
for span in SPANS:
    for arm in ARMS:
        group = rows[(rows.span == span) & (rows.arm_id == arm)]
        for role in ROLES:
            mean_records.append({
                "region": span, "arm_id": arm, "role": role,
                "mean_probability": float(means.loc[(span, arm), f"p_{role}"]),
                "n_pages": 5, "min_page_probability": float(group[f"p_{role}"].min()),
                "max_page_probability": float(group[f"p_{role}"].max()),
                "min_region_tokens": int(group.n_tokens.min()),
                "max_region_tokens": int(group.n_tokens.max()),
                "aggregation": "token mean within each page, then equal-weight mean over five pages",
            })
pd.DataFrame(mean_records).to_csv(DATA / "role-composition-means.csv", index=False)


def build(horizontal):
    if horizontal:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.85), sharey=True)
        fig.subplots_adjust(left=.238, right=.972, top=.736, bottom=.225, wspace=.12)
        fs = 11
    else:
        fig, axes = plt.subplots(2, 1, figsize=(8.3, 6.8))
        fig.subplots_adjust(left=.29, right=.96, top=.81, bottom=.19, hspace=.50)
        fs = 11
    fig.text(.035, .957, "The original vector suppresses CoT and raises User scores",
             fontsize=18.5, fontweight="bold", va="top", color=INK)
    fig.text(.036, .865 if horizontal else .895,
             "GPT-OSS-20B · five matched historical webpages · equal page weights",
             fontsize=10.5, va="top", color=GRAY)
    for i, (ax, span) in enumerate(zip(axes, SPANS)):
        for row, arm in enumerate(ARMS):
            left = 0
            for role in ROLES:
                value = float(means.loc[(span, arm), f"p_{role}"] * 100)
                ax.barh(row, value, left=left, height=.48,
                        color=COLORS[role], edgecolor="white", linewidth=.5, zorder=3)
                if value >= 8:
                    label = "≈100%" if value > 99.9 else f"{value:.0f}%"
                    ax.text(left + value / 2, row, label, ha="center", va="center",
                            fontsize=fs, color="#111111", zorder=4)
                left += value
            if arm == "role_a16":
                ax.text(50, row + .38, "CoT < 0.001%", fontsize=9.5,
                        ha="center", va="center", color=GRAY)
        ax.set(xlim=(0, 100), ylim=(2.56, -.53), yticks=[0, 1, 2],
               xticks=[0, 25, 50, 75, 100], xticklabels=["0", "25", "50", "75", "100%"])
        ax.set_title(f"{'AB'[i]} · {SPAN_LABELS[span]}", fontweight="bold", fontsize=12, pad=12)
        ax.grid(axis="x", color="#d9d9d9", linewidth=.35)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=11)
        ax.tick_params(axis="x", length=2.5, labelsize=10)
        if not horizontal or i == 0:
            ax.set_yticklabels(ARM_LABELS, fontsize=11.2, color=INK)
        ax.set_xlabel("Mean role-probe probability", labelpad=8, fontsize=10.5)
    fig.legend([Patch(facecolor=COLORS[r], edgecolor="none") for r in ROLES],
               [ROLE_LABELS[r] for r in ROLES], ncol=5, frameon=False,
               loc="lower center", bbox_to_anchor=(.605 if horizontal else .58, .055 if horizontal else .025),
               fontsize=11, handlelength=1.1, handleheight=.85, columnspacing=1.4, handletextpad=.45)
    return fig


def save(fig, stem, primary):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    cropped = []
    for item in fig.findobj(matplotlib.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or
                                        box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1):
            cropped.append(item.get_text())
    assert not cropped, cropped
    result = {}
    for ext in (["png", "pdf", "svg"] if primary else ["png"]):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=320)
        result[str(path.relative_to(HERE))] = sha(path)
    plt.close(fig)
    return result


artifacts = save(build(True), OUT / "role-composition", True)
save(build(False), HERE / "private-layouts/role-composition-vertical", False)
manifest = {
    "figure": "role-composition", "stage": "Distill",
    "north_star": "I distinguish suppressing the reasoning-role score from specifically raising the Tool-role score.",
    "lens": ["one decisive comparison", "complete matched cohort", "readable claims", "measurement versus behavior"],
    "source_csv": str(SOURCE), "source_sha256": sha(SOURCE),
    "source_audit": str(AUDIT), "source_audit_sha256": sha(AUDIT),
    "selection": {"cohort": "historical", "pages": sorted(rows.page_id.unique()),
                  "arms": ARMS, "regions": SPANS},
    "n_episodes": 15, "n_page_arm_region_rows": 30,
    "aggregation": "I average token probabilities within each saved region; I then average its five page means equally. I do not renormalize or pool tokens across pages.",
    "probe": "five-class sucat_L12; model.layers[12].post_attention_layernorm output (zero-based index)",
    "steering": "model.layers[11] output (zero-based index); source Tool-content prompt positions; alpha16 for both nonzero arms",
    "readout": "first post-fetch generation; saved episode probabilities, not the engineering gate",
    "region_definition": {"page": "complete tool-content span, excluding tool header",
                          "payload": "forged-passage token mask contained within tool-content span"},
    "control": "zero-dose hook; no-intervention has no saved probe probabilities",
    "limits": "I show a five-page descriptive composition. Probabilities are classifier scores, not ground-truth roles or evidence of a mechanism. The revised vector has no new-page readouts here. I have saved region means, not tensors for reprojecting the probe.",
    "palette": COLORS, "font": "TeX Gyre Termes", "size_inches": [9.2, 4.85],
    "checks": {"matched_pages": True, "matched_seeds": True, "matched_prompts": True,
               "matched_token_counts": True, "episode_hashes": True, "sums_to_one": True,
               "all_five_roles": True, "text_within_canvas": True},
    "plotted_data": {f"data/{p.name}": sha(p) for p in DATA.glob("role-composition-*.csv")},
    "artifacts": artifacts,
}
(DATA / "role-composition-provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"figure": str(OUT / "role-composition.png"),
                  "checks": manifest["checks"]}, indent=2))
