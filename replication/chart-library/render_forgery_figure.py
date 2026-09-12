"""I render a measured four-arm agent example in the paper's Figure 8 style.

I consume displayed-rows.csv and display-manifest.json from an explicit data
folder. I never create example data or infer outcomes from probe scores. The
CSV columns are arm, segment, x, raw_cotness, cotness, token_index, token_text.
The manifest contains primary_probe, four arms with source-labeled segments
and complete excerpts, and optional verbatim attack_texts / companion_texts.

Usage: .venv/bin/python chart-library/render_forgery_figure.py --data DATA_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FONT = HERE / "fonts/termes-ttf"
ARMS = ("plain", "forgery", "steered", "destyled")
COLORS = {
    "user": "#00a6f4", "tool": "#ff637e", "forged_cot": "#ff637e",
    "cot": "#fd9a00", "assistant": "#00d492",
    "ordinary_tool": "#7e6cff", "system": "#90a1b9",
}
INK = {**COLORS, "user": "#0084d1", "cot": "#e17100",
       "assistant": "#009966", "system": "#62748e"}
DARK = "#45556c"
WIDTH = 8.3
LEFT, RIGHT = .64, .22
PLOT_WIDTH = WIDTH - LEFT - RIGHT
EXCERPT_PT, LABEL_PT = 7.4, 7.7


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_style() -> None:
    for style in ("regular", "bold", "italic", "bolditalic"):
        font_manager.fontManager.addfont(FONT / f"texgyretermes-{style}.ttf")
    plt.rcParams.update({
        "font.family": "TeX Gyre Termes", "font.size": 9,
        "text.usetex": False, "text.parse_math": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "axes.linewidth": .5, "savefig.facecolor": "white",
    })


def wrap_physical(text: str, renderer, width_px: float, size: float,
                  *, weight: str = "normal") -> str:
    """I preserve words and literal markup while inserting visual line breaks."""
    font = FontProperties(family="TeX Gyre Termes", size=size, weight=weight)
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = (line + " " + word).strip()
            if line and renderer.get_text_width_height_descent(candidate, font, False)[0] > width_px:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    wrapped = "\n".join(lines)
    if wrapped.split() != text.split():
        raise ValueError("Visual wrapping changed the source words.")
    return wrapped


def measure(fig, text: str, *, size: float, angle: float = 0,
            weight: str = "normal") -> tuple[float, float]:
    artist = fig.text(0, 0, text, fontsize=size, rotation=angle,
                      ha="left", va="top", linespacing=1.17, fontweight=weight)
    box = artist.get_window_extent(fig.canvas.get_renderer())
    artist.remove()
    return box.width / fig.dpi, box.height / fig.dpi


def load_inputs(data: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv = data / "displayed-rows.csv"
    manifest = json.loads((data / "display-manifest.json").read_text())
    rows = pd.read_csv(csv, keep_default_na=False, float_precision="round_trip",
                       dtype={"arm": str, "segment": str, "token_text": str})
    required = {"arm", "segment", "x", "raw_cotness", "cotness", "token_index", "token_text"}
    if required.difference(rows.columns):
        raise ValueError(f"Missing CSV columns: {sorted(required.difference(rows.columns))}")
    if set(rows.arm) != set(ARMS):
        raise ValueError("All four measured arms must be present.")
    arms = manifest.get("arms", [])
    if [a["arm"] for a in arms] != list(ARMS):
        raise ValueError("Manifest arms must be plain, forgery, steered, destyled in that order.")
    probe = manifest.get("primary_probe", {})
    if probe.get("layer") != 16 or probe.get("roles") != 4:
        raise ValueError("Primary Figure 8 panel requires the explicitly identified four-role L16 probe.")
    if not np.isfinite(rows[["x", "raw_cotness", "cotness"]].to_numpy()).all():
        raise ValueError("CSV contains a non-finite value.")
    if not rows.raw_cotness.between(0, 1).all() or not rows.cotness.between(0, 1).all():
        raise ValueError("Probe probabilities must lie in [0,1].")
    for arm in arms:
        part = rows[rows.arm == arm["arm"]].sort_values("x")
        if part.x.duplicated().any() or not np.equal(part.x, np.round(part.x)).all():
            raise ValueError(f"{arm['arm']}: display x positions must be distinct integers.")
        if len(part) > 1 and not np.equal(np.diff(part.x), 1).all():
            raise ValueError(f"{arm['arm']}: concatenate selected content tokens into contiguous x positions.")
        if not arm.get("title") or not arm.get("outcome_label"):
            raise ValueError(f"{arm['arm']}: explicit panel title and measured outcome required.")
        segments = arm.get("segments", [])
        if [str(s["segment"]) for s in segments] != list(dict.fromkeys(part.segment)):
            raise ValueError(f"{arm['arm']}: segment manifest order does not match the CSV.")
        for segment in segments:
            source = segment["source"]
            if source not in COLORS:
                raise ValueError(f"Unknown source category {source!r}.")
            if not segment.get("label") or not segment.get("excerpt"):
                raise ValueError("Each segment needs a source label and complete opening excerpt.")
            excerpt = segment["excerpt"]
            if "…" in excerpt:
                raise ValueError("Main-diagram excerpts must not be cut off with ellipses.")
            if segment.get("full_text") and excerpt not in segment["full_text"]:
                raise ValueError(f"Excerpt is not an exact substring: {arm['arm']}/{segment['segment']}.")
            group = part[part.segment == str(segment["segment"])]
            positions = group.x.to_numpy()
            if len(positions) > 1 and not np.equal(np.diff(positions), 1).all():
                raise ValueError("A source segment must be contiguous, not a reused role label.")
            expected = group.raw_cotness.ewm(alpha=.5, adjust=True).mean().to_numpy()
            if not np.allclose(expected, group.cotness.to_numpy(), atol=2e-6, rtol=0):
                raise ValueError(f"EWMA mismatch: {arm['arm']}/{segment['segment']}.")
    central = {a["arm"]: a for a in arms if a["arm"] in ("forgery", "steered")}
    for segment in central["forgery"]["segments"]:
        if segment["source"] not in ("user", "tool", "forged_cot"):
            continue
        key = str(segment["segment"])
        before = rows[(rows.arm == "forgery") & (rows.segment == key)].sort_values("x")
        after = rows[(rows.arm == "steered") & (rows.segment == key)].sort_values("x")
        columns = ["x", "token_index", "token_text"] + (["token_id"] if "token_id" in rows else [])
        if not before[columns].reset_index(drop=True).equals(after[columns].reset_index(drop=True)):
            raise ValueError(f"The central pair must share identical input tokens and x positions: {key}.")
    return rows, manifest


def plan_excerpts(fig, arm: dict, angle: int) -> dict:
    renderer = fig.canvas.get_renderer()
    choices = []
    for segment in arm["segments"]:
        candidates = []
        for width_in in np.arange(.45, 2.05, .025):
            wrapped = wrap_physical(segment["excerpt"], renderer, width_in * fig.dpi, EXCERPT_PT)
            width, height = measure(fig, wrapped, size=EXCERPT_PT, angle=angle)
            # I make room for a readable label above the excerpt without truncation.
            effective_width = max(.55, width)
            label = wrap_physical(segment["label"], renderer, effective_width * fig.dpi, LABEL_PT,
                                  weight="bold")
            label_width, label_height = measure(fig, label, size=LABEL_PT, weight="bold")
            candidates.append({"wrapped": wrapped, "width": max(width, label_width),
                               "height": height, "label": label, "label_height": label_height})
        choices.append(min(candidates, key=lambda c: (c["width"], c["height"])))
    gap = (PLOT_WIDTH - sum(c["width"] for c in choices)) / max(1, len(choices) - 1)
    if gap < .055:
        raise ValueError(f"{arm['arm']}: complete excerpts do not fit at {EXCERPT_PT} pt; "
                         "select shorter complete opening sentences or fewer complete source segments. "
                         "I will not shrink below this font size or silently cut text.")
    label_height = max(c["label_height"] for c in choices)
    excerpt_height = max(c["height"] for c in choices)
    outcome_note = str(arm.get("outcome_note", ""))
    note = wrap_physical(outcome_note, renderer, PLOT_WIDTH * fig.dpi, 7.7)
    note_height = measure(fig, note, size=7.7)[1] if note else 0
    return {"choices": choices, "gap": gap, "label_height": label_height,
            "excerpt_height": excerpt_height, "note": note, "note_height": note_height,
            "height": .23 + .75 + .12 + label_height + .075 + excerpt_height + .12 + note_height + .28}


def assert_text_inside(fig, artists: list) -> list[dict]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = []
    for label, artist in artists:
        box = artist.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        if box.x0 < -.015 or box.y0 < -.015 or box.x1 > fig.get_figwidth() + .015 or box.y1 > fig.get_figheight() + .015:
            raise ValueError(f"Text would be clipped: {label}: {box.bounds}")
        bounds.append({"label": label, "bounds_inches": list(box.bounds),
                       "font_size_points": artist.get_fontsize()})
    return bounds


def save_formats(fig, prefix: Path, dpi: int) -> dict:
    paths = {}
    for ext in ("png", "pdf", "svg"):
        path = prefix.with_suffix("." + ext)
        fig.savefig(path, dpi=dpi, facecolor="white")
        paths[ext] = {"path": str(path), "sha256": sha(path)}
    return paths


def render_main(rows, manifest, out: Path, angle: int, dpi: int) -> dict:
    planning = plt.figure(figsize=(WIDTH, 10), dpi=120)
    planning.canvas.draw()
    plans = [plan_excerpts(planning, arm, angle) for arm in manifest["arms"]]
    footer = manifest.get("footer", "Selected local example; colors identify token source.")
    if not manifest.get("show_technical_footer", False):footer = ""
    footer = wrap_physical(footer, planning.canvas.get_renderer(), PLOT_WIDTH * planning.dpi, 7.4)
    footer_height = measure(planning, footer, size=7.4)[1] if footer else 0
    plt.close(planning)
    top = .72
    height = top + sum(p["height"] for p in plans) + footer_height + .22
    fig = plt.figure(figsize=(WIDTH, height), dpi=120)
    text_artists, plotted = [], {}

    def text(x, y_from_top, value, **kwargs):
        artist = fig.text(x / WIDTH, 1 - y_from_top / height, value, ha="left", va="top", **kwargs)
        text_artists.append((value[:80], artist))
        return artist

    text(LEFT, .16, manifest.get("title", "Role-probe responses to forged reasoning and steering"),
         fontsize=11.5, fontweight="bold")
    text(LEFT, .40, manifest.get("subtitle", "GPT-OSS-20B · layer 16 · four-role probe · one selected agent example"),
         fontsize=8.3)
    for number, (arm, plan) in enumerate(zip(manifest["arms"], plans)):
        letter = "ABCD"[number]
        panel_title = re.sub(r"^[A-D]\s*[·.]\s*", "", arm["title"])
        text(LEFT, top, f"{letter} · {panel_title}", fontsize=9, fontweight="bold", color=DARK)
        outcome = fig.text((WIDTH - RIGHT) / WIDTH, 1 - top / height, arm["outcome_label"],
                           ha="right", va="top", fontsize=8.3, color=DARK)
        text_artists.append((f"{arm['arm']} outcome", outcome))
        plot_top, plot_height = top + .23, .75
        ax = fig.add_axes([LEFT / WIDTH, 1 - (plot_top + plot_height) / height,
                           PLOT_WIDTH / WIDTH, plot_height / height])
        part = rows[rows.arm == arm["arm"]].sort_values("x")
        ax.set_ylim(-.02, 1.02)
        ax.set_xlim(float(rows.x.min()), float(rows.x.max()))
        ax.set_yticks([0, 1], labels=["0%", "100%"])
        ax.set_xticks([part[part.segment == str(s["segment"])].x.iloc[0] for s in arm["segments"]], labels=[])
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=8, pad=2, length=2, width=.4)
        ax.grid(axis="x", color="#d9d9d9", linewidth=.3)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#333333")
        for segment in arm["segments"]:
            group = part[part.segment == str(segment["segment"])]
            color = COLORS[segment["source"]]
            scatter = ax.scatter(group.x, group.cotness, s=1.42, alpha=.9, color=color, linewidths=0, zorder=3)
            line, = ax.plot(group.x, group.cotness, color=color, linewidth=1.42, alpha=.7, zorder=2)
            expected = group[["x", "cotness"]].to_numpy()
            if not np.array_equal(np.asarray(scatter.get_offsets()), expected):
                raise ValueError("Scatter changed measured values.")
            if not np.array_equal(np.asarray(line.get_xydata()), expected):
                raise ValueError("Line changed measured values.")
        plotted[arm["arm"]] = len(part)
        label_top = plot_top + plot_height + .12
        excerpt_top = label_top + plan["label_height"] + .075
        cursor = LEFT
        for segment, choice in zip(arm["segments"], plan["choices"]):
            group = part[part.segment == str(segment["segment"])]
            midpoint = (group.x.iloc[0] + group.x.iloc[-1]) / 2
            x_plot = LEFT + (midpoint - rows.x.min()) / (rows.x.max() - rows.x.min()) * PLOT_WIDTH
            fig.add_artist(Line2D([x_plot / WIDTH, (cursor + choice["width"] / 2) / WIDTH],
                                  [1 - (plot_top + plot_height + .025) / height,
                                   1 - (label_top - .035) / height],
                                  transform=fig.transFigure, color="#b5b5b5", linewidth=.45))
            text(cursor, label_top, choice["label"], fontsize=LABEL_PT,
                 fontweight="bold", color=INK[segment["source"]], linespacing=1.17)
            text(cursor, excerpt_top, choice["wrapped"], fontsize=EXCERPT_PT,
                 rotation=angle, color=INK[segment["source"]], linespacing=1.17)
            cursor += choice["width"] + plan["gap"]
        note_top = excerpt_top + plan["excerpt_height"] + .12
        if plan["note"]:
            text(LEFT, note_top, plan["note"], fontsize=7.7)
        top += plan["height"]
    y_label = fig.text(.12 / WIDTH, .53, "CoTness", rotation=90,
                      va="center", ha="center", fontsize=9)
    text_artists.append(("CoTness", y_label))
    if footer:text(LEFT, top, footer, fontsize=7.4)
    bounds = assert_text_inside(fig, text_artists)
    artifacts = save_formats(fig, out / "forgery-steering", dpi)
    plt.close(fig)
    return {"artifacts": artifacts, "size_inches": [WIDTH, height],
            "points_per_arm": plotted, "text_bounds": bounds,
            "excerpt_angle_degrees": angle, "min_excerpt_font_points": EXCERPT_PT}


def compact_source_labels(fig, arm: dict, part: pd.DataFrame,
                          x_bounds: tuple[float, float]) -> list[dict]:
    renderer = fig.canvas.get_renderer()
    labels = []
    xmin, xmax = x_bounds
    for segment in arm["segments"]:
        group = part[part.segment == str(segment["segment"])]
        midpoint = (group.x.iloc[0] + group.x.iloc[-1]) / 2
        center = LEFT + (midpoint - xmin) / (xmax - xmin) * PLOT_WIDTH
        # I leave every role label complete; narrow spans can use two lines.
        wrapped = wrap_physical(segment["label"], renderer, 1.30 * fig.dpi, 7.7)
        width, height = measure(fig, wrapped, size=7.7)
        labels.append({"segment": segment, "wrapped": wrapped, "width": width,
                       "height": height, "plot_center": center,
                       "left": max(LEFT, min(center - width / 2, WIDTH - RIGHT - width))})
    for i in range(1, len(labels)):
        labels[i]["left"] = max(labels[i]["left"], labels[i-1]["left"] + labels[i-1]["width"] + .08)
    for i in range(len(labels)-1, -1, -1):
        bound = WIDTH - RIGHT if i == len(labels)-1 else labels[i+1]["left"] - .08
        labels[i]["left"] = min(labels[i]["left"], bound - labels[i]["width"])
    if labels[0]["left"] < LEFT - .01:
        raise ValueError("Complete source labels do not fit in the compact figure.")
    return labels


def shared_input_excerpts(manifest: dict) -> list[dict]:
    if manifest.get("shared_excerpts"):
        shared = manifest["shared_excerpts"]
    else:
        shared, seen = [], set()
        for arm in manifest["arms"]:
            for segment in arm["segments"]:
                if segment["source"] not in ("user", "tool", "forged_cot"):
                    continue
                key = (segment["source"], segment["excerpt"])
                if key not in seen:
                    shared.append(segment)
                    seen.add(key)
    for segment in shared:
        if segment["source"] not in COLORS or not segment.get("excerpt"):
            raise ValueError("Shared excerpts need a known source and nonempty text.")
        if "…" in segment["excerpt"]:
            raise ValueError("Shared excerpts must be complete, without ellipses.")
        if segment.get("full_text") and segment["excerpt"] not in segment["full_text"]:
            raise ValueError("A shared excerpt is not an exact substring of its source.")
    return shared


def render_compact(rows, manifest, out: Path, angle: int, dpi: int) -> dict:
    planning = plt.figure(figsize=(WIDTH, 10), dpi=120)
    planning.canvas.draw()
    shared = shared_input_excerpts(manifest)
    excerpt_plan = plan_excerpts(planning, {"arm": "shared inputs", "segments": shared}, angle)
    label_plans = []
    for arm in manifest["arms"]:
        part = rows[rows.arm == arm["arm"]].sort_values("x")
        label_plans.append(compact_source_labels(planning, arm, part,
                                                 (float(rows.x.min()), float(rows.x.max()))))
    footer = manifest.get("compact_footer", manifest.get("footer", "Selected local example; colors identify token source."))
    if not manifest.get("show_technical_footer", False):footer = ""
    footer = wrap_physical(footer, planning.canvas.get_renderer(), PLOT_WIDTH * planning.dpi, 7.4)
    footer_height = measure(planning, footer, size=7.4)[1] if footer else 0
    plt.close(planning)
    row_heights = [.20 + .64 + .08 + max(s["height"] for s in labels) + .20 for labels in label_plans]
    top = .67
    excerpt_block_height = .22 + excerpt_plan["label_height"] + .075 + excerpt_plan["excerpt_height"] + .20
    height = top + sum(row_heights) + excerpt_block_height + footer_height + .18
    fig = plt.figure(figsize=(WIDTH, height), dpi=120)
    text_artists, plotted = [], {}

    def text(x, y_from_top, value, **kwargs):
        artist = fig.text(x / WIDTH, 1 - y_from_top / height, value, ha="left", va="top", **kwargs)
        text_artists.append((value[:80], artist))
        return artist

    text(LEFT, .15, manifest.get("title", "CoT forgery and steering in a tool-using agent"),
         fontsize=11.5, fontweight="bold")
    text(LEFT, .39, manifest.get("subtitle", "GPT-OSS-20B · layer 16 · four-role probe · one selected agent example"),
         fontsize=8.3)
    plot_band_top = top
    for number, (arm, labels, row_height) in enumerate(zip(manifest["arms"], label_plans, row_heights)):
        panel_title = re.sub(r"^[A-D]\s*[·.]\s*", "", arm["title"])
        panel_artist = text(LEFT, top, f"{'ABCD'[number]} · {panel_title}", fontsize=8.8,
                            fontweight="bold", color=DARK)
        outcome = fig.text((WIDTH - RIGHT) / WIDTH, 1 - top / height, arm["outcome_label"],
                           ha="right", va="top", fontsize=8.0, color=DARK)
        text_artists.append((f"{arm['arm']} outcome", outcome))
        fig.canvas.draw()
        if panel_artist.get_window_extent().x1 + 8 > outcome.get_window_extent().x0:
            raise ValueError("Panel title and outcome overlap; use concise exact labels.")
        plot_top, plot_height = top + .20, .64
        ax = fig.add_axes([LEFT / WIDTH, 1 - (plot_top + plot_height) / height,
                           PLOT_WIDTH / WIDTH, plot_height / height])
        part = rows[rows.arm == arm["arm"]].sort_values("x")
        ax.set_ylim(-.02, 1.02)
        ax.set_xlim(float(rows.x.min()), float(rows.x.max()))
        ax.set_yticks([0, 1], labels=["0%", "100%"])
        ax.set_xticks([part[part.segment == str(s["segment"])].x.iloc[0] for s in arm["segments"]], labels=[])
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=8, pad=2, length=2, width=.4)
        ax.grid(axis="x", color="#d9d9d9", linewidth=.3)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#333333")
        for segment in arm["segments"]:
            group = part[part.segment == str(segment["segment"])]
            color = COLORS[segment["source"]]
            scatter = ax.scatter(group.x, group.cotness, s=1.42, alpha=.9, color=color, linewidths=0, zorder=3)
            line, = ax.plot(group.x, group.cotness, color=color, linewidth=1.42, alpha=.7, zorder=2)
            expected = group[["x", "cotness"]].to_numpy()
            if not np.array_equal(np.asarray(scatter.get_offsets()), expected) or not np.array_equal(line.get_xydata(), expected):
                raise ValueError("Plot artists changed measured coordinates.")
        plotted[arm["arm"]] = len(part)
        label_top = plot_top + plot_height + .08
        for label in labels:
            target_center = label["left"] + label["width"] / 2
            if abs(target_center - label["plot_center"]) > .06:
                fig.add_artist(Line2D([label["plot_center"] / WIDTH, target_center / WIDTH],
                                      [1 - (plot_top + plot_height + .015) / height,
                                       1 - (label_top - .025) / height],
                                      transform=fig.transFigure, color="#b5b5b5", linewidth=.4))
            text(label["left"], label_top, label["wrapped"], fontsize=7.7,
                 color=INK[label["segment"]["source"]], linespacing=1.17)
        top += row_height
    plot_band_bottom = top - .20
    y_label = fig.text(.15 / WIDTH, 1 - (plot_band_top + plot_band_bottom) / (2 * height),
                      "CoTness", rotation=90, va="center", ha="center", fontsize=9)
    text_artists.append(("CoTness", y_label))
    text(LEFT, top, "Input excerpts", fontsize=8.2, fontweight="bold", color=DARK)
    label_top = top + .22
    excerpt_top = label_top + excerpt_plan["label_height"] + .075
    cursor = LEFT
    for segment, choice in zip(shared, excerpt_plan["choices"]):
        text(cursor, label_top, choice["label"], fontsize=LABEL_PT,
             fontweight="bold", color=INK[segment["source"]], linespacing=1.17)
        text(cursor, excerpt_top, choice["wrapped"], fontsize=EXCERPT_PT,
             rotation=angle, color=INK[segment["source"]], linespacing=1.17)
        cursor += choice["width"] + excerpt_plan["gap"]
    top += excerpt_block_height
    if footer:text(LEFT, top, footer, fontsize=7.4)
    bounds = assert_text_inside(fig, text_artists)
    artifacts = save_formats(fig, out / "forgery-steering-compact", dpi)
    plt.close(fig)
    return {"artifacts": artifacts, "size_inches": [WIDTH, height], "points_per_arm": plotted,
            "text_bounds": bounds, "excerpt_angle_degrees": angle,
            "min_excerpt_font_points": EXCERPT_PT, "shared_excerpt_count": len(shared)}


def render_companion(manifest: dict, out: Path, dpi: int) -> dict:
    """I preserve the full injected text separately when it would crowd the chart."""
    blocks = list(manifest.get("attack_texts", [])) + list(manifest.get("companion_texts", []))
    if not blocks:
        return {"artifacts": {}, "note": "No companion text supplied in manifest."}
    width, page_height, body_size = WIDTH, 11.7, 9.3
    page_top, bottom_margin = .78, .35
    probe = plt.figure(figsize=(width, page_height), dpi=120)
    probe.canvas.draw()
    layout = []
    for block in blocks:
        if block.get("source", "tool") not in COLORS:
            raise ValueError("Unknown companion source category.")
        original = block["text"]
        wrapped = wrap_physical(original, probe.canvas.get_renderer(), PLOT_WIDTH * probe.dpi, body_size)
        _, body_height = measure(probe, wrapped, size=body_size)
        layout.append((block, wrapped, .25 + body_height + .28))
    plt.close(probe)
    if any(h > page_height - page_top - bottom_margin for _, _, h in layout):
        raise ValueError("A companion block is longer than one page; split it at a recorded paragraph boundary.")
    pages, current, used = [], [], page_top
    for block in layout:
        if current and used + block[2] > page_height - bottom_margin:
            pages.append(current)
            current, used = [], page_top
        current.append(block)
        used += block[2]
    if current:
        pages.append(current)
    pdf_path = out / "forgery-steering-text.pdf"
    artifacts = []
    with PdfPages(pdf_path) as pdf:
        for number, page in enumerate(pages, 1):
            actual_height = min(page_height, page_top + sum(b[2] for b in page) + bottom_margin)
            fig = plt.figure(figsize=(width, actual_height), dpi=120)
            artists = []
            companion_title = ("Injected text and observed responses" if manifest.get("companion_texts")
                               else "Injected request and forged reasoning")
            title = fig.text(LEFT / width, 1 - .18 / actual_height, companion_title,
                             fontsize=11.5, fontweight="bold", va="top")
            artists.append(("Companion title", title))
            caption = fig.text(LEFT / width, 1 - .44 / actual_height,
                               ("Verbatim research inputs and outputs" if manifest.get("companion_texts") else "Verbatim research inputs")
                               + " · companion to the four-arm role-probe figure",
                               fontsize=8.2, va="top")
            artists.append(("Companion subtitle", caption))
            top = page_top
            for block, wrapped, block_height in page:
                label = fig.text(LEFT / width, 1 - top / actual_height, block["title"],
                                 fontsize=9.4, fontweight="bold", va="top", color=DARK)
                body = fig.text(LEFT / width, 1 - (top + .25) / actual_height, wrapped,
                                fontsize=body_size, va="top", linespacing=1.17,
                                color=INK[block.get("source", "tool")])
                artists.extend([(block["title"], label), (block["title"] + " text", body)])
                top += block_height
            fig.text((width - RIGHT) / width, .15 / actual_height, str(number), fontsize=8, ha="right")
            assert_text_inside(fig, artists)
            pdf.savefig(fig, facecolor="white")
            page_paths = {}
            for ext in ("png", "svg"):
                path = out / f"forgery-steering-text-{number:02d}.{ext}"
                fig.savefig(path, dpi=dpi, facecolor="white")
                page_paths[ext] = {"path": str(path), "sha256": sha(path)}
            artifacts.append(page_paths)
            plt.close(fig)
    return {"pdf": {"path": str(pdf_path), "sha256": sha(pdf_path)},
            "pages": artifacts, "verbatim_blocks": len(blocks), "body_font_points": body_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--angle", type=int, default=35, choices=[0, 35])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--layout", choices=["compact", "detailed", "both"], default="compact")
    parser.add_argument("--technical-footer", action="store_true")
    args = parser.parse_args()
    data = args.data.resolve()
    rows, manifest = load_inputs(data)
    manifest["show_technical_footer"] = args.technical_footer
    if args.validate_only:
        print(json.dumps({"valid": True, "n_rows": len(rows), "arms": list(ARMS)}))
        return
    setup_style()
    out = (args.out or HERE / "figures/figure8-steering" / data.name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "inputs": {name: {"path": str(data / name), "sha256": sha(data / name)}
                   for name in ("displayed-rows.csv", "display-manifest.json")},
        "renderer_sha256": sha(Path(__file__)), "source_contract": str(data.parent / "source-contract.md"),
        "technical_footer_visible": args.technical_footer,
        "font": "TeX Gyre Termes", "point_line_palette": COLORS, "text_palette": INK,
        "smoothing": "Normalized trailing EWMA alpha=0.5, verified independently within each displayed arm/segment.",
        "shading": "None; source Figure8 has no shaded attack spans or uncertainty bands.",
        "x_scale": {"shared_across_arms": True, "minimum": float(rows.x.min()),
                    "maximum": float(rows.x.max()),
                    "note": "Input spans align in the central pair; each arm retains its own continuation and blank tail."},
        "main": (render_compact(rows, manifest, out, args.angle, args.dpi)
                 if args.layout in ("compact", "both")
                 else render_main(rows, manifest, out, args.angle, args.dpi)),
        "companion": render_companion(manifest, out, args.dpi),
    }
    if args.layout == "both":
        result["detailed"] = render_main(rows, manifest, out, args.angle, args.dpi)
    provenance = out / "render-provenance.json"
    provenance.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "main": result["main"]["size_inches"],
                      "points_per_arm": result["main"]["points_per_arm"]}, indent=2))


if __name__ == "__main__":
    main()
