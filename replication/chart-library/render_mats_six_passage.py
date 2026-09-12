"""I render measured MATS conversations with the earlier six-passage figure design.

Examples::
    python3 render_mats_six_passage.py --candidate mats-01
    python3 render_mats_six_passage.py --all --contact-sheet
    python3 render_mats_six_passage.py --candidate mats-01 --selected

I preserve the original probability values and show the same matched tokens in
all three conditions. Candidate selection remains explicit in every figure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
DATA = HERE / "data/mats-six-passage"
OUT = HERE / "figures/mats-six-passage"
FONT = HERE / "fonts/termes-ttf"
CONDITIONS = ["proper_tags", "basic_no_format", "everything_in_user_tags"]
TITLES = ["A · Correct tags", "B · No tags", "C · All text in one User message"]
COLORS = {"user": "#00a6f4", "cot": "#fd9a00", "assistant": "#00d492"}
TEXT = {"user": "#0084d1", "cot": "#e17100", "assistant": "#009966"}
LABELS = {"user": "User", "cot": "CoT", "assistant": "Assistant"}
ROLE_ORDER = ["user", "cot", "assistant", "user", "cot", "assistant"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_style() -> None:
    for variant in ["regular", "bold", "italic", "bolditalic"]:
        font_manager.fontManager.addfont(FONT / f"texgyretermes-{variant}.ttf")
    plt.rcParams.update({
        "font.family": "TeX Gyre Termes", "font.size": 9,
        "text.usetex": False, "text.parse_math": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "axes.linewidth": .5, "savefig.facecolor": "white",
    })


def wrap_tokens(tokens: list[str], line_width: int, max_lines: int = 2) -> str:
    """I retain the authors' token-preserving, two-line excerpt construction."""
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
    lines = [re.sub(r"<br\s*/?>", "\n", line, flags=re.I)
             .replace("\r\n", "\n").replace("\r", "\n")
             .replace("\n", r"\n") for line in lines]
    return "\n".join(lines) + ("…" if used < len(tokens) else "")


def read_candidate(candidate: str, data_root: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", candidate):
        raise ValueError("Candidate must be a directory identifier such as mats-01.")
    path = data_root / candidate / "displayed-rows.csv"
    rows = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    required = {"prompt_key", "display_token_ix", "base_message_type",
                "base_message_ix", "seg_ix", "token", "token_id", "prob"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    assert set(rows.prompt_key) == set(CONDITIONS), "All three conditions are required."
    assert np.isfinite(rows.prob).all() and rows.prob.between(0, 1).all()
    assert not rows.duplicated(["prompt_key", "display_token_ix"]).any()
    if "target_role" in rows:
        assert set(rows.target_role) == {"cot"}
    if "layer_ix" in rows:
        assert set(rows.layer_ix) == {12}
    base = rows[rows.prompt_key == CONDITIONS[0]].sort_values("display_token_ix")
    n = len(base)
    assert n > 0 and base.display_token_ix.tolist() == list(range(1, n + 1))
    assert sorted(base.seg_ix.unique().tolist()) == list(range(1, 7))
    assert base.seg_ix.tolist() == sorted(base.seg_ix.tolist()), "Passages must be contiguous."
    for seg, role in enumerate(ROLE_ORDER, 1):
        part = base[base.seg_ix == seg]
        assert set(part.base_message_type) == {role}, f"Unexpected role in passage {seg}."
        assert set(part.base_message_ix) == {seg}, "Original passage identities must be 1–6."
    matched = ["display_token_ix", "base_message_type", "base_message_ix",
               "seg_ix", "token", "token_id"]
    for condition in CONDITIONS:
        part = rows[rows.prompt_key == condition].sort_values("display_token_ix")
        assert part[matched].reset_index(drop=True).equals(base[matched].reset_index(drop=True)), \
            f"The displayed tokens differ in {condition}."
    return rows, base, path


def passage_labels(base: pd.DataFrame) -> list[tuple]:
    labels, counts = [], {}
    for seg, group in base.groupby("seg_ix", sort=True):
        role = group.base_message_type.iloc[0]
        counts[role] = counts.get(role, 0) + 1
        labels.append((int(group.display_token_ix.iloc[0]), role, counts[role],
                       group.token.tolist(), int(seg), len(group)))
    return labels


def render(candidate: str, selected: bool = False, *, data_root: Path = DATA,
           output_root: Path = OUT, figure_title: str | None = None,
           figure_subtitle: str | None = None, disclosure: str | None = None,
           excerpt_widths: dict[str, int] | None = None,
           fit_excerpts: bool = False) -> dict:
    rows, base, source = read_candidate(candidate, data_root=data_root)
    labels = passage_labels(base)
    n = len(base)
    out = output_root / candidate
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(6.75, 3.6), sharex=True)
    widths = excerpt_widths or {"user": 6, "cot": 12, "assistant": 20}
    excerpts = [wrap_tokens(tokens, widths[role])
                for _, role, _, tokens, _, _ in labels]
    title_text = figure_title or "GPT-OSS-20B role readout on an original MATS conversation"
    subtitle_text = figure_subtitle or f"Layer 12 · four-role probe · {n} displayed tokens from one conversation"
    footer_text = disclosure or (
        "Illustrative example selected from 12 measured candidates; colors identify original passage roles."
        if selected else
        "One of 12 measured illustrative examples; colors identify original passage roles.")
    plotted = {}
    for ax, condition, title in zip(axes, CONDITIONS, TITLES):
        part = rows[rows.prompt_key == condition].sort_values("display_token_ix")
        ax.grid(color="#d9d9d9", linewidth=.3)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.tick_params(width=.4, length=2, pad=3)
        actual = []
        for role in COLORS:
            group = part[part.base_message_type == role]
            artist = ax.scatter(group.display_token_ix, group.prob, s=1.1,
                                linewidths=0, color=COLORS[role], alpha=.9)
            values = np.asarray(artist.get_offsets())
            expected = group[["display_token_ix", "prob"]].to_numpy()
            assert np.array_equal(values, expected), "Plot values differ from the measured CSV."
            actual.extend(values.tolist())
        plotted[condition] = len(actual)
        ax.set_ylim(-.02, 1.02)
        ax.set_xlim(1, n)
        ax.set_yticks([0, 1], labels=["0%", "100%"])
        ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
        ax.set_xticks([item[0] for item in labels])
        ax.tick_params(axis="x", labelbottom=ax == axes[-1])
        if ax == axes[-1]:
            ax.set_xticklabels(excerpts, ha="left", fontsize=7)
            ax.tick_params(axis="x", pad=14)
            for tick, (_, role, _, _, _, _) in zip(ax.get_xticklabels(), labels):
                tick.set_color(TEXT[role])
            for start, role, number, _, _, _ in labels:
                ax.annotate(f"{LABELS[role]} {number}", (start, 0),
                            xycoords=("data", "axes fraction"), xytext=(0, -2),
                            textcoords="offset points", ha="left", va="top",
                            fontsize=6.7, color="#252525")
    fig.text(.11, .975, title_text,
             fontweight="bold", fontsize=9, va="top")
    fig.text(.11, .937, subtitle_text,
             fontsize=8, va="top")
    fig.text(.018, .52, "CoTness", rotation=90, va="center", fontsize=8, fontweight="bold")
    fig.text(.11, .018, footer_text, fontsize=7)
    fig.subplots_adjust(left=.11, right=.97, top=.84, bottom=.20, hspace=.55)
    fig.canvas.draw()
    if fit_excerpts:
        # I fit excerpt text within its passage without moving or dropping data.
        renderer = fig.canvas.get_renderer()
        ticks = axes[-1].get_xticklabels()
        starts = [axes[-1].transData.transform((item[0], 0))[0] for item in labels]
        ends = starts[1:] + [axes[-1].get_window_extent(renderer).x1]
        for tick, start, end in zip(ticks, starts, ends):
            width = tick.get_window_extent(renderer).width
            available = end - start - 3
            if width > available:
                tick.set_fontsize(tick.get_fontsize() * available / width)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for tick, end in zip(ticks, ends):
            assert tick.get_window_extent(renderer).x1 <= end + .1, "Excerpt overlaps the next passage."
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(out / f"{candidate}.{suffix}", dpi=600)
    plt.close(fig)
    metadata = {
        "candidate": candidate,
        "source": str(source), "source_sha256": digest(source),
        "renderer_sha256": digest(Path(__file__)),
        "n_display_tokens_per_condition": n, "n_points_by_condition": plotted,
        "passages": [
            {"seg_ix": seg, "role": role, "role_number": number,
             "start_display_token_ix": start, "n_tokens": count,
             "excerpt": excerpt}
            for (start, role, number, _, seg, count), excerpt in zip(labels, excerpts)
        ],
        "palette": COLORS, "excerpt_palette": TEXT,
        "font": "TeX Gyre Termes", "font_source": str(FONT),
        "font_embedding": "TrueType in PDF; glyph paths in SVG",
        "size_inches": [6.75, 3.6],
        "figure_title": title_text, "figure_subtitle": subtitle_text,
        "excerpt_widths": widths,
        "excerpt_font_sizes": [tick.get_fontsize() for tick in axes[-1].get_xticklabels()],
        "selection_disclosure": disclosure or ("Illustrative example selected from 12 measured candidates." if selected
                                 else "One of 12 measured illustrative examples."),
        "artist_values_exactly_match_csv": True,
        "artifact_sha256": {path.name: digest(path) for path in sorted(out.glob(f"{candidate}.*"))},
    }
    (out / "render-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def contact_sheet(candidates: list[str]) -> Path:
    """I assemble existing measured figures without changing or smoothing their data."""
    from PIL import Image, ImageDraw, ImageFont
    assert candidates, "No measured candidate figures are available."
    columns, thumb_width, gap, label_height = 3, 1080, 30, 48
    thumb_height = round(thumb_width * 3.6 / 6.75)
    rows = math.ceil(len(candidates) / columns)
    sheet = Image.new("RGB", (columns * (thumb_width + gap) + gap,
                              rows * (thumb_height + label_height + gap) + gap), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(FONT / "texgyretermes-bold.ttf"), 32)
    sources = []
    for ix, candidate in enumerate(candidates):
        path = OUT / candidate / f"{candidate}.png"
        with Image.open(path) as figure:
            thumb = figure.convert("RGB").resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = gap + (ix % columns) * (thumb_width + gap)
        y = gap + (ix // columns) * (thumb_height + label_height + gap)
        draw.text((x, y), candidate, fill="#252525", font=font)
        sheet.paste(thumb, (x, y + label_height))
        sources.append({"candidate": candidate, "path": str(path), "sha256": digest(path)})
    destination = OUT / "contact-sheet.png"
    sheet.save(destination)
    (OUT / "contact-sheet-provenance.json").write_text(json.dumps({
        "purpose": "Visual comparison of measured candidates; no statistical selection claim.",
        "sources": sources, "artifact_sha256": digest(destination),
    }, indent=2) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--candidate", help="Measured candidate directory, for example mats-01.")
    choice.add_argument("--all", action="store_true", help="Render every measured candidate.")
    parser.add_argument("--contact-sheet", action="store_true", help="Also assemble all rendered candidates.")
    parser.add_argument("--selected", action="store_true", help="Copy this candidate's exports to the selected directory.")
    args = parser.parse_args()
    if args.selected and not args.candidate:
        parser.error("--selected requires one --candidate.")
    setup_style()
    candidates = ([args.candidate] if args.candidate else
                  [path.parent.name for path in sorted(DATA.glob("*/displayed-rows.csv"))])
    if not candidates:
        parser.error(f"No measured CSVs found under {DATA}; no figure was generated.")
    summaries = [render(candidate, selected=args.selected) for candidate in candidates]
    if args.selected:
        selected = OUT / "selected"
        selected.mkdir(parents=True, exist_ok=True)
        source = OUT / args.candidate
        for suffix in ["png", "pdf", "svg"]:
            shutil.copy2(source / f"{args.candidate}.{suffix}", selected / f"mats-role-readout.{suffix}")
        shutil.copy2(source / "render-provenance.json", selected / "render-provenance.json")
    if args.contact_sheet:
        all_candidates = [path.parent.name for path in sorted(OUT.glob("mats-*/*.png"))
                          if path.parent.name != "selected"]
        print(contact_sheet(all_candidates))
    print(json.dumps([{"candidate": item["candidate"],
                       "tokens_per_condition": item["n_display_tokens_per_condition"],
                       "output": str(OUT / item["candidate"])} for item in summaries], indent=2))


if __name__ == "__main__":
    main()
