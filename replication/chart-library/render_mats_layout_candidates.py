"""I compare readable text layouts using one fixed set of measured MATS scores.

I preserve every plotted value. Two candidates show complete opening sentences;
the third includes every word in all six passages, with Markdown marks rendered
as plain text. These candidates do not replace the current report figure.
"""
from __future__ import annotations

import json
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np

from render_mats_six_passage import (
    HERE, CONDITIONS, TITLES, COLORS, TEXT, ROLE_ORDER, digest,
    passage_labels, read_candidate, setup_style,
)

DATA = HERE / "data/mats-hanan-dialogue-v2"
OUT = HERE / "figures/mats-hanan-layouts"
TITLE = "Reasoning-role scores across dialogue formats"
ROLE_LABELS = ["User 1", "CoT 1", "Assistant 1", "User 2", "CoT 2", "Assistant 2"]


def plain_markdown(text: str) -> str:
    return text.replace("**", "").replace("*", "")


def wrap_physical(text: str, renderer, width: float, font_size: float) -> str:
    """I wrap complete words using their actual Termes glyph widths."""
    font = FontProperties(family="TeX Gyre Termes", size=font_size)
    lines = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = (current + " " + word).strip()
            if current and renderer.get_text_width_height_descent(candidate, font, False)[0] > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    result = "\n".join(lines)
    assert result.split() == text.split(), "Wrapping must preserve every word."
    return result


def add_plots(fig, rows, base, *, bottom: float, top: float):
    labels = passage_labels(base)
    height = (top - bottom - .095) / 3
    axes = []
    for i, (condition, title) in enumerate(zip(CONDITIONS, TITLES)):
        ax = fig.add_axes([.075, top - height - i * (height + .0475), .90, height])
        axes.append(ax)
        part = rows[rows.prompt_key == condition].sort_values("display_token_ix")
        ax.grid(color="#d9d9d9", linewidth=.3)
        ax.set_axisbelow(True)
        ax.set_xticks([item[0] for item in labels], labels=[])
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.tick_params(width=.4, length=2, pad=3)
        for role in COLORS:
            group = part[part.base_message_type == role]
            artist = ax.scatter(group.display_token_ix, group.prob, s=1.1,
                                linewidths=0, color=COLORS[role], alpha=.9)
            assert np.array_equal(np.asarray(artist.get_offsets()),
                                  group[["display_token_ix", "prob"]].to_numpy())
        ax.set_ylim(-.02, 1.02)
        ax.set_xlim(1, len(base))
        ax.set_yticks([0, 1], labels=["0%", "100%"])
        ax.set_title(title, fontsize=9, fontweight="bold", pad=3)
    fig.text(.026, (top + bottom) / 2, "CoTness", rotation=90,
             va="center", fontsize=9, fontweight="bold")
    return axes, labels


def save(fig, name, metadata, text_artists, bottom_crop_inches=0.0):
    destination = OUT / name
    destination.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extents = []
    for label, artist in text_artists:
        bbox = artist.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        bbox = Bbox.from_bounds(bbox.x0, bbox.y0 - bottom_crop_inches, bbox.width, bbox.height)
        assert bbox.x0 >= 0 and bbox.y0 >= 0, f"{label} extends outside the page."
        assert bbox.x1 <= fig.get_figwidth() and bbox.y1 <= fig.get_figheight() - bottom_crop_inches, f"{label} extends outside the page."
        extents.append({"label": label, "bounds_inches": list(bbox.bounds),
                        "font_size_points": artist.get_fontsize()})
    for suffix in ["png", "pdf", "svg"]:
        crop = Bbox.from_bounds(0, bottom_crop_inches, fig.get_figwidth(),
                                fig.get_figheight() - bottom_crop_inches)
        fig.savefig(destination / f"{name}.{suffix}", dpi=600, bbox_inches=crop, pad_inches=0)
    metadata.update({
        "title": TITLE, "n_points_per_panel": 578,
        "points_exactly_match_csv": True,
        "source_sha256": digest(DATA / "displayed-rows.csv"),
        "renderer_sha256": digest(Path(__file__)),
        "palette": COLORS, "text_palette": TEXT,
        "font": "TeX Gyre Termes", "size_inches": [fig.get_figwidth(), fig.get_figheight() - bottom_crop_inches],
        "bottom_whitespace_removed_inches": bottom_crop_inches,
        "text_extents": extents,
        "min_transcript_font_size_points": min(a.get_fontsize() for _, a in text_artists),
        "artifact_sha256": {p.name: digest(p) for p in sorted(destination.glob(f"{name}.*"))},
    })
    (destination / "layout-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plt.close(fig)
    return metadata


def diagonal(rows, base, messages, angle):
    fig = plt.figure(figsize=(11.7, 8.3))
    axes, labels = add_plots(fig, rows, base, bottom=.54, top=.885)
    fig.text(.075, .97, TITLE, fontsize=13, fontweight="bold", va="top")
    fig.text(.075, .937, "GPT-OSS-20B · layer 12 · 578 matched tokens · an original MATS conversation",
             fontsize=10, va="top")
    excerpts = [
        messages[1].split(". ")[0] + ".",
        messages[2].split(". ")[0] + ".",
        plain_markdown(messages[3].split("\n\n")[0]),
        messages[4].rsplit(" What's", 1)[0],
        messages[5].split(". ")[0] + ".",
        plain_markdown(messages[6].split(". ")[0] + "."),
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    col_width = .90 / 6
    font_size = 8.5
    text_artists = []
    for i, (role, label, excerpt) in enumerate(zip(ROLE_ORDER, ROLE_LABELS, excerpts)):
        x = .075 + col_width * i
        start, _, _, _, _, count = labels[i]
        midpoint = axes[-1].transData.transform((start + (count - 1) / 2, 0))[0] / fig.bbox.width
        fig.add_artist(Line2D([midpoint, x + .035], [.53, .495], color="#b5b5b5", linewidth=.5,
                             transform=fig.transFigure))
        fig.text(x, .49, label, fontsize=9, fontweight="bold", va="top")
        wrapped = wrap_physical(excerpt, renderer, fig.bbox.width * col_width * .70, font_size)
        artist = fig.text(x, .46, wrapped, fontsize=font_size, color=TEXT[role],
                          rotation=angle, ha="left", va="top", linespacing=1.25)
        text_artists.append((label, artist))
    fig.text(.075, .033,
             "Complete opening sentences or title; User 2 includes the mood-ring question. Colors identify original passage roles.",
             fontsize=8)
    return save(fig, f"diagonal-{angle}", {
        "layout": "Six complete excerpt blocks rotated and connected to their passages.",
        "angle_degrees": angle, "excerpt_text": excerpts,
        "full_transcript": False,
        "disclosure": "Complete opening sentences or title; User 2 includes the mood-ring question.",
    }, text_artists)


def six_full_columns(rows, base, messages):
    fig = plt.figure(figsize=(11.7, 12.0))
    axes, labels = add_plots(fig, rows, base, bottom=.685, top=.918)
    fig.text(.075, .978, TITLE, fontsize=13, fontweight="bold", va="top")
    fig.text(.075, .953, "GPT-OSS-20B · layer 12 · 578 matched tokens · the complete six-passage MATS exchange",
             fontsize=10, va="top")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    col_width = .90 / 6
    text_artists = []
    full_text = []
    font_size = 8.3
    for i, (role, label, original) in enumerate(zip(ROLE_ORDER, ROLE_LABELS, messages[1:])):
        x = .075 + col_width * i
        start, _, _, _, _, count = labels[i]
        midpoint = axes[-1].transData.transform((start + (count - 1) / 2, 0))[0] / fig.bbox.width
        fig.add_artist(Line2D([midpoint, x + col_width / 2], [.681, .660], color="#b5b5b5", linewidth=.5,
                             transform=fig.transFigure))
        fig.text(x, .65, label, fontsize=9, fontweight="bold", va="top")
        plain = plain_markdown(original)
        full_text.append(plain)
        wrapped = wrap_physical(plain, renderer, fig.bbox.width * (col_width - .014), font_size)
        artist = fig.text(x, .63, wrapped, fontsize=font_size, color=TEXT[role],
                          ha="left", va="top", linespacing=1.15)
        text_artists.append((label, artist))
    fig.text(.075, .027,
             "All six passages are printed in full. The plots show the same matched display subset in all three formats.",
             fontsize=8)
    return save(fig, "six-full-columns", {
        "layout": "Six columns, one per complete passage, connected to the corresponding plot interval.",
        "full_transcript": True, "transcript_text": full_text,
        "text_processing": "Markdown emphasis markers removed for presentation; every original word retained.",
    }, text_artists)


def compact_diagonal(rows, base, messages):
    fig = plt.figure(figsize=(8.3, 6.0))
    axes, labels = add_plots(fig, rows, base, bottom=.515, top=.865)
    next(text for text in fig.texts if text.get_text() == "CoTness").set_x(.012)
    fig.text(.075, .966, TITLE, fontsize=11, fontweight="bold", va="top")
    fig.text(.075, .922, "GPT-OSS-20B · layer 12 · 578 matched tokens · an original MATS conversation",
             fontsize=8.3, va="top")
    excerpts = json.loads((OUT / "diagonal-35/layout-provenance.json").read_text())["excerpt_text"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    font_size = 7.2
    options = []
    for excerpt in excerpts:
        choices = []
        for width in np.arange(45, 136, 1):
            wrapped = wrap_physical(excerpt, renderer, float(width), font_size)
            artist = fig.text(0, .455, wrapped, fontsize=font_size, rotation=35,
                              ha="left", va="top", linespacing=1.17)
            bounds = artist.get_window_extent(renderer)
            choices.append((bounds.width, bounds.height, wrapped))
            artist.remove()
        options.append(min(choices, key=lambda item: (item[0], item[1])))
    left, right = fig.bbox.width * .075, fig.bbox.width * .975
    gaps = (right - left - sum(item[0] for item in options)) / 5
    assert gaps >= 5, f"Only {gaps} pixels remain between complete excerpt blocks."
    text_artists = []
    x_px = left
    for i, (role, label, option) in enumerate(zip(ROLE_ORDER, ROLE_LABELS, options)):
        width, _, wrapped = option
        x = x_px / fig.bbox.width
        start, _, _, _, _, count = labels[i]
        midpoint = axes[-1].transData.transform((start + (count - 1) / 2, 0))[0] / fig.bbox.width
        fig.add_artist(Line2D([midpoint, x + width / fig.bbox.width / 2], [.505, .480],
                             color="#b5b5b5", linewidth=.5, transform=fig.transFigure))
        fig.text(x, .477, label, fontsize=8, fontweight="bold", va="top")
        artist = fig.text(x, .45, wrapped, fontsize=font_size, rotation=35,
                          color=TEXT[role], ha="left", va="top", linespacing=1.17)
        text_artists.append((label, artist))
        x_px += width + gaps
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [a.get_window_extent(renderer) for _, a in text_artists]
    assert all(boxes[i].x1 + 4.9 <= boxes[i+1].x0 for i in range(5))
    lowest_inches = min(b.y0 for b in boxes) / fig.dpi
    footer_inches = lowest_inches - .22
    footer = "Complete opening excerpts; full conversation provided separately."
    fig.text(.075, footer_inches / fig.get_figheight(), footer, fontsize=7.2, va="baseline")
    bottom_crop = max(0, footer_inches - .13)
    return save(fig, "diagonal-35-compact", {
        "layout": "Compact six complete excerpt blocks, optimally wrapped and rotated 35 degrees.",
        "angle_degrees": 35, "excerpt_text": excerpts, "full_transcript": False,
        "disclosure": footer,
        "min_adjacent_text_gap_inches": gaps / fig.dpi,
        "wrapping_choice": "Minimum measured rotated width at a fixed 7.2-point font; every excerpt word retained.",
    }, text_artists, bottom_crop_inches=bottom_crop)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-only", action="store_true")
    args = parser.parse_args()
    setup_style()
    rows, base, _ = read_candidate("mats-hanan-dialogue-v2", data_root=HERE / "data")
    messages = json.loads((DATA / "original-messages.json").read_text())
    if args.compact_only:
        summaries = [compact_diagonal(rows, base, messages)]
    else:
        summaries = [diagonal(rows, base, messages, angle) for angle in [35, 45]]
        summaries.append(six_full_columns(rows, base, messages))
    print(json.dumps([{k: item[k] for k in ["layout", "size_inches", "min_transcript_font_size_points"]}
                      for item in summaries], indent=2))


if __name__ == "__main__":
    main()
