"""I preserve the authors' palette, typography and token-plot conventions."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'outputs/20260911T030050Z'
AUTHORS = Path('/Users/hananather/Desktop/MATS 12.0/prompt-injection-as-role-confusion')
STYLE_SOURCE = AUTHORS / 'r-utils/plots.r'
PALETTE_SOURCE = AUTHORS / 'experiments/role-analysis/04-tomato-probe-results.ipynb'
FONT_DIR = Path('/usr/local/texlive/2019/texmf-dist/fonts/opentype/public/tex-gyre')
for variant in ['regular', 'bold', 'italic', 'bolditalic']:
    font_manager.fontManager.addfont(FONT_DIR / f'texgyretermes-{variant}.otf')
plt.rcParams.update({'font.family': 'TeX Gyre Termes', 'font.size': 9,
                     'text.usetex': False, 'text.parse_math': False, 'pdf.fonttype': 42,
                     'axes.linewidth': .5, 'savefig.facecolor': 'white'})
module_path = HERE.parents[2] / 'appendix_e_figures.py'
spec = importlib.util.spec_from_file_location('saved_appendix_e_figures', module_path)
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)

def style_provenance():
    return {'point_colors': paper.POINT_COLORS, 'text_colors': paper.TEXT_COLORS,
            'font_family': 'TeX Gyre Termes',
            'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in [STYLE_SOURCE, PALETTE_SOURCE]],
            'semantics': 'I color points by original source passage, not by measured target role.',
            'rendering_boundary': 'I use Matplotlib with the exact source palette and font; layout adapts to the comparison and is not pixel-identical to R/ggplot2.'}

def gardening():
    out = HERE / 'paper-style'
    out.mkdir(exist_ok=True)
    four = pd.read_csv(SOURCE/'appendix-e-L12/figure-20-22-displayed-rows.csv', keep_default_na=False)
    overview = pd.read_csv(SOURCE/'appendix-e-L12/figure-7-displayed-rows.csv', keep_default_na=False)
    rendering = paper.render_figures(four, overview, out)
    for ext in ['png', 'pdf']:
        shutil.copyfile(out/f'figure-7-cotness-overview.{ext}', HERE/f'my-h100-gardening-overview.{ext}')
    metadata = json.loads((HERE/'provenance.json').read_text())
    metadata['change'] = 'I restored the authors palette, exact font, point styling, token-text labels, axes and canvas dimensions. Measured data are unchanged.'
    metadata['paper_style'] = style_provenance()
    metadata['rendering'] = rendering
    (HERE/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')

def offsets():
    out = HERE/'probe-offset-illustration'
    rows = pd.read_csv(out/'plotted-probabilities.csv', keep_default_na=False, float_precision='round_trip')
    display = pd.read_csv(SOURCE/'appendix-e-L12/figure-7-displayed-rows.csv', keep_default_na=False)
    display = display[display.prompt_key == 'everything_in_user_tags'].sort_values('display_token_ix')
    annotations = paper.segment_annotations(display, overview=True)
    # I shorten token excerpts for narrower panels without altering any scored rows.
    labels = []
    for seg_ix, group in display.groupby('seg_ix', sort=True):
        role = group.base_message_type.iloc[0]
        width = 4 if role == 'user' else 7 if role == 'cot' else 10
        labels.append(paper.wrap_tokens(group.token.tolist(), width, max_lines=2, ellipsis='…'))
    for fraction in [.01, .05]:
        fig, axes = plt.subplots(4, 3, figsize=(9, 5.8), sharex=True, sharey=True)
        for edit_ix, edited_role in enumerate(['user', 'tool']):
            for col_ix, sign in enumerate([-1, 0, 1]):
                part = rows[(rows.offset_fraction == fraction) & (rows.edited_role == edited_role) & (rows.sign == sign)].sort_values('display_token_ix')
                for metric_ix, target in enumerate(['user', 'tool']):
                    row_ix = edit_ix*2 + metric_ix
                    ax = axes[row_ix, col_ix]
                    for role in ['user', 'cot', 'assistant']:
                        group = part[part.base_message_type == role]
                        ax.scatter(group.display_token_ix, group['p_'+target], s=1.1,
                                   linewidths=0, color=paper.POINT_COLORS[role], alpha=.9)
                    ax.set_ylim(-.02,1.02); ax.set_xlim(1,512)
                    ax.set_yticks([0,1], labels=['0%','100%'])
                    ax.set_xticks([a['start_ix'] for a in annotations])
                    if row_ix in [1,3]:
                        ax.tick_params(axis='x', labelbottom=True)
                        ax.set_xticklabels(labels, ha='left', fontsize=6)
                        for label,a in zip(ax.get_xticklabels(), annotations):label.set_color(paper.TEXT_COLORS[a['role']])
                    else:
                        ax.tick_params(axis='x', labelbottom=False)
                    ax.tick_params(axis='both', width=.4, length=2, pad=2)
                    ax.tick_params(axis='y', labelsize=7)
                    ax.tick_params(axis='x', labelsize=6)
                    ax.grid(axis='both', color='#d9d9d9', linewidth=.3)
                    ax.set_axisbelow(True)
                    for spine in ax.spines.values():spine.set_color('#333333')
                    if col_ix == 0:ax.set_ylabel(target.title()+'ness', fontsize=8, fontweight='bold', labelpad=6)
                    if metric_ix == 0:
                        action = 'Unmodified' if sign == 0 else ('Subtract ' if sign < 0 else 'Add ')+edited_role.title()+' direction'
                        ax.set_title(action, fontsize=8, fontweight='bold', pad=3)
        fig.text(.12,.977, f'My saved gardening activations: ±{fraction:.0%} offsets along User and Tool probe directions',
                 fontsize=9, fontweight='bold')
        fig.text(.12,.949, 'Offline probe-response illustration · gpt-oss-20b · layer 12 · five-role probe · all user tags', fontsize=8)
        fig.text(.12,.069, 'Colors identify the original passages: User, CoT and Assistant, following the paper. All panels retain the same 512 tokens.', fontsize=7.5)
        fig.text(.12,.043, 'I edit saved activations and score them with the same probe. These plots do not show downstream model or behavior changes.', fontsize=7.5)
        fig.text(.12,.017, '“Subtract” means an opposite offset, not erasing a role. Both directions use the same offset norm; strengths are illustrative.', fontsize=7.5)
        fig.subplots_adjust(left=.12, right=.95, top=.89, bottom=.15, wspace=.18, hspace=.56)
        for ext in ['png','pdf']:fig.savefig(out/f'probe-offsets-{int(fraction*100)}pct.{ext}', dpi=300)
        plt.close(fig)
    meta=json.loads((out/'provenance.json').read_text())
    meta['paper_style']=style_provenance()
    (out/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__ == '__main__':
    gardening()
    offsets()
    print('I rendered the gardening overview and both offset comparisons in the paper style.')
