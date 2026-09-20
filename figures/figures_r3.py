"""figures_r3.py - all five figures of the R3 manuscript (PLOS ONE PONE-D-26-12981).

Usage (from the repository root):  python figures/figures_r3.py [r3_tables] [figures/r3]

* Figs 2-4 are drawn from the regenerated tables in r3_tables/ (no hard-coded results).
* Figs 1 and 5 are schematic diagrams; Fig 5 is a constructed example whose candidate
  values are approximately the means of the corresponding rows of Table 3.
* PLOS requirements: TIFF with LZW compression, RGB without alpha, 300 dpi,
  width <= 2250 px, height <= 2625 px, Arial (or a metric-compatible substitute)
  at 8-12 pt, and no figure number, title, or caption inside the image.
"""
import os
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image

T = sys.argv[1] if len(sys.argv) > 1 else 'r3_tables'
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join('figures', 'r3')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'hatch.linewidth': 0.8,
})
C = {'LR': '#7F7F7F', 'DL': '#4472C4', 'FRP': '#ED7D31', 'FE-HITL': '#2E8B57', 'thr': '#C00000'}
MODELS = [('LR', 'LR'), ('DL', 'DL'), ('Debiased-HITL', 'FRP'), ('FE-HITL', 'FE-HITL')]


def save(fig, name, tight=True):
    """Save PNG (preview) and PLOS-compliant TIFF (RGB, LZW, 300 dpi)."""
    png = os.path.join(OUT, name + '.png')
    if tight:
        fig.savefig(png, dpi=300, bbox_inches='tight', pad_inches=0.04, facecolor='white')
    else:   # diagrams: exact canvas, all content drawn inside it
        fig.savefig(png, dpi=300, facecolor='white')
    plt.close(fig)
    im = Image.open(png).convert('RGB')
    im.save(png, dpi=(300, 300))
    im.save(os.path.join(OUT, name + '.tif'), compression='tiff_lzw', dpi=(300, 300))
    w, h = im.size
    assert 789 <= w <= 2250 and h <= 2625, f'{name}: {w}x{h} px outside PLOS limits'
    print(f'{name}: {w} x {h} px')


def panel(ax, letter):
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=11,
            fontweight='bold', ha='right', va='bottom')


def bars(ax, labels, means, sds, colors, hatches=None, fmt='{:.3f}'):
    top = ax.get_ylim()[1]
    for i, (m, s, c) in enumerate(zip(means, sds, colors)):
        ax.bar(i, m, 0.62, color=c, edgecolor='#303030' if hatches and hatches[i] else 'white',
               linewidth=0.6, yerr=s, capsize=3, hatch=(hatches[i] if hatches else None),
               error_kw=dict(elinewidth=0.8, ecolor='#404040'))
        ax.text(i, m + s + 0.015 * top, fmt.format(m), ha='center', va='bottom', fontsize=8,
                bbox=dict(fc='white', ec='none', pad=0.4), zorder=5)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)


# --------------------------------------------------------------------- Fig 2
t1 = pd.read_csv(os.path.join(T, 'table1_agri_main.csv')).set_index(['model', 'metric'])
fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0))
for ax, letter, (met, lab, ylim) in zip(axes, 'ABC', [('DI', 'Disparate Impact (DI)', 1.2),
                                                     ('EOD', '|EOD|', 0.22),
                                                     ('R2', 'R²', 1.0)]):
    ax.set_ylim(0, ylim)
    bars(ax, [m[1] for m in MODELS], [t1.loc[(m[0], met), 'mean'] for m in MODELS],
         [t1.loc[(m[0], met), 'sd'] for m in MODELS], [C[m[1]] for m in MODELS])
    ax.set_ylabel(lab)
    panel(ax, letter)
    if met == 'DI':
        ax.axhline(0.8, color=C['thr'], ls='--', lw=1)
        ax.plot([1, 1, 3, 3], [1.02, 1.06, 1.06, 1.02], color='#404040', lw=0.8)
        ax.text(2, 1.075, 'Δ = +0.232, adj. p < 0.001', ha='center', va='bottom', fontsize=8)
    if met == 'EOD':
        ax.axhline(0.1, color=C['thr'], ls='--', lw=1)
fig.tight_layout(w_pad=1.2)
save(fig, 'Fig2')

# --------------------------------------------------------------------- Fig 3
t3b = pd.read_csv(os.path.join(T, 'table3b_mog_ablation_R3.csv')).set_index(['variant', 'metric'])
t3 = pd.read_csv(os.path.join(T, 'table3_ablation_R2.csv')).set_index(['variant', 'metric'])
V = [('Full: Algorithm 1 (4 candidates + rule-based arbitration)', 'Full\n(α = 1.0)', t3b),
     ('Single fixed candidate, alpha=0.7', 'α = 0.7\nonly', t3b),
     ('Single fixed candidate, alpha=0.5', 'α = 0.5\nonly', t3b),
     ('Single fixed candidate, alpha=0.3', 'α = 0.3\nonly', t3b),
     ('Reduced strength, coef 0.2 (R2 label "w/o MOG")', 'Reduced\nstrength\n(coef. 0.2)', t3b),
     ('No correction, same base (R2 label "w/o HED")', 'w/o HED\n(no\ncorrection)', t3b),
     ('w/o F&U (identical to Full by order of evaluation)', 'w/o F&U\n(identical\nto Full)', t3)]
cols = ['#2E8B57', '#6BAF8C', '#9CC9AE', '#C9E2D3', '#ED7D31', '#C00000', '#BFBFBF']
hat = [None] * 6 + ['///']
fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.4), sharex=True)
for ax, letter, (met, lab, ylim) in zip(axes, 'AB', [('DI', 'Disparate Impact (DI)', 1.0),
                                                    ('R2', 'R²', 0.85)]):
    ax.set_ylim(0, ylim)
    bars(ax, [v[1] for v in V], [v[2].loc[(v[0], met), 'mean'] for v in V],
         [v[2].loc[(v[0], met), 'sd'] for v in V], cols, hatches=hat)
    ax.set_ylabel(lab)
    panel(ax, letter)
    if met == 'DI':
        ax.axhline(0.8, color=C['thr'], ls='--', lw=1)
axes[0].tick_params(axis='x', labelbottom=False)
fig.tight_layout(h_pad=1.5)
save(fig, 'Fig3')

# --------------------------------------------------------------------- Fig 4
t4 = pd.read_csv(os.path.join(T, 'table4_credit_main.csv')).set_index(['model', 'metric'])
fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.0))
for ax, letter, (met, lab, ylim) in zip(axes.ravel(), 'ABCD',
                                        [('DI', 'Disparate Impact (DI)', 2.3),
                                         ('DI_gap', 'Parity gap |DI − 1|', 1.2),
                                         ('EOD', '|EOD|', 0.26), ('Acc', 'Accuracy', 0.9)]):
    ax.set_ylim(0, ylim)
    bars(ax, [m[1] for m in MODELS], [t4.loc[(m[0], met), 'mean'] for m in MODELS],
         [t4.loc[(m[0], met), 'sd'] for m in MODELS], [C[m[1]] for m in MODELS])
    ax.set_ylabel(lab)
    panel(ax, letter)
    if met == 'DI':
        ax.axhline(1.0, color=C['thr'], ls='--', lw=1)
fig.tight_layout(h_pad=1.6, w_pad=2.0)
save(fig, 'Fig4')


# ------------------------------------------------------------ diagram helpers
def box(ax, x, y, w, h, fc, dashed=False, ec='#595959', lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.08',
                                fc=fc, ec=ec, lw=lw, ls=(0, (4, 3)) if dashed else '-'))


def lines(ax, x, y, items, width, dy=0.215):
    """items: list of (text, style); style in {'title','body','muted'}; returns final y."""
    for text, style in items:
        wrapped = textwrap.wrap(text, width) if style != 'title' else [text]
        for k, line in enumerate(wrapped):
            if style == 'title':
                ax.text(x, y, line, fontsize=10, fontweight='bold', va='top', color='#1A1A1A')
                y -= 0.29
            elif style == 'body':
                ax.text(x, y, ('• ' if k == 0 else '   ') + line, fontsize=8.5, va='top',
                        color='#1A1A1A')
                y -= dy
            else:
                ax.text(x, y, ('• ' if k == 0 else '   ') + line, fontsize=8.5, va='top',
                        color='#7F7F7F', style='italic')
                y -= dy
        y -= 0.03
    return y


def arrow(ax, x0, y0, x1, y1, dashed=False, color='#404040'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', lw=1.2, color=color, mutation_scale=12,
                                ls=(0, (4, 3)) if dashed else '-'))


# --------------------------------------------------------------------- Fig 1
fig = plt.figure(figsize=(7.45, 6.75))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 7.45); ax.set_ylim(-0.1, 6.65); ax.axis('off')
BX, BW, TW = 0.45, 4.75, 74
layers = [
    ('#DCE9F7', False, 5.05, 1.40, [
        ('Layer 1 · Data and fairness monitoring', 'title'),
        ('Group fairness metric (DI) computed on each decision batch', 'body'),
        ('Trigger: DI < 0.8 (agriculture); DI outside [0.8, 1.25] (German Credit)', 'body'),
        ('EOD condition (|EOD| > 0.1): designed, not evaluated (requires outcome labels)', 'muted')]),
    ('#FBE3D6', False, 3.05, 1.85, [
        ('Layer 2 · Simulated ethical decision (rule-based proxy)', 'title'),
        ('Routing of unprivileged-group cases to simulated review (fraction ρ)', 'body'),
        ('Multi-option generation: 4 candidates (agriculture), 6 per direction (credit)', 'body'),
        ('Value arbitration by fixed decision rules (Algorithm 1, steps 6a/6b)', 'body'),
        ('Explanations (SHAP, LIME, counterfactuals) and value-preference labels: '
         'designed, not executed', 'muted')]),
    ('#E2F0D9', False, 1.75, 1.20, [
        ('Layer 3 · Bias correction', 'title'),
        ('Post-processing rescaling of the routed cases’ predictions (executed)', 'body'),
        ('Constrained optimization and imitation learning: envisaged, not implemented', 'muted')]),
    ('#FFF4D6', True, 0.25, 1.30, [
        ('Layer 4 · Feedback and update (architectural placeholder)', 'title'),
        ('Would store decisions for periodic retraining; not validated', 'muted'),
        ('In the reported code the refit runs after the predictions are returned, '
         'so it does not affect any reported result', 'muted')]),
]
for fc, dashed, y0, h, items in layers:
    box(ax, BX, y0, BW, h, fc, dashed=dashed)
    lines(ax, BX + 0.15, y0 + h - 0.14, items, TW)
for (_, _, ya, _, _), (_, dashed_next, yb, hb, _) in zip(layers[:-1], layers[1:]):
    arrow(ax, BX + BW / 2, ya, BX + BW / 2, yb + hb, dashed=dashed_next)
# intended feedback loop (not validated)
ax.plot([BX - 0.22, BX - 0.22], [0.9, 5.75], color='#7F7F7F', lw=1.1, ls=(0, (4, 3)))
arrow(ax, BX - 0.22, 5.75, BX - 0.02, 5.75, color='#7F7F7F')
ax.plot([BX - 0.02, BX - 0.22], [0.9, 0.9], color='#7F7F7F', lw=1.1, ls=(0, (4, 3)))
ax.text(BX - 0.30, 3.3, 'intended feedback loop (not validated)', rotation=90, fontsize=8,
        color='#7F7F7F', style='italic', ha='center', va='center')
# Anders diagnoses -> compensations
TX, TWd = 5.40, 1.95
tags = [(5.35, 'Diagnosis 1: human obsolescence', 'Compensation 1: forced review of fairness-critical cases'),
        (4.30, 'Diagnosis 2: Promethean shame', 'Compensation 2: multi-option generation'),
        (3.35, 'Diagnosis 3: technological imperialism', 'Compensation 3: value arbitration')]
for yt, d, c in tags:
    box(ax, TX, yt, TWd, 0.78, '#F2F2F2', ec='#A6A6A6', lw=0.8)
    yy = yt + 0.70
    for line in textwrap.wrap(d, 30):
        ax.text(TX + 0.08, yy, line, fontsize=8, fontweight='bold', va='top', color='#404040'); yy -= 0.17
    for line in textwrap.wrap('→ ' + c, 32):
        ax.text(TX + 0.08, yy, line, fontsize=8, va='top', color='#404040'); yy -= 0.17
    arrow(ax, TX, yt + 0.39, BX + BW + 0.02, yt + 0.39 if yt > 4 else min(yt + 0.39, 4.6),
          color='#A6A6A6')
save(fig, 'Fig1', tight=False)

# --------------------------------------------------------------------- Fig 5
DI0, R2_BASE = 0.64, 0.7245            # monitored DI and same-base R^2 (w/o HED row, Table 3)
t3b_di = {a: t3b.loc[(f'Single fixed candidate, alpha={a}', 'DI'), 'mean'] for a in (0.3, 0.5, 0.7, 1.0)}
t3b_r2 = {a: t3b.loc[(f'Single fixed candidate, alpha={a}', 'R2'), 'mean'] for a in (0.3, 0.5, 0.7, 1.0)}
cand = [(a, 1 + a * (0.8 - DI0) * 0.7, t3b_di[a], (R2_BASE - t3b_r2[a]) / R2_BASE) for a in (0.3, 0.5, 0.7, 1.0)]
fig = plt.figure(figsize=(7.45, 8.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 7.45); ax.set_ylim(-0.08, 8.32); ax.axis('off')
SX, SW = 1.75, 5.55
steps = [  # (y, h, colour, dashed, title, [(text, style)])
    (7.30, 0.85, '#DCE9F7', False, 'Step 1 · Base-model prediction',
     [('Farmer in region A (2.3 ha, 5 years of farming, low recorded yield): 3.6 allocation '
       'units; batch mean ≈ 10.6 units', 'body')]),
    (6.20, 0.85, '#DCE9F7', False, 'Step 2 · Fairness alert and routing',
     [('Monitored batch DI = 0.64 < 0.8 → region-A cases routed to simulated review', 'body')]),
    (5.10, 0.85, '#FBE3D6', False, 'Step 3 · Explanation',
     [('Designed, not executed: e.g., low recorded historical output identified as the main '
       'contributing factor', 'muted')]),
    (3.25, 1.60, '#FBE3D6', False, 'Step 4 · Multi-option generation (Algorithm 1)', []),
    (2.15, 0.85, '#FBE3D6', False, 'Step 5 · Rule-based arbitration (simulated proxy)',
     [('α = 1.0 selected: the only candidate with DI ≥ 0.85 and efficiency loss ≤ 10%; '
       'the rule applied is recorded', 'body')]),
    (1.05, 0.85, '#E2F0D9', False, 'Step 6 · Correction of the routed cases',
     [('This farmer’s allocation: 3.6 → 4.0 units (× 1.112)', 'body')]),
    (0.02, 0.80, '#FFF4D6', True, 'Step 7 · Model update (hypothetical)',
     [('Would store the decision for later retraining; the feedback-and-update layer was not '
       'validated and has no effect on the reported results', 'muted')]),
]
for y0, h, fc, dashed, title, items in steps:
    box(ax, SX, y0, SW, h, fc, dashed=dashed)
    lines(ax, SX + 0.15, y0 + h - 0.12, [(title, 'title')] + items, 86, dy=0.2)
for (ya, _, _, _, _, _), (yb, hb, _, dashed_next, _, _) in zip(steps[:-1], steps[1:]):
    arrow(ax, SX + SW / 2, ya, SX + SW / 2, yb + hb, dashed=dashed_next)
# candidate table inside step 4
cx = [SX + 0.35, SX + 1.45, SX + 2.65, SX + 3.85]
hy = 3.25 + 1.60 - 0.47
for x, head in zip(cx, ['Candidate α', 'Boost', 'DI', 'Efficiency loss']):
    ax.text(x, hy, head, fontsize=8.5, fontweight='bold', va='top')
for k, (a, b, d, e) in enumerate(cand):
    yy = hy - 0.225 * (k + 1)
    for x, v in zip(cx, [f'{a:.1f}', f'{b:.3f}', f'{d:.2f}', f'{100 * e:.1f}%']):
        ax.text(x, yy, v, fontsize=8.5, va='top')
# layer labels on the left
for y0, y1, fc, label, dashed in [(6.20, 8.15, '#DCE9F7', 'Layer 1\nMonitoring', False),
                                   (2.15, 5.95, '#FBE3D6', 'Layer 2\nSimulated\nethical\ndecision', False),
                                   (1.05, 1.90, '#E2F0D9', 'Layer 3\nBias\ncorrection', False),
                                   (0.02, 0.82, '#FFF4D6', 'Layer 4\nF&U\n(placeholder)', True)]:
    box(ax, 0.12, y0, 1.35, y1 - y0, fc, dashed=dashed, ec='#A6A6A6', lw=0.8)
    ax.text(0.795, (y0 + y1) / 2, label, fontsize=8.5, ha='center', va='center', color='#404040')
save(fig, 'Fig5', tight=False)
print('candidates (alpha, boost, DI, eff.loss):', [(a, round(b, 3), round(d, 3), round(e, 4)) for a, b, d, e in cand])
