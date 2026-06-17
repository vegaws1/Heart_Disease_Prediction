"""Generate a clean schematic framework figure -> submission/figs/framework.png"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, 'submission', 'figs')
os.makedirs(FIG, exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

C_DATA = '#e8eef7'; C_DEV = '#dcefe0'; C_PROP = '#fde2e2'; C_EXT = '#fff3d6'; C_STAT = '#ece7f6'
EDGE = '#33333a'


def box(x, y, w, h, text, fc, fs=10, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=fc, ec=EDGE, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            weight=weight, wrap=True)


def arrow(x1, y1, x2, y2, style='-|>', lw=1.6, color=EDGE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, lw=lw, color=color))


ax.text(50, 97, 'Memory-aware tempered fractional learning with filtered swarm feature construction',
        ha='center', fontsize=13, weight='bold')

# Data
box(3, 80, 28, 12, 'Heart disease data\n(11 clinical predictors)\n+ target-safe feature engineering', C_DATA, 10, 'bold')

# Development arm header
ax.text(17, 75.5, 'DEVELOPMENT ARM (leakage-safe, 15 repeats)', ha='center', fontsize=10, weight='bold', color='#1b6b2a')
box(3, 60, 28, 13, 'Stratified outer split\n(train 80% / test 20%)\nFit ALL preprocessing on train only', C_DEV, 9)
box(3, 44, 13.5, 12, 'Tune $(\\alpha,\\lambda)$\nTFGD\n(inner ROC-AUC)', C_DEV, 8.5)
box(17.5, 44, 13.5, 12, 'Tune baselines\n(4-fold CV,\nROC-AUC)', C_DEV, 8.5)

# Proposed core
ax.text(50, 75.5, 'PROPOSED CORE', ha='center', fontsize=10, weight='bold', color='#a11')
box(37, 60, 26, 13, 'TFGD logistic learner\nTempered fractional gradient memory\n$\\to$ variance-reduced updates', C_PROP, 9, 'bold')
box(37, 42, 26, 15, 'Nested PSO feature construction\n+ DEGENERATE-FEATURE FILTER & REPAIR\n(rejects $X-X$, constants, duplicates)\nfitness = inner validation ROC-AUC', C_PROP, 8.5, 'bold')

box(37, 27, 26, 11, 'Baselines: ElasticNet-LR, ExtraTrees,\nHistGB, XGBoost, LightGBM, Stacking', C_DEV, 8.5)

# Stats
ax.text(83, 75.5, 'EVIDENCE', ha='center', fontsize=10, weight='bold', color='#5a3a9a')
box(69, 58, 28, 15, 'Uncertainty-aware comparison\n$\\bullet$ Bootstrap 95% CIs\n$\\bullet$ Paired tests + Holm\n$\\bullet$ Bayesian correlated $t$-test (ROPE)', C_STAT, 9)
box(69, 39, 28, 15, 'Multi-institution\nEXTERNAL VALIDATION\nleave-one-institution-out:\nCleveland / Hungary /\nSwitzerland / Long Beach VA', C_EXT, 9, 'bold')
box(69, 22, 28, 12, '10 metrics: AUC, PR-AUC, F1, MCC,\nBrier, calibration, runtime,\ninterpretability', C_STAT, 8.5)

# arrows
arrow(17, 80, 17, 73)
arrow(17, 60, 17, 56)
arrow(31, 50, 37, 50)              # tuning -> proposed core
arrow(50, 60, 50, 57)             # tfgd -> pso
arrow(50, 42, 50, 38)             # pso -> baselines row level
arrow(63, 65, 69, 65)             # core -> evidence
arrow(63, 46, 69, 46)             # external
arrow(31, 67, 37, 67)             # dev split -> tfgd core
ax.text(50, 6, 'All preprocessing, tuning, and PSO construction are fit on training data only; '
        'the held-out test set / held-out institution is touched once.',
        ha='center', fontsize=9, style='italic', color='#444')

fig.savefig(os.path.join(FIG, 'framework.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("wrote framework.png")
