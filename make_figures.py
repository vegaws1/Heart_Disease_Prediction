"""Regenerate ALL figures from results/ into submission/figs/ (consistent set)."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import (roc_curve, precision_recall_curve, auc as sk_auc,
                             confusion_matrix, roc_auc_score)
from sklearn.calibration import calibration_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'submission', 'figs')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
                     'figure.dpi': 150, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False})

MODELS = ['TFGD', 'TFGD_PSO', 'LogRegEN', 'ExtraTrees', 'HistGB', 'XGBoost',
          'LightGBM', 'Stacking']
COL = {m: c for m, c in zip(MODELS, plt.cm.tab10(np.linspace(0, 1, 10)))}
PROP = '#d62728'  # highlight proposed
ALPHA_GRID = [0.3, 0.5, 0.6, 0.7, 0.9]
LAM_GRID = [0.1, 0.3, 0.5, 0.7, 1.0]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


def main():
    rep = pd.read_csv(os.path.join(RES, 'rep_metrics.csv'))
    summ = pd.read_csv(os.path.join(RES, 'summary.csv'))
    rt = pd.read_csv(os.path.join(RES, 'runtime.csv'))
    paired = pd.read_csv(os.path.join(RES, 'paired_tests.csv'))
    curves = np.load(os.path.join(RES, 'curves_lastrep.npz'))
    loss = np.load(os.path.join(RES, 'loss_curves.npz'))
    with open(os.path.join(RES, 'hyperparams.json')) as f: hyper = json.load(f)
    with open(os.path.join(RES, 'pso_features.json')) as f: psof = json.load(f)
    with open(os.path.join(RES, 'perm_importance.json')) as f: perm = json.load(f)
    with open(os.path.join(RES, 'bayesian.json')) as f: bayes = json.load(f)
    ext = pd.read_csv(os.path.join(RES, 'external_metrics.csv'))
    with open(os.path.join(RES, 'external_bootstrap.json')) as f: extb = json.load(f)

    def smean(model, metric):
        r = summ[(summ.model == model) & (summ.metric == metric)].iloc[0]
        return r['mean'], r['ci_lo'], r['ci_hi'], r['sd']

    # ---- 01 summary bars AUC/F1 with bootstrap CI ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, met in zip(axes, ['AUC', 'F1']):
        means = [smean(m, met)[0] for m in MODELS]
        los = [smean(m, met)[0] - smean(m, met)[1] for m in MODELS]
        his = [smean(m, met)[2] - smean(m, met)[0] for m in MODELS]
        colors = [PROP if m in ('TFGD', 'TFGD_PSO') else '#4c72b0' for m in MODELS]
        ax.bar(range(len(MODELS)), means, yerr=[los, his], capsize=4, color=colors)
        ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, rotation=40, ha='right')
        ax.set_ylim(min(means) - 0.03, max(his[i] + means[i] for i in range(len(MODELS))) + 0.01)
        ax.set_ylabel(f'Mean {met}'); ax.set_title(f'{met} (bars: 95% bootstrap CI)')
    save(fig, '01_summary_bars_auc_f1.png')

    # ---- 02 boxplots ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, met in zip(axes, ['AUC', 'F1']):
        data = [rep[rep.model == m][met].values for m in MODELS]
        bp = ax.boxplot(data, labels=MODELS, patch_artist=True, showmeans=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(PROP if MODELS[i] in ('TFGD', 'TFGD_PSO') else '#4c72b0')
            box.set_alpha(0.6)
        ax.set_xticklabels(MODELS, rotation=40, ha='right')
        ax.set_ylabel(met); ax.set_title(f'{met} across 15 repetitions')
    save(fig, '02_boxplots_auc_f1.png')

    # ---- 03 inference latency vs AUC (deployment trade-off) ----
    fr = pd.read_csv(os.path.join(RES, 'fair_runtime.csv'))
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in MODELS:
        x = fr[fr.model == m]['infer_ms_per1k'].iloc[0]
        yv = smean(m, 'AUC')[0]
        ax.scatter(x, yv, s=130, color=PROP if m in ('TFGD', 'TFGD_PSO') else '#4c72b0',
                   edgecolor='k', zorder=3)
        ax.annotate(m, (x, yv), xytext=(6, 4), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_xlabel('Inference latency (ms per 1000 patients, log scale)')
    ax.set_ylabel('Mean ROC-AUC')
    ax.set_title('Discrimination vs deployment cost: the tempered fractional\n'
                 'models cost 2–3 orders of magnitude less at inference')
    save(fig, '03_runtime_vs_auc.png')

    # ---- 04 paired differences ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    best = summ[summ.metric == 'AUC'].sort_values('mean').iloc[-1]['model']
    for ax, met in zip(axes, ['AUC', 'F1']):
        d1 = rep[rep.model == best].sort_values('rep')[met].values - \
             rep[rep.model == 'TFGD_PSO'].sort_values('rep')[met].values
        d2 = rep[rep.model == 'TFGD_PSO'].sort_values('rep')[met].values - \
             rep[rep.model == 'TFGD'].sort_values('rep')[met].values
        ax.axhline(0, color='k', lw=1)
        ax.plot(d1, 'o-', label=f'{best} − TFGD_PSO', color='#4c72b0')
        ax.plot(d2, 's-', label='TFGD_PSO − TFGD', color=PROP)
        ax.set_xlabel('Repetition'); ax.set_ylabel(f'Δ {met}')
        ax.set_title(f'Paired differences ({met})'); ax.legend()
    save(fig, '04_paired_differences_auc_f1.png')

    # ---- 05/06 hyperparameter frequency heatmaps ----
    for key, fname in [('TFGD', '05_tfgd_hyperparameter_frequency.png'),
                       ('TFGD_PSO', '06_tfgd_pso_hyperparameter_frequency.png')]:
        H = np.zeros((len(ALPHA_GRID), len(LAM_GRID)))
        for a, l in hyper[key]:
            H[ALPHA_GRID.index(a), LAM_GRID.index(l)] += 1
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(H, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(LAM_GRID))); ax.set_xticklabels(LAM_GRID)
        ax.set_yticks(range(len(ALPHA_GRID))); ax.set_yticklabels(ALPHA_GRID)
        ax.set_xlabel('λ (tempering)'); ax.set_ylabel('α (fractional order)')
        ax.set_title(f'{key}: selected (α, λ) frequency')
        for i in range(len(ALPHA_GRID)):
            for j in range(len(LAM_GRID)):
                if H[i, j] > 0:
                    ax.text(j, i, int(H[i, j]), ha='center', va='center',
                            color='w' if H[i, j] < H.max() * 0.6 else 'k')
        fig.colorbar(im, label='count'); ax.grid(False)
        save(fig, fname)

    # ---- 07 confusion matrices (last rep): best vs TFGD_PSO ----
    y = curves['y_true']
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, m in zip(axes, [best, 'TFGD_PSO']):
        yhat = (curves[f'proba_{m}'] >= 0.5).astype(int)
        cm = confusion_matrix(y, yhat, normalize='true')
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center',
                        color='w' if cm[i, j] > 0.5 else 'k', fontsize=14)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['No disease', 'Disease'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['No disease', 'Disease'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(m); ax.grid(False)
    save(fig, '07_confusion_matrices_stacking_vs_tfgd_pso.png')

    # ---- 08 permutation importance ----
    n = min(15, len(perm['features']))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(n), perm['importance_mean'][:n][::-1],
            xerr=perm['importance_sd'][:n][::-1], color='#4c72b0', capsize=3)
    ax.set_yticks(range(n)); ax.set_yticklabels(perm['features'][:n][::-1])
    ax.set_xlabel('Permutation importance (ROC-AUC drop)')
    ax.set_title(f"Permutation importance — {perm['best_classical']} (last repetition)")
    save(fig, '08_permutation_importance_clean.png')

    # ---- 09 PSO feature frequency ----
    freq = psof['frequency'][:14]
    fig, ax = plt.subplots(figsize=(9, 6))
    names = [f[0] for f in freq][::-1]; vals = [f[1] for f in freq][::-1]
    ax.barh(range(len(names)), vals, color='#55a868')
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Times selected across 15 repetitions')
    ax.set_title('PSO-constructed feature frequency')
    save(fig, '09_pso_feature_frequency_clean.png')

    # ---- 10 significance heatmap (Holm-corrected p) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    comps = [c for c in paired['comparison'].unique()]
    M = np.zeros((len(comps), 2))
    for i, c in enumerate(comps):
        for j, met in enumerate(['AUC', 'F1']):
            row = paired[(paired.comparison == c) & (paired.metric == met)]
            M[i, j] = row['holm_p'].iloc[0] if len(row) else np.nan
    im = ax.imshow(M, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['AUC', 'F1'])
    ax.set_yticks(range(len(comps))); ax.set_yticklabels(comps, fontsize=9)
    for i in range(len(comps)):
        for j in range(2):
            ax.text(j, i, f'{M[i,j]:.3f}', ha='center', va='center', fontsize=9)
    ax.set_title('Holm-corrected p-values vs best model'); ax.grid(False)
    fig.colorbar(im, label='Holm p')
    save(fig, '10_significance_heatmap_auc_f1.png')

    # ---- ROC / PR curves ----
    for kind in ['roc', 'pr']:
        fig, ax = plt.subplots(figsize=(7, 6))
        for m in MODELS:
            p = curves[f'proba_{m}']
            if kind == 'roc':
                fpr, tpr, _ = roc_curve(y, p); a = sk_auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{m} ({a:.3f})',
                        lw=2.2 if m in ('TFGD', 'TFGD_PSO') else 1.3,
                        color=PROP if m == 'TFGD_PSO' else None)
            else:
                pr, rc, _ = precision_recall_curve(y, p); a = sk_auc(rc, pr)
                ax.plot(rc, pr, label=f'{m} ({a:.3f})',
                        lw=2.2 if m in ('TFGD', 'TFGD_PSO') else 1.3,
                        color=PROP if m == 'TFGD_PSO' else None)
        if kind == 'roc':
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
            ax.set_title('ROC curves (final repetition)')
        else:
            ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
            ax.set_title('Precision–recall curves (final repetition)')
        ax.legend(fontsize=8, loc='lower left' if kind == 'roc' else 'lower right')
        save(fig, f'{kind}_curves.png')

    # ---- calibration ----
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in ['Stacking', 'HistGB', 'XGBoost', 'TFGD_PSO']:
        if f'proba_{m}' not in curves: continue
        frac, mean_pred = calibration_curve(y, curves[f'proba_{m}'], n_bins=8, strategy='quantile')
        ax.plot(mean_pred, frac, 'o-', label=m, color=PROP if m == 'TFGD_PSO' else None)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed frequency')
    ax.set_title('Calibration (final repetition)'); ax.legend()
    save(fig, 'calibration_plot.png')

    # ---- loss curves ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss['tfgd'], label='TFGD', lw=2)
    ax.plot(loss['tfgd_pso'], label='TFGD_PSO', lw=2, color=PROP)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Training BCE loss')
    ax.set_title('TFGD vs TFGD_PSO optimisation trajectory'); ax.legend()
    save(fig, 'loss_curves.png')

    # ---- 11 external validation forest plot (AUC per centre + CI) ----
    centres = list(extb.keys())
    fig, axes = plt.subplots(1, len(centres), figsize=(4.2 * len(centres), 5.5), sharey=True)
    for ax, c in zip(axes, centres):
        ms = MODELS
        aucs = [extb[c][m]['auc'] for m in ms]
        los = [extb[c][m]['auc'] - extb[c][m]['ci_lo'] for m in ms]
        his = [extb[c][m]['ci_hi'] - extb[c][m]['auc'] for m in ms]
        ypos = np.arange(len(ms))
        cols = [PROP if m in ('TFGD', 'TFGD_PSO') else '#4c72b0' for m in ms]
        ax.errorbar(aucs, ypos, xerr=[los, his], fmt='o', capsize=3, color='k', ls='none')
        ax.scatter(aucs, ypos, color=cols, zorder=3, s=60)
        ax.set_yticks(ypos); ax.set_yticklabels(ms)
        prev = ext[(ext.held_out == c)]['prevalence'].iloc[0]
        nte = ext[(ext.held_out == c)]['n_test'].iloc[0]
        ax.set_title(f'{c}\n(n={nte}, prev={prev:.2f})'); ax.set_xlabel('External AUC (95% CI)')
        ax.axvline(0.5, color='gray', ls=':', alpha=0.6)
    fig.suptitle('Leave-one-institution-out external validation (ROC-AUC)', y=1.02)
    save(fig, '11_external_validation_auc.png')

    # ---- 12 Bayesian posterior region probabilities ----
    contrasts = list(bayes['AUC'].keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, met in zip(axes, ['AUC', 'F1', 'MCC']):
        labels, left, mid, right = [], [], [], []
        for c in contrasts:
            r = bayes[met][c]
            labels.append(c.replace('_vs_', '\n> '))
            left.append(r['p_left']); mid.append(r['p_rope']); right.append(r['p_right'])
        yp = np.arange(len(labels))
        ax.barh(yp, right, color='#2ca02c', label='P(left model better)')
        ax.barh(yp, mid, left=right, color='#bbbbbb', label='P(practically equiv., ROPE±0.01)')
        ax.barh(yp, left, left=np.array(right) + np.array(mid), color='#d62728',
                label='P(right model better)')
        ax.set_yticks(yp); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, 1); ax.set_xlabel('Posterior probability'); ax.set_title(met)
    axes[0].legend(fontsize=7, loc='lower right')
    fig.suptitle('Bayesian correlated t-test: posterior region probabilities', y=1.02)
    save(fig, '12_bayesian_posterior.png')

    # ---- 13 internal vs external AUC ----
    macro = ext.groupby('model')['AUC'].mean()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xint = [smean(m, 'AUC')[0] for m in MODELS]
    xext = [macro[m] for m in MODELS]
    x = np.arange(len(MODELS)); w = 0.38
    ax.bar(x - w / 2, xint, w, label='Internal (repeated hold-out)', color='#4c72b0')
    ax.bar(x + w / 2, xext, w, label='External (macro-avg across 4 institutions)', color='#dd8452')
    ax.set_xticks(x); ax.set_xticklabels(MODELS, rotation=40, ha='right')
    ax.set_ylabel('Mean ROC-AUC'); ax.set_ylim(0.5, 1.0)
    ax.set_title('Internal vs external (inter-institutional) discrimination'); ax.legend()
    save(fig, '13_internal_vs_external.png')

    print("ALL FIGURES DONE ->", FIG)


if __name__ == '__main__':
    main()
