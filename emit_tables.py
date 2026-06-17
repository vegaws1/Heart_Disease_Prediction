"""Emit LaTeX table bodies and a macro block from results/ -> results/tables.tex
Also prints key numbers for prose."""
import os, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
MODELS = ['TFGD', 'TFGD_PSO', 'LogRegEN', 'ExtraTrees', 'HistGB', 'XGBoost',
          'LightGBM', 'Stacking']
DISP = {'TFGD': 'TFGD', 'TFGD_PSO': 'TFGD\\_PSO', 'LogRegEN': 'LogRegEN',
        'ExtraTrees': 'ExtraTrees', 'HistGB': 'HistGB', 'XGBoost': 'XGBoost',
        'LightGBM': 'LightGBM', 'Stacking': 'Stacking'}


def main():
    summ = pd.read_csv(os.path.join(RES, 'summary.csv'))
    rep = pd.read_csv(os.path.join(RES, 'rep_metrics.csv'))
    rt = pd.read_csv(os.path.join(RES, 'runtime.csv'))
    paired = pd.read_csv(os.path.join(RES, 'paired_tests.csv'))
    ext = pd.read_csv(os.path.join(RES, 'external_metrics.csv'))
    extmac = pd.read_csv(os.path.join(RES, 'external_macro.csv'))
    with open(os.path.join(RES, 'bayesian.json')) as f: bayes = json.load(f)
    with open(os.path.join(RES, 'pso_features.json')) as f: psof = json.load(f)
    with open(os.path.join(RES, 'hyperparams.json')) as f: hyper = json.load(f)

    def g(m, met, k='mean'):
        r = summ[(summ.model == m) & (summ.metric == met)].iloc[0]
        return r[k]

    def ms(m, met):
        return f"{g(m,met):.4f} $\\pm$ {g(m,met,'sd'):.4f}"

    def ci(m, met):
        return f"[{g(m,met,'ci_lo'):.4f}, {g(m,met,'ci_hi'):.4f}]"

    out = []
    bestauc = summ[summ.metric == 'AUC'].sort_values('mean').iloc[-1]['model']

    # ---- main results table ----
    out.append("% ====== MAIN RESULTS ======")
    # bold the best per column
    def col_best(met, lo=False):
        vals = {m: g(m, met) for m in MODELS}
        return min(vals, key=vals.get) if lo else max(vals, key=vals.get)
    bests = {met: col_best(met, lo=(met == 'Brier')) for met in ['AUC', 'F1', 'PR_AUC', 'MCC', 'Brier']}
    for m in MODELS:
        cells = []
        for met in ['AUC', 'F1', 'PR_AUC', 'MCC', 'Brier']:
            s = ms(m, met)
            if m == bests[met]:
                s = "\\textbf{" + s + "}"
            cells.append(s)
        out.append(f"{DISP[m]} & " + " & ".join(cells) + " \\\\")

    # ---- main results table WITH 95% CI on AUC and F1 ----
    out.append("\n% ====== MAIN RESULTS WITH CI (AUC,F1) ======")
    for m in MODELS:
        out.append(f"{DISP[m]} & {g(m,'AUC'):.4f} & {ci(m,'AUC')} & {g(m,'F1'):.4f} & {ci(m,'F1')} \\\\")

    # ---- paired tests vs best ----
    out.append(f"\n% ====== PAIRED TESTS vs {bestauc} ======")
    for comp in paired['comparison'].unique():
        a = paired[(paired.comparison == comp) & (paired.metric == 'AUC')].iloc[0]
        f = paired[(paired.comparison == comp) & (paired.metric == 'F1')].iloc[0]
        name = comp.replace('_', '\\_')
        out.append(f"{name} & {a['diff']:.4f} & {a['t_p']:.4f} & {a['holm_p']:.4f} & "
                   f"{f['diff']:.4f} & {f['t_p']:.4f} & {f['holm_p']:.4f} \\\\")

    # ---- runtime ----
    out.append("\n% ====== RUNTIME ======")
    for m in MODELS:
        r = rt[rt.model == m].iloc[0]
        out.append(f"{DISP[m]} & {r['mean_runtime_s']:.2f} \\\\")

    # ---- external per-centre AUC table (model x centre) ----
    out.append("\n% ====== EXTERNAL AUC (model x centre) ======")
    centres = list(ext['held_out'].unique())
    for m in MODELS:
        cells = []
        for c in centres:
            v = ext[(ext.model == m) & (ext.held_out == c)]['AUC'].iloc[0]
            cells.append(f"{v:.3f}")
        mac = extmac[extmac.model == m]['AUC'].iloc[0]
        out.append(f"{DISP[m]} & " + " & ".join(cells) + f" & {mac:.3f} \\\\")

    # ---- external F1 / MCC macro table ----
    out.append("\n% ====== EXTERNAL macro (AUC,F1,PR,MCC) ======")
    for m in MODELS:
        r = extmac[extmac.model == m].iloc[0]
        out.append(f"{DISP[m]} & {r['AUC']:.3f} & {r['F1']:.3f} & {r['PR_AUC']:.3f} & {r['MCC']:.3f} & {r['BalancedAcc']:.3f} \\\\")

    # ---- bayesian table ----
    out.append("\n% ====== BAYESIAN (AUC) ======")
    for c, r in bayes['AUC'].items():
        name = c.replace('_vs_', ' vs ').replace('_', '\\_')
        out.append(f"{name} & {r['mean']:.4f} & [{r['boot_lo']:.4f}, {r['boot_hi']:.4f}] & "
                   f"{r['p_right']:.3f} & {r['p_rope']:.3f} & {r['p_left']:.3f} \\\\")

    with open(os.path.join(RES, 'tables.tex'), 'w') as f:
        f.write("\n".join(out))

    # ---- prose numbers ----
    print("===== KEY NUMBERS FOR PROSE =====")
    print("best by AUC:", bestauc)
    print("pooled dev n: see data")
    for m in ['TFGD', 'TFGD_PSO', bestauc, 'ExtraTrees']:
        print(f"{m}: AUC {ms(m,'AUC')} CI{ci(m,'AUC')} | F1 {ms(m,'F1')} | MCC {ms(m,'MCC')} | Brier {ms(m,'Brier')}")
    print("\nPSO gain over TFGD:")
    for met in ['AUC', 'F1', 'MCC', 'PR_AUC']:
        print(f"  d{met} = {g('TFGD_PSO',met)-g('TFGD',met):+.4f}")
    print("\nBrier TFGD->PSO:", round(g('TFGD', 'Brier'), 4), "->", round(g('TFGD_PSO', 'Brier'), 4))
    print("\nBayesian TFGD_PSO vs TFGD (AUC):", bayes['AUC'].get('TFGD_PSO_vs_TFGD'))
    print("Bayesian best vs TFGD_PSO (AUC):", bayes['AUC'].get(f'{bestauc}_vs_TFGD_PSO'))
    print("\nDegenerate filtered total:", psof['total_degenerate_filtered'],
          "| per rep:", psof['degenerate_filtered_per_rep'])
    print("PSO top features:", psof['frequency'][:6])
    print("\nExternal macro AUC ranking:")
    print(extmac.sort_values('AUC', ascending=False)[['model', 'AUC', 'F1', 'MCC']].to_string(index=False))
    print("\nRuntime:", {m: round(rt[rt.model == m]['mean_runtime_s'].iloc[0], 2) for m in MODELS})
    # hyperparam most common
    from collections import Counter
    for k in ['TFGD', 'TFGD_PSO']:
        cc = Counter([tuple(x) for x in hyper[k]])
        print(f"{k} most common (a,l):", cc.most_common(3))


if __name__ == '__main__':
    main()
