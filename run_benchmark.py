"""
Leakage-safe repeated hold-out benchmark on the development cohort (heart.csv).

Outputs (written to ../results):
  rep_metrics.csv          per-repetition test metrics for every model
  summary.csv              mean, sd, bootstrap 95% CI for every metric/model
  paired_tests.csv         t-test / Wilcoxon / Holm vs best model (AUC, F1)
  bayesian.json            Bayesian correlated t-test for key contrasts
  hyperparams.json         selected TFGD/TFGD_PSO hyperparameters per rep
  runtime.csv              mean runtime per model
  pso_features.json        constructed features + degenerate counts per rep
  curves_lastrep.npz       y_true + predicted probs (last rep) for all models
  loss_curves.npz          TFGD / TFGD_PSO training loss (last rep)
  perm_importance.json     permutation importance of best classical (last rep)
"""
import os, json, time, warnings
import numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.inspection import permutation_importance
from scipy import stats
from statsmodels.stats.multitest import multipletests
import core

warnings.filterwarnings('ignore')
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS, exist_ok=True)

N_REPS = 15
TEST_FRAC = 0.20
SEEDS = list(range(42, 42 + N_REPS))
ALPHA_GRID = [0.3, 0.5, 0.6, 0.7, 0.9]
LAM_GRID = [0.1, 0.3, 0.5, 0.7, 1.0]
TFGD_FINAL = dict(lr=0.1, epochs=300, batch_size=32)
TFGD_TUNE = dict(lr=0.1, epochs=150, batch_size=32)
METRICS = ['AUC', 'F1', 'PR_AUC', 'MCC', 'Brier', 'Accuracy', 'Precision',
           'Recall', 'Specificity', 'BalancedAcc']
MODELS = ['TFGD', 'TFGD_PSO', 'LogRegEN', 'ExtraTrees', 'HistGB', 'XGBoost',
          'LightGBM', 'Stacking']


def tune_tfgd(Xtr, ytr, seed):
    Xin, Xva, yin, yva = train_test_split(Xtr, ytr, test_size=0.25,
                                          stratify=ytr, random_state=seed)
    best, best_auc = None, -1
    for a in ALPHA_GRID:
        for l in LAM_GRID:
            m = core.TFGDLogistic(alpha=a, lam=l, seed=seed, **TFGD_TUNE).fit(Xin, yin)
            auc = core.roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
            if auc > best_auc:
                best_auc, best = auc, (a, l)
    return best


def run_rep(rep, seed, df):
    out = {}
    y = df[core.TARGET].values
    X = df.drop(columns=[core.TARGET])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_FRAC,
                                          stratify=y, random_state=seed)
    # ---- TFGD-family encoding (train-only) ----
    Xtr_e, Xte_e, cols = core.encode_for_tfgd(Xtr, Xte)

    # ---- plain TFGD ----
    t = time.time()
    a, l = tune_tfgd(Xtr_e, ytr, seed)
    mt = core.TFGDLogistic(alpha=a, lam=l, seed=seed, **TFGD_FINAL).fit(Xtr_e, ytr)
    p_tfgd = mt.predict_proba(Xte_e)[:, 1]
    out['TFGD'] = dict(metrics=core.compute_metrics(yte, p_tfgd),
                       runtime=time.time() - t, hp=(a, l), proba=p_tfgd,
                       loss=mt.loss_curve_)

    # ---- TFGD_PSO ----
    t = time.time()
    Xin, Xva, yin, yva = train_test_split(Xtr_e, ytr, test_size=0.25,
                                          stratify=ytr, random_state=seed + 100)
    a2, l2 = tune_tfgd(Xtr_e, ytr, seed + 1)
    pso = core.PSOFeatureConstructor(
        n_features=6, n_particles=8, n_gen=6,
        tfgd_kwargs=dict(alpha=a2, lam=l2, seed=seed, **TFGD_TUNE), seed=seed + 7)
    pso.fit(Xin, yin, Xva, yva)
    Xtr_aug = pso.transform(Xtr_e)
    Xte_aug = pso.transform(Xte_e)
    mp = core.TFGDLogistic(alpha=a2, lam=l2, seed=seed, **TFGD_FINAL).fit(Xtr_aug, ytr)
    p_pso = mp.predict_proba(Xte_aug)[:, 1]
    out['TFGD_PSO'] = dict(metrics=core.compute_metrics(yte, p_pso),
                           runtime=time.time() - t, hp=(a2, l2), proba=p_pso,
                           loss=mp.loss_curve_,
                           features=pso.feature_names(cols),
                           degenerate_filtered=pso.rejected_count_)

    # ---- classical baselines ----
    pre = core.make_preprocessor()
    spaces = core.baseline_search_spaces()
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    fitted, cv_auc = {}, {}
    for name, (est, params) in spaces.items():
        t = time.time()
        pipe = Pipeline([('pre', pre), ('clf', est)])
        rs = RandomizedSearchCV(pipe, params, n_iter=12, scoring='roc_auc',
                                cv=cv, n_jobs=-1, random_state=seed, refit=True)
        rs.fit(Xtr, ytr)
        p = rs.predict_proba(Xte)[:, 1]
        out[name] = dict(metrics=core.compute_metrics(yte, p),
                         runtime=time.time() - t, proba=p)
        fitted[name] = rs.best_estimator_
        cv_auc[name] = rs.best_score_

    # ---- stacking (top-3 classical by inner CV AUC) ----
    t = time.time()
    top3 = sorted(cv_auc, key=cv_auc.get, reverse=True)[:3]
    estimators = [(n, fitted[n]) for n in top3]
    stack = StackingClassifier(estimators=estimators,
                               final_estimator=LogisticRegression(max_iter=2000),
                               cv=4, n_jobs=-1, passthrough=False)
    stack.fit(Xtr, ytr)
    p_stack = stack.predict_proba(Xte)[:, 1]
    out['Stacking'] = dict(metrics=core.compute_metrics(yte, p_stack),
                           runtime=time.time() - t, proba=p_stack, top3=top3)

    # permutation importance of best classical on this rep's test set
    best_classical = max([m for m in cv_auc], key=cv_auc.get)
    out['_best_classical'] = best_classical
    out['_perm_model'] = fitted[best_classical]
    out['_Xte'] = Xte
    out['_yte'] = yte
    out['_cols_tfgd'] = cols
    return out


def main():
    df = core.engineer_features(pd.read_csv(os.path.join(
        os.path.dirname(RESULTS), 'data', 'heart.csv')))
    print(f"Development cohort: n={len(df)}, prevalence={df[core.TARGET].mean():.3f}")

    rows = []
    runtimes = {m: [] for m in MODELS}
    hyper = {'TFGD': [], 'TFGD_PSO': []}
    pso_feats, degen = [], []
    last = None
    t0 = time.time()
    for rep, seed in enumerate(SEEDS):
        tr = time.time()
        res = run_rep(rep, seed, df)
        for m in MODELS:
            r = {'rep': rep, 'seed': seed, 'model': m}
            r.update(res[m]['metrics'])
            rows.append(r)
            runtimes[m].append(res[m]['runtime'])
        hyper['TFGD'].append(res['TFGD']['hp'])
        hyper['TFGD_PSO'].append(res['TFGD_PSO']['hp'])
        pso_feats.append(res['TFGD_PSO']['features'])
        degen.append(res['TFGD_PSO']['degenerate_filtered'])
        last = res
        print(f"  rep {rep+1}/{N_REPS} (seed {seed}) done in {time.time()-tr:.1f}s | "
              f"TFGD AUC={res['TFGD']['metrics']['AUC']:.4f} "
              f"PSO AUC={res['TFGD_PSO']['metrics']['AUC']:.4f} "
              f"Stack AUC={res['Stacking']['metrics']['AUC']:.4f} "
              f"(degen filtered={res['TFGD_PSO']['degenerate_filtered']})")
    print(f"Total benchmark time: {time.time()-t0:.1f}s")

    rep_df = pd.DataFrame(rows)
    rep_df.to_csv(os.path.join(RESULTS, 'rep_metrics.csv'), index=False)

    # ---- summary with bootstrap CIs ----
    srows = []
    for m in MODELS:
        for met in METRICS:
            vals = rep_df[(rep_df.model == m)][met].values
            mean, lo, hi = core.bootstrap_ci_mean(vals, seed=1)
            srows.append(dict(model=m, metric=met, mean=mean, sd=vals.std(ddof=1),
                              ci_lo=lo, ci_hi=hi))
    summ = pd.DataFrame(srows)
    summ.to_csv(os.path.join(RESULTS, 'summary.csv'), index=False)

    # ---- best model by mean AUC ----
    mean_auc = summ[summ.metric == 'AUC'].set_index('model')['mean']
    best = mean_auc.idxmax()
    print(f"Best model by mean AUC: {best} ({mean_auc[best]:.4f})")

    # ---- paired tests vs best (AUC, F1) ----
    prows = []
    for met in ['AUC', 'F1']:
        bvals = rep_df[rep_df.model == best][met].values
        comps, pvals = [], []
        for m in MODELS:
            if m == best:
                continue
            mvals = rep_df[rep_df.model == m][met].values
            diff = bvals.mean() - mvals.mean()
            tt = stats.ttest_rel(bvals, mvals).pvalue
            try:
                ww = stats.wilcoxon(bvals, mvals).pvalue
            except ValueError:
                ww = 1.0
            d = (bvals - mvals)
            cohen = d.mean() / (d.std(ddof=1) + 1e-12)
            comps.append((m, met, diff, tt, ww, cohen))
            pvals.append(tt)
        holm = multipletests(pvals, method='holm')[1]
        for (c, hp) in zip(comps, holm):
            prows.append(dict(metric=c[1], comparison=f'{best} vs {c[0]}',
                              diff=c[2], t_p=c[3], wilcoxon_p=c[4],
                              holm_p=hp, cohen_d=c[5]))
    pd.DataFrame(prows).to_csv(os.path.join(RESULTS, 'paired_tests.csv'), index=False)

    # ---- Bayesian correlated t-test for key contrasts ----
    def vec(m, met):
        return rep_df[rep_df.model == m][met].values
    bayes = {}
    contrasts = [('TFGD_PSO', 'TFGD'), (best, 'TFGD_PSO'), (best, 'TFGD'),
                 ('TFGD_PSO', 'LogRegEN'), (best, 'ExtraTrees')]
    for met in ['AUC', 'F1', 'MCC']:
        bayes[met] = {}
        for a_, b_ in contrasts:
            r = core.bayesian_correlated_ttest(vec(a_, met), vec(b_, met),
                                               rope=0.01, test_fraction=TEST_FRAC)
            mean, lo, hi, pgt = core.bootstrap_ci_paired_diff(vec(a_, met), vec(b_, met), seed=2)
            r.update(dict(boot_diff=mean, boot_lo=lo, boot_hi=hi, boot_p_gt0=pgt))
            bayes[met][f'{a_}_vs_{b_}'] = r
    with open(os.path.join(RESULTS, 'bayesian.json'), 'w') as f:
        json.dump(bayes, f, indent=2)

    # ---- hyperparameters / runtime / pso ----
    with open(os.path.join(RESULTS, 'hyperparams.json'), 'w') as f:
        json.dump({k: [list(x) for x in v] for k, v in hyper.items()}, f, indent=2)
    rt = pd.DataFrame({'model': MODELS,
                       'mean_runtime_s': [np.mean(runtimes[m]) for m in MODELS],
                       'sd_runtime_s': [np.std(runtimes[m], ddof=1) for m in MODELS]})
    rt.to_csv(os.path.join(RESULTS, 'runtime.csv'), index=False)
    allfeat = Counter([f for rep in pso_feats for f in rep])
    with open(os.path.join(RESULTS, 'pso_features.json'), 'w') as f:
        json.dump({'per_rep': pso_feats, 'frequency': allfeat.most_common(),
                   'degenerate_filtered_per_rep': degen,
                   'total_degenerate_filtered': int(sum(degen))}, f, indent=2)

    # ---- last-rep curves / loss / permutation importance ----
    np.savez(os.path.join(RESULTS, 'curves_lastrep.npz'),
             y_true=last['_yte'],
             **{f'proba_{m}': last[m]['proba'] for m in MODELS})
    np.savez(os.path.join(RESULTS, 'loss_curves.npz'),
             tfgd=np.array(last['TFGD']['loss']),
             tfgd_pso=np.array(last['TFGD_PSO']['loss']))
    pim = permutation_importance(last['_perm_model'], last['_Xte'], last['_yte'],
                                 scoring='roc_auc', n_repeats=30, random_state=0, n_jobs=-1)
    feat_names = list(last['_Xte'].columns)
    order = np.argsort(pim.importances_mean)[::-1]
    with open(os.path.join(RESULTS, 'perm_importance.json'), 'w') as f:
        json.dump({'best_classical': last['_best_classical'],
                   'features': [feat_names[i] for i in order],
                   'importance_mean': [float(pim.importances_mean[i]) for i in order],
                   'importance_sd': [float(pim.importances_std[i]) for i in order]},
                  f, indent=2)
    print("All results written to", RESULTS)


if __name__ == '__main__':
    main()
