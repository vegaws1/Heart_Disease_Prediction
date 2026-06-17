"""
Multi-centre external validation by leave-one-institution-out.

For each held-out UCI centre c in {Cleveland, Hungarian, Switzerland, VA}:
  * train the ENTIRE pipeline (preprocessing, tuning, PSO feature construction)
    on the pooled OTHER three centres only;
  * evaluate once on the held-out centre c (never seen in training).

This is genuine inter-institutional external validation with identical feature
schema and zero leakage. Test-set bootstrap CIs quantify per-cohort uncertainty.

Outputs -> ../results/external_metrics.csv, external_bootstrap.json
"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score
import core, data_loader

warnings.filterwarnings('ignore')
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
ALPHA_GRID = [0.3, 0.5, 0.6, 0.7, 0.9]
LAM_GRID = [0.1, 0.3, 0.5, 0.7, 1.0]
TFGD_FINAL = dict(lr=0.1, epochs=300, batch_size=32)
TFGD_TUNE = dict(lr=0.1, epochs=150, batch_size=32)
MODELS = ['TFGD', 'TFGD_PSO', 'LogRegEN', 'ExtraTrees', 'HistGB', 'XGBoost',
          'LightGBM', 'Stacking']
SEED = 42


def tune_tfgd(Xtr, ytr, seed):
    Xin, Xva, yin, yva = train_test_split(Xtr, ytr, test_size=0.25,
                                          stratify=ytr, random_state=seed)
    best, best_auc = None, -1
    for a in ALPHA_GRID:
        for l in LAM_GRID:
            m = core.TFGDLogistic(alpha=a, lam=l, seed=seed, **TFGD_TUNE).fit(Xin, yin)
            auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
            if auc > best_auc:
                best_auc, best = auc, (a, l)
    return best


def boot_auc_ci(y, p, n_boot=5000, seed=0):
    y = np.asarray(y); p = np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def fit_predict_all(Xtr_raw, ytr, Xte_raw, yte):
    preds = {}
    # TFGD family
    Xtr_e, Xte_e, cols = core.encode_for_tfgd(Xtr_raw, Xte_raw)
    a, l = tune_tfgd(Xtr_e, ytr, SEED)
    mt = core.TFGDLogistic(alpha=a, lam=l, seed=SEED, **TFGD_FINAL).fit(Xtr_e, ytr)
    preds['TFGD'] = mt.predict_proba(Xte_e)[:, 1]

    Xin, Xva, yin, yva = train_test_split(Xtr_e, ytr, test_size=0.25,
                                          stratify=ytr, random_state=SEED + 100)
    a2, l2 = tune_tfgd(Xtr_e, ytr, SEED + 1)
    pso = core.PSOFeatureConstructor(
        n_features=6, n_particles=8, n_gen=6,
        tfgd_kwargs=dict(alpha=a2, lam=l2, seed=SEED, **TFGD_TUNE), seed=SEED + 7)
    pso.fit(Xin, yin, Xva, yva)
    mp = core.TFGDLogistic(alpha=a2, lam=l2, seed=SEED, **TFGD_FINAL).fit(
        pso.transform(Xtr_e), ytr)
    preds['TFGD_PSO'] = mp.predict_proba(pso.transform(Xte_e))[:, 1]

    # classical
    pre = core.make_preprocessor()
    spaces = core.baseline_search_spaces()
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    fitted, cv_auc = {}, {}
    for name, (est, params) in spaces.items():
        pipe = Pipeline([('pre', pre), ('clf', est)])
        rs = RandomizedSearchCV(pipe, params, n_iter=12, scoring='roc_auc',
                                cv=cv, n_jobs=-1, random_state=SEED, refit=True)
        rs.fit(Xtr_raw, ytr)
        preds[name] = rs.predict_proba(Xte_raw)[:, 1]
        fitted[name] = rs.best_estimator_
        cv_auc[name] = rs.best_score_
    top3 = sorted(cv_auc, key=cv_auc.get, reverse=True)[:3]
    stack = StackingClassifier(estimators=[(n, fitted[n]) for n in top3],
                               final_estimator=LogisticRegression(max_iter=2000),
                               cv=4, n_jobs=-1)
    stack.fit(Xtr_raw, ytr)
    preds['Stacking'] = stack.predict_proba(Xte_raw)[:, 1]
    return preds


def main():
    cents = data_loader.load_centres()
    names = list(cents.keys())
    # engineer features per centre (target-safe, row-wise)
    eng = {n: core.engineer_features(df) for n, df in cents.items()}

    rows, boot = [], {}
    for held in names:
        train_frames = [eng[n] for n in names if n != held]
        tr = pd.concat(train_frames, ignore_index=True)
        te = eng[held]
        ytr = tr[core.TARGET].values
        yte = te[core.TARGET].values
        Xtr = tr.drop(columns=[core.TARGET])
        Xte = te.drop(columns=[core.TARGET])
        print(f"== External: train on {[n for n in names if n!=held]} "
              f"(n={len(tr)}), test on {held} (n={len(te)}, prev={yte.mean():.3f})")
        preds = fit_predict_all(Xtr, ytr, Xte, yte)
        boot[held] = {}
        for m in MODELS:
            mt = core.compute_metrics(yte, preds[m])
            r = {'held_out': held, 'n_test': len(te), 'prevalence': float(yte.mean()),
                 'model': m}
            r.update(mt)
            rows.append(r)
            mean, lo, hi = boot_auc_ci(yte, preds[m], seed=1)
            boot[held][m] = dict(auc=mt['AUC'], boot_mean=mean, ci_lo=lo, ci_hi=hi)
            print(f"     {m:<11} AUC={mt['AUC']:.4f} [{lo:.3f},{hi:.3f}]  "
                  f"F1={mt['F1']:.4f}  MCC={mt['MCC']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, 'external_metrics.csv'), index=False)
    # macro-average across centres
    macro = df.groupby('model')[['AUC', 'F1', 'PR_AUC', 'MCC', 'Brier',
                                 'BalancedAcc']].mean().reset_index()
    macro.to_csv(os.path.join(RESULTS, 'external_macro.csv'), index=False)
    with open(os.path.join(RESULTS, 'external_bootstrap.json'), 'w') as f:
        json.dump(boot, f, indent=2)
    print("\nMacro-average external AUC by model:")
    print(macro.sort_values('AUC', ascending=False)[['model', 'AUC', 'F1', 'MCC']].to_string(index=False))
    print("External results written to", RESULTS)


if __name__ == '__main__':
    main()
