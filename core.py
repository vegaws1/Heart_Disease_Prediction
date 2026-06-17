"""
Core methodology: clinical feature engineering, the Tempered Fractional Gradient
Descent (TFGD) logistic learner, PSO-based nonlinear feature construction WITH a
degenerate-feature filter/repair operator, baseline model factory with explicit
search spaces, metrics, and statistical tools (bootstrap CIs + Bayesian
correlated t-test).

All transforms are designed to be fit on training data only (leakage-safe).
"""
import numpy as np
import pandas as pd
from math import lgamma, ceil
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, balanced_accuracy_score,
                             accuracy_score, precision_score, recall_score,
                             brier_score_loss, confusion_matrix)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RNG_GLOBAL = 12345

# ---------------------------------------------------------------------------
# 1. Clinical feature engineering (target-safe; uses no label information)
# ---------------------------------------------------------------------------
ORIG_NUM = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
ORIG_CAT = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
TARGET = 'HeartDisease'


def engineer_features(df):
    """Clinically inspired, target-safe feature engineering.

    Returns a NEW dataframe with suspicious-zero indicators and engineered
    interaction variables added. No label is used, so this can be applied to any
    split; statistics used for imputation are still fit only on training data
    downstream inside the model pipelines.
    """
    d = df.copy()

    # (a) suspicious zeros in Cholesterol / RestingBP -> indicators + NaN
    for col in ['Cholesterol', 'RestingBP']:
        ind = f'{col}_was_zero'
        d[ind] = (d[col] == 0).astype(int)
        d.loc[d[col] == 0, col] = np.nan

    # binary / numeric helpers (computed without the label)
    sex_male = (d['Sex'].astype(str) == 'M').astype(float)
    exang = (d['ExerciseAngina'].astype(str) == 'Y').astype(float)
    age = d['Age'].astype(float)
    chol = d['Cholesterol'].astype(float)
    bp = d['RestingBP'].astype(float)
    maxhr = d['MaxHR'].astype(float)
    oldpeak = d['Oldpeak'].astype(float)
    fbs = d['FastingBS'].astype(float)
    cp_asy = (d['ChestPainType'].astype(str) == 'ASY').astype(float)
    slope_flat = (d['ST_Slope'].astype(str) == 'Flat').astype(float)

    eps = 1e-6
    d['Age_SexMale'] = age * sex_male
    d['MaxHR_Age_Ratio'] = maxhr / (age + eps)
    d['Chol_Age_Ratio'] = chol / (age + eps)
    d['BP_Age_Ratio'] = bp / (age + eps)
    d['Oldpeak_ExerciseAngina'] = oldpeak * exang
    d['FastingBS_Chol'] = fbs * chol
    d['Oldpeak_MaxHR'] = oldpeak * maxhr
    d['ChestPain_ExerciseAngina'] = cp_asy * exang
    d['STSlope_ExerciseAngina'] = slope_flat * exang
    return d


ENG_NUM = ['Cholesterol_was_zero', 'RestingBP_was_zero',
           'Age_SexMale', 'MaxHR_Age_Ratio', 'Chol_Age_Ratio', 'BP_Age_Ratio',
           'Oldpeak_ExerciseAngina', 'FastingBS_Chol', 'Oldpeak_MaxHR',
           'ChestPain_ExerciseAngina', 'STSlope_ExerciseAngina']
NUM_COLS = ORIG_NUM + ENG_NUM
CAT_COLS = ORIG_CAT


# ---------------------------------------------------------------------------
# 2. Preprocessing builders (fit on train only)
# ---------------------------------------------------------------------------
def make_preprocessor():
    """ColumnTransformer for classical baselines."""
    num = Pipeline([('imp', SimpleImputer(strategy='median')),
                    ('sc', StandardScaler())])
    cat = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                    ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('num', num, NUM_COLS), ('cat', cat, CAT_COLS)])


def encode_for_tfgd(X_train, X_test):
    """Numeric encoding for TFGD models. Categorical -> integer codes learned on
    train only; unseen test categories -> -1. Median impute + standardise, fit on
    train only. Returns dense float arrays."""
    Xtr = X_train.copy()
    Xte = X_test.copy()
    # categorical -> codes
    for c in CAT_COLS:
        cats = pd.Index(sorted(Xtr[c].dropna().astype(str).unique()))
        mapping = {v: i for i, v in enumerate(cats)}
        Xtr[c] = Xtr[c].astype(str).map(mapping).fillna(-1).astype(float)
        Xte[c] = Xte[c].astype(str).map(mapping).fillna(-1).astype(float)
    cols = NUM_COLS + CAT_COLS
    Xtr = Xtr[cols].astype(float)
    Xte = Xte[cols].astype(float)
    # median impute (train medians)
    med = Xtr.median(numeric_only=True)
    Xtr = Xtr.fillna(med)
    Xte = Xte.fillna(med)
    # standardise (train stats)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0).replace(0, 1.0)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    return Xtr.values, Xte.values, cols


# ---------------------------------------------------------------------------
# 3. Tempered Fractional Gradient Descent (TFGD) logistic regression
# ---------------------------------------------------------------------------
def _tempered_weights(alpha, lam, m):
    """Normalised tempered fractional memory weights w_j, j=0..m-1.
    w_j = Gamma(alpha+j)/(Gamma(alpha)Gamma(j+1)) * exp(-lam j)."""
    j = np.arange(m)
    log_binom = (np.array([lgamma(alpha + jj) for jj in j])
                 - lgamma(alpha)
                 - np.array([lgamma(jj + 1) for jj in j]))
    w = np.exp(log_binom - lam * j)
    s = w.sum()
    return w / s if s > 0 else np.ones(m) / m


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class TFGDLogistic:
    """Logistic regression trained with tempered fractional gradient descent.

    Faithful to the manuscript: a finite history buffer of mini-batch gradients
    is aggregated with normalised tempered fractional weights before each update.
    """

    def __init__(self, alpha=0.5, lam=0.3, lr=0.1, epochs=300, batch_size=32,
                 m_max=30, l2=1e-4, seed=0):
        self.alpha = alpha
        self.lam = lam
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.m_max = m_max
        self.l2 = l2
        self.seed = seed

    def _memory_len(self):
        return int(min(self.m_max, max(5, ceil(5.0 / max(self.lam, 1e-6)))))

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.w = np.zeros(d)
        self.b = 0.0
        m = self._memory_len()
        wts = _tempered_weights(self.alpha, self.lam, m)
        hist_w, hist_b = [], []        # most-recent first
        self.loss_curve_ = []
        bs = min(self.batch_size, n)
        for ep in range(self.epochs):
            idx = rng.permutation(n)
            for start in range(0, n, bs):
                bidx = idx[start:start + bs]
                Xb, yb = X[bidx], y[bidx]
                p = _sigmoid(Xb @ self.w + self.b)
                err = (p - yb)
                gw = Xb.T @ err / len(bidx) + self.l2 * self.w
                gb = err.mean()
                hist_w.insert(0, gw); hist_b.insert(0, gb)
                if len(hist_w) > m:
                    hist_w.pop(); hist_b.pop()
                r = len(hist_w)
                wuse = wts[:r] / wts[:r].sum()
                Gw = np.tensordot(wuse, np.array(hist_w), axes=(0, 0))
                Gb = float(np.dot(wuse, np.array(hist_b)))
                self.w -= self.lr * Gw
                self.b -= self.lr * Gb
            # epoch loss
            pf = _sigmoid(X @ self.w + self.b)
            pf = np.clip(pf, 1e-9, 1 - 1e-9)
            self.loss_curve_.append(float(-np.mean(y * np.log(pf) + (1 - y) * np.log(1 - pf))))
        return self

    def predict_proba(self, X):
        p = _sigmoid(np.asarray(X, float) @ self.w + self.b)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# 4. PSO feature construction with degenerate-feature filter / repair
# ---------------------------------------------------------------------------
# operators: 0 add, 1 sub, 2 mul, 3 protected-div, 4 log-ratio, 5 product-over-sum
OP_NAMES = {0: '+', 1: '-', 2: '*', 3: '/', 4: 'logratio', 5: 'prodsum'}
COMMUTATIVE = {0, 2, 5}
EPS = 1e-8


def _apply_op(f1, f2, op):
    if op == 0:
        return f1 + f2
    if op == 1:
        return f1 - f2
    if op == 2:
        return f1 * f2
    if op == 3:
        return f1 / (f2 + EPS)
    if op == 4:
        return np.log(np.abs(f1) + EPS) - np.log(np.abs(f2) + EPS)
    if op == 5:
        return (f1 * f2) / (f1 + f2 + EPS)
    raise ValueError(op)


def _is_structurally_degenerate(i1, i2, op):
    """Reject constructions that are zero/constant/duplicate BY CONSTRUCTION.
    This is the structural filter that removes e.g. (X - X), (X / X), log-ratio(X,X)."""
    if i1 == i2:
        # self-pairs collapse for all operators except squaring (mul)
        return op != 2
    return False


def _gene_key(i1, i2, op):
    """Canonical key so commutative duplicates are treated as identical."""
    if op in COMMUTATIVE and i1 > i2:
        i1, i2 = i2, i1
    return (i1, i2, op)


class PSOFeatureConstructor:
    """Particle Swarm Optimisation that searches for nonlinear interaction
    features maximising validation ROC-AUC of a TFGD learner.

    A particle is a list of K genes (i1, i2, op). Two filters keep the search
    free of nonsensical variables:
      * structural filter   -> rejects constructions that are constant/zero by
                               construction (self-pair subtraction, division,
                               log-ratio, etc.);
      * numerical filter    -> rejects realised columns that are near-constant
                               (std < tol) or near-duplicate (|r| > 0.999) of an
                               existing or already-accepted feature.
    Whenever a gene fails either filter it is REPAIRED (resampled to a valid,
    non-duplicate gene), so the model never receives a degenerate feature.
    """

    def __init__(self, n_features=6, n_particles=8, n_gen=6,
                 tfgd_kwargs=None, dup_thresh=0.999, var_tol=1e-6, seed=0):
        self.K = n_features
        self.P = n_particles
        self.G = n_gen
        self.tfgd_kwargs = tfgd_kwargs or {}
        self.dup_thresh = dup_thresh
        self.var_tol = var_tol
        self.seed = seed
        self.rejected_count_ = 0  # number of degenerate genes filtered/repaired

    # ---- gene-level validity given a base matrix -------------------------
    def _valid_column(self, col, base_cols, accepted_cols):
        if np.std(col) < self.var_tol or not np.all(np.isfinite(col)):
            return False
        cs = col - col.mean()
        denom = np.linalg.norm(cs) + 1e-12
        for ref in base_cols + accepted_cols:
            rs = ref - ref.mean()
            r = abs(np.dot(cs, rs) / (denom * (np.linalg.norm(rs) + 1e-12)))
            if r > self.dup_thresh:
                return False
        return True

    def _random_gene(self, d, rng):
        i1 = int(rng.integers(d)); i2 = int(rng.integers(d))
        op = int(rng.integers(6))
        return [i1, i2, op]

    def _repair_particle(self, particle, X, rng):
        """Return a validated particle and the realised feature columns; any
        degenerate gene is resampled until valid (bounded attempts)."""
        base_cols = [X[:, j] for j in range(X.shape[1])]
        accepted, cols, seen = [], [], set()
        for gene in particle:
            ok = False
            g = list(gene)
            for _ in range(40):
                i1, i2, op = g
                key = _gene_key(i1, i2, op)
                if (not _is_structurally_degenerate(i1, i2, op)) and key not in seen:
                    col = _apply_op(X[:, i1], X[:, i2], op).astype(float)
                    if np.all(np.isfinite(col)) and self._valid_column(col, base_cols, cols):
                        ok = True
                        break
                self.rejected_count_ += 1
                g = self._random_gene(X.shape[1], rng)
            if ok:
                key = _gene_key(*g)
                seen.add(key)
                accepted.append([_gene_key(*g)[0], _gene_key(*g)[1], g[2]])
                cols.append(_apply_op(X[:, accepted[-1][0]], X[:, accepted[-1][1]], accepted[-1][2]))
        if cols:
            return accepted, np.column_stack(cols)
        return accepted, np.zeros((X.shape[0], 0))

    def _augment(self, X, particle):
        if not particle:
            return X
        cols = [_apply_op(X[:, i1], X[:, i2], op) for (i1, i2, op) in particle]
        add = np.column_stack(cols)
        # standardise added columns
        mu = add.mean(0); sd = add.std(0); sd[sd == 0] = 1.0
        add = (add - mu) / sd
        return np.column_stack([X, add])

    def _fitness(self, particle, Xtr, ytr, Xva, yva):
        acc, _ = self._repair_particle(particle, Xtr, np.random.default_rng(0))
        Xtr_a = self._augment(Xtr, acc)
        Xva_a = self._augment(Xva, acc)
        mdl = TFGDLogistic(**self.tfgd_kwargs).fit(Xtr_a, ytr)
        try:
            return roc_auc_score(yva, mdl.predict_proba(Xva_a)[:, 1]), acc
        except ValueError:
            return 0.5, acc

    def fit(self, Xtr, ytr, Xva, yva):
        rng = np.random.default_rng(self.seed)
        d = Xtr.shape[1]
        # init swarm
        swarm = [[self._random_gene(d, rng) for _ in range(self.K)] for _ in range(self.P)]
        vel = [[[0, 0, 0] for _ in range(self.K)] for _ in range(self.P)]
        pbest = [list(map(list, p)) for p in swarm]
        pbest_fit = []
        for p in swarm:
            f, _ = self._fitness(p, Xtr, ytr, Xva, yva)
            pbest_fit.append(f)
        gi = int(np.argmax(pbest_fit))
        gbest = list(map(list, pbest[gi])); gbest_fit = pbest_fit[gi]
        w, c1, c2 = 0.7, 1.5, 1.5
        for _ in range(self.G):
            for i in range(self.P):
                for k in range(self.K):
                    for dim in range(3):
                        r1, r2 = rng.random(), rng.random()
                        vel[i][k][dim] = (w * vel[i][k][dim]
                                          + c1 * r1 * (pbest[i][k][dim] - swarm[i][k][dim])
                                          + c2 * r2 * (gbest[k][dim] - swarm[i][k][dim]))
                        swarm[i][k][dim] = int(round(swarm[i][k][dim] + vel[i][k][dim]))
                    swarm[i][k][0] = int(np.clip(swarm[i][k][0], 0, d - 1))
                    swarm[i][k][1] = int(np.clip(swarm[i][k][1], 0, d - 1))
                    swarm[i][k][2] = int(np.clip(swarm[i][k][2], 0, 5))
                f, _ = self._fitness(swarm[i], Xtr, ytr, Xva, yva)
                if f > pbest_fit[i]:
                    pbest_fit[i] = f; pbest[i] = list(map(list, swarm[i]))
                    if f > gbest_fit:
                        gbest_fit = f; gbest = list(map(list, swarm[i]))
        # finalise: repaired, validated gene list on the FULL training matrix
        acc, _ = self._repair_particle(gbest, Xtr, rng)
        self.best_genes_ = acc
        self.best_fit_ = gbest_fit
        return self

    def transform(self, X):
        return self._augment(np.asarray(X, float), self.best_genes_)

    def feature_names(self, base_names):
        out = []
        for (i1, i2, op) in self.best_genes_:
            out.append(f'{base_names[i1]} {OP_NAMES[op]} {base_names[i2]}')
        return out


# ---------------------------------------------------------------------------
# 5. Baseline model factory + explicit search spaces (for reproducibility)
# ---------------------------------------------------------------------------
def baseline_search_spaces():
    """Return {name: (estimator, param_distribution)} with EXPLICIT spaces."""
    from scipy.stats import randint, uniform, loguniform
    sp = {}
    sp['LogRegEN'] = (
        LogisticRegression(penalty='elasticnet', solver='saga', max_iter=5000),
        {'clf__C': loguniform(1e-3, 1e2),
         'clf__l1_ratio': uniform(0, 1)})
    sp['ExtraTrees'] = (
        ExtraTreesClassifier(random_state=RNG_GLOBAL, n_jobs=1),
        {'clf__n_estimators': randint(250, 901),
         'clf__max_depth': [None, 4, 6, 8, 10, 15],
         'clf__min_samples_leaf': randint(1, 7),
         'clf__min_samples_split': randint(2, 11),
         'clf__max_features': ['sqrt', 'log2', None, 0.5, 0.8]})
    sp['HistGB'] = (
        HistGradientBoostingClassifier(random_state=RNG_GLOBAL),
        {'clf__learning_rate': loguniform(1e-2, 3e-1),
         'clf__max_iter': randint(150, 601),
         'clf__max_leaf_nodes': randint(15, 64),
         'clf__min_samples_leaf': randint(10, 40),
         'clf__l2_regularization': loguniform(1e-3, 1e1)})
    sp['XGBoost'] = (
        XGBClassifier(random_state=RNG_GLOBAL, n_jobs=1, eval_metric='logloss',
                      tree_method='hist', verbosity=0),
        {'clf__n_estimators': randint(150, 601),
         'clf__max_depth': randint(2, 8),
         'clf__learning_rate': loguniform(1e-2, 3e-1),
         'clf__subsample': uniform(0.6, 0.4),
         'clf__colsample_bytree': uniform(0.6, 0.4),
         'clf__min_child_weight': randint(1, 8),
         'clf__reg_lambda': loguniform(1e-2, 1e1)})
    sp['LightGBM'] = (
        LGBMClassifier(random_state=RNG_GLOBAL, n_jobs=1, verbose=-1),
        {'clf__n_estimators': randint(150, 601),
         'clf__num_leaves': randint(15, 64),
         'clf__learning_rate': loguniform(1e-2, 3e-1),
         'clf__subsample': uniform(0.6, 0.4),
         'clf__colsample_bytree': uniform(0.6, 0.4),
         'clf__min_child_samples': randint(10, 40),
         'clf__reg_lambda': loguniform(1e-2, 1e1)})
    return sp


# ---------------------------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        'Accuracy': accuracy_score(y_true, yhat),
        'Precision': precision_score(y_true, yhat, zero_division=0),
        'Recall': recall_score(y_true, yhat, zero_division=0),
        'Specificity': spec,
        'BalancedAcc': balanced_accuracy_score(y_true, yhat),
        'F1': f1_score(y_true, yhat, zero_division=0),
        'MCC': matthews_corrcoef(y_true, yhat),
        'AUC': roc_auc_score(y_true, p),
        'PR_AUC': average_precision_score(y_true, p),
        'Brier': brier_score_loss(y_true, p),
    }


# ---------------------------------------------------------------------------
# 7. Statistics: bootstrap CIs + Bayesian correlated t-test
# ---------------------------------------------------------------------------
def bootstrap_ci_mean(values, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean of repetition-level values."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def bootstrap_ci_paired_diff(a, b, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean paired difference a-b and P(a>b)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diff.mean()), float(lo), float(hi), float((boot > 0).mean())


def bayesian_correlated_ttest(a, b, rope=0.01, test_fraction=0.2):
    """Bayesian correlated t-test (Benavoli et al., 2017) for repeated hold-out.

    Returns posterior probabilities that the mean difference (a-b) lies below
    -rope (b better), within [-rope, rope] (practically equivalent), or above
    rope (a better). The correlation heuristic rho = test_fraction corrects the
    naive variance for the overlap between resampled training sets.
    """
    x = np.asarray(a, float) - np.asarray(b, float)
    n = len(x)
    mean = x.mean()
    var = x.var(ddof=1)
    rho = test_fraction
    scale2 = var * (1.0 / n + rho / (1.0 - rho))
    if scale2 <= 0:
        # degenerate: all differences identical
        p_left = float(mean < -rope)
        p_rope = float(-rope <= mean <= rope)
        p_right = float(mean > rope)
        return {'p_left': p_left, 'p_rope': p_rope, 'p_right': p_right,
                'mean': float(mean)}
    scale = np.sqrt(scale2)
    df = n - 1
    cdf_hi = stats.t.cdf((rope - mean) / scale, df)
    cdf_lo = stats.t.cdf((-rope - mean) / scale, df)
    p_left = float(cdf_lo)
    p_rope = float(cdf_hi - cdf_lo)
    p_right = float(1 - cdf_hi)
    return {'p_left': p_left, 'p_rope': p_rope, 'p_right': p_right,
            'mean': float(mean), 'scale': float(scale)}
