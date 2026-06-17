"""
Data loader for the multi-centre UCI Heart Disease cohorts.

The widely used Kaggle "Heart Failure Prediction" dataset (heart.csv, 918 rows,
11 predictors) is the de-duplicated union of five classic public cohorts. Working
from the per-centre UCI sources (rather than the pre-pooled CSV) is exactly what
enables genuine, leakage-free, inter-institutional external validation, because
each row traces to a named institution.

This module maps every centre onto the SAME 11-feature schema used by the Kaggle
benchmark:
    Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
    RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
with binary target HeartDisease (0 = no disease, 1 = disease present).
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

RAW_COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

FEATURES = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
TARGET = 'HeartDisease'

CP_MAP = {1: 'TA', 2: 'ATA', 3: 'NAP', 4: 'ASY'}
RESTECG_MAP = {0: 'Normal', 1: 'ST', 2: 'LVH'}
SLOPE_MAP = {1: 'Up', 2: 'Flat', 3: 'Down'}
SEX_MAP = {1: 'M', 0: 'F'}
EXANG_MAP = {1: 'Y', 0: 'N'}

CENTRES = {
    'Cleveland':   'processed.cleveland.data',
    'Hungarian':   'processed.hungarian.data',
    'Switzerland': 'processed.switzerland.data',
    'VA':          'processed.va.data',
}


def _map_centre(path):
    df = pd.read_csv(path, header=None, names=RAW_COLS, na_values='?')
    out = pd.DataFrame()
    out['Age'] = pd.to_numeric(df['age'], errors='coerce')
    out['Sex'] = df['sex'].map(SEX_MAP)
    out['ChestPainType'] = df['cp'].map(CP_MAP)
    out['RestingBP'] = pd.to_numeric(df['trestbps'], errors='coerce')
    out['Cholesterol'] = pd.to_numeric(df['chol'], errors='coerce')
    out['FastingBS'] = pd.to_numeric(df['fbs'], errors='coerce')
    out['RestingECG'] = df['restecg'].map(RESTECG_MAP)
    out['MaxHR'] = pd.to_numeric(df['thalach'], errors='coerce')
    out['ExerciseAngina'] = df['exang'].map(EXANG_MAP)
    out['Oldpeak'] = pd.to_numeric(df['oldpeak'], errors='coerce')
    out['ST_Slope'] = df['slope'].map(SLOPE_MAP)
    num = pd.to_numeric(df['num'], errors='coerce')
    out[TARGET] = (num.fillna(0) > 0).astype(int)
    return out


def load_centres():
    """Return dict {centre_name: DataFrame} on the 11-feature schema."""
    return {name: _map_centre(os.path.join(DATA_DIR, fn)) for name, fn in CENTRES.items()}


def load_pooled(add_centre_col=False):
    """Return the pooled development cohort (all centres concatenated)."""
    cents = load_centres()
    frames = []
    for name, df in cents.items():
        d = df.copy()
        if add_centre_col:
            d['Centre'] = name
        frames.append(d)
    pooled = pd.concat(frames, ignore_index=True)
    return pooled


if __name__ == '__main__':
    cents = load_centres()
    print("Per-centre summary (11-feature schema):")
    print(f"{'Centre':<12}{'n':>5}{'prev':>8}{'%miss':>8}")
    tot = 0
    for name, df in cents.items():
        n = len(df)
        tot += n
        prev = df[TARGET].mean()
        miss = df[FEATURES].isna().mean().mean() * 100
        print(f"{name:<12}{n:>5}{prev:>8.3f}{miss:>8.1f}")
    pooled = load_pooled()
    print(f"{'POOLED':<12}{len(pooled):>5}{pooled[TARGET].mean():>8.3f}"
          f"{pooled[FEATURES].isna().mean().mean()*100:>8.1f}")
    print("\nPer-feature missingness in pooled (%):")
    print((pooled[FEATURES].isna().mean() * 100).round(1).to_string())
    print("\nDtypes:")
    print(pooled.dtypes.to_string())
