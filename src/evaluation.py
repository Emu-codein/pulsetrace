"""Evaluation protocol: grouped CV on the training files + a held-out test set.

Windows from the same record are near-duplicates, so every split here is made
on `record_id` (never on windows). Otherwise the model sees part of a record in
training and is tested on the rest of it, and scores look better than they are.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

TARGETS = ['sys', 'dia']
TRAIN_FILES = [f'part_{i}.mat' for i in range(1, 11)]
TEST_FILES = [f'part_{i}.mat' for i in range(11, 13)]

# AAMI/ISO 81060-2: mean error within 5 mmHg and error SD within 8 mmHg.
# (Applied per window here, so treat it as an approximation of the real test.)
AAMI_ME, AAMI_SD = 5.0, 8.0


def load_clean(path='data/features.parquet', min_peaks=15):
    """Load the cached window table and apply the cleaning rules."""
    df = pd.read_parquet(path)
    df = df[df['n_peaks'] >= min_peaks].dropna(subset=TARGETS)
    return df[df['dia'] < df['sys']].reset_index(drop=True)


def split_train_test(df):
    """Fixed split by source file. The test set is only used at the very end."""
    return df[df['file'].isin(TRAIN_FILES)].reset_index(drop=True), \
           df[df['file'].isin(TEST_FILES)].reset_index(drop=True)


def report(y_true, y_pred, y_dummy):
    """One row per target: model vs the mean-predictor, plus the AAMI check."""
    rows = {}
    for j, t in enumerate(TARGETS):
        err = y_pred[:, j] - y_true[:, j]
        me, sd = err.mean(), err.std()
        rows[t] = {
            'mae': mean_absolute_error(y_true[:, j], y_pred[:, j]),
            'mae_dummy': mean_absolute_error(y_true[:, j], y_dummy[:, j]),
            'r2': r2_score(y_true[:, j], y_pred[:, j]),
            'r2_dummy': r2_score(y_true[:, j], y_dummy[:, j]),
            'me': me,
            'sd': sd,
            'aami_pass': bool(abs(me) <= AAMI_ME and sd <= AAMI_SD),
        }
    return pd.DataFrame(rows).T


def oof_predictions(model, df, features, n_splits=5):
    """Out-of-fold predictions via GroupKFold on record_id.

    Returns (df_used, oof, oof_dummy, fold_id); rows with NaN features are dropped,
    so always use the returned df_used to line things up with the predictions.
    """
    df = df.dropna(subset=features).reset_index(drop=True)
    X = df[features].to_numpy()
    y = df[TARGETS].to_numpy()
    groups = df['record_id'].to_numpy()

    oof = np.empty_like(y, dtype=float)
    oof_dummy = np.empty_like(y, dtype=float)
    fold_id = np.empty(len(df), dtype=int)
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        assert set(groups[tr]).isdisjoint(groups[va]), 'record leaked across folds'
        oof[va] = clone(model).fit(X[tr], y[tr]).predict(X[va])
        oof_dummy[va] = DummyRegressor(strategy='mean').fit(X[tr], y[tr]).predict(X[va])
        fold_id[va] = k
    return df, oof, oof_dummy, fold_id


def cross_validate(model, df, features, n_splits=5):
    """GroupKFold on record_id over the training data. Returns (report, fold MAEs)."""
    df, oof, oof_dummy, fold_id = oof_predictions(model, df, features, n_splits)
    y = df[TARGETS].to_numpy()
    fold_mae = pd.DataFrame([np.abs(oof[fold_id == k] - y[fold_id == k]).mean(axis=0)
                             for k in range(n_splits)], columns=TARGETS)
    return report(y, oof, oof_dummy), fold_mae


def within_record_report(df, pred, min_windows=5):
    """Split model skill into 'which patient is this' vs 'does it track BP changes'.

    Per record, subtract that record's mean from the true and predicted BP:
      within  = the ups and downs inside a recording (the real physiology)
      between = the per-record averages (partly a patient fingerprint)
    within_r2 compares to predicting no change at all (R2 = 0). Records with fewer
    than `min_windows` windows are dropped: their deviations are mostly noise.
    """
    d = df[['record_id'] + TARGETS].reset_index(drop=True).copy()
    for j, t in enumerate(TARGETS):
        d[f'{t}_pred'] = pred[:, j]
    d = d[d.groupby('record_id')['sys'].transform('size') >= min_windows]
    if d.empty:  # nothing long enough to measure
        cols = ['n_records', 'within_sd', 'within_r', 'within_r2', 'between_sd',
                'between_r', 'between_r2', 'within_var_share']
        return pd.DataFrame(np.nan, index=TARGETS, columns=cols).assign(n_records=0)
    g = d.groupby('record_id')

    rows = {}
    for t in TARGETS:
        true_dev = d[t] - g[t].transform('mean')
        pred_dev = d[f'{t}_pred'] - g[f'{t}_pred'].transform('mean')
        rec = g[[t, f'{t}_pred']].mean()
        rows[t] = {
            'n_records': len(rec),
            'within_sd': true_dev.std(),
            'within_r': np.corrcoef(true_dev, pred_dev)[0, 1],
            'within_r2': 1 - ((true_dev - pred_dev) ** 2).sum() / (true_dev ** 2).sum(),
            'between_sd': rec[t].std(),
            'between_r': np.corrcoef(rec[t], rec[f'{t}_pred'])[0, 1],
            'between_r2': r2_score(rec[t], rec[f'{t}_pred']),
            'within_var_share': (true_dev ** 2).sum() / ((d[t] - d[t].mean()) ** 2).sum(),
        }
    return pd.DataFrame(rows).T


def evaluate_test(model, df, features):
    """Fit on all training files, score once on the held-out files."""
    train, test = split_train_test(df.dropna(subset=features))
    m = clone(model).fit(train[features].to_numpy(), train[TARGETS].to_numpy())
    dummy = DummyRegressor(strategy='mean').fit(train[features].to_numpy(), train[TARGETS].to_numpy())
    X, y = test[features].to_numpy(), test[TARGETS].to_numpy()
    return report(y, m.predict(X), dummy.predict(X))
