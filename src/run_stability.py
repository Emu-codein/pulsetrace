"""Does using only feature relationships that are STABLE across files generalise better?

Inside each leave-one-file-out fold, features are chosen using ONLY the 11 training
files (per-file correlation with BP must have a consistent sign), then the model is
fit on them and scored on the held-out file. The held-out file never influences the
selection, so the comparison against the all-features model is fair.

    python -m src.run_stability
"""
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor

from src.evaluation import TARGETS, load_clean
from src.run_lofo import log_lofo, print_line, run
from src.run_ladder import ALL_FEATURES, ridge, xgb

MIN_SIGN_FRAC = 0.9   # fixed in advance, not tuned
MIN_ABS_CORR = 0.1
PTT_ONLY = ['ptt_foot', 'ptt_slope']  # physiology prior (PTT shortens as BP rises)


def stable_features(train, features, target, min_sign_frac=MIN_SIGN_FRAC, min_abs=MIN_ABS_CORR):
    """Features whose per-file correlation with `target` keeps one sign across files."""
    c = train.groupby('file')[features + [target]].corr()[target].unstack().drop(columns=target)
    mean = c.mean()
    same_sign = (np.sign(c) == np.sign(mean)).mean()
    return [f for f in features if same_sign[f] >= min_sign_frac and abs(mean[f]) >= min_abs]


def lofo_selected(df, make_model, select):
    """LOFO where `select(train, target)` picks the features used for each target."""
    files = sorted(df['file'].unique(), key=lambda x: int(x[5:-4]))
    oof = np.full((len(df), 2), np.nan)
    oof_dummy = np.full((len(df), 2), np.nan)
    fold_id = np.zeros(len(df), dtype=int)
    chosen = {t: [] for t in TARGETS}
    for k, f in enumerate(files):
        tr, va = df['file'] != f, df['file'] == f
        fold_id[va.to_numpy()] = k
        for j, t in enumerate(TARGETS):
            feats = select(df[tr], t)
            chosen[t].append(feats)
            oof_dummy[va.to_numpy(), j] = DummyRegressor().fit(df.loc[tr, feats or ['bpm']], df.loc[tr, t]).predict(df.loc[va, feats or ['bpm']])
            if feats:
                oof[va.to_numpy(), j] = clone(make_model()).fit(df.loc[tr, feats], df.loc[tr, t]).predict(df.loc[va, feats])
            else:  # nothing stable enough: fall back to predicting the training mean
                oof[va.to_numpy(), j] = oof_dummy[va.to_numpy(), j]
    return oof, oof_dummy, fold_id, chosen


def main():
    mlflow.set_experiment('pulsetrace-lofo')
    df = load_clean().dropna(subset=ALL_FEATURES).reset_index(drop=True)
    n_folds = df['file'].nunique()

    m, _ = run('xgb-d3', xgb(max_depth=3), 'hr+ptt+morph', df)
    print_line('xgb-d3: all features', m)

    base_xgb = lambda: xgb(max_depth=3).estimator   # single-target XGBRegressor
    experiments = [
        ('xgb-d3-stable', base_xgb, lambda tr, t: stable_features(tr, ALL_FEATURES, t)),
        ('ridge-stable', ridge, lambda tr, t: stable_features(tr, ALL_FEATURES, t)),
        ('ridge-ptt-only', ridge, lambda tr, t: PTT_ONLY),
        ('xgb-d3-ptt-only', base_xgb, lambda tr, t: PTT_ONLY),
    ]
    for name, make_model, select in experiments:
        oof, oof_dummy, fold_id, chosen = lofo_selected(df, make_model, select)
        label = 'stable (chosen per fold)' if 'stable' in name else 'ptt_foot,ptt_slope'
        m, pf = log_lofo(f'{name}', name.split('-')[0], label, df, oof, oof_dummy, fold_id,
                         {'min_sign_frac': MIN_SIGN_FRAC, 'min_abs_corr': MIN_ABS_CORR})
        print_line(name, m)
        if 'stable' in name:
            for t in TARGETS:
                freq = pd.Series([f for feats in chosen[t] for f in feats]).value_counts()
                nones = sum(1 for feats in chosen[t] if not feats)
                print(f'    {t}: selected in N of {n_folds} folds -> ' +
                      (', '.join(f'{k}:{v}' for k, v in freq.items()) or 'none') +
                      (f' | folds with no features: {nones}' if nones else ''))


if __name__ == '__main__':
    main()
