"""Model ladder: Ridge -> RandomForest -> XGBoost, each run logged to MLflow.

    python -m src.run_ladder            # grouped CV on the training files only

All runs use the SAME rows (windows where every feature exists), so differences
come from the model/features, not from which rows survived. The held-out test
set is NOT touched here; see src/final_eval.py.
"""
import os
import subprocess

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.evaluation import (TARGETS, cv_summary, load_clean, oof_predictions,
                            split_train_test, within_record_report)

BASE = ['bpm', 'hrv']
PTT = ['ptt_foot', 'ptt_slope']
MORPH = ['amp', 'rise_time', 'up_slope', 'width50', 'area_ratio']
FEATURE_SETS = {
    'hr': BASE,
    'hr+ptt': BASE + PTT,
    'hr+morph': BASE + MORPH,
    'hr+ptt+morph': BASE + PTT + MORPH,
}
ALL_FEATURES = FEATURE_SETS['hr+ptt+morph']


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def forest():
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=20, n_jobs=-1, random_state=0)


def xgb(max_depth=4, n_estimators=300, learning_rate=0.05, min_child_weight=20):
    return MultiOutputRegressor(XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        min_child_weight=min_child_weight, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=0))


def _git_commit():
    try:
        return subprocess.check_output(['git', '-c', 'safe.directory=*', 'rev-parse', '--short', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 'unknown'


def log_run(run_name, model, feature_set, df, params=None):
    """Grouped CV for one (model, feature set); log params + metrics to MLflow."""
    feats = FEATURE_SETS[feature_set]
    d, oof, oof_dummy, fold_id = oof_predictions(model, df, feats)
    rep, fold_mae = cv_summary(d, oof, oof_dummy, fold_id)
    wr = within_record_report(d, oof)

    metrics = {'n_rows': len(d)}
    for t in TARGETS:
        metrics.update({
            f'cv_{t}_mae': rep.loc[t, 'mae'], f'cv_{t}_mae_dummy': rep.loc[t, 'mae_dummy'],
            f'cv_{t}_r2': rep.loc[t, 'r2'], f'cv_{t}_err_sd': rep.loc[t, 'sd'],
            f'cv_{t}_fold_mae_std': fold_mae[t].std(),
            f'within_{t}_r': wr.loc[t, 'within_r'], f'between_{t}_r': wr.loc[t, 'between_r'],
        })
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({'model': run_name.split(':')[0], 'feature_set': feature_set,
                           'n_features': len(feats), 'features': ','.join(feats),
                           'cv': 'GroupKFold(5) by record_id', **(params or {})})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.set_tags({'git_commit': _git_commit(), 'stage': 'cv'})
    return {'run': run_name, 'feature_set': feature_set, **metrics}


def main():
    mlflow.set_experiment('pulsetrace-ladder')
    train, _ = split_train_test(load_clean())
    train = train.dropna(subset=ALL_FEATURES)  # identical rows for every run

    plan = []
    for fs in FEATURE_SETS:
        plan.append(('ridge', ridge(), fs, {'alpha': 1.0}))
    plan.append(('forest', forest(), 'hr+ptt+morph', {'n_estimators': 100, 'min_samples_leaf': 20}))
    for fs in FEATURE_SETS:
        plan.append(('xgb', xgb(), fs, {'max_depth': 4, 'n_estimators': 300, 'lr': 0.05}))
    for depth, n_est, mcw in [(3, 300, 20), (6, 300, 20), (6, 600, 50)]:
        plan.append((f'xgb-d{depth}-n{n_est}-mcw{mcw}', xgb(depth, n_est, 0.05, mcw), 'hr+ptt+morph',
                     {'max_depth': depth, 'n_estimators': n_est, 'lr': 0.05, 'min_child_weight': mcw}))

    rows = []
    for name, model, fs, params in plan:
        r = log_run(f'{name}:{fs}', model, fs, train, params)
        rows.append(r)
        print(f"{r['run']:34s} sys MAE {r['cv_sys_mae']:.2f}  R2 {r['cv_sys_r2']:.3f} | "
              f"dia MAE {r['cv_dia_mae']:.2f}  R2 {r['cv_dia_r2']:.3f} | "
              f"within r {r['within_sys_r']:.2f}  between r {r['between_sys_r']:.2f}", flush=True)
    print(f"\nrows per run: {rows[0]['n_rows']:.0f} | dummy MAE sys {rows[0]['cv_sys_mae_dummy']:.2f} "
          f"dia {rows[0]['cv_dia_mae_dummy']:.2f}")


if __name__ == '__main__':
    main()
