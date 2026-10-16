"""Leave-one-file-out evaluation over ALL 12 files; final reporting protocol.

    python -m src.run_lofo

Each file is held out once and predicted by a model trained on the other 11, so
every window gets an out-of-file prediction and no separate test set is burned.
Models are a small fixed list (no tuning on these numbers).
"""
import mlflow
import pandas as pd

from src.evaluation import (TARGETS, cv_summary, load_clean, oof_predictions,
                            per_file_report, within_record_report)
from src.run_ladder import ALL_FEATURES, FEATURE_SETS, _git_commit, forest, ridge, xgb


def run(name, model, feature_set, df):
    feats = FEATURE_SETS[feature_set]
    d, oof, oof_dummy, fold_id = oof_predictions(model, df, feats, n_splits=df['file'].nunique())
    rep, _ = cv_summary(d, oof, oof_dummy, fold_id)
    pf = per_file_report(d, oof, oof_dummy)
    wr = within_record_report(d, oof)

    m = {'n_rows': len(d), 'n_files': len(pf)}
    for t in TARGETS:
        m.update({f'lofo_{t}_mae': rep.loc[t, 'mae'], f'lofo_{t}_mae_dummy': rep.loc[t, 'mae_dummy'],
                  f'lofo_{t}_r2': rep.loc[t, 'r2'], f'lofo_{t}_err_sd': rep.loc[t, 'sd'],
                  f'lofo_{t}_gain_mean': pf[f'{t}_gain'].mean(), f'lofo_{t}_gain_std': pf[f'{t}_gain'].std(),
                  f'lofo_{t}_files_better': int((pf[f'{t}_gain'] > 0).sum()),
                  f'within_{t}_r': wr.loc[t, 'within_r'], f'between_{t}_r': wr.loc[t, 'between_r']})
    with mlflow.start_run(run_name=f'{name}:{feature_set}'):
        mlflow.log_params({'model': name, 'feature_set': feature_set, 'features': ','.join(feats),
                           'cv': 'leave-one-file-out'})
        mlflow.log_metrics({k: float(v) for k, v in m.items()})
        mlflow.set_tags({'git_commit': _git_commit(), 'stage': 'lofo'})
    return m, pf


def main():
    mlflow.set_experiment('pulsetrace-lofo')
    df = load_clean().dropna(subset=ALL_FEATURES).reset_index(drop=True)
    plan = [('ridge', ridge(), 'hr'), ('ridge', ridge(), 'hr+ptt+morph'),
            ('forest', forest(), 'hr+ptt+morph'),
            ('xgb-d3', xgb(max_depth=3), 'hr+ptt+morph'), ('xgb-d4', xgb(max_depth=4), 'hr+ptt+morph')]
    per_file = {}
    for name, model, fs in plan:
        m, pf = run(name, model, fs, df)
        per_file[f'{name}:{fs}'] = pf
        print(f"{name+':'+fs:24s} sys MAE {m['lofo_sys_mae']:.2f} (dummy {m['lofo_sys_mae_dummy']:.2f}) "
              f"gain {m['lofo_sys_gain_mean']:+.2f}±{m['lofo_sys_gain_std']:.2f} wins {m['lofo_sys_files_better']}/{m['n_files']} | "
              f"dia MAE {m['lofo_dia_mae']:.2f} (dummy {m['lofo_dia_mae_dummy']:.2f}) "
              f"gain {m['lofo_dia_gain_mean']:+.2f}±{m['lofo_dia_gain_std']:.2f} wins {m['lofo_dia_files_better']}/{m['n_files']}",
              flush=True)
    best = per_file['xgb-d3:hr+ptt+morph']
    pd.set_option('display.width', 200)
    print('\nper-file detail, xgb-d3 (gain = dummy MAE - model MAE):')
    print(best[['n_windows', 'sys_mae', 'sys_dummy', 'sys_gain', 'dia_mae', 'dia_dummy', 'dia_gain']].round(2).to_string())


if __name__ == '__main__':
    main()
