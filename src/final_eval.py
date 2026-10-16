"""Score the CHOSEN model once on the held-out test files (parts 11-12).

Model and features were fixed from grouped CV on parts 1-10 BEFORE running this.
Do not iterate against these numbers: if you change the model after looking, the
test set is no longer held out.

    python -m src.final_eval
"""
import mlflow
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor

from src.evaluation import (TARGETS, load_clean, report, split_train_test,
                            within_record_report)
from src.run_ladder import ALL_FEATURES, _git_commit, xgb

PARAMS = {'max_depth': 6, 'n_estimators': 600, 'learning_rate': 0.05, 'min_child_weight': 50}


def main():
    train, test = split_train_test(load_clean().dropna(subset=ALL_FEATURES))
    model = xgb(PARAMS['max_depth'], PARAMS['n_estimators'], PARAMS['learning_rate'],
                PARAMS['min_child_weight'])
    Xtr, ytr = train[ALL_FEATURES].to_numpy(), train[TARGETS].to_numpy()
    Xte, yte = test[ALL_FEATURES].to_numpy(), test[TARGETS].to_numpy()

    pred = clone(model).fit(Xtr, ytr).predict(Xte)
    dummy = DummyRegressor(strategy='mean').fit(Xtr, ytr).predict(Xte)
    rep = report(yte, pred, dummy)
    wr = within_record_report(test, pred)
    pd.set_option('display.width', 200)
    print(f'train rows {len(train)} | test rows {len(test)} | test records {test.record_id.nunique()}')
    print(rep.round(3).to_string())
    print(wr.round(3).to_string())

    mlflow.set_experiment('pulsetrace-final')
    with mlflow.start_run(run_name='xgb-d6-n600-mcw50:hr+ptt+morph:TEST'):
        mlflow.log_params({'model': 'xgb', 'feature_set': 'hr+ptt+morph', **PARAMS,
                           'features': ','.join(ALL_FEATURES)})
        for t in TARGETS:
            mlflow.log_metrics({f'test_{t}_mae': rep.loc[t, 'mae'], f'test_{t}_mae_dummy': rep.loc[t, 'mae_dummy'],
                                f'test_{t}_r2': rep.loc[t, 'r2'], f'test_{t}_err_sd': rep.loc[t, 'sd'],
                                f'test_{t}_within_r': wr.loc[t, 'within_r'],
                                f'test_{t}_between_r': wr.loc[t, 'between_r']})
        mlflow.set_tags({'git_commit': _git_commit(), 'stage': 'test'})


if __name__ == '__main__':
    main()
