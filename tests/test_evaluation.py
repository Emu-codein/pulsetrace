import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from src.evaluation import cross_validate, report


def _fake_df(n_records=40, windows=25, seed=0):
    rng = np.random.default_rng(seed)
    rid = np.repeat([f'part_1.mat:{i}' for i in range(n_records)], windows)
    level = np.repeat(rng.normal(130, 20, n_records), windows)  # per-record BP level
    return pd.DataFrame({
        'record_id': rid,
        'bpm': rng.normal(80, 10, len(rid)),
        'sys': level + rng.normal(0, 2, len(rid)),
        'dia': level / 2 + rng.normal(0, 2, len(rid)),
    })


def test_no_record_in_both_train_and_val():
    df = _fake_df()
    for tr, va in GroupKFold(5).split(df, groups=df['record_id']):
        assert set(df['record_id'].iloc[tr]).isdisjoint(df['record_id'].iloc[va])


def test_noise_feature_cannot_beat_dummy_under_grouped_cv():
    # bpm is pure noise and BP varies per record; with grouped CV there is nothing to memorise
    rep, folds = cross_validate(LinearRegression(), _fake_df(), ['bpm'])
    assert rep.loc['sys', 'mae'] >= rep.loc['sys', 'mae_dummy'] * 0.97
    assert folds.shape == (5, 2)


def test_report_perfect_prediction():
    y = np.array([[120., 80.], [140., 90.], [110., 70.]])
    rep = report(y, y, np.full_like(y, y.mean(axis=0)[0]))
    assert rep.loc['sys', 'mae'] == 0 and bool(rep.loc['sys', 'aami_pass'])
