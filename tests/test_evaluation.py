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


def _preds(df, kind):
    y = df[['sys', 'dia']].to_numpy()
    if kind == 'perfect':
        return y
    means = df.groupby('record_id')[['sys', 'dia']].transform('mean').to_numpy()
    return means  # knows who the patient is, tracks nothing inside a record


def test_within_record_separates_patient_level_from_tracking():
    from src.evaluation import within_record_report
    df = _fake_df()
    perfect = within_record_report(df, _preds(df, 'perfect'))
    assert perfect.loc['sys', 'within_r2'] > 0.999 and perfect.loc['sys', 'between_r2'] > 0.999

    level_only = within_record_report(df, _preds(df, 'means'))
    assert abs(level_only.loc['sys', 'within_r2']) < 1e-9      # no tracking at all
    assert level_only.loc['sys', 'between_r2'] > 0.999          # but perfect patient level


def test_within_record_drops_short_records():
    from src.evaluation import within_record_report
    short = _fake_df(n_records=10, windows=3, seed=1)
    long_ = _fake_df(n_records=5, windows=10, seed=2)
    long_['record_id'] = 'long:' + long_['record_id']
    both = pd.concat([short, long_], ignore_index=True)
    assert within_record_report(both, _preds(both, 'perfect'), min_windows=5).loc['sys', 'n_records'] == 5

    none = within_record_report(short, _preds(short, 'perfect'), min_windows=5)
    assert none.loc['sys', 'n_records'] == 0 and none['within_r2'].isna().all()
