import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from src.evaluation import cross_validate, report


def _fake_df(n_records=40, windows=25, seed=0):
    rng = np.random.default_rng(seed)
    rid = np.repeat([f'part_{i % 8}.mat:{i}' for i in range(n_records)], windows)
    file = np.repeat([f'part_{i % 8}.mat' for i in range(n_records)], windows)
    level = np.repeat(rng.normal(130, 20, n_records), windows)  # per-record BP level
    return pd.DataFrame({
        'record_id': rid,
        'file': file,
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


def test_default_cv_never_splits_a_file_across_folds():
    from src.evaluation import oof_predictions
    df = _fake_df()
    d, _, _, fold_id = oof_predictions(LinearRegression(), df, ['bpm'])
    assert (pd.Series(fold_id).groupby(d['record_id']).nunique() == 1).all()  # so is each record
    assert (pd.Series(fold_id).groupby(d['file']).nunique() == 1).all()       # each file in exactly one fold


def test_lofo_every_file_predicted_once_and_report_shape():
    from src.evaluation import oof_predictions, per_file_report
    df = _fake_df()
    d, oof, oof_dummy, fold_id = oof_predictions(LinearRegression(), df, ['bpm'], n_splits=df['file'].nunique())
    assert len(np.unique(fold_id)) == df['file'].nunique()           # one fold per file
    pf = per_file_report(d, oof, oof_dummy)
    assert len(pf) == df['file'].nunique() and {'sys_gain', 'dia_gain'} <= set(pf.columns)
    assert np.allclose(pf['sys_gain'], pf['sys_dummy'] - pf['sys_mae'])


def test_stable_features_keeps_consistent_sign_only():
    from src.run_stability import stable_features
    rng = np.random.default_rng(0)
    rows = []
    for f in range(10):
        n = 400
        sys_ = rng.normal(130, 15, n)
        rows.append(pd.DataFrame({
            'file': f'part_{f}.mat', 'sys': sys_,
            'stable': -0.5 * (sys_ - 130) / 15 + rng.normal(0, 1, n),            # always negative
            'flips': (1 if f % 2 else -1) * 0.5 * (sys_ - 130) / 15 + rng.normal(0, 1, n),  # sign flips by file
            'noise': rng.normal(0, 1, n),
        }))
    df = pd.concat(rows, ignore_index=True)
    assert stable_features(df, ['stable', 'flips', 'noise'], 'sys') == ['stable']
