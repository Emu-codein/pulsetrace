import numpy as np

from src.features.build_features import bandpass, get_bp_targets, get_bpm

FS = 125


def test_bandpass_removes_drift_and_keeps_heart_rate_band():
    t = np.arange(0, 60, 1 / FS)
    drift, beat = 5 * np.sin(2 * np.pi * 0.05 * t), np.sin(2 * np.pi * 1.2 * t)
    out = bandpass(drift + beat, fs=FS)
    assert np.abs(out - beat)[FS * 5:-FS * 5].max() < 0.1  # ignore the edges


def test_get_bpm_on_clean_sine():
    t = np.arange(0, 30, 1 / FS)
    record = np.vstack([np.sin(2 * np.pi * 1.2 * t)] * 3)  # 72 bpm
    bpm, n_peaks, _ = get_bpm(record, fs=FS)
    assert abs(bpm - 72) < 2 and 35 <= n_peaks <= 37


def test_get_bp_targets_recovers_known_levels():
    t = np.arange(0, 30, 1 / FS)
    abp = 100 + 20 * np.sin(2 * np.pi * 1.2 * t)  # sys ~120, dia ~80
    sys_, dia = get_bp_targets(abp)
    assert abs(sys_ - 120) < 1 and abs(dia - 80) < 1


def test_get_bp_targets_rejects_flatline_and_garbage():
    assert np.isnan(get_bp_targets(np.full(3750, 90.0))[0])
    t = np.arange(0, 30, 1 / FS)
    low = 40 + 5 * np.sin(2 * np.pi * 1.2 * t)  # sys 45: out of range
    assert np.isnan(get_bp_targets(low)[0])


def test_get_ptt_recovers_known_delay():
    from src.features.build_features import get_ptt
    n, fs, delay = 30 * FS, FS, 0.30
    ecg, ppg = np.zeros(n), np.zeros(n)
    t = np.arange(n) / fs
    for r in np.arange(1.0, 29.0, 0.8):                 # R-peaks every 0.8 s
        ecg[int(r * fs)] = 1.0
        # asymmetric pulse: foot at r+delay, fast rise then slower fall
        k = t - (r + delay)
        ppg += np.where(k > 0, (k / 0.15) * np.exp(1 - k / 0.15), 0)
    ecg = np.convolve(ecg, np.hanning(5), mode='same')
    foot, slope, n_beats = get_ptt(ppg, ecg, fs=fs)
    assert n_beats >= 15
    assert abs(foot - delay) < 0.05
    assert foot <= slope < foot + 0.15                  # steepest point is after the foot


def _pulse_train(tau=0.15, period=0.8, secs=30, noise=0.0, seed=0):
    t = np.arange(secs * FS) / FS
    ppg = np.zeros_like(t)
    for start in np.arange(0.5, secs - 1, period):
        k = t - start
        ppg += np.where(k > 0, (k / tau) * np.exp(1 - k / tau), 0)  # peaks at k = tau
    return ppg + np.random.default_rng(seed).normal(0, noise, len(t))


def test_get_morphology_recovers_rise_time_and_clean_sqi():
    from src.features.build_features import get_morphology
    m = get_morphology(_pulse_train(tau=0.15))
    assert abs(m['rise_time'] - 0.15) < 0.02
    assert m['sqi'] > 0.95 and m['morph_n'] >= 30
    assert 0 < m['area_ratio'] < 1 and m['width50'] > 0


def test_get_morphology_faster_rise_is_steeper_and_noise_lowers_sqi():
    from src.features.build_features import get_morphology
    fast, slow = get_morphology(_pulse_train(tau=0.10)), get_morphology(_pulse_train(tau=0.20))
    assert fast['rise_time'] < slow['rise_time'] and fast['up_slope'] > slow['up_slope']
    noisy = get_morphology(_pulse_train(tau=0.15, noise=0.4))
    assert noisy['sqi'] < get_morphology(_pulse_train(tau=0.15))['sqi']
