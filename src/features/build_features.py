from scipy.signal import butter, find_peaks, sosfiltfilt
import numpy as np
import scipy.io

def bandpass(x, fs=125, low=0.5, high=8.0, order=3):
    """Zero-phase bandpass. HR is 0.5-3 Hz; up to 8 Hz keeps the beat shape.

    sosfiltfilt runs the filter forwards then backwards, so there is no time
    delay (a delay would bias PTT against the ECG later).
    """
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def get_bpm(record, fs= 125, distance=40, prominence=0.5):
    ppg = record[0]
    # PPG is in arbitrary units, so z-score first: prominence is then in std units
    z = (ppg - ppg.mean()) / (ppg.std() + 1e-9)
    peaks,_ = find_peaks(z, distance = distance, prominence = prominence)
    n_peaks = len(peaks)

    if n_peaks < 2 :
        return np.nan, n_peaks, np.nan

    intervals = np.diff(peaks)/fs
    hrv = intervals.std()
    bpm = 60/ intervals.mean()
    return bpm, n_peaks, hrv

def get_ptt(ppg, ecg, fs=125, min_beats=15, lo=0.15, hi=0.7):
    """Median pulse transit time per window, from filtered PPG and ECG.

    For each ECG R-peak we take the NEXT PPG foot before the following R-peak
    (matching by time, not by index, since the two detectors miss different beats).
    Two landmarks on the pulse: the foot (trough) and the max-slope point of the
    upstroke. The slope point is a sharper target and correlates better with BP.
    Returns (ptt_foot, ptt_slope, n_beats) in seconds.
    """
    zp = (ppg - ppg.mean()) / (ppg.std() + 1e-9)
    ze = (ecg - ecg.mean()) / (ecg.std() + 1e-9)
    gap = int(0.3 * fs)
    r, _ = find_peaks(np.abs(ze), distance=gap, prominence=1.0)  # abs: lead polarity varies
    feet, _ = find_peaks(-zp, distance=gap, prominence=0.3)
    peaks, _ = find_peaks(zp, distance=gap, prominence=0.5)

    foot_t, slope_t = [], []
    for a, b in zip(r[:-1], r[1:]):
        f = feet[(feet > a) & (feet < b)]
        if not len(f):
            continue
        f = f[0]
        q = peaks[peaks > f]
        if not len(q) or q[0] >= b + gap:
            continue
        rise = np.diff(zp[f:q[0] + 1])
        if not len(rise):
            continue
        foot_t.append((f - a) / fs)
        slope_t.append((f + rise.argmax() - a) / fs)

    foot_t, slope_t = np.array(foot_t), np.array(slope_t)
    ok = (foot_t > lo) & (foot_t < hi)  # drop T-wave / wrong-beat matches
    if ok.sum() < min_beats:
        return np.nan, np.nan, int(ok.sum())
    return np.median(foot_t[ok]), np.median(slope_t[ok]), int(ok.sum())

MORPH_COLS = ['amp', 'rise_time', 'up_slope', 'width50', 'area_ratio', 'sqi', 'morph_n']

def get_morphology(ppg, fs=125, min_beats=10, n_template=100):
    """Per-beat PPG shape features (foot to foot), aggregated by window median.

    amp        peak height above the foot, in window-std units (PPG units are arbitrary)
    rise_time  foot -> peak, seconds
    up_slope   steepest rise / amp, 1/s (shape only, independent of amplitude)
    width50    time the pulse spends above half its height, seconds
    area_ratio share of the pulse area before the peak (how front-loaded the beat is)
    sqi        median correlation of each beat with the window's median beat
               (template matching): ~1 clean, low = noisy or artefact
    """
    nan = dict.fromkeys(MORPH_COLS, np.nan)
    zp = (ppg - ppg.mean()) / (ppg.std() + 1e-9)
    feet, _ = find_peaks(-zp, distance=int(0.3 * fs), prominence=0.3)

    feats, shapes = [], []
    for f0, f1 in zip(feet[:-1], feet[1:]):
        if not 0.3 <= (f1 - f0) / fs <= 1.5:
            continue
        beat = zp[f0:f1 + 1] - zp[f0]
        pk = beat.argmax()
        amp, area = beat[pk], beat.sum()
        if amp <= 0 or pk == 0 or pk >= len(beat) - 1 or area <= 0:
            continue
        above = np.flatnonzero(beat >= amp / 2)
        feats.append((amp, pk / fs, np.diff(beat[:pk + 1]).max() * fs / amp,
                      (above[-1] - above[0]) / fs, beat[:pk + 1].sum() / area))
        shapes.append(np.interp(np.linspace(0, 1, n_template),
                                np.linspace(0, 1, len(beat)), beat / amp))

    if len(feats) < min_beats:
        return nan
    shapes = np.array(shapes)
    template = np.median(shapes, axis=0)
    sqi = np.median([np.corrcoef(b, template)[0, 1] for b in shapes])
    amp, rise, slope, width, ratio = np.median(feats, axis=0)
    return dict(amp=amp, rise_time=rise, up_slope=slope, width50=width,
                area_ratio=ratio, sqi=sqi, morph_n=len(feats))

def get_bp_targets(abp, distance=40, prominence=10,
                   sbp_range=(60, 250), dbp_range=(30, 150), min_pp=20):
    """Median SBP/DBP for one ABP window; NaN if the window looks like garbage."""
    if len(abp) == 0 or not np.isfinite(abp).all() or np.ptp(abp) < 1e-6:
        return np.nan, np.nan  # dropout / flat-line

    sys_peaks, _ = find_peaks(abp, distance=distance, prominence=prominence)
    dia_peaks, _ = find_peaks(-abp, distance=distance, prominence=prominence)

    if len(sys_peaks) < 2 or len(dia_peaks) < 2:
        return np.nan, np.nan

    systolic = np.median(abp[sys_peaks])
    diastolic = np.median(abp[dia_peaks])

    if not (sbp_range[0] <= systolic <= sbp_range[1]
            and dbp_range[0] <= diastolic <= dbp_range[1]
            and systolic - diastolic >= min_pp):
        return np.nan, np.nan

    return systolic, diastolic

def process_file(filepath, fs=125, distance=40, window_s=30):
    mat = scipy.io.loadmat(filepath)
    p = mat['p']
    n_records = p.shape[1]
    win = int(window_s * fs)

    results = []
    for i in range(n_records):
        record = p[0, i].astype(float)
        if record.shape[1] >= win:
            # filter the whole record, then cut windows (no edge artefacts per window)
            record[0] = bandpass(record[0], fs=fs)
            record[2] = bandpass(record[2], fs=fs, low=5, high=20)  # QRS band, for R-peaks
        for w, start in enumerate(range(0, record.shape[1] - win + 1, win)):
            seg = record[:, start:start + win]
            bpm, n_peaks, hrv = get_bpm(seg, fs=fs, distance=distance)
            systolic, diastolic = get_bp_targets(seg[1], distance=distance)
            ptt_foot, ptt_slope, ptt_n = get_ptt(seg[0], seg[2], fs=fs)
            morph = get_morphology(seg[0], fs=fs)
            results.append({
                'file': filepath,
                'record_idx': i,
                'window_idx': w,
                'n_peaks': n_peaks,
                'bpm': bpm,
                'hrv': hrv,
                'ptt_foot': ptt_foot,
                'ptt_slope': ptt_slope,
                'ptt_n': ptt_n,
                **morph,
                'sys': systolic,
                'dia': diastolic
            })
    return results
