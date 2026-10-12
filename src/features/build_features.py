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
        for w, start in enumerate(range(0, record.shape[1] - win + 1, win)):
            seg = record[:, start:start + win]
            bpm, n_peaks, hrv = get_bpm(seg, fs=fs, distance=distance)
            systolic, diastolic = get_bp_targets(seg[1], distance=distance)
            results.append({
                'file': filepath,
                'record_idx': i,
                'window_idx': w,
                'n_peaks': n_peaks,
                'bpm': bpm,
                'hrv': hrv,
                'sys': systolic,
                'dia': diastolic
            })
    return results
