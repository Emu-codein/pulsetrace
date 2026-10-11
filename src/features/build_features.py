from scipy.signal import find_peaks
import numpy as np
import scipy.io

def get_bpm(record, fs= 125, distance=40):
    ppg = record[0]
    peaks,_ = find_peaks(ppg, distance = distance)
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
        record = p[0, i]
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
