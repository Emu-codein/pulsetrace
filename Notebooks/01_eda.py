# %%
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# %%
mat = scipy.io.loadmat('../data/part_1.mat')
print(mat.keys())

# %%
p = mat['p']
print("p shape:", p.shape)
print("first record shape:", p[0, 0].shape)

# %%
mat

# %%
record = p[0, 0]
ppg, abp, ecg = record[0], record[1], record[2]

fs = 125  # Hz
duration_sec = len(ppg) / fs
print(f"record length: {len(ppg)} samples, {duration_sec:.1f} sec")

fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
axes[0].plot(ppg); axes[0].set_ylabel('PPG')
axes[1].plot(abp); axes[1].set_ylabel('ABP (mmHg)')
axes[2].plot(ecg); axes[2].set_ylabel('ECG')
axes[2].set_xlabel('sample')
plt.tight_layout()
plt.show()

# %%
import numpy as np

peaks, properties = find_peaks(ppg, distance=40)

intervals_samples = np.diff(peaks)          # gap between each peak and the next, in samples
intervals_seconds = intervals_samples / fs  # same gaps, converted to seconds

mean_interval = intervals_seconds.mean()
implied_bpm = 60 / mean_interval

print(f"peaks found: {len(peaks)}")
print(f"mean interval: {mean_interval:.3f} sec  (min {intervals_seconds.min():.3f}, max {intervals_seconds.max():.3f})")
print(f"implied heart rate: {implied_bpm:.1f} bpm")
