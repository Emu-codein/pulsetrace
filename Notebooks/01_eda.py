# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from src.features.build_features import get_bpm, process_file, get_bp_targets
import glob

# %%
mat = scipy.io.loadmat('../data/part_1.mat')
print(mat.keys())

# %%
p = mat['p']
print("p shape:", p.shape)
print("first record shape:", p[0, 0].shape)

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
peaks, properties = find_peaks(ppg, distance=40)

intervals_samples = np.diff(peaks)          # gap between each peak and the next, in samples
intervals_seconds = intervals_samples / fs  # same gaps, converted to seconds

mean_interval = intervals_seconds.mean()
implied_bpm = 60 / mean_interval

print(f"peaks found: {len(peaks)}")
print(f"mean interval: {mean_interval:.3f} sec  (min {intervals_seconds.min():.3f}, max {intervals_seconds.max():.3f})")
print(f"implied heart rate: {implied_bpm:.1f} bpm")

# %%
# built once by: python -m src.features.make_dataset  (one row per 30 s window)
df = pd.read_parquet('../data/features.parquet')

# %%
df

# %%
df['bpm'].describe()

# %%
lengths = []
for i in range(p.shape[1]):
    lengths.append(p[0, i][0].shape[0])

pd.Series(lengths).describe()

# %%
MIN_PEAKS = 15  # windows are 30 s, so 15 peaks = 30 bpm floor


df_clean = df[df['n_peaks'] >= MIN_PEAKS]
# keep severe hypertensive values (no range filter here); get_bp_targets already NaNs garbage windows
df_clean = df_clean.dropna(subset=['sys', 'dia'])
df_clean = df_clean[df_clean['dia'] < df_clean['sys']]
print(f"Dropped {len(df) - len(df_clean)} of {len(df)} windows ({( 1 -len(df_clean)/len(df))*100:.1f}%)")
df_clean['dia'].describe()

# %%
plt.figure(figsize=(10, 5))
plt.hist(df_clean['dia'], bins=50)
plt.xlabel('Diastolic BP (mmHg)')
plt.ylabel('Count')
plt.title('Distribution of Diastolic BP')
plt.show()

# %%
df_clean[df_clean['dia'] > 120]

# %% [markdown]
# # Feature engineering

# %%
get_bp_targets(p[0, 0][1][:30 * 125])  # takes one ABP window now, not a whole record

# %%
df_clean.describe()

# %% [markdown]
# # Model Training

# %%
train_files = [f'part_{i}.mat' for i in range(1, 11)]
test_files = [f'part_{i}.mat' for i in range(11, 13)]

train_df = df_clean[df_clean['file'].isin(train_files)]
test_df = df_clean[df_clean['file'].isin(test_files)]

features = ['bpm', 'hrv']  # n_peaks dropped: it tracks window/record length, not physiology
X_train, y_train = train_df[features], train_df[['sys', 'dia']]
X_test, y_test = test_df[features], test_df[['sys', 'dia']]

print(len(X_train), len(X_test))

# %%
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Naive baseline
dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)

# Real baseline model
model = LinearRegression()
model.fit(X_train, y_train)
model_pred = model.predict(X_test)

for i, target in enumerate(['sys', 'dia']):
    dummy_mae = mean_absolute_error(y_test[target], dummy_pred[:, i])
    model_mae = mean_absolute_error(y_test[target], model_pred[:, i])
    print(f"{target}: dummy MAE = {dummy_mae:.2f}, LinearRegression MAE = {model_mae:.2f}")

# %%
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

for i, target in enumerate(['sys', 'dia']):
    rf_mae = mean_absolute_error(y_test[target], rf_pred[:, i])
    print(f"{target}: RandomForest MAE = {rf_mae:.2f}")

# %% [markdown]
# # Evaluation protocol
# Grouped CV (by `record_id`) on parts 1-10, held-out test on parts 11-12. Logic lives in `src/evaluation.py`.

# %%
from sklearn.linear_model import LinearRegression
from src.evaluation import load_clean, split_train_test, cross_validate, evaluate_test

eval_df = load_clean('../data/features.parquet')
train_eval, _ = split_train_test(eval_df)
FEATURES = ['bpm', 'hrv']

cv_report, fold_mae = cross_validate(LinearRegression(), train_eval, FEATURES)
cv_report.round(3)

# %%
# Held-out test: run this once per finished model, not while iterating
evaluate_test(LinearRegression(), eval_df, FEATURES).round(3)

# %% [markdown]
# ## Within-record check
# Splits skill into *between* records (who is this patient) and *within* a record (does it track BP changes).

# %%
from sklearn.ensemble import RandomForestRegressor
from src.evaluation import oof_predictions, within_record_report

ALL = ['bpm', 'hrv', 'ptt_foot', 'ptt_slope', 'amp', 'rise_time', 'up_slope', 'width50', 'area_ratio']
rf = RandomForestRegressor(100, min_samples_leaf=20, n_jobs=-1, random_state=0)
d_used, oof, _, _ = oof_predictions(rf, train_eval, ALL)
within_record_report(d_used, oof).round(3)
