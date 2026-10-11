"""Parse every data/part_*.mat once and cache the window-level feature table.

Run from the repo root (inside the container: /app):
    python -m src.features.make_dataset
"""
import glob
import os
import sys

import pandas as pd

from src.features.build_features import process_file

DATA_DIR = "data"
OUT_PATH = os.path.join(DATA_DIR, "features.parquet")


def main(data_dir=DATA_DIR, out_path=OUT_PATH):
    paths = sorted(glob.glob(os.path.join(data_dir, "part_*.mat")),
                   key=lambda p: int(os.path.basename(p)[5:-4]))
    rows = []
    for path in paths:
        print(f"parsing {path}", file=sys.stderr, flush=True)
        rows.extend(process_file(path))

    df = pd.DataFrame(rows)
    df["file"] = df["file"].map(os.path.basename)          # 'part_1.mat', not a machine-specific path
    df["record_id"] = df["file"] + ":" + df["record_idx"].astype(str)  # group key for GroupKFold
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df)} windows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
