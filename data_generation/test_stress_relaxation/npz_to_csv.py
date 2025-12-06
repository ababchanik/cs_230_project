#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def npz_to_wide_csv(npz_path):
    """Convert all arrays in an .npz into one wide CSV (arrays stacked column-wise)."""
    npz_path = Path(npz_path)
    out_csv = npz_path.with_suffix(".csv")

    data = np.load(npz_path, allow_pickle=True)

    df_parts = []

    for key in data.files:
        arr = np.array(data[key])
        # Flatten to 2D: (rows, cols)
        arr = np.atleast_2d(arr)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        # Build column labels
        cols = [f"{key}_{i}" for i in range(arr.shape[1])]
        df_part = pd.DataFrame(arr, columns=cols)
        df_parts.append(df_part)

    # Align by row index
    df_all = pd.concat(df_parts, axis=1)
    df_all.to_csv(out_csv, index=False, float_format="%.8e")
    print(f"Saved: {out_csv}  (shape: {df_all.shape})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python npz_to_wide_csv.py file1.npz [file2.npz ...]")
        sys.exit(1)

    for f in sys.argv[1:]:
        npz_to_wide_csv(f)
