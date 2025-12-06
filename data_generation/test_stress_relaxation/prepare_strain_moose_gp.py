#!/usr/bin/env python3
"""
Convert converged_tensors_postprocessed.csv → NPZ for each dt_* folder.

- Fixed Voigt6 order: (xx, yy, zz, yz, xz, xy)
- Averages symmetric shear pairs and checks closeness (xy↔yx, yz↔zy, xz↔zx).

Usage:
----
python prepare_strain_moose_gp.py \
  --base_dir /home/groups/borja/hemiao/projects/small_viscoplasticity/problems/\
small_visco_mcc_aniso_aniso_standard_tests/stress_relaxation_strain_rate_45deg/results_strain_rate \
  --folders dt_1e-1 \
  --prefix stress_relaxation_strain_rate \
  --rtol 1e-10 \
  --atol 1e-12 \
  --on_shear_mismatch warn
---
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

CSV_NAME = "converged_tensors_postprocessed.csv"
VOIGT_ORDER = ("xx", "yy", "zz", "yz", "xz", "xy")
REQUIRED_PREFIXES = ("sigma", "epsilon_e", "epsilon", "delta_epsilon")

def find_cols(df: pd.DataFrame, base: str) -> dict:
    cols = {}
    for comp in ("xx","yy","zz","yz","zy","xz","zx","xy","yx"):
        name = f"{base}_{comp}"
        if name in df.columns:
            cols[comp] = name
    return cols

def get_shear(df: pd.DataFrame, cols_map: dict, a: str, b: str,
              label: str, rtol: float, atol: float, on_mismatch: str) -> np.ndarray:
    T = len(df)
    ca, cb = cols_map.get(a), cols_map.get(b)
    if ca is not None and cb is not None:
        va, vb = df[ca].to_numpy(), df[cb].to_numpy()
        if on_mismatch != "ignore":
            close = np.allclose(va, vb, rtol=rtol, atol=atol, equal_nan=True)
            if not close:
                absdiff = np.abs(va - vb)
                with np.errstate(divide='ignore', invalid='ignore'):
                    reldiff = absdiff / np.maximum(np.abs(vb), atol)
                msg = (f"[{label}] {a} vs {b} not close: "
                       f"max|Δ|={absdiff.max():.3e}, mean|Δ|={absdiff.mean():.3e}, "
                       f"max relΔ={np.nanmax(reldiff):.3e}")
                if on_mismatch == "error":
                    raise ValueError(msg)
                print("  [WARN]", msg)
        return 0.5 * (va + vb)
    if ca is not None:
        return df[ca].to_numpy()
    if cb is not None:
        return df[cb].to_numpy()
    return np.zeros(T, dtype=float)

def to_voigt6(df: pd.DataFrame, cols_map: dict, label: str,
              rtol: float, atol: float, on_mismatch: str) -> np.ndarray:
    T = len(df)
    out = np.zeros((T, 6), dtype=float)
    if "xx" in cols_map: out[:, 0] = df[cols_map["xx"]].to_numpy()
    if "yy" in cols_map: out[:, 1] = df[cols_map["yy"]].to_numpy()
    if "zz" in cols_map: out[:, 2] = df[cols_map["zz"]].to_numpy()
    out[:, 3] = get_shear(df, cols_map, "yz", "zy", label, rtol, atol, on_mismatch)
    out[:, 4] = get_shear(df, cols_map, "xz", "zx", label, rtol, atol, on_mismatch)
    out[:, 5] = get_shear(df, cols_map, "xy", "yx", label, rtol, atol, on_mismatch)
    return out

def load_and_convert(csv_path: Path, rtol: float, atol: float, on_mismatch: str):
    print(f"\nProcessing {csv_path}")
    df = pd.read_csv(csv_path)
    sig  = to_voigt6(df, find_cols(df, "sigma"),         "sigma",         rtol, atol, on_mismatch)
    epse = to_voigt6(df, find_cols(df, "epsilon_e"),     "epsilon_e",     rtol, atol, on_mismatch)
    eps  = to_voigt6(df, find_cols(df, "epsilon"),       "epsilon",       rtol, atol, on_mismatch)
    deps = to_voigt6(df, find_cols(df, "delta_epsilon"), "delta_epsilon", rtol, atol, on_mismatch)
    if "state" not in df.columns:
        raise KeyError("'state' column not found")
    phase = df["state"].astype(str).str.lower().to_numpy()
    return sig, epse, eps, deps, phase

def main():
    ap = argparse.ArgumentParser(
        description="Convert CSV to NPZ for each dt_* folder (Voigt: xx,yy,zz,yz,xz,xy)"
    )
    ap.add_argument("--base_dir", required=True, help="Directory containing dt_* subfolders")
    ap.add_argument("--folders", nargs="*", default=["dt_1e-1","dt_1e-2","dt_1e-4", "dt_1e-6"])
    ap.add_argument("--prefix", default="stress_relaxation_strain_rate")
    ap.add_argument("--rtol", type=float, default=1e-10, help="Relative tol for shear symmetry")
    ap.add_argument("--atol", type=float, default=1e-12, help="Absolute tol for shear symmetry")
    ap.add_argument("--on_shear_mismatch", choices=("warn","error","ignore"),
                    default="warn", help="Behavior if xy!=yx etc.")
    args = ap.parse_args()

    base = Path(args.base_dir)
    if not base.is_dir():
        print(f"Base directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    cwd = Path.cwd()
    for sub in args.folders:
        csv_path = base / sub / CSV_NAME
        if not csv_path.is_file():
            print(f"[SKIP] Missing {csv_path}")
            continue
        try:
            sig, epse, eps, deps, phase = load_and_convert(
                csv_path, args.rtol, args.atol, args.on_shear_mismatch
            )
            # Initial zero state row
            zero6 = np.zeros((1,6), dtype=float)

            sig  = np.vstack([zero6, sig])
            epse = np.vstack([zero6, epse])
            eps  = np.vstack([zero6, eps])
            deps = np.vstack([zero6, deps])

            phase = np.concatenate([np.array(["elastic"]), phase])

        except Exception as e:
            print(f"[ERROR] {csv_path}: {e}", file=sys.stderr)
            continue

        out_path = cwd / f"{args.prefix}_{sub}.npz"
        np.savez(
            out_path,
            sig=sig, epse=epse, eps=eps, deps=deps, phase=phase,
            voigt_order=np.array(VOIGT_ORDER)
        )
        print(f"[OK] Saved {out_path.name} in {cwd}")

if __name__ == "__main__":
    main()
