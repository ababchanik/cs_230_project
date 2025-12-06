import argparse
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prepare_strain_random_walk import prepare_strain_random_walk
from update_stress_ti_mcc_small import update_cauchy_stress
from tensor_utils import (
    voigt_from_rank2,
    rank2_from_voigt,
)

def build_dataset(m: int, T: int, inc_min: float = 1e-6, inc_max: float = 1e-4, seed: int | None = 0):
    eps, deps = prepare_strain_random_walk(m=m, T=T, inc_max=inc_max, seed=seed)
    sig  = np.zeros((m, T+1, 6), dtype=float)
    epse = np.zeros((m, T+1, 6), dtype=float)
    phase = np.empty((m, T+1), dtype="U7")
    for i in tqdm(range(m), desc="instances"):
        phase[i, 0] = "elastic"
        for t in range(T):
            epsilone_old = rank2_from_voigt(epse[i, t, :])
            depsilon     = rank2_from_voigt(deps[i, t+1, :])
            epsilone, sigma, status, _ = update_cauchy_stress(epsilone_old, depsilon)
            sig[i, t+1, :]  = voigt_from_rank2(sigma)
            epse[i, t+1, :] = voigt_from_rank2(epsilone)
            phase[i, t+1]   = status
    return eps, deps, epse, sig, phase

def parse_args():
    ap = argparse.ArgumentParser(description="Generate stress/strain dataset and save to NPZ.")
    ap.add_argument("--m", type=int, default=1000)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--inc_min", type=float, default=1e-6)
    ap.add_argument("--inc_max", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data_stress_strain_random_walk.npz")
    return ap.parse_args()

def main():
    args = parse_args()
    eps, deps, epse, sig, phase = build_dataset(args.m, args.T, args.inc_min, args.inc_max, args.seed)
    np.savez(args.out, eps=eps, deps=deps, epse=epse, sig=sig, phase=phase, meta=dict(m=args.m, T=args.T, seed=args.seed))
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
