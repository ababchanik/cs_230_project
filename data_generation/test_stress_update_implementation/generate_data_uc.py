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

def build_dataset():
    data_path = "/home/groups/borja/hemiao/sequential_models/ti_mcc_small/data_generation/test_stress_update_implementation/stress_relaxation_strain_rate_dt_1e-2.npz"
    data = np.load(data_path, allow_pickle=True)
    eps  = data["eps"]      # shape (m, T, 6) if batched or (T, 6) otherwise
    deps = data["deps"]     # shape (m, T, 6) if batched or (T, 6) otherwise
    print(f"Loaded eps, deps from {data_path}: {eps.shape=}, {deps.shape=}")

    T, _ = eps.shape

    sig_test = np.zeros((T, 6), dtype=float)
    sig_star_test  = np.zeros((T, 6), dtype=float)
    epse_test = np.zeros((T, 6), dtype=float)
    phase_test = np.empty((T), dtype="U7")
    for t in range(T):
        print("\nt=", t)
        if t == 0:
            epsilone_old = rank2_from_voigt(np.zeros((6), dtype=float), shear_div2=False)
        else:
            epsilone_old = rank2_from_voigt(epse_test[t-1, :], shear_div2=False)
        depsilon = rank2_from_voigt(deps[t, :], shear_div2=False)
        print("depsilon\n", depsilon)
        print("before local Newton epsilone\n", epsilone_old)
        epsilone, sigma, status, sigma_star = update_cauchy_stress(depsilon, epsilone_old)
        print("after local Newton epsilone\n", epsilone)
        sig_test[t, :] = voigt_from_rank2(sigma, shear_term_x2=False)
        sig_star_test[t, :] = voigt_from_rank2(sigma_star, shear_term_x2=False)
        epse_test[t, :] = voigt_from_rank2(epsilone, shear_term_x2=False)
        phase_test[t] = status
    return epse_test, sig_test, phase_test, sig_star_test

def parse_args():
    ap = argparse.ArgumentParser(description="Generate stress/strain dataset and save to NPZ.")
    ap.add_argument("--out", type=str, default="stress_relaxation_strain_rate_dt_1e-1_test.npz")
    return ap.parse_args()

def main():
    args = parse_args()
    # NEED TO MODIFY THE SHAPE HERE
    epse_test, sig_test, phase_test, sig_star_test = build_dataset()
    np.savez(args.out, epse_test=epse_test, sig_test=sig_test, phase_test=phase_test, sig_star_test=sig_star_test)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
