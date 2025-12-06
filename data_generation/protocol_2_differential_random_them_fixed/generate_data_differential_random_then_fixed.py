"""
    Random-walk dataset where strain is held fixed after the first plastic step,
    thus allowing stress relaxation.

    Workflow per instance i:
      - Use one_satep_strain_random_walk to generate candidate increments deps_rw.
      - March in time, updating stress with update_cauchy_stress.
      - Until we see status != "elastic", we apply the random increments.
      - Once plasticity is detected, we:
          * set all LATER deps to zero
          * keep eps constant from that point on
          * copy sig, epse, phase forward in time.
"""

import argparse
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

from compute_invariants import compute_invariants

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from update_stress_ti_mcc_small import update_cauchy_stress
from tensor_utils import (
    voigt_from_rank2,
    rank2_from_voigt,
)

def one_step_strain_differential_random_walk(rng,
                                             vol_inc_min: float = 1e-5, 
                                             vol_inc_max: float = 1e-4,
                                             she_inc_min: float = 1e-6, 
                                             she_inc_max: float = 1e-5):
    """
    Return deps with shape (6,).
    """
    assert vol_inc_max >= vol_inc_min >= 0.0 and she_inc_max >= she_inc_min >= 0.0

    signs = rng.choice([-1.0, 1.0], size=(6,))
    mags  = np.concatenate([rng.uniform(vol_inc_min, vol_inc_max, size=(3,)),
                           rng.uniform(she_inc_min, she_inc_max, size=(3,))])
    deps  = signs * mags

    return deps

def build_dataset(m: int,
                  T: int, 
                  vol_inc_min: float = 1e-5, 
                  vol_inc_max: float = 1e-4,
                  she_inc_min: float = 1e-6, 
                  she_inc_max: float = 1e-5,
                  seed: int | None = 0,
                  max_tries: int = 1000):
    # In the while loop we should get a different candidate increment
    rng = np.random.default_rng(seed)
    
    # Total and incremental strain use (admissile from random walk)
    eps = np.zeros((m, T+1, 6), dtype=float)
    deps = np.zeros((m, T+1, 6), dtype=float)
    # Allocate arrays for properties to be updated
    sig  = np.zeros((m, T+1, 6), dtype=float)
    epse = np.zeros((m, T+1, 6), dtype=float)
    epsp = np.zeros((m, T+1, 6), dtype=float)
    phase = np.empty((m, T+1), dtype="U16")
    sig_star  = np.zeros((m, T+1, 6), dtype=float)

    # Initial conditions
    # Assme stress free. epse = 0, sig = 0, eps = 0
    phase[:, 0] = "elastic"  

    for i in tqdm(range(m), desc="instances"):
        plastic_reached = False

        for t in range(T):
            # Restore
            epsilone_old = rank2_from_voigt(epse[i, t, :])

            if plastic_reached:
                # Once plasticity has occurred, hold strain fixed
                deps[i, t+1, :] = np.zeros((6), dtype=float)
                eps[i, t+1, :] = eps[i, t, :]
            else:
                # Elastic, need to find admissible strain increment
                incr_found = False
                tries = 0
                while not incr_found and tries < max_tries:
                    # Proposed random-walk increments (candidates) at time t+1
                    deps_rw_cur = one_step_strain_differential_random_walk(rng=rng, \
                                                                           vol_inc_min=vol_inc_min, vol_inc_max=vol_inc_max, \
                                                                           she_inc_min=she_inc_min, she_inc_max=she_inc_max)
                    # Compute admissible p and p_star
                    (p, _), (p_star, _) = compute_invariants(epsilone_old, rank2_from_voigt(deps_rw_cur))
                    # Check admissibility
                    if (p < 0.0) and (p_star < 0.0):
                        incr_found = True
                    else:
                        tries += 1
                deps[i, t+1, :] = deps_rw_cur
                eps[i, t+1, :]  = eps[i, t, :] + deps_rw_cur
                if not incr_found:
                    raise RuntimeError("Could not find admissible increment within max_tries of\t", max_tries)

            # Update
            depsilon = rank2_from_voigt(deps[i, t+1, :])
            
            # Stress update
            epsilone, sigma, status, sigma_star = update_cauchy_stress(epsilone_old, depsilon)

            # Update the dataset
            sig[i, t+1, :]      = voigt_from_rank2(sigma)
            epse[i, t+1, :]     = voigt_from_rank2(epsilone)
            epsp[i, t+1, :]     = eps[i, t+1, :] - epse[i, t+1, :]
            phase[i, t+1]       = status
            sig_star[i, t+1, :] = voigt_from_rank2(sigma_star)

            # If this step is plastic, mark and freeze from the next step on
            if status == "plastic":
                plastic_reached = True

    return eps, deps, epse, sig, phase, epsp, sig_star

def parse_args():
    ap = argparse.ArgumentParser(description="Generate stress/strain dataset and save to NPZ.")
    ap.add_argument("--m", type=int, default=1000)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--vol_inc_min", type=float, default=1e-5)
    ap.add_argument("--vol_inc_max", type=float, default=1e-4)
    ap.add_argument("--she_inc_min", type=float, default=1e-6)
    ap.add_argument("--she_inc_max", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tries", type=int, default=1000)
    ap.add_argument("--out", type=str, default="data_stress_strain_differential_random_then_fixed.npz")
    return ap.parse_args()

def main():
    args = parse_args()
    eps, deps, epse, sig, phase, epsp, sig_star = build_dataset(args.m, args.T,\
                                                                args.vol_inc_min, args.vol_inc_max, \
                                                                args.she_inc_min, args.she_inc_max, \
                                                                args.seed, args.max_tries)
    np.savez(args.out, eps=eps, deps=deps, epse=epse, sig=sig, phase=phase, epsp=epsp, sig_star=sig_star, meta=dict(m=args.m, T=args.T, seed=args.seed))
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
