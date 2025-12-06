#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import sys

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from update_stress_ti_mcc_small import update_cauchy_stress
from tensor_utils import (
    voigt_from_rank2,
    rank2_from_voigt,
)

def build_dataset(data_path: str):
    """
    Build a stress-relaxation trajectory by *replaying* the strain history
    (eps, deps) that was extracted from a MOOSE stress-relaxation run.

    No random walk or synthetic strain increments are generated here.

    Args
    ----
    data_path : str
        Path to NPZ file produced by prepare_strain_moose_gp.py.
        It must contain at least: eps, deps, phase.

    Returns
    -------
    eps       : (T, 6) total strain (Voigt)    [from NPZ]
    deps      : (T, 6) total strain increment  [from NPZ]
    epse      : (T, 6) elastic strain computed by update_cauchy_stress
    sig       : (T, 6) Cauchy stress from update_cauchy_stress
    phase     : (T,)   status strings from update_cauchy_stress
    sig_star  : (T, 6) auxiliary stress from update_cauchy_stress
    phase_ref : (T,)   original MOOSE phase/state (lowercase strings)
    """

    data = np.load(data_path, allow_pickle=True)
    eps       = data["eps"]      # (T, 6)
    deps      = data["deps"]     # (T, 6)
    phase_ref = data["phase"]    # (T,)

    T, D = eps.shape
    assert D == 6, f"Expected Voigt-6, got D={D}"
    print(f"Loaded {data_path}: {eps.shape=}, {deps.shape=}, {phase_ref.shape=}")

    # Storage for small-model results
    epse     = np.zeros((T, 6), dtype=float)
    epsp     = np.zeros((T, 6), dtype=float)
    sig      = np.zeros((T, 6), dtype=float)
    sig_star = np.zeros((T, 6), dtype=float)
    phase    = np.empty((T,), dtype="U16")

    for t in range(T):
        # Previous elastic strain tensor in rank-2 form
        if t == 0:
            epsilone_old = rank2_from_voigt(
                np.zeros(6, dtype=float),
                shear_div2=False
            )
        else:
            epsilone_old = rank2_from_voigt(epse[t-1, :], shear_div2=False)

        # Use given deps from NPZ
        deps_voigt = deps[t, :]
        depsilon   = rank2_from_voigt(deps_voigt, shear_div2=False)

        epsilone, sigma, status, sigma_star_rank2 = update_cauchy_stress(depsilon, epsilone_old)

        # Save results in Voigt form
        epse[t, :]     = voigt_from_rank2(epsilone,           shear_term_x2=False)
        epsp[t, :]     = eps[t, :] - epse[t, :]
        sig[t, :]      = voigt_from_rank2(sigma,              shear_term_x2=False)
        sig_star[t, :] = voigt_from_rank2(sigma_star_rank2,   shear_term_x2=False)
        phase[t]       = status

    return eps, deps, epse, sig, phase, epsp, sig_star, phase_ref


def parse_args():
    ap = argparse.ArgumentParser(description="Replay MOOSE stress-relaxation strain history and generate stress data with TI-MCC-small update.")
    ap.add_argument("--data", type=str, required=False, default=("/home/groups/borja/hemiao/sequential_models/ti_mcc_small/data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz"), help="Input NPZ file with eps, deps, phase from MOOSE.",)
    ap.add_argument("--out", type=str, default="stress_relaxation_strain_rate_dt_1e-1.npz", help="Output NPZ file name",)
    return ap.parse_args()


def main():
    args = parse_args()

    (eps, deps,
     epse, sig, phase,
     epsp, sig_star, phase_ref) = build_dataset(args.data)

    np.savez(
        args.out,
        eps=eps,
        deps=deps,
        epse=epse,
        sig=sig,
        phase=phase,        # status from TI-MCC-small python function
        epsp=epsp,
        sig_star=sig_star,
        phase_ref=phase_ref # original MOOSE state labels
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
