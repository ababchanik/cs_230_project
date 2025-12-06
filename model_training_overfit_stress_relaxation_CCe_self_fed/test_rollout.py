#!/usr/bin/env python3
"""
Rollout test script for TI-MCC small-strain sequence model on a SINGLE path.

- Loads eps, epse, deps, sig from a .npz WITHOUT 'split'
- Uses model with inputs (eps_in, epse_in, deps_in, sig_trial_in)
- Uses ground-truth epse for history window [t0-L, ..., t0-1]
- From t0 onward, uses previously predicted epse in the rolling buffer
- Predicts over [t0, t0 + H_test)
- Saves ONLY predicted sig_pred and epse_pred + metadata (L, H_model, H_test, t0)

Usage:
    python test_rollout.py \
        --data ../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz \
        --ckpt model_overfit.pt \
        --L 4 \
        --H 2 \
        --hid 128 \
        --H_test 20 \
        --out_npz rollout_results_L4_H2_Htest20.npz
"""

import argparse
import numpy as np
import torch

from model import SeqModel
from constitutive_update import assemble_elastic_tensor_voigt, elastic_update


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H_model = args.H

    # ------------------------------------------------------------
    # Elastic tensor
    # ------------------------------------------------------------
    CCe_voigt_np = assemble_elastic_tensor_voigt("material_parameters.csv")
    CCe_voigt = torch.from_numpy(CCe_voigt_np).float().to(device)

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    data = np.load(args.data)
    required = ["eps", "epse", "deps", "sig"]
    if not all(k in data.files for k in required):
        raise RuntimeError(f"Dataset must contain: {required}")

    eps_all  = data["eps"]
    epse_all = data["epse"]
    deps_all = data["deps"]
    sig_all  = data["sig"]

    # Allow both (T,6) and (1,T,6)
    if eps_all.ndim == 2:
        eps_all  = eps_all[None, ...]
        epse_all = epse_all[None, ...]
        deps_all = deps_all[None, ...]
        sig_all  = sig_all[None, ...]

    eps_all  = eps_all[0]
    epse_all = epse_all[0]
    deps_all = deps_all[0]
    sig_all  = sig_all[0]
    T = sig_all.shape[0]

    # ------------------------------------------------------------
    # Choose t0 and H_test
    # ------------------------------------------------------------
    t0 = args.L
    if t0 <= 0 or t0 >= T:
        raise ValueError(f"Invalid t0 = {t0} for T = {T} (L={args.L}).")

    H_test = args.H_test if args.H_test is not None else T - t0
    if t0 + H_test > T:
        raise ValueError(
            f"H_test={H_test} with t0={t0} exceeds T={T} "
            f"(need t0 + H_test <= T)."
        )

    print(f"[INFO] Rollout:")
    print(f"       T={T}, L={args.L}, H={H_model}, H_test={H_test}, t0={t0}")

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    model = SeqModel(H=H_model, hid=args.hid).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # ------------------------------------------------------------
    # Output prediction arrays
    # ------------------------------------------------------------
    sig_pred_roll  = np.zeros((H_test, 6), dtype=np.float32)
    epse_pred_roll = np.zeros((H_test, 6), dtype=np.float32)

    # Rolling buffer for epse
    epse_roll = epse_all.copy()

    # Model calls to cover rollout horizon
    n_blocks = int(np.ceil(H_test / H_model))

    for k in range(n_blocks):
        g_start = k * H_model
        g_end   = min((k + 1) * H_model, H_test)
        block_len = g_end - g_start

        if block_len <= 0:
            continue

        # Index of first predicted step of this block
        t_block_start = t0 + g_start

        # Input window indices
        in_start = t_block_start - args.L
        in_end   = t_block_start

        # Inputs
        eps_in  = torch.from_numpy(eps_all[in_start:in_end]).float().unsqueeze(0).to(device)
        deps_in = torch.from_numpy(deps_all[in_start:in_end]).float().unsqueeze(0).to(device)
        epse_in = torch.from_numpy(epse_roll[in_start:in_end]).float().unsqueeze(0).to(device)

        # Trial
        sig_trial_in = elastic_update(epse_in + deps_in, CCe_voigt)

        # Model rollout
        with torch.no_grad():
            sig_pred_block, epse_pred_block = model(eps_in, epse_in, deps_in, sig_trial_in)

        sig_pred_block_np  = sig_pred_block.squeeze(0).cpu().numpy()
        epse_pred_block_np = epse_pred_block.squeeze(0).cpu().numpy()

        # Store predictions
        sig_pred_roll[g_start:g_end]  = sig_pred_block_np[:block_len]
        epse_pred_roll[g_start:g_end] = epse_pred_block_np[:block_len]

        # Update rolling epse
        t_pred_start = t0 + g_start
        t_pred_end   = t0 + g_end
        epse_roll[t_pred_start:t_pred_end] = epse_pred_block_np[:block_len]

    # ------------------------------------------------------------
    # Save NPZ (WITHOUT truth arrays)
    # ------------------------------------------------------------
    out_path = args.out_npz
    np.savez(
        out_path,
        sig_pred=sig_pred_roll,        # (H_test, 6)
        epse_pred=epse_pred_roll,      # (H_test, 6)
        L=args.L,
        H_model=H_model,
        H_test=H_test,
        t0=t0,
    )

    print(f"[OK] Saved rollout predictions to: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Rollout prediction for single relaxation path.")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--H_test", type=int, default=None)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--out_npz", type=str, default="rollout_results.npz")

    args = ap.parse_args()
    main(args)
