#!/usr/bin/env python3
"""
Rollout test script for TI-MCC small-strain sequence model on a SINGLE specimen
from a (possibly multi-trajectory) NPZ.

- Loads eps, epsp, deps, sig from a .npz WITHOUT 'split'
- Supports data shaped either as:
      (T, 6)       : single trajectory  (treated as m=1)
      (m, T, 6)    : multiple trajectories
- Uses model with inputs:
      Encoder: history [eps_hist, epsp_hist, sig_hist] over L steps
      Decoder: future deps over H_model steps
- Uses ground-truth epsp for history window [t0-L, ..., t0-1]
- From t0 onward, uses previously predicted epsp in the rolling buffer
- Predicts over [t0, t0 + H_test) for a chosen specimen i_spec
- Saves ONLY predicted sig_pred and epsp_pred + metadata (L, H_model, H_test, t0, i_spec)

Usage:
    python test_rollout.py \
        --data ../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz \
        --ckpt model_overfit.pt \
        --L 4 \
        --H 2 \
        --hid 128 \
        --H_test 20 \
        --i_spec 0 \
        --out_npz rollout_results_L4_H2_Htest20_spec0.npz
"""
import argparse
import numpy as np
import torch

from model import SeqModel
from constitutive_update import assemble_elastic_tensor_voigt, elastic_update


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H_model = args.H
    P = args.P if args.P is not None else H_model
    if not (1 <= P <= H_model):
        raise ValueError(f"P must satisfy 1 ≤ P ≤ H_model. Got P={P}, H_model={H_model}")

    # ------------------------------------------------------------
    # Elastic tensor
    # ------------------------------------------------------------
    CCe_voigt_np = assemble_elastic_tensor_voigt("material_parameters.csv")
    CCe_voigt = torch.from_numpy(CCe_voigt_np).float().to(device)

    # ------------------------------------------------------------
    # Load dataset (multi-trajectory aware)
    # ------------------------------------------------------------
    data = np.load(args.data)
    required = ["eps", "epsp", "deps", "sig"]
    if not all(k in data.files for k in required):
        raise RuntimeError(f"Dataset must contain: {required}")

    eps_all  = data["eps"]
    epsp_all = data["epsp"]
    deps_all = data["deps"]
    sig_all  = data["sig"]

    # Allow both (T,6) and (m,T,6)
    if eps_all.ndim == 2:
        # Single trajectory -> treat as m=1
        eps_all  = eps_all[None, ...]
        epsp_all = epsp_all[None, ...]
        deps_all = deps_all[None, ...]
        sig_all  = sig_all[None, ...]

    if eps_all.ndim != 3:
        raise RuntimeError(
            f"Expected eps to be (T,6) or (m,T,6), got shape {eps_all.shape}"
        )

    m, T, D = eps_all.shape
    if D != 6:
        raise RuntimeError(f"Expected Voigt-6 last dimension, got D={D}")

    # ------------------------------------------------------------
    # Choose specimen index
    # ------------------------------------------------------------
    i_spec = args.i_spec
    if not (0 <= i_spec < m):
        raise ValueError(f"i_spec={i_spec} out of range for m={m} trajectories.")

    eps   = eps_all[i_spec]    # (T, 6)
    epsp  = epsp_all[i_spec]   # (T, 6)
    deps  = deps_all[i_spec]   # (T, 6)
    sig   = sig_all[i_spec]    # (T, 6)
    T     = sig.shape[0]

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

    print(f"[INFO] Rollout for specimen i_spec={i_spec}:")
    print(f"       m={m}, T={T}, L={args.L}, H={H_model}, H_test={H_test}, t0={t0}, P={P}")

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
    epsp_pred_roll = np.zeros((H_test, 6), dtype=np.float32)

    # Rolling buffer for epsp (full trajectory copy)
    epsp_roll = epsp.copy()

    # Number of model calls to cover rollout horizon (stride P)
    n_blocks = int(np.ceil(H_test / P))

    for k in range(n_blocks):
        g_start = k * P
        if g_start >= H_test:
            break  # we've already covered [0, H_test)

        g_end = min(g_start + H_model, H_test)
        block_len = g_end - g_start
        if block_len <= 0:
            continue

        # Index of first predicted step of this block (global rollout index)
        t_block_start = t0 + g_start

        # Encoder history window
        in_start = t_block_start - args.L
        in_end   = t_block_start
        if in_start < 0:
            raise RuntimeError(f"in_start={in_start} is negative.")

        eps_hist  = torch.from_numpy(eps[in_start:in_end]).float().unsqueeze(0).to(device)
        epsp_hist = torch.from_numpy(epsp_roll[in_start:in_end]).float().unsqueeze(0).to(device)
        sig_hist  = elastic_update(eps_hist - epsp_hist, CCe_voigt)

        # Decoder future window
        f_start = t_block_start
        f_end   = t_block_start + H_model
        if f_end > T:
            # Not enough future deps for this block; skip it
            continue

        deps_future = torch.from_numpy(deps[f_start:f_end]).float().unsqueeze(0).to(device)

        # Model rollout block
        with torch.no_grad():
            sig_pred_block, epsp_pred_block = model(eps_hist, epsp_hist, sig_hist, deps_future)

        sig_pred_block_np  = sig_pred_block.squeeze(0).cpu().numpy()   # (H_model, 6)
        epsp_pred_block_np = epsp_pred_block.squeeze(0).cpu().numpy()  # (H_model, 6)

        # Store predictions for rollout indices [g_start, g_end)
        sig_pred_roll[g_start:g_end]  = sig_pred_block_np[:block_len]
        epsp_pred_roll[g_start:g_end] = epsp_pred_block_np[:block_len]

        # Update rolling epsp in global time [t0+g_start, t0+g_end)
        t_pred_start = t0 + g_start
        t_pred_end   = t0 + g_end
        epsp_roll[t_pred_start:t_pred_end] = epsp_pred_block_np[:block_len]

    # ------------------------------------------------------------
    # Save NPZ (WITHOUT truth arrays)
    # ------------------------------------------------------------
    out_path = args.out_npz
    np.savez(
        out_path,
        sig_pred=sig_pred_roll,        # (H_test, 6) over [t0 ... t0+H_test)
        epsp_pred=epsp_pred_roll,      # (H_test, 6) over [t0 ... t0+H_test)
        L=args.L,
        H_model=H_model,
        H_test=H_test,
        t0=t0,
        i_spec=i_spec,
    )

    print(f"[OK] Saved rollout predictions for i_spec={i_spec} to: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Rollout prediction for single relaxation path (multi-trajectory NPZ).")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--H_test", type=int, default=None,
                    help="Rollout horizon length; if None, uses T - t0.")
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--out_npz", type=str, default="rollout_results.npz")
    ap.add_argument(
        "--P",
        type=int,
        default=None,
        help=(
            "Stride in rollout horizon between model calls (1 ≤ P ≤ H). "
            "If None, defaults to P = H (non-overlapping blocks)."
        ),
    )
    ap.add_argument(
        "--i_spec",
        type=int,
        default=0,
        help="Specimen index in the NPZ file (0-based)."
    )

    args = ap.parse_args()
    main(args)
