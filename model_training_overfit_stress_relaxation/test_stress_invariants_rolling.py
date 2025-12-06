#!/usr/bin/env python3
"""
Test script: visualize p–q stress invariant trajectories predicted by the
TI-MCC small-strain sequence model for a SINGLE stress-relaxation path.

For the single specimen (i_spec = 0), the script:
  • loads eps, epse, deps, sig from a .npz WITHOUT 'split'
  • chooses a start time t0 = L
  • runs the model in rolling fashion to cover horizon H_test:
        - use past L steps to predict next H stresses
        - shift window forward by H and repeat
        - truncate to total horizon H_test
  • computes p, q from Voigt-6 (fictitious) stress
  • plots ground-truth vs predicted q–p paths over [t0, t0+H_test)
  • overlays MCC ellipse (optional)
  • crops the plot to Quadrant II (p<0, q>0)

Usage example (for your overfit model):
    python test_stress_invariants_rolling.py \
        --data ../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz \
        --ckpt model_overfit.pt \
        --L 2 --H 2 --hid 128 \
        --H_test 16 \
        --M 2.0 --pc -6.0 \
        --out_dir ./figs_pq_relax
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import SeqModel
from tensor_utils import TensorUtils  # not strictly needed here, but kept for consistency


# ============================================================
#   p–q invariant helpers (unchanged)
# ============================================================
def fictitious_from_real(sig, theta_deg=45, c1=0.82, c2=-0.33, c3=0.36):
    # Prepare Projection operator PP
    I = np.eye(3)
    II  = 0.5 * (np.einsum('ik,jl->ijkl', I, I) +
                 np.einsum('il,jk->ijkl', I, I))

    theta = np.deg2rad(theta_deg)
    n = np.array([np.cos(theta), np.sin(theta), 0.0])
    m = np.outer(n, n)

    I_oplus_m  = np.einsum('jl,ik->ijkl', I, m)
    m_oplus_I  = np.einsum('jl,ik->ijkl', m, I)
    I_ominus_m = np.einsum('il,jk->ijkl', I, m)
    m_ominus_I = np.einsum('il,jk->ijkl', m, I)
    m_oplus_m  = np.einsum('jl,ik->ijkl', m, m)
    m_ominus_m = np.einsum('il,jk->ijkl', m, m)

    PP = (
        c1 * II
        + (c2 / 2.0) * (m_oplus_m + m_ominus_m)
        + (c3 / 4.0) * (I_oplus_m + m_oplus_I + I_ominus_m + m_ominus_I)
    )

    sigma = voigt6_to_tensor3x3(sig)  # (...,3,3)
    sigma_star = np.einsum("ijkl,...kl->...ij", PP, sigma)  # (...,3,3)
    sig_star = tensor3x3_to_voigt6(sigma_star)  # (...,6)
    return sig_star


def p_q_from_voigt(sig_v):
    """
    Compute mean stress p and deviatoric q from Voigt stress (..., 6).
    Returns arrays with leading shape sig_v.shape[:-1].

    Assumes Voigt order: (xx, yy, zz, yz, zx, xy)
    """
    s = sig_v.reshape(-1, 6)
    p = (s[:, 0] + s[:, 1] + s[:, 2]) / 3.0

    sxx, syy, szz = s[:, 0] - p, s[:, 1] - p, s[:, 2] - p
    shear = s[:, 3:6]  # yz, zx, xy under this ordering
    s2 = (
        sxx**2 + syy**2 + szz**2
        + 2.0 * (shear[:, 0]**2 + shear[:, 1]**2 + shear[:, 2]**2)
    )
    q = np.sqrt(1.5 * s2)
    return p.reshape(sig_v.shape[:-1]), q.reshape(sig_v.shape[:-1])


def compute_mcc_ellipse(M, pc, n_points=300):
    """
    MCC ellipse in p–q plane: q^2 = - M^2 p (p - pc).
    Only returns positive-q branch.
    """
    p_min = min(0.0, pc)
    p_max = max(0.0, pc)
    p_ell = np.linspace(p_max, p_min, n_points)

    q2 = - (M**2) * p_ell * (p_ell - pc)
    q = np.sqrt(np.maximum(q2, 0.0))
    return p_ell, q


def show_quadrant_II(ax, pad=0.05):
    """
    Crop to Quadrant II (p<0, q>0) based on plotted data.
    """
    xs, ys = [], []
    for ln in ax.lines:
        x, y = ln.get_xdata(), ln.get_ydata()
        if x is None or y is None or len(x) == 0:
            continue
        xs.append(np.asarray(x))
        ys.append(np.asarray(y))

    if not xs:
        ax.set_xlim(-1, 0)
        ax.set_ylim(0, 1)
        return

    X = np.concatenate(xs)
    Y = np.concatenate(ys)

    mask = (X < 0) & (Y > 0)
    if mask.any():
        X2, Y2 = X[mask], Y[mask]
    else:
        X2, Y2 = X, Y

    xmin = X2.min()
    xmax = 0.0
    ymin = 0.0
    ymax = Y2.max()
    ax.set_xlim(xmin - pad*abs(xmin), xmax + pad*abs(xmin))
    ax.set_ylim(ymin - pad*ymax, ymax + pad*ymax)
    ax.set_aspect("equal", adjustable="box")


def voigt6_to_tensor3x3(v):
    """
    Convert Voigt-6 stress to full 3×3 tensor form.

    Assumes Voigt order: (xx, yy, zz, yz, zx, xy).
    v : np.ndarray (..., 6)
    Returns: np.ndarray (..., 3, 3)
    """
    v = np.asarray(v)
    if v.shape[-1] != 6:
        raise ValueError(f"Expected last dim=6, got {v.shape[-1]}")

    out = np.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype)
    out[..., 0, 0] = v[..., 0]  # xx
    out[..., 1, 1] = v[..., 1]  # yy
    out[..., 2, 2] = v[..., 2]  # zz
    out[..., 1, 2] = out[..., 2, 1] = v[..., 3]  # yz
    out[..., 0, 2] = out[..., 2, 0] = v[..., 4]  # zx
    out[..., 0, 1] = out[..., 1, 0] = v[..., 5]  # xy
    return out


def tensor3x3_to_voigt6(T):
    """
    Convert full 3×3 stress tensor(s) to Voigt-6 form.

    Voigt order: (xx, yy, zz, yz, zx, xy).
    T : np.ndarray (..., 3, 3)
    Returns: np.ndarray (..., 6)
    """
    T = np.asarray(T)
    if T.shape[-1] != 3 or T.shape[-2] != 3:
        raise ValueError(f"Expected (...,3,3), got {T.shape}")

    out = np.zeros(T.shape[:-2] + (6,), dtype=T.dtype)
    out[..., 0] = T[..., 0, 0]  # xx
    out[..., 1] = T[..., 1, 1]  # yy
    out[..., 2] = T[..., 2, 2]  # zz
    out[..., 3] = T[..., 1, 2]  # yz
    out[..., 4] = T[..., 2, 0]  # zx
    out[..., 5] = T[..., 0, 1]  # xy
    return out


# ============================================================
#   Main evaluation logic for SINGLE specimen (no 'split')
# ============================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H_model = args.H

    # ------------------------------------------------------------
    # Load dataset (single path)
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

    # One specimen: (1, T, 6)
    eps_all  = eps_all[0]   # (T,6)
    epse_all = epse_all[0]
    deps_all = deps_all[0]
    sig_all  = sig_all[0]
    T = sig_all.shape[0]

    # Choose start time t0 = L, and default H_test so that we stay in-bounds
    t0 = args.L
    if t0 <= 0 or t0 >= T:
        raise ValueError(f"Invalid t0 = {t0} for T = {T} (L={args.L}).")

    if args.H_test is None:
        H_test = T - t0
    else:
        H_test = args.H_test
        if t0 + H_test > T:
            raise ValueError(
                f"H_test={H_test} with t0={t0} exceeds data length T={T} "
                f"(need t0 + H_test <= T)."
            )

    print(f"[INFO] Single specimen, T={T}, L={args.L}, H={H_model}, H_test={H_test}, t0={t0}")

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    model = SeqModel(H=H_model, hid=args.hid).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # ------------------------------------------------------------
    # Ground truth over [t0, t0+H_test)
    # ------------------------------------------------------------
    sl_out = slice(t0, t0 + H_test)
    sig_true_np = sig_all[sl_out]  # (H_test, 6)

    # ------------------------------------------------------------
    # Rolling prediction over H_test using H_model per call
    # ------------------------------------------------------------
    sig_pred_roll = np.zeros_like(sig_true_np)  # (H_test, 6)
    n_blocks = int(np.ceil(H_test / H_model))

    for k in range(n_blocks):
        g_start = k * H_model
        g_end   = min((k + 1) * H_model, H_test)
        block_len = g_end - g_start
        if block_len <= 0:
            continue

        t_block_start = t0 + g_start
        in_start = t_block_start - args.L
        in_end   = t_block_start

        eps_in  = torch.from_numpy(eps_all[in_start:in_end]).float().unsqueeze(0).to(device)
        epse_in = torch.from_numpy(epse_all[in_start:in_end]).float().unsqueeze(0).to(device)
        deps_in = torch.from_numpy(deps_all[in_start:in_end]).float().unsqueeze(0).to(device)

        with torch.no_grad():
            sig_pred_block, _ = model(eps_in, epse_in, deps_in)  # (1, H_model, 6)

        sig_pred_block_np = sig_pred_block.squeeze(0).cpu().numpy()  # (H_model, 6)
        sig_pred_roll[g_start:g_end] = sig_pred_block_np[:block_len]

    # ------------------------------------------------------------
    # Compute p–q invariants in real and fictitious space
    # ------------------------------------------------------------
    sig_star_pred = fictitious_from_real(sig_pred_roll, theta_deg=45)
    sig_star_true = fictitious_from_real(sig_true_np,  theta_deg=45)

    p_pred, q_pred = p_q_from_voigt(sig_pred_roll)
    p_true, q_true = p_q_from_voigt(sig_true_np)

    p_star_pred, q_star_pred = p_q_from_voigt(sig_star_pred)
    p_star_true, q_star_true = p_q_from_voigt(sig_star_true)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))

    # MCC ellipse
    p_ell, q_ell = compute_mcc_ellipse(args.M, args.pc)
    ax.plot(p_ell, q_ell, "k--", lw=1.5, label=f"M={args.M}, pc={args.pc}")

    ax.plot(p_true, q_true, "-o", lw=1.0, ms=3, label="ground truth q-p")
    ax.plot(p_pred, q_pred, "-s", lw=1.0, ms=3, label="prediction q-p")
    ax.plot(p_star_true, q_star_true, "-o", lw=1.0, ms=3, label="ground truth q*-p*")
    ax.plot(p_star_pred, q_star_pred, "-s", lw=1.0, ms=3, label="prediction q*-p*")

    # Mark start and end of the true trajectory
    ax.scatter(p_true[0],  q_true[0],  marker="o", s=25)
    ax.scatter(p_true[-1], q_true[-1], marker="s", s=25)

    ax.set_title(f"q–p path (single relaxation, t0={t0}, H={H_model}, H_test={H_test})")
    ax.set_xlabel("p (mean stress)")
    ax.set_ylabel("q (deviatoric stress)")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, frameon=False)

    show_quadrant_II(ax)
    fig.tight_layout()

    fname = os.path.join(args.out_dir, f"pq_t0_{t0:04d}.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")
    print(f"All q–p figures saved in: {args.out_dir}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Test invariant p–q trajectories for single relaxation path.")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--L", type=int, default=2,
                    help="Input window length (past steps). Must match training.")
    ap.add_argument("--H", type=int, default=2,
                    help="Model prediction horizon per call. Must match training.")
    ap.add_argument("--H_test", type=int, default=None,
                    help="Total evaluation horizon from t0=L. Defaults to T-L if not set.")
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--out_dir", type=str, default="./figs_pq_relax")

    # MCC parameters
    ap.add_argument("--M",  type=float, default=2.0)
    ap.add_argument("--pc", type=float, default=-6.0)

    args = ap.parse_args()
    main(args)
