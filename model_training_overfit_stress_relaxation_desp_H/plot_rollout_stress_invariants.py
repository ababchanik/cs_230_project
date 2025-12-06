#!/usr/bin/env python3
"""
Plot script: visualize p–q stress invariant trajectories predicted by the
TI-MCC small-strain sequence model for a SINGLE stress-relaxation path.

This version is PLOT-ONLY:

  • loads full eps, epsp, deps, sig from a .npz WITHOUT 'split' (data_truth)
  • loads rollout predictions from data_pred (sig_pred, epsp_pred, L, H_model, H_test, t0)
  • extracts ground-truth sig over [0, T)  <-- full trajectory
  • computes p, q from Voigt-6 (real and fictitious stresses)
  • plots ground-truth (0..T) vs predicted q–p and q*–p* paths
  • overlays MCC ellipse
  • crops the plot to Quadrant II (p<0, q>0)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

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
    m_ominus_I = np.einsum('il,jk->ijkl', m, m)
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
#   Main: plot-only p–q for single trajectory
# ============================================================
def main(args):
    # ------------------------------------------------------------
    # Load prediction NPZ (rollout file)
    # ------------------------------------------------------------
    roll = np.load(args.data_pred)

    required_pred = ["sig_pred", "epsp_pred", "L", "H_model", "H_test", "t0"]
    if not all(k in roll.files for k in required_pred):
        raise RuntimeError(
            f"Prediction NPZ {args.data_pred} must contain keys {required_pred}."
        )

    sig_pred_np  = roll["sig_pred"]   # (H_test, 6)
    L       = int(roll["L"])
    H_model = int(roll["H_model"])
    H_test  = int(roll["H_test"])
    t0      = int(roll["t0"])

    print("[INFO] Loaded data_pred:", args.data_pred)
    print(f"       L={L}, H_model={H_model}, H_test={H_test}, t0={t0}")
    print(f"       sig_pred shape = {sig_pred_np.shape}")

    # ------------------------------------------------------------
    # Load full ground-truth trajectory
    # ------------------------------------------------------------
    data = np.load(args.data_truth)
    required_truth = ["eps", "epsp", "deps", "sig"]
    if not all(k in data.files for k in required_truth):
        raise RuntimeError(
            f"Ground-truth NPZ {args.data_truth} must contain keys {required_truth}."
        )

    sig_all = data["sig"]   # (T,6) or (1,T,6)

    if sig_all.ndim == 2:
        sig_all = sig_all[None, ...]

    sig_all = sig_all[0]    # (T,6)
    T = sig_all.shape[0]
    print(f"[INFO] Ground-truth trajectory length T={T}")

    # still sanity-check that rollout horizon fits in T
    if t0 + H_test > T:
        raise RuntimeError(f"Rollout horizon t0+H_test = {t0 + H_test} exceeds T={T}.")

    # Full ground truth 0..T-1
    sig_true_full = sig_all.copy()          # (T,6)

    # Optional: window of truth matching prediction (not used for plotting,
    # but kept here in case you want it later)
    sl_out = slice(t0, t0 + H_test)
    sig_true_win = sig_all[sl_out].copy()   # (H_test, 6)

    # ------------------------------------------------------------
    # Compute p–q invariants in real and fictitious space
    #   – truth over 0..T-1
    #   – prediction over its H_test steps
    # ------------------------------------------------------------
    sig_star_pred      = fictitious_from_real(sig_pred_np, theta_deg=args.theta_deg)
    sig_star_true_full = fictitious_from_real(sig_true_full, theta_deg=args.theta_deg)

    p_pred, q_pred             = p_q_from_voigt(sig_pred_np)
    p_true_full, q_true_full   = p_q_from_voigt(sig_true_full)

    p_star_pred, q_star_pred         = p_q_from_voigt(sig_star_pred)
    p_star_true_full, q_star_true_full = p_q_from_voigt(sig_star_true_full)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))

    # MCC ellipse
    p_ell, q_ell = compute_mcc_ellipse(args.M, args.pc)
    ax.plot(p_ell, q_ell, "k--", lw=1.5, label=f"M={args.M}, pc={args.pc}")

    # Truth over full 0..T trajectory
    ax.plot(p_true_full, q_true_full, "-o", lw=1.0, ms=3,
            label="ground truth q-p (0..T)")

    # Prediction over [t0, t0+H_test)
    ax.plot(p_pred, q_pred, "-s", lw=1.0, ms=3,
            label="prediction q-p (t0..t0+H_test)")

    # Fictitious invariants, full truth vs prediction
    ax.plot(p_star_true_full, q_star_true_full, "-o", lw=1.0, ms=3,
            label="ground truth q*-p* (0..T)")
    ax.plot(p_star_pred, q_star_pred, "-s", lw=1.0, ms=3,
            label="prediction q*-p* (t0..t0+H_test)")

    # Mark start and end of the full true trajectory
    ax.scatter(p_true_full[0],  q_true_full[0],  marker="o", s=25)
    ax.scatter(p_true_full[-1], q_true_full[-1], marker="s", s=25)

    ax.set_title(
        f"q–p path (single relaxation, t0={t0}, "
        f"H_model={H_model}, H_test={H_test})"
    )
    ax.set_xlabel("p (mean stress)")
    ax.set_ylabel("q (deviatoric stress)")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, frameon=False)

    show_quadrant_II(ax)
    fig.tight_layout()

    fname = os.path.join(args.out_dir, f"pq_rollout_t0_{t0:04d}.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {fname}")
    print(f"[OK] All q–p figures saved in: {args.out_dir}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Plot invariant p–q trajectories for a single relaxation path (rollout vs truth).")
    ap.add_argument("--data_truth", type=str, required=True, help="Ground-truth NPZ with eps, epsp, deps, sig (single trajectory).")
    ap.add_argument("--data_pred", type=str, required=True, help="Rollout NPZ with sig_pred, epsp_pred, L, H_model, H_test, t0.")
    ap.add_argument("--out_dir", type=str, default="./figs_pq_relax", help="Output directory for figure.")
    ap.add_argument("--M",  type=float, default=2.0, help="MCC slope parameter M.")
    ap.add_argument("--pc", type=float, default=-6.0, help="MCC preconsolidation pressure pc.")
    ap.add_argument("--theta_deg", type=float, default=45.0, help="Projection angle θ (deg) for fictitious stress.")
    args = ap.parse_args()
    main(args)
