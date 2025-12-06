#!/usr/bin/env python3
"""
Usage:
python plot_rollout_stress_strain.py \
  --data_truth ../data_generation/.../stress_relaxation.npz \
  --data_pred ./rollout_results_L4_H2_Htest20.npz \
  --out_dir ./figs_stress_strain

This script:
  - loads the full ground-truth trajectory from data_truth
  - loads rollout predictions from data_pred

  Figure 1 (stress–strain, 2×3 panels):
    - For I = 0..5 in Voigt order (xx, yy, zz, yz, zx, xy):
        truth:      σ_I^truth(ε_I^truth), over the entire horizon [0, T)
        prediction: σ_I^pred(ε_I^truth),  over the rollout horizon [t0, t0+H_test)

  Figure 2 (plastic strain–strain, 2×3 panels):
    - For I = 0..5 in Voigt order:
        truth:      ε_{e,I}^truth(ε_I^truth), over the entire horizon [0, T)
        prediction: ε_{e,I}^pred(ε_I^truth),  over the rollout horizon [t0, t0+H_test)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


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

    sig_pred_np  = roll["sig_pred"]    # (H_test, 6)
    epsp_pred_np = roll["epsp_pred"]   # (H_test, 6)

    L       = int(roll["L"])
    H_model = int(roll["H_model"])
    H_test  = int(roll["H_test"])
    t0      = int(roll["t0"])

    print("[INFO] Loaded data_pred:", args.data_pred)
    print(f"       L={L}, H_model={H_model}, H_test={H_test}, t0={t0}")
    print(f"       sig_pred shape  = {sig_pred_np.shape}")
    print(f"       epsp_pred shape = {epsp_pred_np.shape}")

    # ------------------------------------------------------------
    # Load full ground-truth trajectory
    # ------------------------------------------------------------
    truth = np.load(args.data_truth)
    required_truth = ["eps", "epsp", "deps", "sig"]
    if not all(k in truth.files for k in required_truth):
        raise RuntimeError(
            f"Ground-truth NPZ {args.data_truth} must contain keys {required_truth}."
        )

    eps_all  = truth["eps"]    # (T,6) or (1,T,6)
    epsp_all = truth["epsp"]   # (T,6) or (1,T,6)
    sig_all  = truth["sig"]    # (T,6) or (1,T,6)

    # Allow both (T,6) and (1,T,6)
    if eps_all.ndim == 2:
        eps_all  = eps_all[None, ...]
        epsp_all = epsp_all[None, ...]
        sig_all  = sig_all[None, ...]

    eps_all  = eps_all[0]   # (T,6)
    epsp_all = epsp_all[0]
    sig_all  = sig_all[0]

    T = eps_all.shape[0]
    print(f"[INFO] Ground-truth trajectory length T={T}")

    if t0 + H_test > T:
        raise RuntimeError(f"Rollout horizon t0+H_test = {t0 + H_test} exceeds T={T}.")

    # Time indices
    t_full = np.arange(T)                       # 0 .. T-1
    t_roll = np.arange(t0, t0 + H_test)         # t0 .. t0+H_test-1

    # ------------------------------------------------------------
    # Slices over rollout horizon (for prediction and horizon truth)
    # ------------------------------------------------------------
    eps_roll       = eps_all[t0:t0 + H_test]    # (H_test, 6) ε_I^truth on rollout horizon
    # We still use sig_all/epsp_all over [0,T) for "entire horizon" truth curves.

    # ------------------------------------------------------------
    # Prepare output directory
    # ------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    # Voigt component indices and labels
    # Order: xx, yy, zz, yz, zx, xy  -> I = 0..5
    voigt_indices = [0, 1, 2, 3, 4, 5]
    panel_labels = [
        r'$\sigma_{11}$ vs $\varepsilon_{11}$',
        r'$\sigma_{22}$ vs $\varepsilon_{22}$',
        r'$\sigma_{33}$ vs $\varepsilon_{33}$',
        r'$\sigma_{23}$ vs $\varepsilon_{23}$',
        r'$\sigma_{13}$ vs $\varepsilon_{13}$',
        r'$\sigma_{12}$ vs $\varepsilon_{12}$',
    ]
    sigma_tex = [r'\sigma_{11}', r'\sigma_{22}', r'\sigma_{33}',
                 r'\sigma_{23}', r'\sigma_{13}', r'\sigma_{12}']
    eps_tex   = [r'\varepsilon_{11}', r'\varepsilon_{22}', r'\varepsilon_{33}',
                 r'\varepsilon_{23}', r'\varepsilon_{13}', r'\varepsilon_{12}']

    # For Figure 2 (epsp), define nicer labels
    epsp_tex = [r'\varepsilon_{p,11}', r'\varepsilon_{p,22}', r'\varepsilon_{p,33}',
                r'\varepsilon_{p,23}', r'\varepsilon_{p,13}', r'\varepsilon_{p,12}']

    # ------------------------------------------------------------
    # Figure 1: Stress–strain comparison
    #   - truth over [0, T)
    #   - prediction over [t0, t0+H_test)
    # ------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    fig1.suptitle(
        "Figure 1: Stress–strain comparison in Voigt form\n"
        f"truth: [0, T={T}), prediction: [t0={t0}, t0+H_test={t0+H_test}) "
        f"(L={L}, H_model={H_model}, H_test={H_test})",
        fontsize=12
    )

    for idx, ax in enumerate(axes1.ravel()):
        I = voigt_indices[idx]

        # Truth over entire horizon
        x_full = eps_all[:, I]      # ε_I^truth over [0,T)
        y_full = sig_all[:, I]      # σ_I^truth over [0,T)

        # Prediction over rollout horizon
        x_roll = eps_roll[:, I]     # ε_I^truth over [t0, t0+H_test)
        y_pred = sig_pred_np[:, I]  # σ_I^pred over [t0, t0+H_test)

        ax.plot(x_full, y_full, '-o', lw=1.0, ms=3, label="ground truth (full)")
        ax.plot(x_roll, y_pred,  '-s', lw=1.0, ms=3, label="prediction (rollout)")

        ax.set_xlabel(rf'${eps_tex[idx]}$')
        ax.set_ylabel(rf'${sigma_tex[idx]}$')
        ax.set_title(panel_labels[idx], fontsize=10)
        ax.tick_params(labelsize=8)

    handles1, labels1 = axes1[0, 0].get_legend_handles_labels()
    fig1.legend(handles1, labels1, loc="lower center", ncol=2, fontsize=9, frameon=False)

    out_path1 = os.path.join(args.out_dir, f"stress_strain_rollout_voigt_t0_{t0:04d}.png")
    fig1.savefig(out_path1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"[OK] Saved {out_path1}")

    # ------------------------------------------------------------
    # Figure 2: Plastic strain–time comparison
    #   - epsp truth over [0, T)
    #   - epsp prediction over [t0, t0+H_test)
    #   x-axis = time step index
    # ------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    fig2.suptitle(
        "Figure 2: Plastic strain vs time step (Voigt components)\n"
        f"truth: [0, T={T}), prediction: [t0={t0}, t0+H_test={t0+H_test}) "
        f"(L={L}, H_model={H_model}, H_test={H_test})",
        fontsize=12
    )

    for idx, ax in enumerate(axes2.ravel()):
        I = voigt_indices[idx]

        # Truth over entire horizon: epsp vs time step
        y_full_epsp = epsp_all[:, I]           # ε_{e,I}^truth over [0,T)

        # Prediction over rollout horizon: epsp vs time step
        y_pred_epsp = epsp_pred_np[:, I]       # ε_{e,I}^pred over [t0,t0+H_test)

        ax.plot(t_full, y_full_epsp, '-o', lw=1.0, ms=3,
                label="ground truth (epsp, full)")
        ax.plot(t_roll, y_pred_epsp,  '-s', lw=1.0, ms=3,
                label="prediction (epsp, rollout)")

        ax.set_xlabel(r'$t$ (time step)')
        ax.set_ylabel(rf'${epsp_tex[idx]}$')
        ax.set_title(rf'${epsp_tex[idx]}$ vs time step', fontsize=10)
        ax.tick_params(labelsize=8)

    handles2, labels2 = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc="lower center", ncol=2, fontsize=9, frameon=False)

    out_path2 = os.path.join(args.out_dir, f"epsp_time_rollout_voigt_t0_{t0:04d}.png")
    fig2.savefig(out_path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"[OK] Saved {out_path2}")

    print(f"[OK] All figures saved in:", args.out_dir)
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot Voigt-form stress–strain curves comparing rollout prediction vs ground truth.")
    ap.add_argument("--data_truth", type=str, required=True, help="Path to ground-truth NPZ with eps, epsp, deps, sig.")
    ap.add_argument("--data_pred", type=str, required=True, help="Path to rollout NPZ with sig_pred, epsp_pred, L, H_model, H_test, t0.")
    ap.add_argument("--out_dir", type=str, default="./figs_stress_strain", help="Directory for output figure.")
    args = ap.parse_args()
    main(args)
