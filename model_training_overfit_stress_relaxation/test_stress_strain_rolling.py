#!/usr/bin/env python3
"""
Usage:
python test_stress_strain_rolling.py \
  --data ../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz \
  --ckpt model_overfit.pt \
  --L 2 \
  --H 2 \
  --hid 128 \
  --H_test 16 \
  --out_dir ./figs_relax
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import SeqModel
from tensor_utils import TensorUtils  # provides from_voigt6(...)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Horizon definitions ----
    H_model = args.H
    # For single trajectory, H_test defaults to "as far as we can go"
    # starting from t0 = L.
    # We'll set H_test later after knowing T if args.H_test is None.

    # ---- Load single-path data (Voigt-6) ----
    data = np.load(args.data)
    required = ["eps", "epse", "deps", "sig"]
    if not all(k in data.files for k in required):
        raise RuntimeError(f"Missing arrays; need {required} in {args.data}")

    eps_all   = data["eps"]    # (T,6) or (1,T,6)
    epse_all  = data["epse"]
    deps_all  = data["deps"]
    sig_all   = data["sig"]

    # Allow both (T,6) and (1,T,6)
    if eps_all.ndim == 2:
        eps_all   = eps_all[None, ...]
        epse_all  = epse_all[None, ...]
        deps_all  = deps_all[None, ...]
        sig_all   = sig_all[None, ...]

    # Now assume exactly one specimen: (1, T, 6)
    eps_all   = eps_all[0]   # (T,6)
    epse_all  = epse_all[0]
    deps_all  = deps_all[0]
    sig_all   = sig_all[0]

    # Basic shape sanity
    if not (eps_all.shape == epse_all.shape == deps_all.shape == sig_all.shape):
        raise RuntimeError("All fields must have the same shape (T, 6).")

    T = eps_all.shape[0]

    # ---- Choose start time and evaluation horizon ----
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

    # ---- Load model ----
    model = SeqModel(H=H_model, hid=args.hid).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # ---- Ground-truth horizon [t0, t0 + H_test) ----
    sl_out = slice(t0, t0 + H_test)
    sig_true_v_np = sig_all[sl_out]   # (H_test, 6)
    eps_true_v_np = eps_all[sl_out]   # (H_test, 6)

    # ---- Rolling prediction over H_test using blocks of size H_model ----
    sig_pred_roll_np = np.zeros_like(sig_true_v_np)  # (H_test, 6)
    n_blocks = int(np.ceil(H_test / H_model))

    for k in range(n_blocks):
        # Block indices in [0, H_test)
        g_start = k * H_model
        g_end   = min((k + 1) * H_model, H_test)
        block_len = g_end - g_start
        if block_len <= 0:
            continue

        # Corresponding time index in the specimen: t_block_start = t0 + g_start
        t_block_start = t0 + g_start

        # Inputs: last L steps before t_block_start
        in_start = t_block_start - args.L
        in_end   = t_block_start

        eps_in  = torch.from_numpy(eps_all[in_start:in_end]).float().unsqueeze(0).to(device)
        epse_in = torch.from_numpy(epse_all[in_start:in_end]).float().unsqueeze(0).to(device)
        deps_in = torch.from_numpy(deps_all[in_start:in_end]).float().unsqueeze(0).to(device)

        with torch.no_grad():
            sig_pred_block, _ = model(eps_in, epse_in, deps_in)  # (1, H_model, 6)

        sig_pred_block_np = sig_pred_block.squeeze(0).cpu().numpy()  # (H_model, 6)

        # Fill the portion we actually need for this block
        sig_pred_roll_np[g_start:g_end] = sig_pred_block_np[:block_len]

    # ---- Convert Voigt-6 to 3×3 for plotting ----
    sig_pred_v = torch.from_numpy(sig_pred_roll_np).float()    # (H_test, 6)
    sig_true_v = torch.from_numpy(sig_true_v_np).float()       # (H_test, 6)
    eps_true_v = torch.from_numpy(eps_true_v_np).float()       # (H_test, 6)

    sig_pred = TensorUtils.from_voigt6(sig_pred_v.unsqueeze(0)).squeeze(0)  # (H_test, 3, 3)
    sig_true = TensorUtils.from_voigt6(sig_true_v.unsqueeze(0)).squeeze(0)  # (H_test, 3, 3)
    eps_true = TensorUtils.from_voigt6(eps_true_v.unsqueeze(0)).squeeze(0)  # (H_test, 3, 3)

    # ---- Prepare output dir ----
    os.makedirs(args.out_dir, exist_ok=True)

    # 3×3 panel order and labels
    comp_labels = [
        (0, 0, r'$\sigma_{11}$ vs $\varepsilon_{11}$'),
        (1, 1, r'$\sigma_{22}$ vs $\varepsilon_{22}$'),
        (2, 2, r'$\sigma_{33}$ vs $\varepsilon_{33}$'),
        (1, 2, r'$\sigma_{23}$ vs $\varepsilon_{23}$'),
        (0, 2, r'$\sigma_{13}$ vs $\varepsilon_{13}$'),
        (0, 1, r'$\sigma_{12}$ vs $\varepsilon_{12}$'),
        (2, 1, r'$\sigma_{32}$ vs $\varepsilon_{32}$'),
        (2, 0, r'$\sigma_{31}$ vs $\varepsilon_{31}$'),
        (1, 0, r'$\sigma_{21}$ vs $\varepsilon_{21}$'),
    ]

    # ---- Plot 3×3 stress–strain grid ----
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    fig.suptitle(
        f"Single relaxation path, t0={t0} "
        f"(H_model={H_model}, H_test={H_test})",
        fontsize=12,
    )

    for (i, j, label), ax in zip(comp_labels, axes.ravel()):
        x = eps_true[:, i, j].numpy()
        y_true = sig_true[:, i, j].numpy()
        y_pred = sig_pred[:, i, j].numpy()

        ax.plot(x, y_true, '-o', lw=1.0, ms=3, label='ground truth')
        ax.plot(x, y_pred, '-s', lw=1.0, ms=3, label='prediction')
        ax.set_xlabel(rf'$\varepsilon_{{{i+1}{j+1}}}$')
        ax.set_ylabel(rf'$\sigma_{{{i+1}{j+1}}}$')
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9, frameon=False)

    fname = os.path.join(
        args.out_dir,
        f"stress_strain_t0_{t0:04d}.png"
    )
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")
    print(f"All figures saved in: {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot stress–strain curves for a single relaxation trajectory."
    )
    ap.add_argument("--data", type=str, required=True,
                    help="Path to npz with keys {eps, epse, deps, sig} (single trajectory)")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to trained model .pt")
    ap.add_argument("--L", type=int, default=2,
                    help="Lookback window length (must match training)")
    ap.add_argument("--H", type=int, default=2,
                    help="Model prediction horizon per call (must match training)")
    ap.add_argument("--H_test", type=int, default=None,
                    help="Total evaluation horizon; defaults to T-L if not set")
    ap.add_argument("--hid", type=int, default=128,
                    help="Hidden size (must match training)")
    ap.add_argument("--out_dir", type=str, default="./figs_relax",
                    help="Output directory for figures")
    args = ap.parse_args()
    main(args)
