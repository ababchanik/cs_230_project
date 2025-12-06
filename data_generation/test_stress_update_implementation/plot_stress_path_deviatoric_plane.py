#!/usr/bin/env python3
"""
Reproduce p–q and π-plane plots from NPZ files that contain stresses in Voigt-6 form.

Assumptions
-----------
- Voigt order is [xx, yy, zz, yz, xz, xy].
- "Real" stress path comes from file A, key 'sig' (T,6).
- "Fictitious" stress path comes from file B, key 'sig_test' (T,6).
- Compressional stresses are negative; p = tr(sigma)/3 follows that sign convention.

Usage
-----
python plot_stress_path_deviatoric_plane.py \
    --gold_npz  stress_relaxation_strain_rate_dt_1e-1.npz \
    --test_npz  stress_relaxation_strain_rate_dt_1e-1_test.npz \
    --gold_real_key  sig \
    --test_real_key  sig_test \
    --test_star_key  sig_star_test \
    --M 2.0 \
    --pc -6.0 \
    --out_prefix ccx_uc_viscoplastic

Outputs
-------
- <out_prefix>_postprocessed.csv : optional combined table of p, q, p*, q*, principal values, etc.
- <out_prefix>_PQ.png            : p–q with MCC ellipse and star path overlaid
- <out_prefix>_piplane.png       : π-plane principal-stress projection (and optional MCC circle)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------- Voigt helpers -------------------------
# Assume Voigt order: [xx, yy, zz, yz, xz, xy]
def voigt6_to_tensor(v6):
    return np.array([
        [v6[0], v6[5], v6[4]],
        [v6[5], v6[1], v6[3]],
        [v6[4], v6[3], v6[2]]
    ], dtype=float)

def invariants_from_voigt(v6):
    s = voigt6_to_tensor(v6)
    p = np.trace(s) / 3.0
    dev = s - p * np.eye(3)
    q = np.sqrt(1.5) * np.linalg.norm(dev, ord='fro')
    return p, q

def principals_from_voigt(v6):
    s = voigt6_to_tensor(v6)
    s = 0.5 * (s + s.T)
    w = np.linalg.eigvalsh(s)
    return np.sort(w)

def principals_to_primeprime(sig123):
    Q1 = np.array([
        [np.sqrt(2)/2, 0, np.sqrt(2)/2],
        [0, 1, 0],
        [-np.sqrt(2)/2, 0, np.sqrt(2)/2]
    ])
    Q2 = np.array([
        [1, 0, 0],
        [0, np.sqrt(2)/np.sqrt(3), 1/np.sqrt(3)],
        [0, -1/np.sqrt(3), np.sqrt(2)/np.sqrt(3)]
    ])
    Q = (Q1 @ Q2).T
    return Q @ sig123

# ------------------------- MCC helpers -------------------------
def mcc_q_of_p(p, M, pc):
    return M * np.sqrt(np.maximum(0.0, p * (pc - p)))

def mcc_pi_radius_from_q(q):
    return np.sqrt(2.0/3.0) * q

# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_npz", required=True)
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--gold_real_key", default="sig")
    ap.add_argument("--test_real_key", default="sig_test")
    ap.add_argument("--test_star_key", default="sig_test_star")
    ap.add_argument("--M", type=float, default=2.0)
    ap.add_argument("--pc", type=float, default=-6.0)
    ap.add_argument("--out_prefix", default="ccx_uc_viscoplastic")
    args = ap.parse_args()

    # Load
    d_gold = np.load(args.gold_npz, allow_pickle=True)
    d_test = np.load(args.test_npz, allow_pickle=True)
    gold_sig_real = d_gold[args.gold_real_key]
    test_sig_real = d_test[args.test_real_key]
    test_sig_star = d_test[args.test_star_key]

    T = min(gold_sig_real.shape[0], test_sig_real.shape[0])
    gold_sig_real = gold_sig_real[:T]
    test_sig_real = test_sig_real[:T]
    test_sig_star = test_sig_star[:T]

    # Compute invariants
    gold_p_real, gold_q_real = np.zeros(T), np.zeros(T)
    test_p_real, test_q_real = np.zeros(T), np.zeros(T)
    test_p_star, test_q_star = np.zeros(T), np.zeros(T)
    for t in range(T):
        gold_p_real[t], gold_q_real[t] = invariants_from_voigt(gold_sig_real[t])
        test_p_real[t], test_q_real[t] = invariants_from_voigt(test_sig_real[t])
        test_p_star[t], test_q_star[t] = invariants_from_voigt(test_sig_star[t])

    # Principal stresses and prime-prime
    gold_sig_pp_real = np.zeros((T, 3))
    test_sig_pp_real = np.zeros((T, 3))
    test_sig_pp_star = np.zeros((T, 3))
    for t in range(T):
        gold_vals_r = principals_from_voigt(gold_sig_real[t])
        test_vals_r = principals_from_voigt(test_sig_real[t])
        test_vals_s = principals_from_voigt(test_sig_star[t])
        gold_sig_pp_real[t] = principals_to_primeprime(gold_vals_r)
        test_sig_pp_real[t] = principals_to_primeprime(test_vals_r)
        test_sig_pp_star[t] = principals_to_primeprime(test_vals_s)

    # === FIGURE 1: p–q plane ===
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # -- MCC ellipse
    pgrid = np.linspace(0.0, args.pc, 400)
    qgrid = args.M * np.sqrt(np.maximum(0.0, pgrid * (args.pc - pgrid)))
    axs[0].plot(pgrid,  qgrid, "-", color="k", lw=2)
    axs[0].plot(pgrid, -qgrid, "-", color="k", lw=2)
    axs[0].text(0.05, 0.95, r"$q^2/M^2 + p(p-p_c)=0$",
                transform=axs[0].transAxes, va="top", ha="left")

    axs[0].plot(gold_p_real, gold_q_real, "-ok", lw=2, mfc="none", label="gold: q vs p")
    axs[0].plot(test_p_real, test_q_real, "--s", lw=2, mfc="none", label="test: q vs p")
    axs[0].plot(test_p_star, test_q_star, "--x", lw=2, mfc="none", label="test: q* vs p*")
    axs[0].set_xlabel("Invariant p")
    axs[0].set_ylabel("Invariant q")
    axs[0].set_title("Figure 1: Stress Invariant Curves")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_aspect('equal', adjustable='box')
    #axs[0].set_xlim(args.pc * 1.05, 0.5)
    #axs[0].set_ylim(0, abs(args.pc) * args.M * 0.6)
    from mcc_pi_plane import show_quadrant_II
    show_quadrant_II(axs[0]) 

    # === FIGURE 2: π-plane ===
    axs[1].plot(gold_sig_pp_real[:, 0], gold_sig_pp_real[:, 1], "-ok", lw=2, mfc="none",
                label=r"gold: $\sigma_{2}^{\prime\prime}$ vs $\sigma_{1}^{\prime\prime}$")
    axs[1].plot(test_sig_pp_real[:, 0], test_sig_pp_real[:, 1], "--x", lw=2, mfc="none",
                label=r"test: $\sigma_{2}^{\prime\prime}$ vs $\sigma_{1}^{\prime\prime}$")
    axs[1].plot(test_sig_pp_star[:, 0], test_sig_pp_star[:, 1], "--x", lw=2, mfc="none",
                label=r"test: $\sigma_{2}^{*\prime\prime}$ vs $\sigma_{1}^{*\prime\prime}$")

    # Overlay MCC π-plane circle (approx)
    '''
    p_last = test_p_star[-1]
    q_last = test_q_star[-1]
    R = mcc_pi_radius_from_q(q_last)
    th = np.linspace(0, 2 * np.pi, 400)
    axs[1].plot(R * np.cos(th), R * np.sin(th), "k-", lw=1.5, label="MCC (π-plane)")

    axs[1].set_xlabel(r"$\sigma_{1}^{\prime\prime}$")
    axs[1].set_ylabel(r"$\sigma_{2}^{\prime\prime}$")
    axs[1].set_title("Figure 2: Principal Stresses in Prime-Prime Space")
    axs[1].grid(True)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_xlim(-abs(R) * 1.5, abs(R) * 1.5)
    axs[1].set_ylim(-abs(R) * 1.5, abs(R) * 1.5)
    '''

    try:
        from mcc_pi_plane import show_quadrant_II
        show_quadrant_II(axs[1])
    except Exception:
        axs[1].invert_xaxis()

    axs[1].legend()
    plt.tight_layout()

    out_file = args.out_prefix + "_PQ_piplane.png"
    fig.savefig(out_file, dpi=600)
    print(f"[+] Saved: {out_file}")

if __name__ == "__main__":
    main()
