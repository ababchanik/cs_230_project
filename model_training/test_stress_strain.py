# # # test_stress_strain.py
# # #!/usr/bin/env python3
# # import argparse
# # import os
# # import numpy as np
# # import torch
# # import matplotlib.pyplot as plt

# # from model import SeqModel
# # from tensor_utils import TensorUtils  # provides from_voigt6(...)

# # def build_eval_window_indices(data, L, H, split_key='split', label='eval'):
# #     """
# #     Build (i_spec, t) pairs for windows only from specimens labeled `label`.
# #     Expects labeled dataset with `split` array and time series arrays shaped (m, T, 6).
# #     """
# #     if split_key not in data.files:
# #         raise RuntimeError("Dataset must be labeled; missing 'split' array.")

# #     labels = data[split_key]                      # (m,)
# #     if 'eps' not in data.files:
# #         raise RuntimeError("Dataset missing 'eps' array needed to infer T.")
# #     m, T, D = data['eps'].shape
# #     if D != 6:
# #         raise RuntimeError(f"Expected Voigt-6 last dimension; got {D}")

# #     eval_specs = np.where(labels == label)[0]
# #     if eval_specs.size == 0:
# #         raise RuntimeError(f"No specimens labeled '{label}' in this dataset.")

# #     idxs = []
# #     for i in eval_specs:
# #         for t in range(L, T - H):
# #             idxs.append((i, t))
# #     return idxs, m, T

# # def main(args):
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #     # ---- Load labeled data (Voigt-6) ----
# #     data = np.load(args.data)
# #     required = ["eps", "epse", "deps", "sig", "split"]
# #     if not all(k in data.files for k in required):
# #         raise RuntimeError(f"Missing arrays; need {required} in {args.data}")

# #     eps_all   = data["eps"]    # (m, T, 6)
# #     epse_all  = data["epse"]   # (m, T, 6)
# #     deps_all  = data["deps"]   # (m, T, 6)
# #     sig_all   = data["sig"]    # (m, T, 6)

# #     # Basic shape sanity
# #     if not (eps_all.shape == epse_all.shape == deps_all.shape == sig_all.shape):
# #         raise RuntimeError("All fields must have the same shape (m, T, 6).")

# #     # ---- Build eval windows ----
# #     eval_idxs, m, T = build_eval_window_indices(data, args.L, args.H, split_key="split", label="eval")
# #     if len(eval_idxs) == 0:
# #         raise RuntimeError("No eval windows available with current L/H.")

# #     rng = np.random.default_rng(args.seed)
# #     chosen = rng.choice(len(eval_idxs), size=min(args.n_samples, len(eval_idxs)), replace=False)
# #     chosen = [eval_idxs[i] for i in chosen]  # list of (i_spec, t)

# #     # ---- Load model ----
# #     model = SeqModel(H=args.H, hid=args.hid).to(device)
# #     ckpt = torch.load(args.ckpt, map_location=device)
# #     model.load_state_dict(ckpt)
# #     model.eval()

# #     # ---- Prepare output dir ----
# #     os.makedirs(args.out_dir, exist_ok=True)

# #     # 3×3 panel order and labels
# #     comp_labels = [
# #         (0,0,r'$\sigma_{11}$ vs $\varepsilon_{11}$'),
# #         (1,1,r'$\sigma_{22}$ vs $\varepsilon_{22}$'),
# #         (2,2,r'$\sigma_{33}$ vs $\varepsilon_{33}$'),
# #         (1,2,r'$\sigma_{23}$ vs $\varepsilon_{23}$'),
# #         (0,2,r'$\sigma_{13}$ vs $\varepsilon_{13}$'),
# #         (0,1,r'$\sigma_{12}$ vs $\varepsilon_{12}$'),
# #         (2,1,r'$\sigma_{32}$ vs $\varepsilon_{32}$'),
# #         (2,0,r'$\sigma_{31}$ vs $\varepsilon_{31}$'),
# #         (1,0,r'$\sigma_{21}$ vs $\varepsilon_{21}$'),
# #     ]

# #     for (i_spec, t) in chosen:
# #         sl_in  = slice(t - args.L, t)
# #         sl_out = slice(t, t + args.H)

# #         # Inputs: (1, L, 6) in Voigt-6
# #         eps_in   = torch.from_numpy(eps_all[i_spec,  sl_in]).float().unsqueeze(0).to(device)
# #         epse_in  = torch.from_numpy(epse_all[i_spec, sl_in]).float().unsqueeze(0).to(device)
# #         deps_in  = torch.from_numpy(deps_all[i_spec, sl_in]).float().unsqueeze(0).to(device)

# #         # Predict σ and ε^e over horizon (Voigt-6)
# #         with torch.no_grad():
# #             sig_pred_v, epse_pred_v = model(eps_in, epse_in, deps_in)  # (1, H, 6) each

# #         sig_pred_v  = sig_pred_v.squeeze(0)   # (H, 6)
# #         epse_pred_v = epse_pred_v.squeeze(0)  # (H, 6)

# #         # Ground-truth over horizon (Voigt-6)
# #         sig_true_v = torch.from_numpy(sig_all[i_spec,  sl_out]).float()   # (H, 6)
# #         eps_true_v = torch.from_numpy(eps_all[i_spec,  sl_out]).float()   # (H, 6)

# #         # Convert to 3×3 for plotting
# #         sig_pred = TensorUtils.from_voigt6(sig_pred_v.unsqueeze(0)).squeeze(0)   # (H, 3, 3)
# #         sig_true = TensorUtils.from_voigt6(sig_true_v.unsqueeze(0)).squeeze(0)   # (H, 3, 3)
# #         eps_true = TensorUtils.from_voigt6(eps_true_v.unsqueeze(0)).squeeze(0)   # (H, 3, 3)

# #         # ---- Plot 3×3 stress–strain grid ----
# #         fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
# #         fig.suptitle(f"Eval specimen {i_spec}, window start t={t} (H={args.H})", fontsize=12)

# #         for (i, j, label), ax in zip(comp_labels, axes.ravel()):
# #             x = eps_true[:, i, j].numpy()
# #             y_true = sig_true[:, i, j].numpy()
# #             y_pred = sig_pred[:, i, j].cpu().numpy()
            
# #             # === ADD RMSE CALCULATION HERE ===
# #             rmse = np.sqrt(np.mean((y_pred - y_true)**2))
# #             # =================================
            
# #             ax.plot(x, y_true, '-o', lw=1.0, ms=3, label='ground truth')
# #             ax.plot(x, y_pred, '-s', lw=1.0, ms=3, label='prediction')
            
# #             # === ADD RMSE TEXT TO PLOT ===
# #             ax.text(0.05, 0.95, f'RMSE: {rmse:.4f}', 
# #                     transform=ax.transAxes, 
# #                     fontsize=8,
# #                     verticalalignment='top',
# #                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
# #             # =============================
            
# #             ax.set_xlabel(rf'$\varepsilon_{{{i+1}{j+1}}}$')
# #             ax.set_ylabel(rf'$\sigma_{{{i+1}{j+1}}}$')
# #             ax.set_title(label, fontsize=10)
# #             ax.tick_params(labelsize=8)

# #         handles, labels = axes[0, 0].get_legend_handles_labels()
# #         fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9, frameon=False)

# #         fname = os.path.join(args.out_dir, f"eval_specimen_{i_spec:03d}_t{t:04d}.png")
# #         fig.savefig(fname, dpi=300, bbox_inches='tight')
# #         plt.close(fig)
# #         print(f"Saved {fname}")

# #     print(f"All figures saved in: {args.out_dir}")

# # if __name__ == "__main__":
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--data", type=str, required=True, help="Path to *labeled* npz with keys {eps, epse, deps, sig, split}")
# #     ap.add_argument("--ckpt", type=str, required=True, help="Path to trained model .pt")
# #     ap.add_argument("--L", type=int, default=10, help="Lookback window length")
# #     ap.add_argument("--H", type=int, default=10, help="Prediction horizon")
# #     ap.add_argument("--hid", type=int, default=128, help="Hidden size (must match training)")
# #     ap.add_argument("--n_samples", type=int, default=5, help="Number of eval windows to plot")
# #     ap.add_argument("--seed", type=int, default=1234, help="Random seed for window selection")
# #     ap.add_argument("--out_dir", type=str, default="./figs", help="Output directory for figures")
# #     args = ap.parse_args()
# #     main(args)


# # #!/usr/bin/env python3
# # import argparse
# # import os
# # import numpy as np
# # import torch
# # import matplotlib.pyplot as plt
# # from matplotlib.gridspec import GridSpec

# # from model import SeqModel
# # from tensor_utils import TensorUtils

# # def build_eval_window_indices(data, L, H, split_key='split', label='eval'):
# #     """
# #     Build (i_spec, t) pairs for windows only from specimens labeled `label`.
# #     """
# #     if split_key not in data.files:
# #         raise RuntimeError("Dataset must be labeled; missing 'split' array.")

# #     labels = data[split_key]
# #     if 'eps' not in data.files:
# #         raise RuntimeError("Dataset missing 'eps' array needed to infer T.")
# #     m, T, D = data['eps'].shape
# #     if D != 6:
# #         raise RuntimeError(f"Expected Voigt-6 last dimension; got {D}")

# #     eval_specs = np.where(labels == label)[0]
# #     if eval_specs.size == 0:
# #         raise RuntimeError(f"No specimens labeled '{label}' in this dataset.")

# #     idxs = []
# #     for i in eval_specs:
# #         for t in range(L, T - H):
# #             idxs.append((i, t))
# #     return idxs, m, T

# # def compute_invariants(tensor_3x3):
# #     """
# #     Compute stress invariants.
# #     Returns: p (mean stress), q (von Mises stress)
# #     """
# #     # Convert to numpy if torch tensor
# #     if isinstance(tensor_3x3, torch.Tensor):
# #         tensor_3x3 = tensor_3x3.cpu().numpy()
    
# #     # I1 = trace(σ)
# #     I1 = np.trace(tensor_3x3, axis1=-2, axis2=-1)
    
# #     # Mean stress p = I1/3
# #     p = I1 / 3.0
    
# #     # Deviatoric stress: s = σ - pI
# #     identity = np.eye(3)
# #     if tensor_3x3.ndim == 3:
# #         identity = identity[np.newaxis, :, :]
    
# #     s = tensor_3x3 - p[:, np.newaxis, np.newaxis] * identity
    
# #     # J2 = 0.5 s_ij s_ij
# #     J2 = 0.5 * np.einsum('...ij,...ij->...', s, s)
    
# #     # von Mises stress q = sqrt(3*J2)
# #     q = np.sqrt(3 * J2)
    
# #     return p, q

# # def compute_strain_invariants(tensor_3x3):
# #     """
# #     Compute strain invariants.
# #     Returns: ε_v (volumetric strain), ε_q (deviatoric strain)
# #     """
# #     if isinstance(tensor_3x3, torch.Tensor):
# #         tensor_3x3 = tensor_3x3.cpu().numpy()
    
# #     # Volumetric strain ε_v = trace(ε)
# #     ε_v = np.trace(tensor_3x3, axis1=-2, axis2=-1)
    
# #     # Deviatoric strain: e = ε - (ε_v/3)I
# #     identity = np.eye(3)
# #     if tensor_3x3.ndim == 3:
# #         identity = identity[np.newaxis, :, :]
    
# #     e = tensor_3x3 - (ε_v[:, np.newaxis, np.newaxis] / 3.0) * identity
    
# #     # Second invariant of deviatoric strain
# #     ε_q = np.sqrt((2/3) * np.einsum('...ij,...ij->...', e, e))
    
# #     return ε_v, ε_q

# # def main(args):
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #     # ---- Load labeled data (Voigt-6) ----
# #     data = np.load(args.data)
# #     required = ["eps", "epse", "deps", "sig", "split"]
# #     if not all(k in data.files for k in required):
# #         raise RuntimeError(f"Missing arrays; need {required} in {args.data}")

# #     eps_all   = data["eps"]    # (m, T, 6)
# #     epse_all  = data["epse"]   # (m, T, 6)
# #     deps_all  = data["deps"]   # (m, T, 6)
# #     sig_all   = data["sig"]    # (m, T, 6)

# #     # Basic shape sanity
# #     if not (eps_all.shape == epse_all.shape == deps_all.shape == sig_all.shape):
# #         raise RuntimeError("All fields must have the same shape (m, T, 6).")

# #     # ---- Build eval windows ----
# #     eval_idxs, m, T = build_eval_window_indices(data, args.L, args.H, split_key="split", label="eval")
# #     if len(eval_idxs) == 0:
# #         raise RuntimeError("No eval windows available with current L/H.")

# #     rng = np.random.default_rng(args.seed)
# #     chosen = rng.choice(len(eval_idxs), size=min(args.n_samples, len(eval_idxs)), replace=False)
# #     chosen = [eval_idxs[i] for i in chosen]

# #     # ---- Load model ----
# #     model = SeqModel(H=args.H, hid=args.hid).to(device)
# #     ckpt = torch.load(args.ckpt, map_location=device)
# #     model.load_state_dict(ckpt)
# #     model.eval()

# #     # ---- Prepare output dir ----
# #     os.makedirs(args.out_dir, exist_ok=True)

# #     # Component labels for 3x3 tensors
# #     comp_labels = [
# #         (0,0,r'$\sigma_{11}$ vs $\varepsilon_{11}$'),
# #         (1,1,r'$\sigma_{22}$ vs $\varepsilon_{22}$'),
# #         (2,2,r'$\sigma_{33}$ vs $\varepsilon_{33}$'),
# #         (1,2,r'$\sigma_{23}$ vs $\varepsilon_{23}$'),
# #         (0,2,r'$\sigma_{13}$ vs $\varepsilon_{13}$'),
# #         (0,1,r'$\sigma_{12}$ vs $\varepsilon_{12}$'),
# #         (2,1,r'$\sigma_{32}$ vs $\varepsilon_{32}$'),
# #         (2,0,r'$\sigma_{31}$ vs $\varepsilon_{31}$'),
# #         (1,0,r'$\sigma_{21}$ vs $\varepsilon_{21}$'),
# #     ]

# #     for (i_spec, t) in chosen:
# #         sl_in  = slice(t - args.L, t)
# #         sl_out = slice(t, t + args.H)

# #         # Inputs: (1, L, 6) in Voigt-6
# #         eps_in   = torch.from_numpy(eps_all[i_spec,  sl_in]).float().unsqueeze(0).to(device)
# #         epse_in  = torch.from_numpy(epse_all[i_spec, sl_in]).float().unsqueeze(0).to(device)
# #         deps_in  = torch.from_numpy(deps_all[i_spec, sl_in]).float().unsqueeze(0).to(device)

# #         # Predict σ and ε^e over horizon (Voigt-6)
# #         with torch.no_grad():
# #             sig_pred_v, epse_pred_v = model(eps_in, epse_in, deps_in)

# #         sig_pred_v  = sig_pred_v.squeeze(0)   # (H, 6)
# #         epse_pred_v = epse_pred_v.squeeze(0)  # (H, 6)

# #         # Ground-truth over horizon (Voigt-6)
# #         sig_true_v = torch.from_numpy(sig_all[i_spec,  sl_out]).float()   # (H, 6)
# #         eps_true_v = torch.from_numpy(eps_all[i_spec,  sl_out]).float()   # (H, 6)

# #         # Convert to 3×3 tensors
# #         sig_pred = TensorUtils.from_voigt6(sig_pred_v.unsqueeze(0)).squeeze(0)   # (H, 3, 3)
# #         sig_true = TensorUtils.from_voigt6(sig_true_v.unsqueeze(0)).squeeze(0)   # (H, 3, 3)
# #         eps_true = TensorUtils.from_voigt6(eps_true_v.unsqueeze(0)).squeeze(0)   # (H, 3, 3)

# #         # ===================================================================
# #         # PLOT 1: 3×3 Stress vs Strain Grid (Your original plot with RMSE)
# #         # ===================================================================
# #         fig1, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
# #         fig1.suptitle(f"Eval specimen {i_spec}, window start t={t} (H={args.H})", fontsize=14)

# #         component_errors = []
# #         for (i, j, label), ax in zip(comp_labels, axes.ravel()):
# #             x = eps_true[:, i, j].numpy()
# #             y_true = sig_true[:, i, j].numpy()
# #             y_pred = sig_pred[:, i, j].cpu().numpy()
            
# #             # RMSE calculation
# #             rmse = np.sqrt(np.mean((y_pred - y_true)**2))
# #             mae = np.mean(np.abs(y_pred - y_true))
# #             component_errors.append((i, j, rmse, mae))
            
# #             ax.plot(x, y_true, '-o', lw=1.5, ms=4, label='True', alpha=0.7)
# #             ax.plot(x, y_pred, '-s', lw=1.5, ms=4, label='Pred', alpha=0.7)
            
# #             # Add error text
# #             ax.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}', 
# #                     transform=ax.transAxes, 
# #                     fontsize=7,
# #                     verticalalignment='top',
# #                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
# #             ax.set_xlabel(rf'$\varepsilon_{{{i+1}{j+1}}}$')
# #             ax.set_ylabel(rf'$\sigma_{{{i+1}{j+1}}}$')
# #             ax.set_title(label, fontsize=10)
# #             ax.tick_params(labelsize=8)
# #             ax.grid(True, alpha=0.3)

# #         handles, labels = axes[0, 0].get_legend_handles_labels()
# #         fig1.legend(handles, labels, loc='lower center', ncol=2, fontsize=10)
        
# #         fname1 = os.path.join(args.out_dir, f"specimen_{i_spec:03d}_t{t:04d}_stress_strain.png")
# #         fig1.savefig(fname1, dpi=300, bbox_inches='tight')
# #         plt.close(fig1)
# #         print(f"Saved stress-strain plot: {fname1}")


# #         # ===================================================================
# #         # PLOT 4: Combined Stress-Strain Trajectory (3D or 2D projections)
# #         # ===================================================================
# #         fig4, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
# #         fig4.suptitle(f"Stress-Strain Trajectories (specimen {i_spec})", fontsize=14)
        
# #         # 4a: Normal stresses vs strains
# #         ax = axes[0, 0]
# #         colors = plt.cm.viridis(np.linspace(0, 1, args.H))
# #         for step in range(args.H):
# #             ax.scatter(eps_true[step, 0, 0].numpy(), sig_true[step, 0, 0].numpy(), 
# #                       color=colors[step], s=50, alpha=0.6)
# #             ax.scatter(eps_true[step, 1, 1].numpy(), sig_true[step, 1, 1].numpy(), 
# #                       color=colors[step], s=50, alpha=0.6, marker='s')
# #             ax.scatter(eps_true[step, 2, 2].numpy(), sig_true[step, 2, 2].numpy(), 
# #                       color=colors[step], s=50, alpha=0.6, marker='^')
        
# #         ax.plot(eps_true[:, 0, 0].numpy(), sig_pred[:, 0, 0].cpu().numpy(), 'r--', lw=1.5, label='Pred σ_11')
# #         ax.plot(eps_true[:, 1, 1].numpy(), sig_pred[:, 1, 1].cpu().numpy(), 'b--', lw=1.5, label='Pred σ_22')
# #         ax.plot(eps_true[:, 2, 2].numpy(), sig_pred[:, 2, 2].cpu().numpy(), 'g--', lw=1.5, label='Pred σ_33')
# #         ax.set_xlabel('Strain')
# #         ax.set_ylabel('Stress')
# #         ax.set_title('Normal Stress vs Strain')
# #         ax.legend(fontsize=8)
# #         ax.grid(True, alpha=0.3)
        
# #         # 4b: Shear stresses vs strains
# #         ax = axes[0, 1]
# #         for step in range(args.H):
# #             ax.scatter(eps_true[step, 0, 1].numpy(), sig_true[step, 0, 1].numpy(), 
# #                       color=colors[step], s=50, alpha=0.6)
# #             ax.scatter(eps_true[step, 1, 2].numpy(), sig_true[step, 1, 2].numpy(), 
# #                       color=colors[step], s=50, alpha=0.6, marker='s')
        
# #         ax.plot(eps_true[:, 0, 1].numpy(), sig_pred[:, 0, 1].cpu().numpy(), 'r--', lw=1.5, label='Pred σ_12')
# #         ax.plot(eps_true[:, 1, 2].numpy(), sig_pred[:, 1, 2].cpu().numpy(), 'b--', lw=1.5, label='Pred σ_23')
# #         ax.set_xlabel('Shear Strain')
# #         ax.set_ylabel('Shear Stress')
# #         ax.set_title('Shear Stress vs Strain')
# #         ax.legend(fontsize=8)
# #         ax.grid(True, alpha=0.3)
        
# #         # 4c: Invariant space (p-q plot)
# #         ax = axes[1, 0]
# #         # Compute invariants
# #         p_true, q_true = compute_invariants(sig_true.numpy())
# #         p_pred, q_pred = compute_invariants(sig_pred.cpu().numpy())
        
# #         ax.plot(p_true, q_true, 'b-o', lw=1.5, ms=4, label='True', alpha=0.7)
# #         ax.plot(p_pred, q_pred, 'r--s', lw=1.5, ms=4, label='Pred', alpha=0.7)
# #         ax.set_xlabel('Mean Stress p')
# #         ax.set_ylabel('Deviatoric Stress q')
# #         ax.set_title('Stress Path in p-q Space')
# #         ax.legend()
# #         ax.grid(True, alpha=0.3)
        
# #         # 4d: Stress-strain phase diagram for one component
# #         ax = axes[1, 1]
# #         # Use σ_11 vs ε_11
# #         σ_11_true = sig_true[:, 0, 0].numpy()
# #         ε_11_true = eps_true[:, 0, 0].numpy()
# #         σ_11_pred = sig_pred[:, 0, 0].cpu().numpy()
        
# #         # Plot as parametric curve (stress vs strain)
# #         ax.plot(ε_11_true, σ_11_true, 'b-o', lw=1.5, ms=4, label='True', alpha=0.7)
# #         ax.plot(ε_11_true, σ_11_pred, 'r--s', lw=1.5, ms=4, label='Pred', alpha=0.7)
        
# #         # Add expected elastic line (if you know Young's modulus)
# #         # For isotropic linear elasticity: σ = E * ε
# #         # Uncomment and adjust E if known:
# #         # E = 2000.0  # Example Young's modulus
# #         # ε_range = np.linspace(min(ε_11_true), max(ε_11_true), 50)
# #         # ax.plot(ε_range, E * ε_range, 'k:', lw=1, label='Elastic', alpha=0.5)
        
# #         ax.set_xlabel('ε_11')
# #         ax.set_ylabel('σ_11')
# #         ax.set_title('σ_11 vs ε_11 (with expected elastic line)')
# #         ax.legend(fontsize=8)
# #         ax.grid(True, alpha=0.3)
        
# #         fname4 = os.path.join(args.out_dir, f"specimen_{i_spec:03d}_t{t:04d}_trajectories.png")
# #         fig4.savefig(fname4, dpi=300, bbox_inches='tight')
# #         plt.close(fig4)
# #         print(f"Saved trajectory plots: {fname4}")

# #         # ===================================================================
# #         # PLOT 5: Error Analysis Summary
# #         # ===================================================================
# #         fig5, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
# #         fig5.suptitle(f"Error Analysis (specimen {i_spec})", fontsize=14)
        
# #         # 5a: Component-wise RMSE bar chart
# #         ax = axes[0]
# #         components = [f'({i},{j})' for i, j, _, _ in component_errors]
# #         rmse_values = [rmse for _, _, rmse, _ in component_errors]
        
# #         bars = ax.bar(range(len(components)), rmse_values)
# #         ax.set_xlabel('Component (i,j)')
# #         ax.set_ylabel('RMSE')
# #         ax.set_title('Component-wise RMSE')
# #         ax.set_xticks(range(len(components)))
# #         ax.set_xticklabels(components, rotation=45)
        
# #         # Add value labels on bars
# #         for bar, val in zip(bars, rmse_values):
# #             ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
# #                    f'{val:.4f}', ha='center', va='bottom', fontsize=7)
        
# #         # 5b: Prediction vs Truth scatter (all components)
# #         ax = axes[1]
# #         all_true = []
# #         all_pred = []
# #         for i in range(3):
# #             for j in range(3):
# #                 all_true.extend(sig_true[:, i, j].numpy())
# #                 all_pred.extend(sig_pred[:, i, j].cpu().numpy())
        
# #         all_true = np.array(all_true)
# #         all_pred = np.array(all_pred)
        
# #         ax.scatter(all_true, all_pred, alpha=0.5, s=10)
        
# #         # Add perfect prediction line
# #         min_val = min(all_true.min(), all_pred.min())
# #         max_val = max(all_true.max(), all_pred.max())
# #         ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Perfect')
        
# #         # Calculate overall R²
# #         ss_res = np.sum((all_pred - all_true)**2)
# #         ss_tot = np.sum((all_true - np.mean(all_true))**2)
# #         r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
# #         ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
# #                 transform=ax.transAxes, 
# #                 fontsize=10,
# #                 verticalalignment='top',
# #                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
# #         ax.set_xlabel('True Stress')
# #         ax.set_ylabel('Predicted Stress')
# #         ax.set_title(f'All Components: Predicted vs True (R²={r2:.4f})')
# #         ax.legend()
# #         ax.grid(True, alpha=0.3)
# #         ax.axis('equal')
        
# #         fname5 = os.path.join(args.out_dir, f"specimen_{i_spec:03d}_t{t:04d}_error_analysis.png")
# #         fig5.savefig(fname5, dpi=300, bbox_inches='tight')
# #         plt.close(fig5)
# #         print(f"Saved error analysis: {fname5}")

# #         # ===================================================================
# #         # Save summary statistics to text file
# #         # ===================================================================
# #         summary_file = os.path.join(args.out_dir, f"specimen_{i_spec:03d}_t{t:04d}_summary.txt")
# #         with open(summary_file, 'w') as f:
# #             f.write(f"Evaluation Summary for specimen {i_spec}, window start t={t}\n")
# #             f.write("="*60 + "\n\n")
            
# #             f.write("COMPONENT-WISE ERRORS:\n")
# #             f.write("Component |     RMSE     |      MAE\n")
# #             f.write("-"*40 + "\n")
# #             for i, j, rmse, mae in component_errors:
# #                 f.write(f"  ({i},{j})   |  {rmse:.6f}  |  {mae:.6f}\n")
            
# #             f.write("\nOVERALL STATISTICS:\n")
# #             f.write(f"Mean RMSE: {np.mean([e[2] for e in component_errors]):.6f}\n")
# #             f.write(f"Max RMSE: {np.max([e[2] for e in component_errors]):.6f}\n")
# #             f.write(f"R² score: {r2:.6f}\n")
            
# #             # Stress invariants comparison
# #             f.write("\nSTRESS INVARIANTS:\n")
# #             f.write("Step | p_true | p_pred | q_true | q_pred\n")
# #             f.write("-"*50 + "\n")
# #             for step in range(min(5, args.H)):  # First 5 steps
# #                 f.write(f"{step:4d} | {p_true[step]:.4f} | {p_pred[step]:.4f} | {q_true[step]:.4f} | {q_pred[step]:.4f}\n")
        
# #         print(f"Saved summary: {summary_file}")
# #         print("-"*60)

# #     print(f"\nAll analysis complete! Files saved in: {args.out_dir}")

# # if __name__ == "__main__":
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--data", type=str, required=True, help="Path to *labeled* npz with keys {eps, epse, deps, sig, split}")
# #     ap.add_argument("--ckpt", type=str, required=True, help="Path to trained model .pt")
# #     ap.add_argument("--L", type=int, default=10, help="Lookback window length")
# #     ap.add_argument("--H", type=int, default=10, help="Prediction horizon")
# #     ap.add_argument("--hid", type=int, default=128, help="Hidden size (must match training)")
# #     ap.add_argument("--n_samples", type=int, default=5, help="Number of eval windows to plot")
# #     ap.add_argument("--seed", type=int, default=1234, help="Random seed for window selection")
# #     ap.add_argument("--out_dir", type=str, default="./figs_comprehensive", help="Output directory for figures")
# #     args = ap.parse_args()
# #     main(args)



# #!/usr/bin/env python3
# import argparse
# import os
# import numpy as np
# import torch
# import matplotlib.pyplot as plt

# from model import SeqModel
# from tensor_utils import TensorUtils

# def build_eval_window_indices(data, L, H, split_key='split', label='eval'):
#     """
#     Build (i_spec, t) pairs for windows only from specimens labeled `label`.
#     """
#     if split_key not in data.files:
#         raise RuntimeError("Dataset must be labeled; missing 'split' array.")

#     labels = data[split_key]
#     if 'eps' not in data.files:
#         raise RuntimeError("Dataset missing 'eps' array needed to infer T.")
#     m, T, D = data['eps'].shape
#     if D != 6:
#         raise RuntimeError(f"Expected Voigt-6 last dimension; got {D}")

#     eval_specs = np.where(labels == label)[0]
#     if eval_specs.size == 0:
#         raise RuntimeError(f"No specimens labeled '{label}' in this dataset.")

#     idxs = []
#     for i in eval_specs:
#         for t in range(L, T - H):
#             idxs.append((i, t))
#     return idxs, m, T

# def main(args):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # ---- Load labeled data (Voigt-6) ----
#     data = np.load(args.data)
#     required = ["eps", "epse", "deps", "sig", "split"]
#     if not all(k in data.files for k in required):
#         raise RuntimeError(f"Missing arrays; need {required} in {args.data}")

#     eps_all   = data["eps"]    # (m, T, 6)
#     epse_all  = data["epse"]   # (m, T, 6)
#     deps_all  = data["deps"]   # (m, T, 6)
#     sig_all   = data["sig"]    # (m, T, 6)

#     # Basic shape sanity
#     if not (eps_all.shape == epse_all.shape == deps_all.shape == sig_all.shape):
#         raise RuntimeError("All fields must have the same shape (m, T, 6).")

#     # ---- Build eval windows ----
#     eval_idxs, m, T = build_eval_window_indices(data, args.L, args.H, split_key="split", label="eval")
#     if len(eval_idxs) == 0:
#         raise RuntimeError("No eval windows available with current L/H.")

#     rng = np.random.default_rng(args.seed)
#     chosen = rng.choice(len(eval_idxs), size=min(args.n_samples, len(eval_idxs)), replace=False)
#     chosen = [eval_idxs[i] for i in chosen]

#     # ---- Load model ----
#     model = SeqModel(H=args.H, hid=args.hid).to(device)
#     ckpt = torch.load(args.ckpt, map_location=device)
#     model.load_state_dict(ckpt)
#     model.eval()

#     # ---- Prepare output dir ----
#     os.makedirs(args.out_dir, exist_ok=True)

#     # CORRECT Voigt ordering labels for your data
#     # Your Voigt order: 11, 22, 33, 12, 23, 13
#     voigt_labels = [
#         (0, 0, 'σ_11', 'ε_11'),
#         (1, 1, 'σ_22', 'ε_22'), 
#         (2, 2, 'σ_33', 'ε_33'),
#         (0, 1, 'σ_12', 'ε_12'),
#         (1, 2, 'σ_23', 'ε_23'),
#         (0, 2, 'σ_13', 'ε_13')
#     ]

#     for (i_spec, t) in chosen:
#         sl_in  = slice(t - args.L, t)
#         sl_out = slice(t, t + args.H)

#         # Inputs: (1, L, 6) in Voigt-6
#         eps_in   = torch.from_numpy(eps_all[i_spec,  sl_in]).float().unsqueeze(0).to(device)
#         epse_in  = torch.from_numpy(epse_all[i_spec, sl_in]).float().unsqueeze(0).to(device)
#         deps_in  = torch.from_numpy(deps_all[i_spec, sl_in]).float().unsqueeze(0).to(device)

#         # Predict σ and ε^e over horizon (Voigt-6)
#         with torch.no_grad():
#             sig_pred_v, epse_pred_v = model(eps_in, epse_in, deps_in)

#         sig_pred_v  = sig_pred_v.squeeze(0)   # (H, 6)
#         epse_pred_v = epse_pred_v.squeeze(0)  # (H, 6)

#         # Ground-truth over horizon (Voigt-6)
#         sig_true_v = torch.from_numpy(sig_all[i_spec,  sl_out]).float()   # (H, 6)
#         eps_true_v = torch.from_numpy(eps_all[i_spec,  sl_out]).float()   # (H, 6)

#         # ===================================================================
#         # PLOT: 6 Voigt Components Stress vs Strain (2x3 grid)
#         # ===================================================================
#         fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
#         fig.suptitle(f"Specimen {i_spec}, t={t} to t+{args.H} (H={args.H})", fontsize=14)

#         for idx, (i, j, stress_label, strain_label) in enumerate(voigt_labels):
#             ax = axes[idx // 3, idx % 3]
            
#             # Get data for this Voigt component
#             x = eps_true_v[:, idx].numpy()      # strain component
#             y_true = sig_true_v[:, idx].numpy()  # true stress component  
#             y_pred = sig_pred_v[:, idx].cpu().numpy()  # predicted stress component
            
#             # Calculate errors
#             rmse = np.sqrt(np.mean((y_pred - y_true)**2))
#             mae = np.mean(np.abs(y_pred - y_true))
            
#             # Plot
#             ax.plot(x, y_true, '-o', lw=1.5, ms=4, label='True', alpha=0.7)
#             ax.plot(x, y_pred, '-s', lw=1.5, ms=4, label='Pred', alpha=0.7)
            
#             # Add error text
#             ax.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}', 
#                     transform=ax.transAxes, 
#                     fontsize=8,
#                     verticalalignment='top',
#                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
#             ax.set_xlabel(f'{strain_label}')
#             ax.set_ylabel(f'{stress_label}')
#             ax.set_title(f'{stress_label} vs {strain_label}')
#             ax.tick_params(labelsize=8)
#             ax.grid(True, alpha=0.3)
#             ax.legend(fontsize=8)

#         fname = os.path.join(args.out_dir, f"specimen_{i_spec:03d}_t{t:04d}_stress_strain.png")
#         fig.savefig(fname, dpi=300, bbox_inches='tight')
#         plt.close(fig)
#         print(f"Saved stress-strain plot: {fname}")

#         # ===================================================================
#         # Save summary statistics
#         # ===================================================================
#         summary_file = os.path.join(args.out_dir, f"specimen_{i_spec:03d}_t{t:04d}_summary.txt")
#         with open(summary_file, 'w') as f:
#             f.write(f"Evaluation Summary for specimen {i_spec}, window start t={t}\n")
#             f.write("="*60 + "\n\n")
            
#             f.write("VOIGT COMPONENT ERRORS (order: 11, 22, 33, 12, 23, 13):\n")
#             f.write("-"*60 + "\n")
#             f.write(f"{'Component':<8} {'RMSE':<12} {'MAE':<12} {'True Range':<20} {'Pred Range':<20}\n")
#             f.write("-"*60 + "\n")
            
#             for idx, (i, j, stress_label, strain_label) in enumerate(voigt_labels):
#                 true_comp = sig_true_v[:, idx].numpy()
#                 pred_comp = sig_pred_v[:, idx].cpu().numpy()
#                 true_range = f"[{true_comp.min():.3f}, {true_comp.max():.3f}]"
#                 pred_range = f"[{pred_comp.min():.3f}, {pred_comp.max():.3f}]"
                
#                 # Recalculate errors for summary
#                 rmse = np.sqrt(np.mean((pred_comp - true_comp)**2))
#                 mae = np.mean(np.abs(pred_comp - true_comp))
                
#                 f.write(f"{stress_label:<8} {rmse:<12.6f} {mae:<12.6f} {true_range:<20} {pred_range:<20}\n")
        
#         print(f"Saved summary: {summary_file}")
#         print("-"*60)

#     print(f"\nAll analysis complete! Files saved in: {args.out_dir}")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data", type=str, required=True, help="Path to *labeled* npz with keys {eps, epse, deps, sig, split}")
#     ap.add_argument("--ckpt", type=str, required=True, help="Path to trained model .pt")
#     ap.add_argument("--L", type=int, default=10, help="Lookback window length")
#     ap.add_argument("--H", type=int, default=10, help="Prediction horizon")
#     ap.add_argument("--hid", type=int, default=128, help="Hidden size (must match training)")
#     ap.add_argument("--n_samples", type=int, default=5, help="Number of eval windows to plot")
#     ap.add_argument("--seed", type=int, default=1234, help="Random seed for window selection")
#     ap.add_argument("--out_dir", type=str, default="./figs_clean", help="Output directory for figures")
#     args = ap.parse_args()
#     main(args)





# # #### with rolling...? - weird plots

# # #!/usr/bin/env python3
# # import argparse
# # import os
# # import numpy as np
# # import torch
# # import matplotlib.pyplot as plt

# # from model import SeqModel
# # from tensor_utils import TensorUtils

# # def build_eval_window_indices(data, L, H, split_key='split', label='eval'):
# #     """
# #     Build (i_spec, t) pairs for windows only from specimens labeled `label`.
# #     """
# #     if split_key not in data.files:
# #         raise RuntimeError("Dataset must be labeled; missing 'split' array.")

# #     labels = data[split_key]
# #     if 'eps' not in data.files:
# #         raise RuntimeError("Dataset missing 'eps' array needed to infer T.")
# #     m, T, D = data['eps'].shape
# #     if D != 6:
# #         raise RuntimeError(f"Expected Voigt-6 last dimension; got {D}")

# #     eval_specs = np.where(labels == label)[0]
# #     if eval_specs.size == 0:
# #         raise RuntimeError(f"No specimens labeled '{label}' in this dataset.")

# #     idxs = []
# #     for i in eval_specs:
# #         for t in range(L, T - H):
# #             idxs.append((i, t))
# #     return idxs, m, T

# # def rolling_predictions(model, data, specimen_idx, t_start, L, H, H_test, device):
# #     """
# #     Generate rolling predictions (same logic as your partner's test_rollout.py).
    
# #     Returns:
# #         sig_pred_all: (H_test, 6) - Rolling predictions
# #         eps_true_all: (H_test, 6) - Corresponding true strains
# #         sig_true_all: (H_test, 6) - Corresponding true stresses
# #     """
# #     eps_all = data["eps"]
# #     epse_all = data["epse"]
# #     deps_all = data["deps"]
# #     sig_all = data["sig"]
    
# #     # Initialize arrays
# #     sig_pred_all = np.zeros((H_test, 6))
# #     eps_true_all = np.zeros((H_test, 6))
# #     sig_true_all = np.zeros((H_test, 6))
    
# #     # Number of rolling steps needed
# #     n_blocks = int(np.ceil(H_test / H))
    
# #     for k in range(n_blocks):
# #         # Block indices in [0, H_test)
# #         g_start = k * H
# #         g_end = min((k + 1) * H, H_test)
# #         block_len = g_end - g_start
        
# #         if block_len <= 0:
# #             continue
        
# #         # Time index in the full sequence
# #         t_block_start = t_start + g_start
        
# #         # Input: last L steps before t_block_start
# #         in_start = t_block_start - L
# #         in_end = t_block_start
        
# #         # Get inputs
# #         eps_in = torch.from_numpy(eps_all[specimen_idx, in_start:in_end]).float().unsqueeze(0).to(device)
# #         epse_in = torch.from_numpy(epse_all[specimen_idx, in_start:in_end]).float().unsqueeze(0).to(device)
# #         deps_in = torch.from_numpy(deps_all[specimen_idx, in_start:in_end]).float().unsqueeze(0).to(device)
        
# #         # Predict
# #         with torch.no_grad():
# #             sig_pred_block, _ = model(eps_in, epse_in, deps_in)  # (1, H, 6)
        
# #         sig_pred_block_np = sig_pred_block.squeeze(0).cpu().numpy()  # (H, 6)
        
# #         # Store predictions
# #         sig_pred_all[g_start:g_end] = sig_pred_block_np[:block_len]
        
# #         # Store corresponding ground truth
# #         eps_true_all[g_start:g_end] = eps_all[specimen_idx, t_block_start:t_block_start+block_len]
# #         sig_true_all[g_start:g_end] = sig_all[specimen_idx, t_block_start:t_block_start+block_len]
    
# #     return sig_pred_all, eps_true_all, sig_true_all

# # def single_predictions(model, data, specimen_idx, t_start, L, H, device):
# #     """
# #     Single forward pass predictions (your original approach).
    
# #     Returns:
# #         sig_pred: (H, 6) - Single prediction
# #         eps_true: (H, 6) - True strains for horizon
# #         sig_true: (H, 6) - True stresses for horizon
# #     """
# #     eps_all = data["eps"]
# #     epse_all = data["epse"]
# #     deps_all = data["deps"]
# #     sig_all = data["sig"]
    
# #     # Input: last L steps before t_start
# #     in_start = t_start - L
# #     in_end = t_start
    
# #     # Output: H steps starting from t_start
# #     out_start = t_start
# #     out_end = t_start + H
    
# #     # Get inputs
# #     eps_in = torch.from_numpy(eps_all[specimen_idx, in_start:in_end]).float().unsqueeze(0).to(device)
# #     epse_in = torch.from_numpy(epse_all[specimen_idx, in_start:in_end]).float().unsqueeze(0).to(device)
# #     deps_in = torch.from_numpy(deps_all[specimen_idx, in_start:in_end]).float().unsqueeze(0).to(device)
    
# #     # Predict
# #     with torch.no_grad():
# #         sig_pred, _ = model(eps_in, epse_in, deps_in)  # (1, H, 6)
    
# #     sig_pred = sig_pred.squeeze(0).cpu().numpy()  # (H, 6)
    
# #     # Get ground truth
# #     eps_true = eps_all[specimen_idx, out_start:out_end]
# #     sig_true = sig_all[specimen_idx, out_start:out_end]
    
# #     return sig_pred, eps_true, sig_true

# # def main(args):
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #     # ---- Load labeled data ----
# #     data = np.load(args.data)
# #     required = ["eps", "epse", "deps", "sig", "split"]
# #     if not all(k in data.files for k in required):
# #         raise RuntimeError(f"Missing arrays; need {required}")

# #     eps_all = data["eps"]
# #     m, T, D = eps_all.shape
    
# #     print(f"Dataset info: {m} specimens, T={T} time steps, D={D} Voigt components")
    
# #     if args.rolling and args.H_test is None:
# #         # Auto-set H_test for rolling predictions
# #         args.H_test = min(50, T - args.L)  # Up to 50 steps or max available
# #         print(f"Auto-set H_test = {args.H_test} for rolling predictions")
    
# #     # ---- Load model ----
# #     model = SeqModel(H=args.H, hid=args.hid).to(device)
# #     ckpt = torch.load(args.ckpt, map_location=device)
# #     model.load_state_dict(ckpt)
# #     model.eval()

# #     # ---- Prepare output dir ----
# #     os.makedirs(args.out_dir, exist_ok=True)
    
# #     # Voigt order: 11, 22, 33, 12, 23, 13
# #     voigt_labels = [
# #         (0, 'σ_11', 'ε_11'),
# #         (1, 'σ_22', 'ε_22'), 
# #         (2, 'σ_33', 'ε_33'),
# #         (3, 'σ_12', 'ε_12'),
# #         (4, 'σ_23', 'ε_23'),
# #         (5, 'σ_13', 'ε_13')
# #     ]

# #     # ---- Select specimens to evaluate ----
# #     if "split" in data.files:
# #         eval_indices = np.where(data["split"] == "eval")[0]
# #     else:
# #         eval_indices = np.arange(m)
    
# #     if len(eval_indices) == 0:
# #         eval_indices = np.arange(min(args.n_samples, m))
# #     else:
# #         eval_indices = eval_indices[:min(args.n_samples, len(eval_indices))]
    
# #     print(f"Evaluating {len(eval_indices)} specimens")
    
# #     for specimen_idx in eval_indices:
# #         print(f"\n=== Specimen {specimen_idx} ===")
        
# #         # Choose start time
# #         if args.rolling:
# #             # For rolling, start at t = L
# #             t_start = args.L
# #             horizon = args.H_test
# #             print(f"Rolling predictions: t_start={t_start}, H_test={horizon}")
            
# #             # Get rolling predictions
# #             sig_pred, eps_true, sig_true = rolling_predictions(
# #                 model, data, specimen_idx, t_start, 
# #                 args.L, args.H, args.H_test, device
# #             )
# #         else:
# #             # For single prediction, use a random start time
# #             max_start = T - args.H - 1
# #             t_start = np.random.randint(args.L, max_start)
# #             horizon = args.H
# #             print(f"Single prediction: t_start={t_start}, H={horizon}")
            
# #             # Get single predictions
# #             sig_pred, eps_true, sig_true = single_predictions(
# #                 model, data, specimen_idx, t_start, args.L, args.H, device
# #             )
        
# #         # ===================================================================
# #         # PLOT 1: Stress vs Strain for each Voigt component
# #         # ===================================================================
# #         fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
# #         title_mode = "Rolling" if args.rolling else "Single"
# #         fig1.suptitle(f"Specimen {specimen_idx}: {title_mode} Predictions (L={args.L}, H={args.H}, horizon={horizon})", 
# #                      fontsize=14)
        
# #         component_errors = []
        
# #         for idx, (comp_idx, stress_label, strain_label) in enumerate(voigt_labels):
# #             ax = axes[idx // 3, idx % 3]
            
# #             # Get data for this component
# #             x = eps_true[:, comp_idx]
# #             y_true = sig_true[:, comp_idx]
# #             y_pred = sig_pred[:, comp_idx]
            
# #             # Calculate errors
# #             rmse = np.sqrt(np.mean((y_pred - y_true)**2))
# #             mae = np.mean(np.abs(y_pred - y_true))
# #             component_errors.append((stress_label, rmse, mae))
            
# #             # Plot
# #             ax.plot(x, y_true, '-o', lw=1.5, ms=3, label='True', alpha=0.7)
# #             ax.plot(x, y_pred, '-s', lw=1.5, ms=3, label='Pred', alpha=0.7)
            
# #             # Add error text
# #             ax.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}', 
# #                     transform=ax.transAxes, 
# #                     fontsize=8,
# #                     verticalalignment='top',
# #                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
# #             ax.set_xlabel(f'{strain_label}')
# #             ax.set_ylabel(f'{stress_label}')
# #             ax.set_title(f'{stress_label} vs {strain_label}')
# #             ax.grid(True, alpha=0.3)
# #             ax.legend(fontsize=8)
        
# #         plt.tight_layout()
# #         mode_str = "rolling" if args.rolling else "single"
# #         fname1 = os.path.join(args.out_dir, f"specimen_{specimen_idx:03d}_{mode_str}_stress_strain.png")
# #         fig1.savefig(fname1, dpi=300, bbox_inches='tight')
# #         plt.close(fig1)
# #         print(f"  Saved: {fname1}")
        
# #         # ===================================================================
# #         # PLOT 2: Components vs Time
# #         # ===================================================================
# #         fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
# #         fig2.suptitle(f"Specimen {specimen_idx}: Stress vs Time ({title_mode})", fontsize=14)
        
# #         time_steps = np.arange(horizon)
        
# #         for idx, (comp_idx, stress_label, _) in enumerate(voigt_labels):
# #             ax = axes[idx // 3, idx % 3]
            
# #             # Plot stress vs time
# #             ax.plot(time_steps, sig_true[:, comp_idx], 'b-', lw=2, label=f'True {stress_label}', alpha=0.7)
# #             ax.plot(time_steps, sig_pred[:, comp_idx], 'r--', lw=2, label=f'Pred {stress_label}', alpha=0.7)
            
# #             ax.set_xlabel('Time Step')
# #             ax.set_ylabel(f'{stress_label}')
# #             ax.set_title(f'{stress_label} vs Time')
# #             ax.grid(True, alpha=0.3)
# #             ax.legend(fontsize=8)
        
# #         plt.tight_layout()
# #         fname2 = os.path.join(args.out_dir, f"specimen_{specimen_idx:03d}_{mode_str}_stress_vs_time.png")
# #         fig2.savefig(fname2, dpi=300, bbox_inches='tight')
# #         plt.close(fig2)
# #         print(f"  Saved: {fname2}")
        
# #         # ===================================================================
# #         # PLOT 3: Error Analysis
# #         # ===================================================================
# #         fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# #         fig3.suptitle(f"Specimen {specimen_idx}: Error Analysis ({title_mode})", fontsize=14)
        
# #         # Left: Component-wise RMSE
# #         stress_labels = [label for label, _, _ in component_errors]
# #         rmse_values = [rmse for _, rmse, _ in component_errors]
        
# #         bars = ax1.bar(range(len(stress_labels)), rmse_values, color='skyblue', edgecolor='black')
# #         ax1.set_xlabel('Component')
# #         ax1.set_ylabel('RMSE')
# #         ax1.set_title('Component-wise RMSE')
# #         ax1.set_xticks(range(len(stress_labels)))
# #         ax1.set_xticklabels(stress_labels, rotation=45)
        
# #         for bar, val in zip(bars, rmse_values):
# #             ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
# #                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
# #         # Right: Overall scatter plot
# #         ax2.scatter(sig_true.flatten(), sig_pred.flatten(), alpha=0.3, s=10)
        
# #         min_val = min(sig_true.min(), sig_pred.min())
# #         max_val = max(sig_true.max(), sig_pred.max())
# #         ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Perfect')
        
# #         # Calculate overall R²
# #         all_true = sig_true.flatten()
# #         all_pred = sig_pred.flatten()
# #         ss_res = np.sum((all_pred - all_true)**2)
# #         ss_tot = np.sum((all_true - np.mean(all_true))**2)
# #         r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
# #         ax2.text(0.05, 0.95, f'R² = {r2:.4f}\nMean RMSE = {np.mean(rmse_values):.4f}', 
# #                 transform=ax2.transAxes, 
# #                 fontsize=10,
# #                 verticalalignment='top',
# #                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
# #         ax2.set_xlabel('True Stress (All Components)')
# #         ax2.set_ylabel('Predicted Stress (All Components)')
# #         ax2.set_title(f'All Components: Predicted vs True')
# #         ax2.legend()
# #         ax2.grid(True, alpha=0.3)
# #         ax2.axis('equal')
        
# #         plt.tight_layout()
# #         fname3 = os.path.join(args.out_dir, f"specimen_{specimen_idx:03d}_{mode_str}_error_analysis.png")
# #         fig3.savefig(fname3, dpi=300, bbox_inches='tight')
# #         plt.close(fig3)
# #         print(f"  Saved: {fname3}")
        
# #         # ===================================================================
# #         # Save summary
# #         # ===================================================================
# #         summary_file = os.path.join(args.out_dir, f"specimen_{specimen_idx:03d}_{mode_str}_summary.txt")
# #         with open(summary_file, 'w') as f:
# #             f.write(f"Evaluation Summary for specimen {specimen_idx}\n")
# #             f.write(f"Mode: {title_mode} predictions\n")
# #             f.write(f"Parameters: L={args.L}, H={args.H}, horizon={horizon}\n")
# #             f.write("="*60 + "\n\n")
            
# #             f.write("COMPONENT ERRORS:\n")
# #             f.write(f"{'Component':<8} {'RMSE':<12} {'MAE':<12}\n")
# #             f.write("-"*32 + "\n")
# #             for label, rmse, mae in component_errors:
# #                 f.write(f"{label:<8} {rmse:<12.6f} {mae:<12.6f}\n")
            
# #             f.write("\nOVERALL STATISTICS:\n")
# #             f.write(f"Mean RMSE: {np.mean(rmse_values):.6f}\n")
# #             f.write(f"R² score: {r2:.6f}\n")
# #             f.write(f"Total points: {len(all_true)}\n")
        
# #         print(f"  Saved summary: {summary_file}")

# #     print(f"\nAll evaluations complete! Files saved in: {args.out_dir}")

# # if __name__ == "__main__":
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--data", type=str, required=True, help="Path to npz with keys {eps, epse, deps, sig, split}")
# #     ap.add_argument("--ckpt", type=str, required=True, help="Path to trained model .pt")
# #     ap.add_argument("--L", type=int, default=10, help="Lookback window length")
# #     ap.add_argument("--H", type=int, default=10, help="Prediction horizon (model capability)")
# #     ap.add_argument("--H_test", type=int, default=None, help="Total evaluation horizon for rolling predictions")
# #     ap.add_argument("--rolling", action="store_true", help="Use rolling predictions (requires H_test)")
# #     ap.add_argument("--hid", type=int, default=128, help="Hidden size (must match training)")
# #     ap.add_argument("--n_samples", type=int, default=5, help="Number of specimens to evaluate")
# #     ap.add_argument("--out_dir", type=str, default="./figs", help="Output directory for figures")
# #     args = ap.parse_args()
    
# #     if args.rolling and args.H_test is None:
# #         ap.error("--rolling requires --H_test to be specified")
    
# #     main(args)