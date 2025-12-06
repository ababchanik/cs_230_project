# # #!/usr/bin/env python3
# # # debug_plots.py
# # import torch
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import argparse
# # import os
# # from model import SeqModel

# # def plot_debug_minimal(data_path, ckpt_path, loss_type, L=50, H=30, hid=128):

# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #     # Load model
# #     model = SeqModel(H=H, hid=hid).to(device)
# #     model.load_state_dict(torch.load(ckpt_path, map_location=device))
# #     model.eval()

# #     # Load data
# #     data = np.load(data_path)

# #     # First eval specimen
# #     eval_indices = np.where(data['split'] == 'eval')[0]
# #     i_spec = eval_indices[0]
# #     t = 50

# #     # Inputs
# #     eps_in  = torch.from_numpy(data['eps'][i_spec, t-L:t]).float().unsqueeze(0).to(device)
# #     epse_in = torch.from_numpy(data['epse'][i_spec, t-L:t]).float().unsqueeze(0).to(device)
# #     deps_in = torch.from_numpy(data['deps'][i_spec, t-L:t]).float().unsqueeze(0).to(device)

# #     # Predictions
# #     with torch.no_grad():
# #         sig_pred, epse_pred = model(eps_in, epse_in, deps_in)

# #     sig_pred = sig_pred.squeeze(0).cpu()
# #     sig_true = torch.from_numpy(data['sig'][i_spec, t:t+H]).float()

# #     # Output directory
# #     out_dir = f"debug_{loss_type}"
# #     os.makedirs(out_dir, exist_ok=True)

# #     # ------------------------------------------------------------
# #     # Plot 1: Time series of all 6 stress components
# #     # ------------------------------------------------------------
# #     fig1, axes1 = plt.subplots(3, 2, figsize=(12, 10))
# #     fig1.suptitle(f'Stress Components Over Time ({loss_type} loss)', fontsize=14)

# #     component_names = ['σ₁₁','σ₂₂','σ₃₃','σ₂₃','σ₁₃','σ₁₂']

# #     for ax, name, idx in zip(axes1.ravel(), component_names, range(6)):
# #         ax.plot(sig_true[:, idx].numpy(), 'o-', label='True', linewidth=1.5)
# #         ax.plot(sig_pred[:, idx].numpy(), 's-', label='Pred', linewidth=1.5)
# #         ax.set_xlabel('Time Step')
# #         ax.set_ylabel(name)
# #         ax.grid(True, alpha=0.3)
# #         ax.legend(fontsize=9)

# #     plt.tight_layout()
# #     plt.savefig(f"{out_dir}/stress_time_series.png", dpi=150, bbox_inches='tight')
# #     plt.close(fig1)

# #     # ------------------------------------------------------------
# #     # Plot 2: Error analysis
# #     # ------------------------------------------------------------
# #     fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8))

# #     # Error over time
# #     abs_error = torch.abs(sig_true - sig_pred).mean(dim=1)
# #     axes2[0].plot(abs_error.numpy(), 'r-', linewidth=2)
# #     axes2[0].fill_between(np.arange(H), 0, abs_error.numpy(), alpha=0.3, color='red')
# #     axes2[0].set_xlabel("Time Step")
# #     axes2[0].set_ylabel("Mean Abs Error")
# #     axes2[0].set_title("Prediction Error Over Time")
# #     axes2[0].grid(True, alpha=0.3)

# #     # Component-wise mean error
# #     component_errors = torch.abs(sig_true - sig_pred).mean(dim=0)
# #     bars = axes2[1].bar(range(6), component_errors.numpy())
# #     axes2[1].set_xticks(range(6))
# #     axes2[1].set_xticklabels(component_names)
# #     axes2[1].set_ylabel("Mean Abs Error")
# #     axes2[1].set_title("Component-wise Error")
# #     axes2[1].grid(True, alpha=0.3, axis='y')

# #     for bar, val in zip(bars, component_errors.numpy()):
# #         axes2[1].text(
# #             bar.get_x() + bar.get_width()/2.,
# #             bar.get_height(),
# #             f"{val:.4f}",
# #             ha="center",
# #             va="bottom",
# #             fontsize=8
# #         )

    
# #     plt.tight_layout()
# #     plt.savefig(f"{out_dir}/error_analysis.png", dpi=150, bbox_inches='tight')
# #     plt.close(fig2)

    
# #     print(f"\nSaved:")
# #     print(f"  {out_dir}/stress_time_series.png")
# #     print(f"  {out_dir}/error_analysis.png\n")

# # def main():
# #     parser = argparse.ArgumentParser(description="Minimal debug plots")
# #     parser.add_argument("--data", type=str, required=True)
# #     parser.add_argument("--ckpt", type=str, required=True)
# #     parser.add_argument("--loss", type=str, required=True)
# #     parser.add_argument("--L", type=int, default=50)
# #     parser.add_argument("--H", type=int, default=30)
# #     parser.add_argument("--hid", type=int, default=128)
# #     args = parser.parse_args()

# #     plot_debug_minimal(
# #         data_path=args.data,
# #         ckpt_path=args.ckpt,
# #         loss_type=args.loss,
# #         L=args.L,
# #         H=args.H,
# #         hid=args.hid
# #     )

# # if __name__ == "__main__":
# #     main()






# ###### OLD BUT WORKS???
# #!/usr/bin/env python3
# # debug_plots.py
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import argparse
# import os
# from model import SeqModel

# def plot_debug_minimal(data_path, ckpt_path, loss_type, L=50, H=30, hid=128):

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load model
#     model = SeqModel(H=H, hid=hid).to(device)
#     model.load_state_dict(torch.load(ckpt_path, map_location=device))
#     model.eval()

#     # Load data
#     data = np.load(data_path)

#     # First eval specimen
#     eval_indices = np.where(data['split'] == 'eval')[0]
#     i_spec = eval_indices[0]
#     t = 50

#     # Inputs
#     eps_in  = torch.from_numpy(data['eps'][i_spec, t-L:t]).float().unsqueeze(0).to(device)
#     epse_in = torch.from_numpy(data['epse'][i_spec, t-L:t]).float().unsqueeze(0).to(device)
#     deps_in = torch.from_numpy(data['deps'][i_spec, t-L:t]).float().unsqueeze(0).to(device)

#     # Predictions
#     with torch.no_grad():
#         sig_pred, epse_pred = model(eps_in, epse_in, deps_in)

#     sig_pred = sig_pred.squeeze(0).cpu()
#     sig_true = torch.from_numpy(data['sig'][i_spec, t:t+H]).float()
    
#     # Also get strain data for stress-strain plots
#     eps_true = torch.from_numpy(data['eps'][i_spec, t:t+H]).float()
#     epse_true = torch.from_numpy(data['epse'][i_spec, t:t+H]).float()

#     # Output directory
#     out_dir = f"debug_{loss_type}"
#     os.makedirs(out_dir, exist_ok=True)

#     # ------------------------------------------------------------
#     # Plot 1: Time series of all 6 stress components
#     # ------------------------------------------------------------
#     fig1, axes1 = plt.subplots(3, 2, figsize=(12, 10))
#     fig1.suptitle(f'Stress Components Over Time ({loss_type} loss)', fontsize=14)

#     component_names = ['σ₁₁','σ₂₂','σ₃₃','σ₂₃','σ₁₃','σ₁₂']

#     for ax, name, idx in zip(axes1.ravel(), component_names, range(6)):
#         ax.plot(sig_true[:, idx].numpy(), 'o-', label='True', linewidth=1.5, markersize=4)
#         ax.plot(sig_pred[:, idx].numpy(), 's-', label='Pred', linewidth=1.5, markersize=4)
#         ax.set_xlabel('Time Step')
#         ax.set_ylabel(name)
#         ax.grid(True, alpha=0.3)
#         ax.legend(fontsize=9)

#     plt.tight_layout()
#     plt.savefig(f"{out_dir}/stress_time_series.png", dpi=150, bbox_inches='tight')
#     plt.close(fig1)

#     # ------------------------------------------------------------
#     # NEW PLOT: Stress vs Strain for each component
#     # ------------------------------------------------------------
#     fig3, axes3 = plt.subplots(3, 2, figsize=(12, 10))
#     fig3.suptitle(f'Stress vs Strain ({loss_type} loss)', fontsize=14)
    
#     strain_names = ['ε₁₁','ε₂₂','ε₃₃','ε₂₃','ε₁₃','ε₁₂']
    
#     for ax, strain_name, stress_name, idx in zip(axes3.ravel(), strain_names, component_names, range(6)):
#         # Get strain and stress data
#         strain = eps_true[:, idx].numpy()
#         true_stress = sig_true[:, idx].numpy()
#         pred_stress = sig_pred[:, idx].numpy()
        
#         # Sort by strain for cleaner plots
#         sort_idx = np.argsort(strain)
#         strain_sorted = strain[sort_idx]
#         true_sorted = true_stress[sort_idx]
#         pred_sorted = pred_stress[sort_idx]
        
#         ax.plot(strain_sorted, true_sorted, 'o-', label='True', linewidth=1.5, markersize=4)
#         ax.plot(strain_sorted, pred_sorted, 's-', label='Pred', linewidth=1.5, markersize=4)
#         ax.set_xlabel(strain_name)
#         ax.set_ylabel(stress_name)
#         ax.grid(True, alpha=0.3)
#         ax.legend(fontsize=9)
        
#         # Add correlation coefficient
#         corr = np.corrcoef(true_stress, pred_stress)[0, 1]
#         ax.text(0.05, 0.95, f'R² = {corr:.3f}', transform=ax.transAxes,
#                 fontsize=9, verticalalignment='top',
#                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
#     plt.tight_layout()
#     plt.savefig(f"{out_dir}/stress_vs_strain.png", dpi=150, bbox_inches='tight')
#     plt.close(fig3)

#     # ------------------------------------------------------------
#     # NEW PLOT: Stress-Strain Hysteresis (if any)
#     # ------------------------------------------------------------
#     fig4, axes4 = plt.subplots(2, 3, figsize=(15, 8))
#     fig4.suptitle(f'Stress-Strain Trajectories ({loss_type} loss)', fontsize=14)
    
#     for idx, ax in enumerate(axes4.ravel()):
#         if idx >= 6:
#             break
            
#         strain = eps_true[:, idx].numpy()
#         true_stress = sig_true[:, idx].numpy()
#         pred_stress = sig_pred[:, idx].numpy()
        
#         # Color by time (blue to red)
#         colors = np.arange(len(strain))
        
#         # True trajectory
#         sc1 = ax.scatter(strain, true_stress, c=colors, cmap='viridis', 
#                         s=20, alpha=0.7, label='True')
#         # Predicted trajectory
#         sc2 = ax.scatter(strain, pred_stress, c=colors, cmap='plasma', 
#                         s=20, alpha=0.7, label='Pred', marker='s')
        
#         ax.set_xlabel(strain_names[idx])
#         ax.set_ylabel(component_names[idx])
#         ax.grid(True, alpha=0.3)
#         ax.legend(fontsize=8)
        
#         # Add arrows to show direction
#         if len(strain) > 1:
#             # True trajectory arrow
#             dx = strain[-1] - strain[0]
#             dy = true_stress[-1] - true_stress[0]
#             ax.arrow(strain[0], true_stress[0], dx*0.8, dy*0.8, 
#                     head_width=0.1*abs(dy), head_length=0.1*abs(dx),
#                     fc='blue', ec='blue', alpha=0.5)
            
#             # Predicted trajectory arrow
#             dy_pred = pred_stress[-1] - pred_stress[0]
#             ax.arrow(strain[0], pred_stress[0], dx*0.8, dy_pred*0.8,
#                     head_width=0.1*abs(dy_pred), head_length=0.1*abs(dx),
#                     fc='red', ec='red', alpha=0.5)
    
#     # Add colorbars
#     plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
#     # Add colorbar for true
#     cbar_ax1 = fig4.add_axes([0.91, 0.55, 0.02, 0.3])
#     cbar1 = plt.colorbar(sc1, cax=cbar_ax1)
#     cbar1.set_label('Time Step (True)')
    
#     # Add colorbar for pred
#     cbar_ax2 = fig4.add_axes([0.91, 0.15, 0.02, 0.3])
#     cbar2 = plt.colorbar(sc2, cax=cbar_ax2)
#     cbar2.set_label('Time Step (Pred)')
    
#     plt.savefig(f"{out_dir}/stress_strain_trajectories.png", dpi=150, bbox_inches='tight')
#     plt.close(fig4)

#     # ------------------------------------------------------------
#     # NEW PLOT: Elastic Stress vs Elastic Strain
#     # ------------------------------------------------------------
#     fig5, axes5 = plt.subplots(3, 2, figsize=(12, 10))
#     fig5.suptitle(f'Elastic Stress vs Elastic Strain ({loss_type} loss)', fontsize=14)
    
#     for ax, strain_name, stress_name, idx in zip(axes5.ravel(), strain_names, component_names, range(6)):
#         # Get ELASTIC strain and stress
#         elastic_strain = epse_true[:, idx].numpy()
#         stress = sig_true[:, idx].numpy()  # Total stress (should relate to elastic strain)
#         pred_stress = sig_pred[:, idx].numpy()
        
#         # Sort by elastic strain
#         sort_idx = np.argsort(elastic_strain)
#         elastic_strain_sorted = elastic_strain[sort_idx]
#         stress_sorted = stress[sort_idx]
#         pred_sorted = pred_stress[sort_idx]
        
#         ax.plot(elastic_strain_sorted, stress_sorted, 'o-', label='True', linewidth=1.5, markersize=4)
#         ax.plot(elastic_strain_sorted, pred_sorted, 's-', label='Pred', linewidth=1.5, markersize=4)
#         ax.set_xlabel(f'εᵉ {strain_name[-2:]}')
#         ax.set_ylabel(stress_name)
#         ax.grid(True, alpha=0.3)
#         ax.legend(fontsize=9)
        
#         # Check linearity (elastic relation should be linear)
#         if len(elastic_strain) > 1:
#             # Fit line to true data
#             coeffs = np.polyfit(elastic_strain, stress, 1)
#             x_fit = np.array([elastic_strain.min(), elastic_strain.max()])
#             y_fit = np.polyval(coeffs, x_fit)
#             ax.plot(x_fit, y_fit, 'g--', alpha=0.5, label=f'Fit: slope={coeffs[0]:.2e}')
#             ax.legend(fontsize=8)
    
#     plt.tight_layout()
#     plt.savefig(f"{out_dir}/elastic_stress_vs_strain.png", dpi=150, bbox_inches='tight')
#     plt.close(fig5)

#     # ------------------------------------------------------------
#     # Plot 2: Error analysis
#     # ------------------------------------------------------------
#     fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8))

#     # Error over time
#     abs_error = torch.abs(sig_true - sig_pred).mean(dim=1)
#     axes2[0].plot(abs_error.numpy(), 'r-', linewidth=2)
#     axes2[0].fill_between(np.arange(H), 0, abs_error.numpy(), alpha=0.3, color='red')
#     axes2[0].set_xlabel("Time Step")
#     axes2[0].set_ylabel("Mean Abs Error")
#     axes2[0].set_title("Prediction Error Over Time")
#     axes2[0].grid(True, alpha=0.3)

#     # Component-wise mean error
#     component_errors = torch.abs(sig_true - sig_pred).mean(dim=0)
#     bars = axes2[1].bar(range(6), component_errors.numpy())
#     axes2[1].set_xticks(range(6))
#     axes2[1].set_xticklabels(component_names)
#     axes2[1].set_ylabel("Mean Abs Error")
#     axes2[1].set_title("Component-wise Error")
#     axes2[1].grid(True, alpha=0.3, axis='y')

#     for bar, val in zip(bars, component_errors.numpy()):
#         axes2[1].text(
#             bar.get_x() + bar.get_width()/2.,
#             bar.get_height(),
#             f"{val:.4f}",
#             ha="center",
#             va="bottom",
#             fontsize=8
#         )
    
#     plt.tight_layout()
#     plt.savefig(f"{out_dir}/error_analysis.png", dpi=150, bbox_inches='tight')
#     plt.close(fig2)

#     # ------------------------------------------------------------
#     # Summary statistics
#     # ------------------------------------------------------------
#     print(f"\n=== Summary for {loss_type} loss ===")
#     print(f"Mean Absolute Error: {torch.mean(torch.abs(sig_true - sig_pred)).item():.6e}")
#     print(f"Max Error: {torch.max(torch.abs(sig_true - sig_pred)).item():.6e}")
    
#     # Check if stress is changing
#     stress_range = torch.max(sig_true) - torch.min(sig_true)
#     print(f"Stress range in target: {stress_range.item():.6e}")
    
#     if stress_range < 1e-6:
#         print("⚠️  WARNING: Stress appears constant!")
#     else:
#         print("✅ Good: Stress is changing")
    
#     # Check correlation
#     for idx in range(6):
#         true_vals = sig_true[:, idx].numpy()
#         pred_vals = sig_pred[:, idx].numpy()
#         corr = np.corrcoef(true_vals, pred_vals)[0, 1]
#         print(f"  Component {idx+1}: R² = {corr:.4f}")
    
#     print(f"\nSaved plots to {out_dir}/:")
#     print(f"  stress_time_series.png")
#     print(f"  stress_vs_strain.png")
#     print(f"  stress_strain_trajectories.png")
#     print(f"  elastic_stress_vs_strain.png")
#     print(f"  error_analysis.png")

# def main():
#     parser = argparse.ArgumentParser(description="Minimal debug plots")
#     parser.add_argument("--data", type=str, required=True)
#     parser.add_argument("--ckpt", type=str, required=True)
#     parser.add_argument("--loss", type=str, required=True)
#     parser.add_argument("--L", type=int, default=50)
#     parser.add_argument("--H", type=int, default=30)
#     parser.add_argument("--hid", type=int, default=128)
#     args = parser.parse_args()

#     plot_debug_minimal(
#         data_path=args.data,
#         ckpt_path=args.ckpt,
#         loss_type=args.loss,
#         L=args.L,
#         H=args.H,
#         hid=args.hid
#     )

# if __name__ == "__main__":
#     main()





# # newest?
# # debug_plots.py
# #!/usr/bin/env python3
# import argparse
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from model import SeqModel
# from loss_functions import compute_loss, SpectralDecomposition, LieAlgebraUtils
# from tensor_utils import TensorUtils

# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Load model
#     model = SeqModel(H=args.H, hid=args.hid).to(device)
#     model.load_state_dict(torch.load(args.ckpt, map_location=device))
#     model.eval()
    
#     # Load data
#     data = np.load(args.data)
    
#     # Get a single example
#     eps = torch.from_numpy(data["eps"][0:1, :args.L]).float().to(device)
#     epse = torch.from_numpy(data["epse"][0:1, :args.L]).float().to(device)
#     deps = torch.from_numpy(data["deps"][0:1, :args.L]).float().to(device)
#     sig_true = torch.from_numpy(data["sig"][0:1, args.L:args.L+args.H]).float().to(device)
    
#     # Forward pass
#     with torch.no_grad():
#         sig_pred, epse_pred = model(eps, epse, deps)
    
#     # Test the loss function
#     loss = compute_loss(
#         sig_pred=sig_pred,
#         sig_true=sig_true,
#         loss_type=args.loss
#     )
    
#     print(f"Loss value: {loss.item():.6f}")
    
#     # Debug spectral decomposition
#     if args.loss in ["lie", "fro"]:
#         B, H, _ = sig_pred.shape
#         sig_pred_3x3 = TensorUtils.from_voigt6(sig_pred.reshape(B * H, 6)).reshape(B, H, 3, 3)
#         sig_true_3x3 = TensorUtils.from_voigt6(sig_true.reshape(B * H, 6)).reshape(B, H, 3, 3)
        
#         # Get eigenvalues and eigenvectors
#         eigvals_pred, eigvecs_pred = SpectralDecomposition.eigen_decomposition(sig_pred_3x3)
#         eigvals_true, eigvecs_true = SpectralDecomposition.eigen_decomposition(sig_true_3x3)
        
#         print(f"\nEigenvalues (pred): {eigvals_pred[0, 0]}")
#         print(f"Eigenvalues (true): {eigvals_true[0, 0]}")
        
#         # Check eigenvector norms
#         norms_pred = torch.linalg.norm(eigvecs_pred, dim=-2)
#         norms_true = torch.linalg.norm(eigvecs_true, dim=-2)
#         print(f"\nEigenvector norms (pred): {norms_pred[0, 0]}")
#         print(f"Eigenvector norms (true): {norms_true[0, 0]}")
        
#         # Form rotation matrices
#         R_pred = LieAlgebraUtils.rotation_matrix_from_eigenvectors(eigvecs_pred)
#         R_true = LieAlgebraUtils.rotation_matrix_from_eigenvectors(eigvecs_true)
        
#         # Check orthogonality
#         RTR_pred = torch.matmul(R_pred.transpose(-1, -2), R_pred)
#         RTR_true = torch.matmul(R_true.transpose(-1, -2), R_true)
        
#         print(f"\nRᵀR (pred, first element):\n{RTR_pred[0, 0]}")
#         print(f"\nRᵀR (true, first element):\n{RTR_true[0, 0]}")
        
#         if args.loss == "lie":
#             # Test Lie algebra
#             w_pred = LieAlgebraUtils.log_map_simple(R_pred)
#             w_true = LieAlgebraUtils.log_map_simple(R_true)
#             print(f"\nLie algebra components (pred): {w_pred[0, 0]}")
#             print(f"Lie algebra components (true): {w_true[0, 0]}")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data", type=str, required=True)
#     ap.add_argument("--ckpt", type=str, required=True)
#     ap.add_argument("--loss", type=str, required=True)
#     ap.add_argument("--L", type=int, default=2)
#     ap.add_argument("--H", type=int, default=2)
#     ap.add_argument("--hid", type=int, default=128)
#     args = ap.parse_args()
#     main(args)