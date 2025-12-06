import torch
import torch.nn.functional as F
from tensor_utils import TensorUtils

import torch

def overlap_consistency_loss(global_key: torch.Tensor,
                             pred: torch.Tensor,
                             eps: float = 1e-12) -> torch.Tensor:
    """
    Penalize disagreement among predictions that correspond to the same
    (trajectory, time) pair.

    Parameters
    ----------
    global_key : (B, H) LongTensor
        Encodes the global identity of each predicted point.
        Typical encoding: traj_id * SCALE + time_index, or just time_index
        in the single-trajectory case.
    pred : (B, H, D) Tensor
        Predicted values (e.g. stress in Voigt-6).
    eps : float
        Small number to avoid division by zero when normalizing by counts.

    Returns
    -------
    loss : scalar Tensor
        MSE between each overlapping prediction and its group mean.
        Zero if there are no overlapping keys in the batch.
    """
    B, H, D = pred.shape

    # Flatten batch + horizon
    keys_flat = global_key.reshape(-1)        # (B*H,)
    pred_flat = pred.reshape(-1, D)           # (B*H, D)

    # Group by unique keys
    #   uniq   : (K,) unique keys
    #   inv    : (B*H,) index of each element's group in [0, K)
    #   counts : (K,) how many times each key appears
    uniq, inv, counts = keys_flat.unique(return_inverse=True,
                                         return_counts=True)
    mask_multi = counts > 1                  # (K,) True if key appears ≥2 times

    # If no overlapping points, return zero loss
    if not mask_multi.any():
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    K = uniq.size(0)

    # Sum of predictions for each group
    group_sum = torch.zeros(K, D, device=pred.device, dtype=pred.dtype)
    group_sum.index_add_(0, inv, pred_flat)   # accumulate along group index

    # Mean prediction for each group
    counts_f = counts.to(pred.dtype).unsqueeze(1)   # (K,1)
    group_mean = group_sum / (counts_f + eps)       # (K,D)

    # Expand means back to (B*H,D) using inv
    mean_flat = group_mean[inv]

    # Only keep entries belonging to keys that actually overlap
    valid = mask_multi[inv]                         # (B*H,)

    # MSE between individual predictions and their group mean
    return mse_loss(pred_flat[valid], mean_flat[valid])

def mse_loss(sig_pred, sig_true):
    """
    Inputs: sig_pred, sig_true: (B,H,6)
    """
    return F.mse_loss(sig_pred, sig_true)




# # --------------- Loss functions from Heider Y, Wang K, Sun W (2020) paper  -----------------------
# # future work

# """
# Loss functions from the paper for stress prediction in Voigt-6 format.

# Includes:
#     - loss_voigt_mse               : DM L0 
#     - loss_eigenvalues_lie         : DM L_SO3 (Lie algebra)
# """

# import torch
# import torch.nn.functional as F
# from tensor_utils import TensorUtils


# def voigt6_to_tensor33(v6):
#     """
#     v6: (N,6)
#     returns: (N,3,3)
#     """
#     return TensorUtils.from_voigt6(v6)


# def eig_sorted(A):
#     """
#     A: (N,3,3) symmetric
#     Returns:
#         lam (N,3) sorted descending
#         R   (N,3,3) corresponding eigenvectors
#     """
#     lam, R = torch.linalg.eigh(A)
#     idx = torch.argsort(lam, dim=1, descending=True)
#     lam_sorted = torch.gather(lam, 1, idx)
    
#     # Sort eigenvectors correctly
#     R_sorted = torch.zeros_like(R)
#     for i in range(R.xshape[0]):
#         R_sorted[i] = R[i, :, idx[i]]
#     return lam_sorted, R_sorted


# def safe_so3_log(R):
#     """
#     Safe version of so3_log with error checking.
#     R: (N,3,3) rotation matrices
#     Returns: w: (N,3) rotation vectors in Lie algebra
#     """
#     N = R.shape[0]
#     device = R.device
    
#     # Check for NaN/Inf
#     if torch.any(torch.isnan(R)) or torch.any(torch.isinf(R)):
#         print("WARNING: so3_log: R contains NaN/Inf, returning zeros")
#         return torch.zeros((N, 3), device=device)
    
#     # Force symmetry  
#     # Some stress tensors might not yield perfectly orthogonal eigenvectorsl 
#     with torch.no_grad():
#         U, S, V = torch.svd(R)
#         R_ortho = torch.matmul(U, V.transpose(1, 2))
    
#     # Use the orthogonalized version
#     R = R_ortho
    
#     # Make sure trace is valid
#     tr = (R[:,0,0] + R[:,1,1] + R[:,2,2] - 1) / 2
#     tr = torch.clamp(tr, -1.0 + 1e-7, 1.0 - 1e-7)  
    
#     theta = torch.acos(tr)
#     eps = 1e-8
    
#     # Skew-symmetric part
#     S_mat = R - R.transpose(1,2)
#     w_raw = 0.5 * torch.stack([S_mat[:,2,1], S_mat[:,0,2], S_mat[:,1,0]], dim=1)
    
#     mask_small = theta < eps
#     mask_large = ~mask_small
    
#     w = torch.zeros_like(w_raw)
    
#     # Small angles: w ≈ skew(R)/2
#     w[mask_small] = w_raw[mask_small]
    
#     # Large angles: w = (theta/(2*sin(theta))) * skew(R)
#     if mask_large.any():
#         theta_large = theta[mask_large].unsqueeze(1)
#         # avoid division by zero
#         sin_theta = torch.sin(theta_large + eps)
#         scale = theta_large / (2 * sin_theta + eps)
#         w[mask_large] = w_raw[mask_large] * scale
    
#     if torch.any(torch.isnan(w)) or torch.any(torch.isinf(w)):
#         print("WARNING: so3_log: result contains NaN/Inf, clipping")
#         w = torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=-1.0)
    
#     return w


# def lie_loss(sig_pred, sig_true, epsp_pred=None, epsp_true=None, 
#              lambda_pstrain=1e4, lambda_eigen=1.0, lambda_rot=1.0,
#              debug=False): 
#     """
    
#     Paper uses: L = L_lam + L_rot (with lambda_eigen = lambda_rot = 1.0)
    
#     also include plastic strain loss with lambda_pstrain.
    
#     Args:
#         sig_pred, sig_true: (B,H,6) stress tensors
#         epsp_pred, epsp_true: (B,H,6) plastic strain tensors (optional)
#         lambda_pstrain: weight for plastic strain MSE loss
#         lambda_eigen: weight for eigenvalue MSE
#         lambda_rot: weight for rotation (Lie algebra) loss
#     """
#     # Handle input shapes
#     if sig_pred.dim() == 3:
#         B, H, _ = sig_pred.shape
#         sig_pred_flat = sig_pred.reshape(-1, 6)
#         sig_true_flat = sig_true.reshape(-1, 6)
#     else:
#         sig_pred_flat = sig_pred
#         sig_true_flat = sig_true
    
#     if debug:
#         print(f"lie_loss: Input shapes: {sig_pred.shape} -> {sig_pred_flat.shape}")
#         print(f"Stress range: pred [{sig_pred_flat.min():.3f}, {sig_pred_flat.max():.3f}], "
#               f"true [{sig_true_flat.min():.3f}, {sig_true_flat.max():.3f}]")
    
#     # Convert to 3x3 tensors
#     A_pred = voigt6_to_tensor33(sig_pred_flat)
#     A_true = voigt6_to_tensor33(sig_true_flat)
    
#     if debug:
#         print(f"Tensor shapes: A_pred {A_pred.shape}, A_true {A_true.shape}")
#         # Check symmetry
#         sym_err_pred = torch.max(torch.abs(A_pred - A_pred.transpose(1, 2)))
#         sym_err_true = torch.max(torch.abs(A_true - A_true.transpose(1, 2)))
#         print(f"Symmetry error: pred={sym_err_pred:.2e}, true={sym_err_true:.2e}")
    
#     # Eigen-decomposition
#     lam_p, R_p = eig_sorted(A_pred)
#     lam_t, R_t = eig_sorted(A_true)
    
#     if debug:
#         print(f"Eigenvalues: pred [{lam_p.min():.3f}, {lam_p.max():.3f}], "
#               f"true [{lam_t.min():.3f}, {lam_t.max():.3f}]")
#         # Check for NaN in eigenvectors
#         if torch.any(torch.isnan(R_p)) or torch.any(torch.isnan(R_t)):
#             print("ERROR: NaN in eigenvectors!")

#     # Convert to 3x3 tensors
#     A_pred = voigt6_to_tensor33(sig_pred_flat)
#     A_true = voigt6_to_tensor33(sig_true_flat)
    
#     # Eigen-decomposition
#     lam_p, R_p = eig_sorted(A_pred)  # eigenvalues sorted descending
#     lam_t, R_t = eig_sorted(A_true)
    
#     # Eigenvalue loss (MSE)
#     L_eigen = F.mse_loss(lam_p, lam_t) * lambda_eigen
    
#     # Rotation loss (Lie algebra
#     # Residual rotation: R_delta = R_true * R_pred^T
#     R_delta = torch.matmul(R_t, R_p.transpose(1,2))
    
#     # Rotation vector in Lie algebra 
#     w = safe_so3_log(R_delta)  # shape (N,3)
    
#     # Rotation loss: ||w||^2 (squared geodesic distance on SO(3))
#     L_rot = torch.mean(torch.sum(w * w, dim=1)) * lambda_rot
    
#     # Total stress loss from paper
#     stress_loss = L_eigen + L_rot
    
#     # Plastic strain loss (optional, MSE)
#     plastic_loss = 0.0
#     if epsp_pred is not None and epsp_true is not None:
#         plastic_loss = mse_loss(epsp_pred, epsp_true) * lambda_pstrain
    
#     return stress_loss + plastic_loss


# # just the Lie loss (no plastic strain at all)
# def pure_lie_loss(sig_pred, sig_true, lambda_eigen=1.0, lambda_rot=1.0):
#     """
#     Pure Lie loss without plastic strain term.
#     Exactly as in paper: L = L_lam + L_rot
#     """
#     if sig_pred.dim() == 3:
#         sig_pred_flat = sig_pred.reshape(-1, 6)
#         sig_true_flat = sig_true.reshape(-1, 6)
#     else:
#         sig_pred_flat = sig_pred
#         sig_true_flat = sig_true
    
#     A_pred = voigt6_to_tensor33(sig_pred_flat)
#     A_true = voigt6_to_tensor33(sig_true_flat)
    
#     lam_p, R_p = eig_sorted(A_pred)
#     lam_t, R_t = eig_sorted(A_true)
    
#     # Eigenvalue loss
#     L_eigen = F.mse_loss(lam_p, lam_t) * lambda_eigen
    
#     # Rotation loss in Lie algebra
#     R_delta = torch.matmul(R_t, R_p.transpose(1,2))
#     w = safe_so3_log(R_delta)
#     L_rot = torch.mean(torch.sum(w * w, dim=1)) * lambda_rot
    
#     return L_eigen + L_rot




