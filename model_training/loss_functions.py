# loss functions

#!/usr/bin/env python3
"""
loss_functions.py - Lie loss per the paper
Paper: "SO(3)-invariance of informed-graph-based deep neural network for anisotropic elastoplastic materials"
Equation 15: 𝒟ₒᴹᴸ = 1/2(𝒟σᴹᴸ + 𝒟ᴿᴹᴸ)
"""
import torch
import torch.nn.functional as F
from tensor_utils import TensorUtils



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

    
# def compute_lie_loss(sig_pred, sig_true):
#     """
#     Lie loss function as described in the paper (Equation 15).
    
#     Paper uses:
#     1. Eigenvalue MSE loss: 𝒟σᴹᴸ = 1/N ∑[σᵢ - σᵢᴹ]²
#     2. Rotation loss: 𝒟ᴿᴹᴸ = 1/N ∑Φᵢ where Φ is the geodesic distance in Lie algebra
#     3. Total loss: 𝒟ₒᴹᴸ = 1/2(𝒟σᴹᴸ + 𝒟ᴿᴹᴸ)  ← 50/50 weighting!
    
#     sig_pred, sig_true: (batch_size, H, 6) in Voigt notation
#     """
#     batch_size, H, _ = sig_pred.shape
    
#     # Reshape to 2D for processing
#     sig_pred_flat = sig_pred.reshape(-1, 6)
#     sig_true_flat = sig_true.reshape(-1, 6)
    
#     # Convert to 3x3 tensors
#     sig_pred_3x3 = TensorUtils.voigt6_to_tensor3x3(sig_pred_flat)
#     sig_true_3x3 = TensorUtils.voigt6_to_tensor3x3(sig_true_flat)
    
#     # --------------------------------------------------
#     # 1. EIGENVALUE LOSS (𝒟σᴹᴸ in paper)
#     # --------------------------------------------------
#     # Get eigenvalues (stress principal values)
#     # Note: paper uses stress eigenvalues, NOT scaled values
#     eigvals_pred, _ = torch.linalg.eigh(sig_pred_3x3)
#     eigvals_true, _ = torch.linalg.eigh(sig_true_3x3)
    
#     # Eigenvalue MSE loss (Equation 15 middle term)
#     # Paper: 𝒟σᴹᴸ = 1/N ∑[σᵢ - σᵢᴹ]²
#     eigenvalue_loss = F.mse_loss(eigvals_pred, eigvals_true)
    
#     # --------------------------------------------------
#     # 2. ROTATION LOSS (𝒟ᴿᴹᴸ in paper)
#     # --------------------------------------------------
#     # Get rotation matrices from eigenvectors
#     with torch.no_grad():
#         # Use eigen decomposition for rotation extraction
#         _, eigvecs_pred = torch.linalg.eigh(sig_pred_3x3)
#         _, eigvecs_true = torch.linalg.eigh(sig_true_3x3)
        
#         # Ensure proper rotation matrices (det = +1)
#         R_pred = eigvecs_pred.clone()
#         R_true = eigvecs_true.clone()
        
#         # Fix determinants if needed (ensure proper rotations)
#         det_pred = torch.linalg.det(R_pred)
#         det_true = torch.linalg.det(R_true)
        
#         # Flip sign of first eigenvector if determinant is negative
#         if torch.any(det_pred < 0):
#             mask = det_pred < 0
#             if R_pred.dim() == 3:  # batched
#                 R_pred[mask, :, 0] = -R_pred[mask, :, 0]
#             else:  # single
#                 R_pred[:, 0] = -R_pred[:, 0]
        
#         if torch.any(det_true < 0):
#             mask = det_true < 0
#             if R_true.dim() == 3:  # batched
#                 R_true[mask, :, 0] = -R_true[mask, :, 0]
#             else:  # single
#                 R_true[:, 0] = -R_true[:, 0]
    
#     # Detach rotations since we don't want gradients through eigen decomposition
#     R_pred = R_pred.detach()
#     R_true = R_true.detach()
    
#     # --------------------------------------------------
#     # 3. GEODESIC DISTANCE IN LIE ALGEBRA (Φ in paper)
#     # --------------------------------------------------
#     # Paper Equation 10: φₗᵢₑ = ‖log(R₁R₂ᵀ)‖ = ‖W₁ - W₂‖
#     # where W = log(R) is in Lie algebra so(3)
    
#     # Compute relative rotation
#     R_rel = torch.bmm(R_pred, R_true.transpose(-1, -2))
    
#     # Compute rotation angle using trace formula
#     # Paper Equation 7: Θ = arccos[½(tr(R) - 1)]
#     trace = R_rel.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    
#     # Clamp for numerical stability
#     cos_theta = 0.5 * (trace - 1.0)
#     cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    
#     # Compute rotation angle Θ
#     theta = torch.acos(cos_theta)  # in [0, π]
    
#     # For geodesic distance in Lie algebra:
#     # Paper Equation 10: φₗᵢₑ = ‖log(R₁R₂ᵀ)‖ = Θ (for rotation angle)
#     # Actually, φₗᵢₑ = Θ, since ‖log(R)‖ = Θ for rotations
    
#     # The geodesic distance is just theta (rotation angle)
#     geodesic_dist = theta  # φₗᵢₑ in paper
    
#     # Rotation loss: average geodesic distance squared? Let's check paper...
#     # Paper Equation 15: 𝒟ᴿᴹᴸ = 1/N ∑Φᵢ
#     # Looking at context, Φᵢ is the geodesic distance for sample i
#     rotation_loss = geodesic_dist.mean()
  
#     # But the paper says "𝒟ᴿᴹᴸ = 1/N ∑Φᵢ" 
#     rotation_loss = (geodesic_dist ** 2).mean()
    
#     # --------------------------------------------------
#     # 4. TOTAL LOSS (Equation 15)
#     # --------------------------------------------------
#     # Paper: 𝒟ₒᴹᴸ = ½(𝒟σᴹᴸ + 𝒟ᴿᴹᴸ) ← 50/50 weighting!
#     total_loss = 0.5 * (eigenvalue_loss + rotation_loss)
    
#     return total_loss

def compute_lie_loss(sig_pred, sig_true):
    """
    Lie loss function as described in the paper (Equation 15).
    
    Gradient flow through eigen decomposition.
    """
    batch_size, H, _ = sig_pred.shape
    
    # Reshape to 2D for processing
    sig_pred_flat = sig_pred.reshape(-1, 6)
    sig_true_flat = sig_true.reshape(-1, 6)
    
    # Convert to 3x3 tensors
    sig_pred_3x3 = TensorUtils.voigt6_to_tensor3x3(sig_pred_flat)
    sig_true_3x3 = TensorUtils.voigt6_to_tensor3x3(sig_true_flat)
    
    # --------------------------------------------------
    # 1. EIGENVALUE LOSS (𝒟σᴹᴸ in paper)
    # --------------------------------------------------
    # Get eigenvalues AND eigenvectors WITH GRADIENTS
    eigvals_pred, eigvecs_pred = torch.linalg.eigh(sig_pred_3x3)
    eigvals_true, eigvecs_true = torch.linalg.eigh(sig_true_3x3)
    
    # Eigenvalue MSE loss
    eigenvalue_loss = F.mse_loss(eigvals_pred, eigvals_true)
    
    # --------------------------------------------------
    # 2. ROTATION LOSS (𝒟ᴿᴹᴸ in paper)
    # --------------------------------------------------
    # Clone eigenvectors to avoid inplace modification issues
    R_pred = eigvecs_pred.clone()
    R_true = eigvecs_true.clone()
    
    # Ensure proper rotation matrices (det = +1)
    # FIX: Don't modify inplace, create new tensors
    det_pred = torch.linalg.det(R_pred)
    det_true = torch.linalg.det(R_true)
    
    # Fix determinants WITHOUT inplace operations
    if torch.any(det_pred < 0):
        mask = det_pred < 0
        if R_pred.dim() == 3:  # batched
            # Create new tensor for modified rotations
            R_pred_fixed = R_pred.clone()
            R_pred_fixed[mask, :, 0] = -R_pred_fixed[mask, :, 0]
            R_pred = R_pred_fixed
        else:  # single
            R_pred_fixed = R_pred.clone()
            R_pred_fixed[:, 0] = -R_pred_fixed[:, 0]
            R_pred = R_pred_fixed
    
    # For ground truth, we can detach (no gradients needed for ground truth)
    R_true_detached = R_true.detach().clone()
    if torch.any(det_true < 0):
        mask = det_true < 0
        if R_true.dim() == 3:  # batched
            R_true_detached[mask, :, 0] = -R_true_detached[mask, :, 0]
        else:  # single
            R_true_detached[:, 0] = -R_true_detached[:, 0]
    
    # --------------------------------------------------
    # 3. GEODESIC DISTANCE IN LIE ALGEBRA (Φ in paper)
    # --------------------------------------------------
    # Compute relative rotation
    R_rel = torch.bmm(R_pred, R_true_detached.transpose(-1, -2))
    
    # Compute rotation angle using trace formula
    trace = R_rel.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    
    # Clamp for numerical stability
    cos_theta = 0.5 * (trace - 1.0)
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    
    # Compute rotation angle Θ
    theta = torch.acos(cos_theta)  # in [0, π]
    
    # The geodesic distance is theta (rotation angle)
    geodesic_dist = theta
    
    # Paper Equation 15: 𝒟ᴿᴹᴸ = 1/N ∑Φᵢ
    # Use average geodesic distance (not squared)
    rotation_loss = geodesic_dist.mean()
    
    # --------------------------------------------------
    # 4. TOTAL LOSS (Equation 15)
    # --------------------------------------------------
    total_loss = 0.5 * (eigenvalue_loss + rotation_loss)
    
    return total_loss

def compute_voigt_loss(sig_pred, sig_true):
    """
    Equation 11: Component-based loss (Voigt notation).
    
    Paper: 𝒟σᴹᴸ = 1/N ∑[σ̄ᵢ - σ̄ᵢᴹ]²
    
    Where:
    - σ̄ᵢ are true stress components scaled to [0, 1]
    - σ̄ᵢᴹ are predicted stress components (sigmoid output in [0, 1])
    
    This is the baseline "black box" loss using tensor components directly.
    """
    # Equation 11: Simple MSE on stress components
    # Paper scales components to [0, 1] during preprocessing
    # If your data is already scaled, use directly:
    loss = torch.mean((sig_pred - sig_true)**2)
    
    return loss

def compute_loss(sig_pred, sig_true, loss_type="lie", epse_pred=None, epse_true=None, scaling_factors=None):
    """
    Main loss function dispatcher.
    
    Args:
        sig_pred: Predicted stress (batch_size, H, 6) - already in [0,1] from sigmoid
        sig_true: True stress (batch_size, H, 6) - needs scaling to [0,1]
        loss_type: "lie" (geodesic) or "voigt" 
        epse_pred, epse_true: Elastic strain (optional)
        scaling_factors: Dict with min/max for scaling ground truth
    """
    if loss_type == "lie":
        # Lie loss doesn't need scaling factors per paper
        return compute_lie_loss(sig_pred, sig_true)
    elif loss_type == "voigt":
        # Voigt loss: scale only ground truth to [0,1]
        if scaling_factors is not None:
            # Scale ONLY the ground truth
            sig_true_scaled = scale_tensor(sig_true, scaling_factors)
            # sig_pred is already in [0,1] from sigmoid
            return F.mse_loss(sig_pred, sig_true_scaled)
        else:
            # If no scaling factors, use as-is (but this may not converge well)
            return F.mse_loss(sig_pred, sig_true)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}. Use 'lie' or 'voigt'")

def compute_scaling_factors(dataset, device='cpu'):
    """
    Compute scaling factors for eigenvalues and Lie algebra components
    from training dataset.
    
    Parameters
    ----------
    dataset : TrajWindowDataset
        Training dataset
    device : str
        Device to compute on
        
    Returns
    -------
    dict
        Scaling factors
    """
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_eigvals = []
    all_w = []
    
    for eps, epse, deps, sig_true, epse_true in loader:
        sig_true = sig_true.to(device)
        sig_3x3 = TensorUtils.voigt6_to_tensor3x3(sig_true)
        
        # Get eigenvalues
        eigvals, eigvecs = TensorUtils.spectral_decomposition(sig_3x3)
        all_eigvals.append(eigvals)
        
        # Get rotation matrices
        R = eigvecs
        # Ensure right-handed
        det_R = torch.linalg.det(R)
        if torch.any(det_R < 0):
            mask = det_R < 0
            R[mask, :, 0] = -R[mask, :, 0]
        
        # Get Lie algebra components
        w = TensorUtils.rotation_to_lie_algebra(R)
        all_w.append(w)
    
    # Concatenate all batches
    all_eigvals = torch.cat(all_eigvals, dim=0)
    all_w = torch.cat(all_w, dim=0)
    
    # Compute min/max
    scaling_factors = {
        'eigval_min': all_eigvals.min(dim=0)[0].cpu().numpy(),
        'eigval_max': all_eigvals.max(dim=0)[0].cpu().numpy(),
        'w_min': all_w.min(dim=0)[0].cpu().numpy(),
        'w_max': all_w.max(dim=0)[0].cpu().numpy(),
    }
    
    return scaling_factors