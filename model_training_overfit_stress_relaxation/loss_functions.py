
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

# --------------- FROM THE ACTUAL PAPER ------------------------------

"""
Loss functions from the paper for stress prediction in Voigt-6 format.

Includes:
    - loss_voigt_mse               : DM L0 (baseline)
    - loss_eigenvalues_eigenvectors: DM L_SN
    - loss_eigenvalues_euler       : DM L_Euler
    - loss_eigenvalues_lie         : DM L_SO3 (Lie algebra)
"""


def voigt6_to_tensor33(v6):
    """
    v6: (N,6)
    returns: (N,3,3)
    """
    return TensorUtils.from_voigt6(v6)



# eigen-decomposition with sorted eigenvalues (descending)
def eig_sorted(A):
    """
    A: (N,3,3) symmetric
    Returns:
        lam (N,3) sorted descending
        R   (N,3,3) corresponding eigenvectors
    """
    lam, R = torch.linalg.eigh(A)
    idx = torch.argsort(lam, dim=1, descending=True)
    lam_sorted = torch.gather(lam, 1, idx)
    R_sorted = torch.stack([R[i, :, idx[i]] for i in range(R.shape[0])], dim=0)
    return lam_sorted, R_sorted


# ZYX Euler angles from rotation matrix
def rotation_to_euler(R):
    """
    R: (N,3,3)
    Returns: (N,3) Euler angles phi, theta, psi
    """
    sy = torch.clamp(-R[:,2,0], -1.0, 1.0)
    theta = torch.asin(sy)
    phi   = torch.atan2(R[:,2,1], R[:,2,2])
    psi   = torch.atan2(R[:,1,0], R[:,0,0])
    return torch.stack([phi, theta, psi], dim=1)


# periodic angle distance
def angle_dist(a, b):
    diff = torch.abs(a - b)
    return torch.min(diff, 2 * torch.pi - diff)


# so(3) logarithm map
def so3_log(R):
    """
    R: (N,3,3)
    Returns: w: (N,3) rotation vector
    """
    tr = torch.clamp((R[:,0,0] + R[:,1,1] + R[:,2,2] - 1) / 2, -1.0, 1.0)
    phi = torch.acos(tr)
    eps = 1e-7

    S = R - R.transpose(1,2)
    w_raw = 0.5 * torch.stack([S[:,2,1], S[:,0,2], S[:,1,0]], dim=1)

    mask = phi > eps
    scale = phi / (torch.sin(phi + eps))
    scale = scale.unsqueeze(1)

    w = torch.zeros_like(w_raw)
    w[mask] = w_raw[mask] * scale[mask]
    w[~mask] = w_raw[~mask]  # small-angle approximation
    return w


# LOSS FUNCTIONS

# DM L0: simple MSE on Voigt-6
# -------------------------------------------------------------------------
def loss_voigt_mse(sig_pred, sig_true):
    """
    Inputs: sig_pred, sig_true: (B,H,6)
    """
    return F.mse_loss(sig_pred, sig_true)


# DM L_SN: eigenvalues + eigenvectors
# -------------------------------------------------------------------------
def loss_eigenvalues_eigenvectors(sig_pred, sig_true):
    B,H,_ = sig_pred.shape
    N = B*H

    A_pred = voigt6_to_tensor33(sig_pred.reshape(N,6))
    A_true = voigt6_to_tensor33(sig_true.reshape(N,6))

    lam_p, R_p = eig_sorted(A_pred)
    lam_t, R_t = eig_sorted(A_true)

    L_lam = F.mse_loss(lam_p, lam_t)
    L_R   = torch.mean((R_p - R_t)**2)

    return L_lam + L_R


# DM L_Euler: eigenvalues + Euler-angle distance
# -------------------------------------------------------------------------
def loss_eigenvalues_euler(sig_pred, sig_true):
    B,H,_ = sig_pred.shape
    N = B*H

    A_pred = voigt6_to_tensor33(sig_pred.reshape(N,6))
    A_true = voigt6_to_tensor33(sig_true.reshape(N,6))

    lam_p, R_p = eig_sorted(A_pred)
    lam_t, R_t = eig_sorted(A_true)

    eul_p = rotation_to_euler(R_p)
    eul_t = rotation_to_euler(R_t)

    L_lam = F.mse_loss(lam_p, lam_t)
    L_ang = torch.mean(angle_dist(eul_p, eul_t)**2)

    return L_lam + L_ang


# DM L_SO3: eigenvalues + so(3) log
# -------------------------------------------------------------------------
def loss_eigenvalues_lie(sig_pred, sig_true):
    B,H,_ = sig_pred.shape
    N = B*H

    A_pred = voigt6_to_tensor33(sig_pred.reshape(N,6))
    A_true = voigt6_to_tensor33(sig_true.reshape(N,6))

    lam_p, R_p = eig_sorted(A_pred)
    lam_t, R_t = eig_sorted(A_true)

    L_lam = F.mse_loss(lam_p, lam_t)

    # rotation residual: R_delta = R_true * R_pred^T
    R_delta = torch.matmul(R_t, R_p.transpose(1,2))
    w = so3_log(R_delta)
    L_rot = torch.mean(torch.sum(w*w, dim=1))

    return L_lam + L_rot


# DM L_FRO: eigenvalues + Frobenius distance from identity
# -------------------------------------------------------------------------
def loss_eigenvalues_fro(sig_pred, sig_true):
    """
    Inputs:
        sig_pred, sig_true: (B,H,6) Voigt-6 stress tensors
    Returns:
        scalar loss: eigenvalue MSE + Frobenius distance for rotations
    """
    B, H, _ = sig_pred.shape
    N = B * H

    # Convert Voigt-6 to 3x3 tensors
    A_pred = TensorUtils.from_voigt6(sig_pred.reshape(N,6))
    A_true = TensorUtils.from_voigt6(sig_true.reshape(N,6))

    # Eigen-decomposition
    lam_p, R_p = eig_sorted(A_pred)
    lam_t, R_t = eig_sorted(A_true)

    # Eigenvalue loss
    L_lam = F.mse_loss(lam_p, lam_t)

    # Frobenius distance term
    R_delta = torch.matmul(R_t, R_p.transpose(1,2))  # R_true * R_pred^T
    trace = torch.diagonal(R_delta, dim1=-2, dim2=-1).sum(-1)  # shape (N,)
    L_rot = torch.sqrt(2.0 * (3.0 - trace)).mean()  # Eq. (18)

    return L_lam + L_rot
