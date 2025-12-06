
# # --------------------- Lie loss - test 1 - from Heider Y, Wang K, Sun W (2020) paper -------------
#!/usr/bin/env python3
import argparse
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
import numpy as np

from model import SeqModel
from data_loader import TrajWindowDataset
from loss_functions import mse_loss
from constitutive_update import assemble_elastic_tensor_voigt, elastic_update

# ============================================================
# Lie Algebra Utilities for SO(3) Rotations
# ============================================================
def tensor_to_eigen_rotation(tensor_batch):
    """
    Convert batch of symmetric tensors (Voigt notation) to eigenvalues and rotation matrices.
    
    Args:
        tensor_batch: (B, 6) tensor in Voigt notation [σ11, σ22, σ33, σ23, σ13, σ12]
    
    Returns:
        eigenvalues: (B, 3) principal values
        rotation: (B, 3, 3) rotation matrix (eigenvectors as columns)
    """
    B = tensor_batch.shape[0]
    
    # Convert Voigt to full 3x3 tensor
    tensor_3x3 = torch.zeros(B, 3, 3, device=tensor_batch.device)
    
    # Diagonal components
    tensor_3x3[:, 0, 0] = tensor_batch[:, 0]
    tensor_3x3[:, 1, 1] = tensor_batch[:, 1]
    tensor_3x3[:, 2, 2] = tensor_batch[:, 2]
    
    # Off-diagonal components (symmetric)
    tensor_3x3[:, 1, 2] = tensor_batch[:, 3]
    tensor_3x3[:, 2, 1] = tensor_batch[:, 3]
    
    tensor_3x3[:, 0, 2] = tensor_batch[:, 4]
    tensor_3x3[:, 2, 0] = tensor_batch[:, 4]
    
    tensor_3x3[:, 0, 1] = tensor_batch[:, 5]
    tensor_3x3[:, 1, 0] = tensor_batch[:, 5]
    
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = torch.linalg.eigh(tensor_3x3)
    
    # Sort eigenvalues in descending order (most tensile first)
    # and corresponding eigenvectors
    sorted_indices = torch.argsort(eigvals, dim=1, descending=True)
    
    sorted_eigvals = torch.zeros_like(eigvals)
    sorted_eigvecs = torch.zeros_like(eigvecs)
    
    for i in range(B):
        sorted_eigvals[i] = eigvals[i, sorted_indices[i]]
        sorted_eigvecs[i] = eigvecs[i, :, sorted_indices[i]]
    
    return sorted_eigvals, sorted_eigvecs

def rotation_to_lie_algebra(R, eps=1e-8):
    """
    Convert rotation matrix to skew-symmetric matrix in Lie algebra.
    NUMERICALLY STABLE VERSION.
    """
    B = R.shape[0]
    W = torch.zeros_like(R)
    
    for i in range(B):
        Ri = R[i]
        
        # Ensure Ri is a valid rotation matrix (orthogonal)
        # Add safety checks
        if torch.any(torch.isnan(Ri)) or torch.any(torch.isinf(Ri)):
            W[i] = torch.zeros(3, 3, device=R.device)
            continue
            
        # Compute rotation angle Θ with safe clamping
        cos_theta = (torch.trace(Ri) - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
        theta = torch.acos(cos_theta)
        
        # Handle edge cases
        if theta < eps:
            # Identity or very small rotation
            W[i] = torch.zeros(3, 3, device=R.device)
        elif torch.abs(theta - torch.pi) < eps:
            # θ ≈ π case
            # Find eigenvector for eigenvalue 1 (more stable)
            try:
                eigvals, eigvecs = torch.linalg.eigh(Ri)
                # Find eigenvector for eigenvalue closest to 1
                idx = torch.argmin(torch.abs(eigvals - 1.0))
                v = eigvecs[:, idx]
                
                # Ensure v is unit length
                v = v / (torch.norm(v) + eps)
                
                # Create skew-symmetric matrix [v]
                v_skew = torch.zeros(3, 3, device=R.device)
                v_skew[0, 1] = -v[2]
                v_skew[0, 2] = v[1]
                v_skew[1, 0] = v[2]
                v_skew[1, 2] = -v[0]
                v_skew[2, 0] = -v[1]
                v_skew[2, 1] = v[0]
                
                W[i] = torch.pi * v_skew
            except:
                W[i] = torch.zeros(3, 3, device=R.device)
        else:
            # General case with safe division
            sin_theta = torch.sin(theta)
            if torch.abs(sin_theta) < eps:
                # sin(theta) too small, use approximation
                W[i] = 0.5 * (Ri - Ri.T)  # First-order approximation
            else:
                R_minus_RT = Ri - Ri.T
                # Safe division
                scale = theta / (2 * sin_theta + eps)
                W[i] = scale * R_minus_RT
    
    return W

def lie_distance_loss(R_true, R_pred, epsilon=1e-8):
    """
    Compute geodesic distance loss in Lie algebra (Eq. 10 of the paper).
    
    Args:
        R_true: (B, 3, 3) true rotation matrices
        R_pred: (B, 3, 3) predicted rotation matrices
    
    Returns:
        loss: scalar, mean geodesic distance
    """
    B = R_true.shape[0]
    
    # Compute relative rotation: R_rel = R_true @ R_pred^T
    R_rel = torch.bmm(R_true, R_pred.transpose(1, 2))
    
    # Convert to Lie algebra
    W_rel = rotation_to_lie_algebra(R_rel)
    
    # Compute Frobenius norm: ||W_rel||_F
    # For skew-symmetric matrices, this is sqrt(2) * ||w|| where w is the axial vector
    # But we can compute directly
    norm_sq = torch.sum(W_rel * W_rel, dim=(1, 2))
    norm = torch.sqrt(norm_sq + epsilon)
    
    # Geodesic distance (Eq. 10): φ_Lie = ||log(R1 R2^T)|| = ||W_rel||
    return torch.mean(norm)

def lie_algebra_loss(tensor_true, tensor_pred, lambda_eig=1.0, lambda_rot=1.0):
    """
    Combined loss function using Lie algebra approach (DG2_Lie in paper).
    
    Args:
        tensor_true: (B, 6) true stress tensor in Voigt notation
        tensor_pred: (B, 6) predicted stress tensor in Voigt notation
        lambda_eig: weight for eigenvalue loss
        lambda_rot: weight for rotation loss
    
    Returns:
        total_loss: scalar combined loss
    """
    # Get eigenvalues and rotation matrices
    eigvals_true, R_true = tensor_to_eigen_rotation(tensor_true)
    eigvals_pred, R_pred = tensor_to_eigen_rotation(tensor_pred)
    
    # Eigenvalue loss (MSE on normalized eigenvalues)
    eigvals_true_norm = torch.linalg.norm(eigvals_true, dim=1, keepdim=True)
    eigvals_pred_norm = torch.linalg.norm(eigvals_pred, dim=1, keepdim=True)
    
    # Normalize to avoid scale issues
    eigvals_true_normalized = eigvals_true / (eigvals_true_norm + 1e-8)
    eigvals_pred_normalized = eigvals_pred / (eigvals_pred_norm + 1e-8)
    
    eig_loss = torch.mean((eigvals_true_normalized - eigvals_pred_normalized) ** 2)
    
    # Rotation loss using Lie algebra geodesic distance
    rot_loss = lie_distance_loss(R_true, R_pred)
    
    # Combined loss (Eq. 15 in paper)
    total_loss = lambda_eig * eig_loss + lambda_rot * rot_loss
    
    return total_loss, eig_loss, rot_loss

# ============================================================
# Logging utility
# ============================================================
def log_device_info(log_path="train_log.txt"):
    """
    Append a timestamped entry describing which device is used.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)
        msg = f"[{now}] Using GPU: {gpu_name} (device id: {gpu_id})"
    else:
        msg = f"[{now}] Using CPU"

    with open(log_path, "a") as f:
        f.write(msg + "\n")
    print(msg)
    return msg

def log_epoch_loss(epoch, loss_fit, loss_overlap, loss_self, loss_eig, loss_rot, total_loss, log_path="train_log.txt"):
    """
    Append timestamped loss info to the log file.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = (
        f"[{now}] [TRAIN] Epoch {epoch:03d} | "
        f"fit={loss_fit:.6e}, "
        f"overlap={loss_overlap:.6e}, "
        f"self={loss_self:.6e}, "
        f"eig={loss_eig:.6e}, "
        f"rot={loss_rot:.6e}, "
        f"total={total_loss:.6e}"
    )
    with open(log_path, "a") as f:
        f.write(msg + "\n")
    print(msg)
    return msg

def log_eval_loss(epoch, mse_sig, mse_epsp, mse_eig, mse_rot, log_path="train_log.txt"):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = (
        f"[{now}] [EVAL] Epoch {epoch:03d} | "
        f"MSE: sig={mse_sig:.6e}, epsp={mse_epsp:.6e}, "
        f"eig={mse_eig:.6e}, rot={mse_rot:.6e}"
    )
    with open(log_path, "a") as f:
        f.write(msg + "\n")
    print(msg)
    return msg

# ============================================================
# Overlap consistency loss (adapted for Lie algebra)
# ============================================================
def overlap_consistency_loss_lie(global_key, tensor_pred):
    """
    Penalize disagreement among predictions with the same (traj_id, time)
    using Lie algebra loss.
    
    Args:
        global_key: (B, H) encoded as traj_id * SCALE + time
        tensor_pred: (B, H, 6) predicted tensors in Voigt notation
    """
    B, H, _ = tensor_pred.shape
    keys_flat = global_key.reshape(-1)   # (B*H,)
    tensor_flat = tensor_pred.reshape(-1, 6)  # (B*H, 6)
    
    # Group by unique keys
    uniq, inv, counts = keys_flat.unique(
        return_inverse=True, return_counts=True
    )
    mask_multi = counts > 1
    
    if not mask_multi.any():
        return torch.zeros((), device=tensor_pred.device, requires_grad=True)
    
    # For each unique key with multiple predictions, compute mean tensor
    K = uniq.size(0)
    group_sum = torch.zeros(K, 6, device=tensor_pred.device)
    group_sum.index_add_(0, inv, tensor_flat)
    group_mean = group_sum / counts.float().unsqueeze(1)
    
    # Get predictions that belong to groups with multiple entries
    valid = mask_multi[inv]
    pred_multi = tensor_flat[valid]
    mean_multi = group_mean[inv[valid]]
    
    # Compute Lie algebra loss between predictions and their group mean
    loss_multi, _, _ = lie_algebra_loss(mean_multi, pred_multi)
    
    return loss_multi

# ============================================================
# Full-dev eval: MSE over ALL windows in a loader
# ============================================================
def eval_full_mse(model, dataloader, args, device, CCe_voigt):
    """
    Compute global MSE(sig), MSE(epsp), and Lie-based metrics.
    """
    model.eval()
    L, H, S = args.L, args.H, args.S
    
    sse_sig = sse_epsp = 0.0
    sse_eig = sse_rot = 0.0
    n_sig = n_epsp = n_eig = n_rot = 0
    
    with torch.no_grad():
        for eps_hist, epsp_hist, deps_future, sig_y, epsp_y, t0, traj_idx in dataloader:
            eps_hist = eps_hist.to(device)
            epsp_hist = epsp_hist.to(device)
            deps_future = deps_future.to(device)
            sig_y = sig_y.to(device)
            epsp_y = epsp_y.to(device)
            
            # Slice as in training
            eps_hist_L = eps_hist[:, :L, :]
            epsp_hist_L = epsp_hist[:, :L, :]
            deps_future_H = deps_future[:, :H, :]
            sig_y_H = sig_y[:, :H, :]
            epsp_y_H = epsp_y[:, :H, :]
            
            # Elastic predictor
            sig_hist = elastic_update(eps_hist_L - epsp_hist_L, CCe_voigt)
            
            # Forward model
            sig_pred, epsp_pred = model(eps_hist_L, epsp_hist_L, sig_hist, deps_future_H)
            
            B = sig_pred.shape[0]
            
            # Flatten for MSE
            sig_pred_flat = sig_pred.reshape(-1, 6)
            epsp_pred_flat = epsp_pred.reshape(-1, 6)
            sig_y_flat = sig_y_H.reshape(-1, 6)
            epsp_y_flat = epsp_y_H.reshape(-1, 6)
            
            # Standard MSE
            sse_sig += ((sig_pred_flat - sig_y_flat) ** 2).sum().item()
            sse_epsp += ((epsp_pred_flat - epsp_y_flat) ** 2).sum().item()
            
            n_sig += sig_pred_flat.numel()
            n_epsp += epsp_pred_flat.numel()
            
            # Lie-based metrics (for stress only)
            for i in range(B * H):
                sig_true_i = sig_y_flat[i:i+1]
                sig_pred_i = sig_pred_flat[i:i+1]
                
                eig_loss_i, rot_loss_i = 0.0, 0.0
                try:
                    # Compute Lie loss for this sample
                    total_loss_i, eig_loss_i, rot_loss_i = lie_algebra_loss(
                        sig_true_i, sig_pred_i
                    )
                    sse_eig += eig_loss_i.item()
                    sse_rot += rot_loss_i.item()
                    n_eig += 1
                    n_rot += 1
                except:
                    # Skip if numerical issues
                    continue
    
    mse_sig = sse_sig / n_sig if n_sig > 0 else 0.0
    mse_epsp = sse_epsp / n_epsp if n_epsp > 0 else 0.0
    mse_eig = sse_eig / n_eig if n_eig > 0 else 0.0
    mse_rot = sse_rot / n_rot if n_rot > 0 else 0.0
    
    return mse_sig, mse_epsp, mse_eig, mse_rot

# ============================================================
# Training with Lie loss
# ============================================================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_device_info()
    
    # Assemble elastic tensor
    CCe_voigt_np = assemble_elastic_tensor_voigt("material_parameters.csv")
    CCe_voigt = torch.from_numpy(CCe_voigt_np).float().to(device)
    
    # WandB initialization
    wandb.init(
        project="cs230-ti-mcc-lie-loss",
        entity="hemiaozh-stanford-university",
        name=(
            f"L{args.L:02d}_H{args.H:02d}_"
            f"lie_eig{args.lambda_eig:g}_rot{args.lambda_rot:g}"
        ),
        config={
            "L": args.L, "H": args.H, "S": args.S,
            "batch_size": args.bs, "learning_rate": args.lr,
            "epochs": args.epochs, "lambda_eig": args.lambda_eig,
            "lambda_rot": args.lambda_rot, "lambda_overlap": args.lambda_overlap,
            "lambda_self": args.lambda_self,
        },
    )
    
    # Datasets
    ds_tr = TrajWindowDataset(args.data, L=args.L, H=args.H, S=args.S, split='train')
    ds_va = TrajWindowDataset(args.data, L=args.L, H=args.H, S=args.S, split='dev')
    
    tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=False, num_workers=8)
    va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, num_workers=8)
    
    model = SeqModel(H=args.H, hid=args.hid).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    
    KEY_SCALE = 1_000_000
    
    # Initial evaluation
    mse_sig_va, mse_epsp_va, mse_eig_va, mse_rot_va = eval_full_mse(
        model, va, args, device, CCe_voigt
    )
    print(f"[EVAL] INITIAL | MSE: sig={mse_sig_va:.3e}, epsp={mse_epsp_va:.3e}, "
          f"eig={mse_eig_va:.3e}, rot={mse_rot_va:.3e}")
    
    wandb.log({
        "epoch": 0,
        "eval_dev_mse_sig": mse_sig_va,
        "eval_dev_mse_epsp": mse_epsp_va,
        "eval_dev_mse_eig": mse_eig_va,
        "eval_dev_mse_rot": mse_rot_va,
    }, step=0)
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_fit_sum = loss_overlap_sum = loss_self_sum = 0.0
        loss_eig_sum = loss_rot_sum = total_loss_sum = 0.0
        
        for eps_hist, epsp_hist, deps_future, sig_y, epsp_y, t0, traj_idx in tr:
            # Move to device
            eps_hist = eps_hist.to(device)
            epsp_hist = epsp_hist.to(device)
            deps_future = deps_future.to(device)
            sig_y = sig_y.to(device)
            epsp_y = epsp_y.to(device)
            t0 = t0.to(device).long()
            traj_idx = traj_idx.to(device).long()
            
            L, H, S = args.L, args.H, args.S
            
            # Slice windows
            eps_hist_L = eps_hist[:, :L, :]
            epsp_hist_L = epsp_hist[:, :L, :]
            deps_future_H = deps_future[:, :H, :]
            sig_y_H = sig_y[:, :H, :]
            epsp_y_H = epsp_y[:, :H, :]
            
            # Elastic predictor
            sig_hist = elastic_update(eps_hist_L - epsp_hist_L, CCe_voigt)
            
            # Forward model
            sig_pred, epsp_pred = model(eps_hist_L, epsp_hist_L, sig_hist, deps_future_H)
            
            B = sig_pred.shape[0]
            
            # Flatten
            sig_pred_flat = sig_pred.reshape(B * H, 6)
            epsp_pred_flat = epsp_pred.reshape(B * H, 6)
            sig_y_flat = sig_y_H.reshape(B * H, 6)
            epsp_y_flat = epsp_y_H.reshape(B * H, 6)
            
            # 1. Fit loss with Lie algebra
            # Stress loss: combine eigenvalue and rotation components
            loss_stress, loss_eig, loss_rot = lie_algebra_loss(
                sig_y_flat, sig_pred_flat,
                lambda_eig=args.lambda_eig,
                lambda_rot=args.lambda_rot
            )
            
            # Plastic strain loss (MSE)
            loss_plastic = mse_loss(epsp_pred_flat, epsp_y_flat)
            
            # Total fit loss with nonlinear transform
            loss_fit = loss_stress + args.lambda_pstrain * loss_plastic
            loss_fit = -torch.exp(-loss_fit) + loss_fit
            
            # 2. Overlap consistency loss (using Lie algebra)
            step_idx = torch.arange(1, H + 1, device=device).unsqueeze(0)
            global_t = t0.unsqueeze(1) + step_idx
            global_key = traj_idx.unsqueeze(1) * KEY_SCALE + global_t
            
            loss_overlap = overlap_consistency_loss_lie(global_key, sig_pred)
            
            # 3. Self-fed loss (if enabled)
            loss_self = torch.tensor(0.0, device=device)
            if (S > 0) and (args.lambda_self > 0.0):
                epsp_self = epsp_hist_L.clone()
                epsp_self[:, :-S, :] = epsp_self[:, S:, :]
                epsp_self[:, -S:, :] = epsp_pred[:, :S, :]
                
                eps_self = eps_hist[:, S:L+S, :]
                sig_self = elastic_update(eps_self - epsp_self, CCe_voigt)
                deps_self = deps_future[:, S:H+S, :]
                
                sig_self_pred, epsp_self_pred = model(
                    eps_self, epsp_self, sig_self, deps_self
                )
                
                sig_y_S = sig_y[:, S:S+H, :]
                epsp_y_S = epsp_y[:, S:S+H, :]
                
                sig_self_flat = sig_self_pred.reshape(B * H, 6)
                sig_y_S_flat = sig_y_S.reshape(B * H, 6)
                epsp_self_pred_flat = epsp_self_pred.reshape(B * H, 6)
                epsp_y_S_flat = epsp_y_S.reshape(B * H, 6)
                
                # Use Lie loss for stress in self-fed too
                loss_self_stress, _, _ = lie_algebra_loss(
                    sig_y_S_flat, sig_self_flat
                )
                loss_self_plastic = mse_loss(epsp_self_pred_flat, epsp_y_S_flat)
                
                loss_self = loss_self_stress + args.lambda_pstrain * loss_self_plastic
            
            # Total loss
            loss = (
                loss_fit
                + args.lambda_overlap * loss_overlap
                + args.lambda_self * loss_self
            )
            
            # Optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # Record losses
            loss_fit_sum += loss_fit.item()
            loss_overlap_sum += loss_overlap.item()
            loss_self_sum += loss_self.item()
            loss_eig_sum += loss_eig.item()
            loss_rot_sum += loss_rot.item()
            total_loss_sum += loss.item()
        
        # Average losses
        num_batches = len(tr)
        avg_fit = loss_fit_sum / num_batches
        avg_overlap = loss_overlap_sum / num_batches
        avg_self = loss_self_sum / num_batches
        avg_eig = loss_eig_sum / num_batches
        avg_rot = loss_rot_sum / num_batches
        avg_total = total_loss_sum / num_batches
        
        # Log every epoch to wandb
        wandb.log({
            "epoch": epoch,
            "fit_loss": avg_fit,
            "overlap_loss": avg_overlap,
            "self_loss": avg_self,
            "eig_loss": avg_eig,
            "rot_loss": avg_rot,
            "total_loss": avg_total,
        }, step=epoch)
        
        # Print every 50 epochs for faster training feedback
        if epoch % 50 == 0:
            log_epoch_loss(epoch, avg_fit, avg_overlap, avg_self, 
                          avg_eig, avg_rot, avg_total)
        
        # Evaluation every 100 epochs
        if epoch % 100 == 0:
            mse_sig_dev, mse_epsp_dev, mse_eig_dev, mse_rot_dev = eval_full_mse(
                model, va, args, device, CCe_voigt
            )
            log_eval_loss(epoch, mse_sig_dev, mse_epsp_dev, 
                         mse_eig_dev, mse_rot_dev)
            
            wandb.log({
                "eval_dev_mse_sig": mse_sig_dev,
                "eval_dev_mse_epsp": mse_epsp_dev,
                "eval_dev_mse_eig": mse_eig_dev,
                "eval_dev_mse_rot": mse_rot_dev,
            }, step=epoch)
            
            model.train()
    
    # Save model
    torch.save(model.state_dict(), args.out)
    print(f"[OK] Training completed. Model saved to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./data_stress_strain_labeled.npz")
    ap.add_argument("--out", type=str, default="./model_lie_loss.pt")
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--S", type=int, default=1)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hid", type=int, default=128)
    
    # Loss weights
    ap.add_argument("--lambda_pstrain", type=float, default=1e4)
    ap.add_argument("--lambda_eig", type=float, default=1.0,
                    help="Weight for eigenvalue loss in Lie algebra approach")
    ap.add_argument("--lambda_rot", type=float, default=1.0,
                    help="Weight for rotation loss in Lie algebra approach")
    ap.add_argument("--lambda_overlap", type=float, default=0.1)
    ap.add_argument("--lambda_self", type=float, default=0.1)
    
    args = ap.parse_args()
    assert 0 <= args.S <= min(args.L, args.H), "S must satisfy 0 ≤ S ≤ min(L, H)"
    
    train(args)