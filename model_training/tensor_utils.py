#!/usr/bin/env python3
"""
tensor_utils.py
----------------------------------
Common tensor utilities for symmetric 3×3 tensors used in mechanics.

Functions:
    - to_voigt6(T):      (…,3,3) → (…,6)
    - from_voigt6(v):    (…,6) → (…,3,3)
    - invariants_voigt6(v): compute [I1, J2] invariants from Voigt-6
"""

import torch

class TensorUtils:
    @staticmethod
    def to_voigt6(T):
        """
        Convert symmetric 3×3 tensor(s) to Voigt-6 vector form.

        Parameters
        ----------
        T : torch.Tensor (..., 3, 3)
            Input symmetric tensor(s)

        Returns
        -------
        torch.Tensor (..., 6)
            Voigt representation [xx, yy, zz, yz, xz, xy]
        """
        xx, yy, zz = T[...,0,0], T[...,1,1], T[...,2,2]
        yz = 0.5 * (T[...,1,2] + T[...,2,1])
        xz = 0.5 * (T[...,0,2] + T[...,2,0])
        xy = 0.5 * (T[...,0,1] + T[...,1,0])
        return torch.stack([xx, yy, zz, yz, xz, xy], dim=-1)

    @staticmethod
    def from_voigt6(v):
        """
        Convert Voigt-6 vector form back to symmetric 3×3 tensor(s).

        Parameters
        ----------
        v : torch.Tensor (..., 6)
            Input in Voigt notation [xx, yy, zz, yz, xz, xy]

        Returns
        -------
        torch.Tensor (..., 3, 3)
            Symmetric tensor(s)
        """
        xx, yy, zz, yz, xz, xy = v.unbind(-1)
        out = torch.zeros(v.shape[:-1] + (3,3), dtype=v.dtype, device=v.device)
        out[...,0,0] = xx; out[...,1,1] = yy; out[...,2,2] = zz
        out[...,1,2] = yz; out[...,2,1] = yz
        out[...,0,2] = xz; out[...,2,0] = xz
        out[...,0,1] = xy; out[...,1,0] = xy
        return out

    @staticmethod
    def invariants_voigt6(v):
        """
        Compute first and second deviatoric invariants [I1, J2].

        Parameters
        ----------
        v : torch.Tensor (..., 6)
            Input symmetric tensor(s) in Voigt form

        Returns
        -------
        torch.Tensor (..., 2)
            [I1, J2]
        """
        xx, yy, zz, yz, xz, xy = v.unbind(-1)
        I1 = xx + yy + zz
        sxx, syy, szz = xx - I1/3, yy - I1/3, zz - I1/3
        J2 = 0.5 * (sxx**2 + syy**2 + szz**2) + (yz**2 + xz**2 + xy**2)
        return torch.stack([I1, J2], dim=-1)












#### more functions for loss functions new
    @staticmethod
    def voigt6_to_tensor3x3(v):
        """
        Convert Voigt-6 stress to full 3×3 tensor form.
        
        Assumes Voigt order: (xx, yy, zz, yz, xz, xy).
        v : torch.Tensor (..., 6)
        Returns: torch.Tensor (..., 3, 3)
        """
        if v.shape[-1] != 6:
            raise ValueError(f"Expected last dim=6, got {v.shape[-1]}")
        
        out = torch.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype, device=v.device)
        # Diagonals
        out[..., 0, 0] = v[..., 0]  # xx
        out[..., 1, 1] = v[..., 1]  # yy
        out[..., 2, 2] = v[..., 2]  # zz
        # Off-diagonals (symmetric)
        out[..., 1, 2] = out[..., 2, 1] = v[..., 3]  # yz
        out[..., 0, 2] = out[..., 2, 0] = v[..., 4]  # xz
        out[..., 0, 1] = out[..., 1, 0] = v[..., 5]  # xy
        return out
    
    @staticmethod
    def tensor3x3_to_voigt6(T):
        """
        Convert full 3×3 stress tensor(s) to Voigt-6 form.
        
        Voigt order: (xx, yy, zz, yz, xz, xy).
        T : torch.Tensor (..., 3, 3)
        Returns: torch.Tensor (..., 6)
        """
        if T.shape[-1] != 3 or T.shape[-2] != 3:
            raise ValueError(f"Expected (...,3,3), got {T.shape}")
        
        out = torch.zeros(T.shape[:-2] + (6,), dtype=T.dtype, device=T.device)
        out[..., 0] = T[..., 0, 0]  # xx
        out[..., 1] = T[..., 1, 1]  # yy
        out[..., 2] = T[..., 2, 2]  # zz
        out[..., 3] = T[..., 1, 2]  # yz
        out[..., 4] = T[..., 2, 0]  # xz
        out[..., 5] = T[..., 0, 1]  # xy
        return out
    
    @staticmethod
    # In tensor_utils.py (or test_utils.py), update the tensor_to_rotation method:

    @staticmethod
    def tensor_to_rotation(tensor):
        """
        Convert a symmetric 3x3 tensor to a rotation matrix from its eigenvectors.
        FIXED: No in-place operations on eigenvectors to preserve gradient flow.
        """
        # Ensure symmetry WITHOUT in-place
        tensor_sym = 0.5 * (tensor + tensor.transpose(-1, -2))
        
        # Eigen decomposition
        eigvals, eigvecs = torch.linalg.eigh(tensor_sym)
        
        # Clone eigenvectors to avoid in-place modification
        R = eigvecs.clone()
        
        # Ensure right-handed coordinate system WITHOUT in-place modification
        # Instead of modifying in-place, create a new tensor
        det = torch.linalg.det(R)
        
        if torch.any(det < 0):
            # Create mask for negative determinants
            mask = det < 0
            
            if R.dim() == 2:  # single matrix
                if mask:
                    # Create new tensor with flipped first column
                    R_fixed = R.clone()
                    R_fixed[:, 0] = -R_fixed[:, 0]
                    return R_fixed
            else:  # batched
                # Create copy and modify only where needed
                R_fixed = R.clone()
                R_fixed[mask, :, 0] = -R_fixed[mask, :, 0]
                return R_fixed
        
        return R
    
    # Add to tensor_utils.py

    @staticmethod
    def rotation_to_euler(R, convention='ZXZ'):
        """
        Convert rotation matrix to Euler angles.
        
        Paper uses proper ranges:
        φ, θ ∈ [-π, π) and ψ ∈ [-π/2, π/2)
        
        convention: 'ZXZ' (common in mechanics) or 'ZYZ'
        """
        if R.dim() == 2:
            R = R.unsqueeze(0)
        
        batch_size = R.shape[0]
        euler = torch.zeros(batch_size, 3, device=R.device)
        
        if convention == 'ZXZ':
            # Z-X-Z convention
            for i in range(batch_size):
                # Check for gimbal lock
                if abs(R[i, 2, 2]) > 0.9999:
                    # Gimbal lock
                    euler[i, 0] = torch.atan2(R[i, 0, 1], R[i, 0, 0])
                    euler[i, 1] = 0.0
                    euler[i, 2] = 0.0
                else:
                    theta = torch.acos(R[i, 2, 2])
                    psi = torch.atan2(R[i, 2, 0] / torch.sin(theta), 
                                    -R[i, 2, 1] / torch.sin(theta))
                    phi = torch.atan2(R[i, 0, 2] / torch.sin(theta), 
                                    R[i, 1, 2] / torch.sin(theta))
                    euler[i] = torch.stack([phi, theta, psi])
        
        # Apply proper ranges as per paper
        # φ, θ ∈ [-π, π)
        euler[:, 0] = torch.remainder(euler[:, 0] + torch.pi, 2 * torch.pi) - torch.pi
        euler[:, 1] = torch.remainder(euler[:, 1] + torch.pi, 2 * torch.pi) - torch.pi
        # ψ ∈ [-π/2, π/2)
        euler[:, 2] = torch.clamp(euler[:, 2], -torch.pi/2, torch.pi/2 - 1e-7)
        
        return euler.squeeze()

    # @staticmethod
    # def euler_distance(euler1, euler2):
    #     """
    #     Compute Euclidean distance between Euler angles with proper wrapping.
        
    #     Paper Equation 5: φ_Eu = √[d(φ₁,φ₂)² + d(θ₁,θ₂)² + d(ψ₁,ψ₂)²]
    #     where d(a₁, a₂) = min{|a₁ - a₂|, 2π - |a₁ - a₂|} ∈ [0, π]
    #     """
    #     # Compute differences
    #     diff = euler1 - euler2
        
    #     # Apply wrapping: d(a,b) = min(|a-b|, 2π - |a-b|)
    #     abs_diff = torch.abs(diff)
    #     wrapped_diff = 2 * torch.pi - abs_diff
    #     min_diff = torch.min(abs_diff, wrapped_diff)
        
    #     # Euclidean distance
    #     distance = torch.sqrt(torch.sum(min_diff ** 2, dim=-1))
        
    #     return distance


# Add to tensor_utils.py

    import torch
    import torch.nn.functional as F
    from torch.linalg import eigh

    @staticmethod
    def rotation_to_lie_algebra(R, eps=1e-7):
        """
        Convert rotation matrix R ∈ SO(3) to Lie algebra (skew-symmetric) representation.
        
        Implements Eq. (8) from the paper: Logarithmic mapping from SO(3) to so(3).
        
        Parameters
        ----------
        R : torch.Tensor (..., 3, 3)
            Rotation matrix in SO(3)
        eps : float
            Small epsilon for numerical stability
            
        Returns
        -------
        torch.Tensor (..., 3)
            Lie algebra coordinates [w1, w2, w3] such that:
            W = [[0, -w3, w2],
                 [w3, 0, -w1],
                 [-w2, w1, 0]]
        """
        # Ensure R is in SO(3) within numerical tolerance
        if torch.any(torch.abs(torch.linalg.det(R) - 1.0) > 1e-4):
            print(f"Warning: Some rotation matrices have det ≠ 1. Mean det: {torch.linalg.det(R).mean().item()}")
        
        # Compute rotation angle using Eq. (7)
        # tr(R) = 1 + 2*cos(Θ)
        trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1)
        cos_theta = 0.5 * (trace - 1.0)
        
        # Clamp to avoid numerical issues
        cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
        theta = torch.acos(cos_theta)
        
        # Initialize output
        batch_shape = R.shape[:-2]
        w = torch.zeros(batch_shape + (3,), dtype=R.dtype, device=R.device)
        
        # Handle small angles (θ ≈ 0) - use Taylor expansion
        small_angle_mask = theta < eps
        large_angle_mask = ~small_angle_mask
        
        if torch.any(small_angle_mask):
            # For θ ≈ 0: W ≈ (R - R^T)/2 using small angle approximation
            R_small = R[small_angle_mask]
            W_small = 0.5 * (R_small - R_small.transpose(-1, -2))
            # Extract vector from skew-symmetric matrix
            # W = [[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]]
            w_small = torch.stack([
                -W_small[..., 1, 2],
                W_small[..., 0, 2],
                -W_small[..., 0, 1]
            ], dim=-1)
            w[small_angle_mask] = w_small
        
        if torch.any(large_angle_mask):
            # For θ > 0: W = (θ/(2 sin θ)) (R - R^T)
            R_large = R[large_angle_mask]
            theta_large = theta[large_angle_mask]
            
            # Avoid division by zero for θ near π
            sin_theta = torch.sin(theta_large)
            
            # Handle near-π case
            near_pi_mask = torch.abs(sin_theta) < eps
            if torch.any(near_pi_mask):
                # For θ = π: special case, R has eigenvalue 1 with eigenvector v
                # W = ±π * [v] where [v] is skew-symmetric form
                R_near_pi = R_large[near_pi_mask]
                # Find eigenvector with eigenvalue 1
                eigvals, eigvecs = torch.linalg.eigh(R_near_pi)
                # Find index where eigenvalue is close to 1
                idx = torch.argmin(torch.abs(eigvals - 1.0), dim=-1)
                v = eigvecs[torch.arange(eigvecs.shape[0]), :, idx]
                # Create skew-symmetric form
                W_near_pi = torch.zeros_like(R_near_pi)
                W_near_pi[..., 0, 1] = -v[..., 2]
                W_near_pi[..., 0, 2] = v[..., 1]
                W_near_pi[..., 1, 0] = v[..., 2]
                W_near_pi[..., 1, 2] = -v[..., 0]
                W_near_pi[..., 2, 0] = -v[..., 1]
                W_near_pi[..., 2, 1] = v[..., 0]
                
                # Extract w from W
                w_near_pi = torch.stack([
                    -W_near_pi[..., 1, 2],
                    W_near_pi[..., 0, 2],
                    -W_near_pi[..., 0, 1]
                ], dim=-1) * torch.pi
                
                # Update output for near-π cases
                large_idx = torch.where(large_angle_mask)[0][near_pi_mask]
                w[large_idx] = w_near_pi
            
            # Handle regular large angles (θ ∈ (0,π), θ ≠ π)
            regular_mask = ~near_pi_mask
            if torch.any(regular_mask):
                R_regular = R_large[regular_mask]
                theta_regular = theta_large[regular_mask]
                sin_theta_regular = torch.sin(theta_regular)
                
                # Compute W using Eq. (8): W = (θ/(2 sin θ)) (R - R^T)
                factor = theta_regular.unsqueeze(-1).unsqueeze(-1) / (2.0 * sin_theta_regular.unsqueeze(-1).unsqueeze(-1))
                W_regular = factor * (R_regular - R_regular.transpose(-1, -2))
                
                # Extract w from W
                w_regular = torch.stack([
                    -W_regular[..., 1, 2],
                    W_regular[..., 0, 2],
                    -W_regular[..., 0, 1]
                ], dim=-1)
                
                # Update output for regular cases
                large_idx = torch.where(large_angle_mask)[0][regular_mask]
                w[large_idx] = w_regular
        
        return w
    
    @staticmethod
    def spectral_decomposition(tensor):
        """
        Perform spectral decomposition of symmetric 3×3 tensor.
        
        Returns eigenvalues and eigenvectors.
        
        Parameters
        ----------
        tensor : torch.Tensor (..., 3, 3)
            Symmetric tensor
            
        Returns
        -------
        eigvals : torch.Tensor (..., 3)
            Eigenvalues in ascending order
        eigvecs : torch.Tensor (..., 3, 3)
            Eigenvectors as columns
        """
        # Ensure symmetry
        tensor_sym = 0.5 * (tensor + tensor.transpose(-1, -2))
        
        # Eigen decomposition (eigh for symmetric matrices)
        eigvals, eigvecs = torch.linalg.eigh(tensor_sym)
        
        return eigvals, eigvecs
    
    @staticmethod
    def scaled_lie_loss(sig_pred, sig_true, scaling_factors=None):
        """
        Compute Lie loss between predicted and true stress tensors.
        
        Implements Eq. (17) from the paper with scaling.
        
        Parameters
        ----------
        sig_pred : torch.Tensor (..., 6) or (..., 3, 3)
            Predicted stress in Voigt or full tensor form
        sig_true : torch.Tensor (..., 6) or (..., 3, 3)
            True stress in same form
        scaling_factors : dict, optional
            Scaling factors for eigenvalues and w components
            
        Returns
        -------
        torch.Tensor
            Loss value
        """
        # Convert to 3x3 tensors if needed
        if sig_pred.shape[-1] == 6:
            sig_pred_3x3 = TensorUtils.voigt6_to_tensor3x3(sig_pred)
            sig_true_3x3 = TensorUtils.voigt6_to_tensor3x3(sig_true)
        else:
            sig_pred_3x3 = sig_pred
            sig_true_3x3 = sig_true
        
        # Spectral decomposition
        eigvals_pred, eigvecs_pred = TensorUtils.spectral_decomposition(sig_pred_3x3)
        eigvals_true, eigvecs_true = TensorUtils.spectral_decomposition(sig_true_3x3)
        
        # Get rotation matrices from eigenvectors
        # Ensure right-handed coordinate system
        R_pred = eigvecs_pred
        R_true = eigvecs_true
        
        # Ensure det(R) = 1 (right-handed)
        det_pred = torch.linalg.det(R_pred)
        det_true = torch.linalg.det(R_true)
        
        # Flip sign of first eigenvector if det = -1
        if torch.any(det_pred < 0):
            mask = det_pred < 0
            if R_pred.dim() == 2:
                R_pred[mask, :, 0] = -R_pred[mask, :, 0]
            else:
                R_pred[mask, :, 0] = -R_pred[mask, :, 0]
        
        if torch.any(det_true < 0):
            mask = det_true < 0
            if R_true.dim() == 2:
                R_true[mask, :, 0] = -R_true[mask, :, 0]
            else:
                R_true[mask, :, 0] = -R_true[mask, :, 0]
        
        # Convert to Lie algebra
        w_pred = TensorUtils.rotation_to_lie_algebra(R_pred)
        w_true = TensorUtils.rotation_to_lie_algebra(R_true)
        
        # Apply scaling if provided (for eigenvalues and w components)
        if scaling_factors is not None:
            # Scale eigenvalues
            if 'eigval_min' in scaling_factors and 'eigval_max' in scaling_factors:
                eig_min = torch.tensor(scaling_factors['eigval_min'], device=eigvals_pred.device)
                eig_max = torch.tensor(scaling_factors['eigval_max'], device=eigvals_pred.device)
                eig_range = eig_max - eig_min
                eig_range[eig_range == 0] = 1.0  # Avoid division by zero
                
                eigvals_pred_scaled = (eigvals_pred - eig_min) / eig_range
                eigvals_true_scaled = (eigvals_true - eig_min) / eig_range
            else:
                eigvals_pred_scaled = eigvals_pred
                eigvals_true_scaled = eigvals_true
            
            # Scale w components
            if 'w_min' in scaling_factors and 'w_max' in scaling_factors:
                w_min = torch.tensor(scaling_factors['w_min'], device=w_pred.device)
                w_max = torch.tensor(scaling_factors['w_max'], device=w_pred.device)
                w_range = w_max - w_min
                w_range[w_range == 0] = 1.0
                
                w_pred_scaled = (w_pred - w_min) / w_range
                w_true_scaled = (w_true - w_min) / w_range
            else:
                # Auto-scale based on data
                w_pred_scaled = (w_pred - w_pred.min()) / (w_pred.max() - w_pred.min())
                w_true_scaled = (w_true - w_true.min()) / (w_true.max() - w_true.min())
        else:
            eigvals_pred_scaled = eigvals_pred
            eigvals_true_scaled = eigvals_true
            w_pred_scaled = w_pred
            w_true_scaled = w_true
        
        # Compute MSE for eigenvalues (Eq. 15: D_σ^ML)
        eigval_loss = F.mse_loss(eigvals_pred_scaled, eigvals_true_scaled)
        
        # Compute MSE for Lie algebra components (Eq. 17: Φ_i)
        w_loss = F.mse_loss(w_pred_scaled, w_true_scaled)
        
        # Combined loss (Eq. 15: D_{σR}^ML = 0.5*(D_σ^ML + D_R^ML))
        combined_loss = 0.5 * (eigval_loss + w_loss)
        
        return {
            'loss': combined_loss,
            'eigval_loss': eigval_loss,
            'w_loss': w_loss,
            'w_pred': w_pred,
            'w_true': w_true,
            'eigvals_pred': eigvals_pred,
            'eigvals_true': eigvals_true
        }
