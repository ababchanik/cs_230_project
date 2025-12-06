import numpy as np
import pandas as pd 
import torch

# Material loader
def load_material_params(csv_path="material_parameters.csv"):
    """
    Load material parameters from a CSV file and return them as a dict.
    Assumes the CSV has columns "Parameter" and "Value".
    """
    df = pd.read_csv(csv_path)
    params = dict(zip(df["Parameter"], df["Value"]))
    return params


# Rank-four TI elasticity constructor CCe
def elasticity_tensor(lame_lambda, lame_mu_T, lame_mu_L, a, b, theta_deg):
    """
    Build transversely isotropic rank-four elastic tensor CCe_ijkl (3x3x3x3).
    """
    I = np.eye(3)
    IxI = np.einsum('ij,kl->ijkl', I, I)
    II  = 0.5 * (np.einsum('ik,jl->ijkl', I, I) +
                 np.einsum('il,jk->ijkl', I, I))

    theta = np.deg2rad(theta_deg)
    n = np.array([np.cos(theta), np.sin(theta), 0.0])
    m = np.outer(n, n)

    mxI = np.einsum('ij,kl->ijkl', m, I)
    Ixm = np.einsum('ij,kl->ijkl', I, m)
    mxm = np.einsum('ij,kl->ijkl', m, m)

    I_oplus_m  = np.einsum('jl,ik->ijkl', I, m)
    m_oplus_I  = np.einsum('jl,ik->ijkl', m, I)
    I_ominus_m = np.einsum('il,jk->ijkl', I, m)
    m_ominus_I = np.einsum('il,jk->ijkl', m, I)
    m_oplus_m  = np.einsum('jl,ik->ijkl', m, m)
    m_ominus_m = np.einsum('il,jk->ijkl', m, m)

    CCe = (
        lame_lambda * IxI + 2.0 * lame_mu_T * II
        + a * (mxI + Ixm) + b * mxm
        + (lame_mu_L - lame_mu_T) * (
            I_oplus_m + m_oplus_I + I_ominus_m + m_ominus_I
        )
    )

    return CCe


# Rank-four elasticity C_ijkl → 6x6 Voigt matrix
def voigt_mapping():
    """
    Return mapping a -> (i,j) with 0-based indices
    for Voigt ordering: [11, 22, 33, 23, 13, 12].
    """
    return {
        0: (0, 0),
        1: (1, 1),
        2: (2, 2),
        3: (1, 2),
        4: (0, 2),
        5: (0, 1),
    }


def rank4_to_voigt6x6(C4):
    """
    Convert rank-four tensor C_ijkl (3x3x3x3) to a 6x6 Voigt matrix C_voigt
    for Voigt ordering [11, 22, 33, 23, 13, 12].

    This is a plain tensor-Voigt mapping with NO factor-2 scaling
    for shear; the factor 2 enters in the double contraction instead.
    """
    C4 = np.asarray(C4, dtype=float)
    if C4.shape != (3, 3, 3, 3):
        raise ValueError(f"C4 must have shape (3,3,3,3); got {C4.shape}")

    # Check minor symmetry
    if not (np.allclose(C4, C4.transpose(1, 0, 2, 3)) and
            np.allclose(C4, C4.transpose(0, 1, 3, 2))):
        raise ValueError("C4 must be minor-symmetric in (i,j) and (k,l).")

    C_voigt = np.zeros((6, 6), dtype=float)
    vmap = voigt_mapping()

    for a, (i, j) in vmap.items():
        for b, (k, l) in vmap.items():
            C_voigt[a, b] = C4[i, j, k, l]

    return C_voigt

# Assemble elastic tensor from CSV params and convert to Voigt
def assemble_elastic_tensor_voigt(csv_path="material_parameters.csv"):
    """
    Read material parameters, build TI rank-four elasticity, and convert
    to a 6x6 Voigt matrix CCe_voigt.

    Expects the CSV to contain at least:
      lame_lambda, lame_mu_T, lame_mu_L, a, b, theta_deg
    """
    params = load_material_params(csv_path)

    lame_lambda = params["lame_lambda"]
    lame_mu_T   = params["lame_mu_T"]
    lame_mu_L   = params["lame_mu_L"]
    a           = params["a"]
    b           = params["b"]
    theta_deg   = params["bedding_plane_orientation"]

    CCe = elasticity_tensor(lame_lambda, lame_mu_T, lame_mu_L, a, b, theta_deg)
    CCe_voigt = rank4_to_voigt6x6(CCe)

    return CCe_voigt


# Elastic update in Voigt form (double contraction with factor 2 on shear)
def elastic_update(strain_voigt: torch.Tensor,
                   CCe_voigt: torch.Tensor) -> torch.Tensor:
    """
    Given strain in Voigt form, return elastic stress in Voigt form.

    Voigt convention:
        eps[..., :] = [e11, e22, e33, e23, e13, e12]
        sig[..., :] = [s11, s22, s33, s23, s13, s12]

    We implement sigma_i = sum_j C_ij * eps_j with an explicit factor 2
    on shear columns j = 3,4,5 (23, 13, 12) to account for the double
    contraction on symmetric tensors.

    Parameters
    ----------
    strain_voigt : torch.Tensor
        Shape (..., 6), already on the correct device.
    CCe_voigt : torch.Tensor
        Shape (6, 6), elastic stiffness in Voigt form, same device / dtype.

    Returns
    -------
    stress_voigt : torch.Tensor
        Same shape as strain_voigt, same device / dtype.
    """
    if not torch.is_tensor(strain_voigt) or not torch.is_tensor(CCe_voigt):
        raise TypeError("elastic_update expects torch.Tensors for both arguments.")

    eps = strain_voigt
    C = CCe_voigt

    if C.shape != (6, 6):
        raise ValueError(f"CCe_voigt must have shape (6,6), got {C.shape}")
    if eps.shape[-1] != 6:
        raise ValueError(f"strain_voigt last dim must be 6, got {eps.shape}")

    # Factor 2 on shear strain components: j = 3,4,5
    shear_factor = torch.tensor([1., 1., 1., 2., 2., 2.],
                                dtype=eps.dtype,
                                device=eps.device)        # (6,)

    eps_scaled = eps * shear_factor  # (..., 6)

    # Double contraction in Voigt form:
    sig = torch.einsum("...j,ij->...i", eps_scaled, C)

    return sig
