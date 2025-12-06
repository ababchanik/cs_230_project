import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tensor_utils import (
    voigt_from_rank2,
    rank2_from_voigt,
)

def load_material_params(csv_path="material_parameters.csv"):
    """
    Load material parameters from a CSV file and return them as a dictionary.
    Also constructs the transversely isotropic microstructure tensor m_tensor.
    """
    # Read CSV and convert to dict
    df = pd.read_csv(csv_path)
    params = dict(zip(df["Parameter"], df["Value"]))

    return params

    import numpy as np

def p_q_from_tensor(sigma):
    """
    Compute mean stress p and deviatoric invariant q from a 3x3 stress tensor.
    """
    sig = np.asarray(sigma)
    # mean stress
    p = np.trace(sig, axis1=-2, axis2=-1) / 3.0          # shape (...)
    # deviatoric part: s = sig - p * I
    I = np.eye(3, dtype=sig.dtype)
    s = sig - p[..., None, None] * I                     # broadcast I
    # s_ij s_ij
    s2 = np.einsum("...ij,...ij->...", s, s)             # shape (...)
    # q = sqrt(3/2 s_ij s_ij)
    q = np.sqrt(1.5 * s2)

    return p, q


# --------special rank-four tensor -------
def ti_r4t(lame_lambda, lame_mu_T, lame_mu_L, a, b, 
           c1, c2, c3,
           theta_deg):
    
    # preapre special r4t
    I = np.eye(3)
    IxI = np.einsum('ij,kl->ijkl', I, I)
    II  = 0.5 * (np.einsum('ik,jl->ijkl', I, I) + np.einsum('il,jk->ijkl', I, I))
    AA = 3. * II - IxI
    aa = (1. / 3.) * I

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
        + (lame_mu_L - lame_mu_T) * (I_oplus_m + m_oplus_I + I_ominus_m + m_ominus_I)
    )

    PP = (
        c1 * II
        + (c2 / 2.0) * (m_oplus_m + m_ominus_m)
        + (c3 / 4.0) * (I_oplus_m + m_oplus_I + I_ominus_m + m_ominus_I)
    )

    AA_star = np.einsum('ijmn,mnpq,pqkl->ijkl', PP, AA, PP)
    aa_star = np.einsum('ijmn,mn->ij', PP, aa)

    return CCe, AA_star, aa_star, PP

# ------------------ main MOOSE-style function ------------------

def compute_invariants(depsilon, epsilone):
    # load material and local Newton paraeters
    params = load_material_params()
    lame_lambda = params["lame_lambda"]
    lame_mu_T   = params["lame_mu_T"]
    lame_mu_L   = params["lame_mu_L"]
    a           = params["a"]
    b           = params["b"]
    pc          = params["initial_preconsolidation_pressure"]
    M           = params["M"]
    eta         = params["eta"]
    dt          = params["dt"]
    c1          = params["c1"]
    c2          = params["c2"]    
    c3          = params["c3"]
    theta_deg   = params["bedding_plane_orientation"]
    rtol        = params["rtol"]
    stol        = params["stol"]
    ftol        = params["ftol"]
    local_newton_max_its = params["local_newton_max_its"]

    CCe, AA_star, aa_star, PP = ti_r4t(
        lame_lambda, lame_mu_T, lame_mu_L, a, b,
        c1, c2, c3,
        theta_deg
    ) 
    
    epsilone += depsilon
    sigma = np.einsum('ijkl,kl->ij', CCe, epsilone)
    sigma_star = np.einsum('ijkl,kl->ij', PP, sigma)

    return p_q_from_tensor(sigma), p_q_from_tensor(sigma_star)
