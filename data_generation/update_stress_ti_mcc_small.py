import numpy as np
import pandas as pd

from tensor_utils import (
    outer_product,
    symm_outer_product,
    double_contraction_r2t_r2t,
    double_dot_r4t_r2t,
    voigt_from_rank4,
    voigt_from_rank2,
    rank2_from_voigt,
    is_minor_symmetric
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

# ------------------ MCC helper functions ------------------
def yield_function(sigma, pc, M, AA_star, aa_star):
    """
    Yield function f
    """
    term1 = (1.0 / (2.0 * M**2)) * np.einsum('ij,ijkl,kl->', sigma, AA_star, sigma)
    p_star = np.einsum('ij,ij->', aa_star, sigma)
    term2 = p_star * (p_star - pc)
    return term1 + term2

def dyield_function_dstress(sigma, pc, M, AA_star, aa_star):
    """
    Gradient of yield function df/dsigma
    """
    term1 = double_dot_r4t_r2t(AA_star, sigma) / (M ** 2)
    term2 = (2.0 * double_contraction_r2t_r2t(aa_star, sigma) - pc) * aa_star
    return term1 + term2

def ddyield_function_ddstress(M, AA_star, aa_star):
    """
    Second derivative (Hessian) of yiled funciton 
    d^2f/dsigma^2

    Returns np.array of size (3, 3, 3, 3)
    """
    return AA_star / (M ** 2) + 2.0 * np.einsum('ij,kl->ijkl', aa_star, aa_star)

# --------residual and local tangent -------
def residual_vector_perzyna(sigma, sigma_trial, pc, M, eta, dt, AA_star, aa_star, CCe):
    """
    Residual vector
    r(sigma) = sigma - sigma_trial + (dt/eta) * CCe : ( f * df/dsigma )

    Returns a 6-vector in Voigt form
    """
    f = yield_function(sigma, pc, M, AA_star, aa_star)
    df = dyield_function_dstress(sigma, pc, M, AA_star, aa_star)  # rank-2
    rho = sigma - sigma_trial + (dt / eta) * double_dot_r4t_r2t(CCe, f * df)
    return voigt_from_rank2(rho)

def local_tangent_perzyna(sigma, dt, eta, pc, M, AA_star, aa_star, CCe):
    """
    Local Newton Jacobian matrix
    dr/dsigma = I6 + voigt( (dt/eta) * Ce : ( f*d^2F/dsigma^2 + (dF/dsigma)⊗(dF/dsigma) ) )

    Returns a 6×6 matrix in Voigt form
    """
    f = yield_function(sigma, pc, M, AA_star, aa_star)
    df = dyield_function_dstress(sigma, pc, M, AA_star, aa_star)
    d2f = ddyield_function_ddstress(M, AA_star, aa_star)

    df_outer_df = outer_product(df, df)
    hessian_plus_outerproduct  = np.einsum('ijkl,klmn->ijmn', CCe, f * d2f + df_outer_df)

    return (dt / eta) * voigt_from_rank4(hessian_plus_outerproduct, shear_x2_col=True) + np.eye(6)

# ------------------ main MOOSE-style function ------------------

def update_cauchy_stress(depsilon, epsilone):
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

    # check yield
    yield_fcn = yield_function(sigma, pc, M, AA_star, aa_star)
    if yield_fcn <= ftol:
    # - - - - - - - - - Elastic - - - - - - - - - 
        return epsilone, sigma, "elastic", \
               np.einsum('ijkl,kl->ij', PP, sigma)

    # - - - - - - - - - Plastic - - - - - - - - - 
    sigma_trial = sigma.copy()
    residual = residual_vector_perzyna(sigma, sigma_trial, pc, M, eta, dt, AA_star, aa_star, CCe)  # shape (6,)
    r_err0 = float(np.linalg.norm(residual))  # initial residual norm
    r_err  = 1.0
    s_err  = r_err0
    iter = 0

    while (r_err > rtol) and (s_err > stol) and (iter < local_newton_max_its):
        iter += 1
        if iter >= local_newton_max_its:
            print(f"r_err = {r_err:.6e}")
            print(f"s_err = {s_err:.6e}")
            raise RuntimeError("Gauss point Newton iteration didn't converge.")

        # Local tangent (6x6) at the current sigma
        local_tangent = local_tangent_perzyna(sigma, dt, eta, pc, M, AA_star, aa_star, CCe)  # shape (6,6)

        # Solve J * Dx = residual
        Dx = np.linalg.solve(local_tangent, residual)    # shape (6,)

        # Update sigma in Voigt form, then convert back to tensor
        sig_voigt = voigt_from_rank2(sigma) - Dx
        sigma = rank2_from_voigt(sig_voigt)      # back to (3x3)

        # Recompute residual at new sigma
        residual = residual_vector_perzyna(sigma, sigma_trial, pc, M, eta, dt, AA_star, aa_star, CCe)
        s_err = float(np.linalg.norm(residual))
        r_err = s_err / (r_err0 + 1e-30)

    # Upon convergence update elastic strain
    CCe_mat = voigt_from_rank4(CCe, shear_x2_col=True)
    sigma_voigt = voigt_from_rank2(sigma)
    eps_e_voigt = np.linalg.solve(CCe_mat, sigma_voigt)
    epsilone = rank2_from_voigt(eps_e_voigt)

    return epsilone, sigma, "plastic", \
           np.einsum('ijkl,kl->ij', PP, sigma)
