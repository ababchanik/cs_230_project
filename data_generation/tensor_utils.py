"""
tensor_utils.py
---------------
Contains general-purpose tensor algebra and Voigt conversion utilities
for use in MOOSE-style constitutive update algorithms.
"""

import numpy as np

# ------------------ small utilities ------------------

def outer_product(A, B):
    """Fourth-order tensor from dyadic (outer) product of two rank-2 tensors."""
    return np.einsum('ij,kl->ijkl', A, B)


def symm_outer_product(A, B):
    """Symmetric fourth-order tensor: 0.5*(A⊗B + A⊗B^T)."""
    return 0.5 * (np.einsum('ik,jl->ijkl', A, B) + np.einsum('il,jk->ijkl', A, B))


def double_contraction_r2t_r2t(A, B):
    """Double contraction between two rank-2 tensors: A:B."""
    return np.einsum('ij,ij', A, B)


def double_dot_r4t_r2t(A4, B2):
    """Double dot product between a rank-4 and rank-2 tensor: (A4:B2)."""
    return np.einsum('ijkl,kl->ij', A4, B2)


# ------------------ helper functions for Voigt conversions ------------------

def voigt_from_rank4(A, shear_x2_row=False, shear_x2_col=False):
    """Convert a rank-4 tensor to 6×6 Voigt matrix form.
    
    If `shear_x2_row` or `shear_x2_col` is True, the corresponding
    shear rows/columns (Voigt indices 3–5) are multiplied by 2,
    matching engineering shear strain conventions.
    """
    if not is_minor_symmetric(A):
        raise ValueError("Only minor symmetric rank-four tensors are convertible to Voigt form.")

    voigt_map = [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]
    V = np.zeros((6,6))
    for I, (i, j) in enumerate(voigt_map):
        for J, (k, l) in enumerate(voigt_map):
            V[I, J] = (
                (1.0 + float(shear_x2_row * (I >= 3)))
                * (1.0 + float(shear_x2_col * (J >= 3)))
                * A[i, j, k, l]
            )
    return V

def voigt_from_rank2(T, shear_term_x2=False):
    """
    Convert a 3×3 symmetric rank-2 tensor to a 6-component Voigt vector.
    
    If `shear_term_x2` is True, the shear components (yz, zx, xy)
    are multiplied by 2. This matches the engineering strain convention.
    """
    if not np.array_equal(T, T.T):
        raise ValueError("Only symmetric rank-two tensors are convertible to Voigt form.")

    V = np.array([
        T[0, 0],
        T[1, 1],
        T[2, 2],
        T[1, 2],
        T[0, 2],
        T[0, 1],
    ], dtype=float)

    if shear_term_x2:
        V[3:] *= 2.0   # multiply yz, zx, xy by 2

    return V

def rank2_from_voigt(v, shear_div2=False):
    """
    Convert a 6-component Voigt vector back to a 3×3 symmetric tensor.
    
    If `shear_div2` is True, the shear components (yz, zx, xy)
    are divided by 2 — reversing the engineering shear convention.
    """
    v = np.array(v, dtype=float).copy()

    if shear_div2:
        v[3:] /= 2.0  # divide yz, zx, xy by 2

    return np.array([
        [v[0], v[5], v[4]],
        [v[5], v[1], v[3]],
        [v[4], v[3], v[2]],
    ])

def is_minor_symmetric(r4t, tol=1e-12):
    """
    Check minor symmetry of a rank-4 tensor:
        A_ijkl = A_jikl  and  A_ijkl = A_ijlk
    Returns True if both symmetries hold within given tolerance.
    """
    r4t = np.asarray(r4t)
    assert r4t.shape == (3, 3, 3, 3), f"Expected (3,3,3,3), got {r4t.shape}"

    # Check both minor symmetries using elementwise comparison
    cond_ij = np.allclose(r4t, np.swapaxes(r4t, 0, 1), atol=tol)
    cond_kl = np.allclose(r4t, np.swapaxes(r4t, 2, 3), atol=tol)

    return bool(cond_ij and cond_kl)


if __name__ == "__main__":
    I = np.eye(3)
    A = np.random.rand(3,3)
    B = np.random.rand(3,3)
    print("outer_product shape:", outer_product(A, B).shape)
    print("symm_outer_product shape:", symm_outer_product(A, B).shape)
    print("voigt_from_rank4 shape:", voigt_from_rank4(outer_product(I, I)).shape)
    print("rank2_from_voigt:", rank2_from_voigt(voigt_from_rank2(I)))
