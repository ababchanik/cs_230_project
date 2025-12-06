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
