import numpy as np
import torch
from torch.utils.data import Dataset


class SingleRelaxationWindowDataset(Dataset):
    """
    Build (L,H) sliding windows from a *single* stress–relaxation path stored in:

        data_generation/test_stress_relaxation/
            stress_relaxation_strain_rate_dt_1e-1.npz

    Expected arrays inside the .npz:
        eps:  (T,6) or (1,T,6)
        epse: (T,6) or (1,T,6)
        deps: (T,6) or (1,T,6)
        sig:  (T,6) or (1,T,6)

    No train/dev/test splitting. All windows are used for training.
    """

    def __init__(self, npz_path, L, H):
        super().__init__()

        data = np.load(npz_path)

        eps  = data["eps"]
        epse = data["epse"]
        deps = data["deps"]
        sig  = data["sig"]

        # Allow both (T,6) and (1,T,6)
        if eps.ndim == 2:
            eps  = eps[None, ...]
            epse = epse[None, ...]
            deps = deps[None, ...]
            sig  = sig[None, ...]

        # Now shapes are (1, T, 6)
        eps_i  = eps[0]
        epse_i = epse[0]
        deps_i = deps[0]
        sig_i  = sig[0]

        T = eps_i.shape[0]
        self.L, self.H = L, H

        xs_eps, xs_epse, xs_deps = [], [], []
        ys_sig, ys_epse = [], []
        t0_list = []

        # Build windows:  [t-L+1 ... t] → [t+1 ... t+H]
        for t in range(L - 1, T - H):
            x = slice(t - L + 1, t + 1)
            y = slice(t + 1,     t + 1 + H)

            xs_eps.append(eps_i[x])
            xs_epse.append(epse_i[x])
            xs_deps.append(deps_i[x])
            ys_sig.append(sig_i[y])
            ys_epse.append(epse_i[y])
            t0_list.append(t)

        self.eps   = torch.tensor(np.stack(xs_eps),   dtype=torch.float32)
        self.epse  = torch.tensor(np.stack(xs_epse),  dtype=torch.float32)
        self.deps  = torch.tensor(np.stack(xs_deps),  dtype=torch.float32)
        self.sig_y = torch.tensor(np.stack(ys_sig),   dtype=torch.float32)
        self.epse_y= torch.tensor(np.stack(ys_epse),  dtype=torch.float32)
        self.t0    = torch.tensor(t0_list, dtype=torch.long)

    def __len__(self):
        return self.eps.shape[0]

    def __getitem__(self, idx):
        return (
            self.eps[idx],     # (L,6)
            self.epse[idx],    # (L,6)
            self.deps[idx],    # (L,6)
            self.sig_y[idx],   # (H,6)
            self.epse_y[idx],  # (H,6)
            self.t0[idx],
        )
