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
        epsp: (T,6) or (1,T,6)
        deps: (T,6) or (1,T,6)
        sig:  (T,6) or (1,T,6)

    For each central time t, we construct:

        history (encoder input):
            eps_hist:  eps[t-L+1 : t+S]          -> shape (L+S, 6)
            epsp_hist: epsp[t-L+1 : t+S]         -> shape (L+S, 6)

        future (decoder input + targets), with overlap S:
            deps_future: deps[t+1 : t+1+H+S]     -> shape (H+S, 6)
            sig_y:       sig[t+1 : t+1+H+S]      -> shape (H+S, 6)
            epsp_y:      epsp[t+1 : t+1+H+S]     -> shape (H+S, 6)

    No train/dev/test splitting. All windows are used for training.

    Parameters
    ----------
    npz_path : str
        Path to NPZ file.
    L : int
        Base input length (history length you conceptually care about).
    H : int
        Base prediction horizon (length you conceptually care about).
    S : int
        Extra overlap length.
    """

    def __init__(self, npz_path, L, H, S=0):
        super().__init__()

        data = np.load(npz_path)

        eps  = data["eps"]
        epsp = data["epsp"]
        deps = data["deps"]
        sig  = data["sig"]

        # Allow both (T,6) and (1,T,6)
        if eps.ndim == 2:
            eps  = eps[None, ...]
            epsp = epsp[None, ...]
            deps = deps[None, ...]
            sig  = sig[None, ...]

        # Now shapes are (1, T, 6)
        eps_i  = eps[0]
        epsp_i = epsp[0]
        deps_i = deps[0]
        sig_i  = sig[0]

        T = eps_i.shape[0]
        self.L = L
        self.H = H
        self.S = S
        self.L_total = L + S
        self.H_total = H + S

        xs_eps, xs_epsp, xs_deps = [], [], []
        xs_deps_future = []
        ys_sig, ys_epsp = [], []
        t0_list = []

        # Valid t must satisfy:
        #   history start  : t - L + 1 >= 0
        #   future end     : t + (H + S) <= T - 1
        #   => t >= L - 1
        #   => t <= T - (H + S) - 1
        #
        # In Python's range:
        #   for t in range(L - 1, T - (H + S)):
        for t in range(self.L - 1, T - self.H_total):
            # history indices [t-L+1 ... t+S] (inclusive)
            # Python slice: start = t-L+1, end = (t+S) + 1
            h_start = t - self.L + 1
            h_end   = t + self.S + 1              # exclusive
            h_slice = slice(h_start, h_end)       # length = L + S = L_total

            # future indices [t+1 ... t+1+H+S) (exclusive)
            f_start = t + 1
            f_end   = t + 1 + self.H_total        # exclusive
            f_slice = slice(f_start, f_end)       # length = H + S = H_total

            # history: encoder inputs
            xs_eps.append(eps_i[h_slice])         # (L_total, 6)
            xs_epsp.append(epsp_i[h_slice])       # (L_total, 6)

            # future: decoder input + targets
            xs_deps_future.append(deps_i[f_slice])  # (H_total, 6)
            ys_sig.append(sig_i[f_slice])           # (H_total, 6)
            ys_epsp.append(epsp_i[f_slice])         # (H_total, 6)

            t0_list.append(t)

        # Stack and convert to tensors
        self.eps         = torch.tensor(np.stack(xs_eps),          dtype=torch.float32)
        self.epsp        = torch.tensor(np.stack(xs_epsp),         dtype=torch.float32)
        self.deps_future = torch.tensor(np.stack(xs_deps_future),  dtype=torch.float32)
        self.sig_y       = torch.tensor(np.stack(ys_sig),          dtype=torch.float32)
        self.epsp_y      = torch.tensor(np.stack(ys_epsp),         dtype=torch.float32)
        self.t0          = torch.tensor(t0_list,                   dtype=torch.long)

    def __len__(self):
        return self.eps.shape[0]

    def __getitem__(self, idx):
        return (
            self.eps[idx],         # (L_total, 6)  history eps
            self.epsp[idx],        # (L_total, 6)  history epsp
            self.deps_future[idx], # (H_total, 6)  future deps
            self.sig_y[idx],       # (H_total, 6)  future sig
            self.epsp_y[idx],      # (H_total, 6)  future epsp
            self.t0[idx],          # scalar (central index t)
        )
