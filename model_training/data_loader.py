
# import torch
# from torch.utils.data import Dataset
# import numpy as np

# class TrajWindowDataset(Dataset):
#     """
#     Multi-specimen dataset matching the windowing logic of SingleRelaxationWindowDataset,
#     but keeping the same outputs expected by main.py and the model.
#     """

#     def __init__(self, npz_path, L=10, H=10, S=0, split='train'):
#         if split not in {'train', 'dev', 'eval'}:
#             raise ValueError("split must be one of {'train','dev','eval'}")

#         data = np.load(npz_path)
#         needed = ['eps', 'epse', 'deps', 'sig', 'split']
#         if not all(k in data.files for k in needed):
#             raise RuntimeError(f"{npz_path} must contain {needed}")

#         self.eps   = data['eps']      # (m, T, 6)
#         self.epse  = data['epse']     # (m, T, 6)
#         self.deps  = data['deps']     # (m, T, 6)
#         self.sig   = data['sig']      # (m, T, 6)
#         labels     = data['split']    # (m,)

#         self.L = L
#         self.H = H
#         self.S = S
#         self.L_total = L + S
#         self.H_total = H + S

#         m, T, _ = self.sig.shape

#         # choose specimens for split
#         idx_specs = np.where(labels == split)[0]
#         if len(idx_specs) == 0:
#             raise RuntimeError(f"No specimens labeled '{split}'")

#         # valid t matching single-trajectory logic:
#         # t ∈ [L−1 , T − (H+S) − 1]
#         self.idxs = []
#         for i in idx_specs:
#             for t in range(L - 1, T - (H + S)):
#                 self.idxs.append((i, t))

#     def __len__(self):
#         return len(self.idxs)

#     def __getitem__(self, idx):
#         i_spec, t = self.idxs[idx]

#         # history:  [t-L+1 ... t+S]
#         h_start = t - self.L + 1
#         h_end   = t + self.S + 1

#         # future:   [t+1 ... t+1+H+S]
#         f_start = t + 1
#         f_end   = t + 1 + self.H_total

#         eps_hist  = torch.from_numpy(self.eps [i_spec, h_start:h_end]).float()
#         epse_hist = torch.from_numpy(self.epse[i_spec, h_start:h_end]).float()
#         deps_hist = torch.from_numpy(self.deps[i_spec, h_start:h_end]).float()

#         sig_y  = torch.from_numpy(self.sig [i_spec, f_start:f_end]).float()
#         epse_y = torch.from_numpy(self.epse[i_spec, f_start:f_end]).float()

#         # ⚠️ IMPORTANT:
#         # main.py and model expect only these 5 outputs
#         return eps_hist, epse_hist, deps_hist, sig_y, epse_y


### Hemiao works
import torch
from torch.utils.data import Dataset
import numpy as np

class TrajWindowDataset(Dataset):
    """
    Expected .npz contents
    ----------------------
      eps:   (m, T, 6)   total strain
      epsp:  (m, T, 6)   plastic strain
      deps:  (m, T, 6)   total strain increments
      sig:   (m, T, 6)   Cauchy stress
      split: (m,)        strings in {'train', 'dev', 'eval'}

    Window definition (per specimen i, central time index t)
    --------------------------------------------------------
    Hyperparameters:
      L : base history length
      H : base prediction horizon
      S : extra overlap length (0 <= S <= min(L, H))

    For each valid central index t, we build:

      History (encoder input, length L+S):
        eps_hist  = eps[i,  t-L+1 : t+S+1]    -> (L+S, 6)
        epsp_hist = epsp[i, t-L+1 : t+S+1]    -> (L+S, 6)

      Future (decoder input + targets, length H+S):
        deps_future = deps[i,  t+1 : t+1+H+S] -> (H+S, 6)
        sig_y       = sig[i,   t+1 : t+1+H+S] -> (H+S, 6)
        epsp_y      = epsp[i,  t+1 : t+1+H+S] -> (H+S, 6)
    """

    def __init__(self, npz_path, L=4, H=4, S=1, split='train'):
        if split not in {'train', 'dev', 'eval'}:
            raise ValueError("split must be one of {'train','dev','eval'}")

        if S < 0 or S > min(L, H):
            raise ValueError(f"S must satisfy 0 <= S <= min(L, H). Got L={L}, H={H}, S={S}")

        data = np.load(npz_path)
        required = ['eps', 'epsp', 'deps', 'sig', 'split']
        if not all(k in data.files for k in required):
            raise RuntimeError(f"{npz_path} must contain {required}")

        self.eps   = data['eps']      # (m, T, 6)
        self.epsp  = data['epsp']     # (m, T, 6)
        self.deps  = data['deps']     # (m, T, 6)
        self.sig   = data['sig']      # (m, T, 6)
        labels     = data['split']    # (m,)

        m, T, D = self.sig.shape
        if D != 6:
            raise RuntimeError(f"Expected Voigt-6 last dimension, got {D}")

        self.L, self.H, self.S = L, H, S

        # select specimens with the requested label
        idx_specs = np.where(labels == split)[0]
        if idx_specs.size == 0:
            raise RuntimeError(f"No specimens labeled '{split}' found in {npz_path}")

        # build (specimen, central_time t) indices
        idxs = []
        for i in idx_specs:
            T_i = self.sig[i].shape[0]  # total steps in i, in case specimens have different lengths

            # t in [L-1, T_i - (H+S) - 1]
            for t in range(L - 1, T_i - (H + S)):
                idxs.append((i, t))

        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        i_spec, t = self.idxs[i]
        L, H, S = self.L, self.H, self.S

        # History indices: [t-L+1 ... t+S] inclusive
        h_start = t - L + 1
        h_end   = t + S + 1        # exclusive

        # Future indices: [t+1 ... t+1+H+S) exclusive
        f_start = t + 1
        f_end   = t + 1 + H + S

        eps_hist     = torch.from_numpy(self.eps[i_spec,  h_start:h_end]).float()   # (L+S, 6)
        epsp_hist    = torch.from_numpy(self.epsp[i_spec, h_start:h_end]).float()   # (L+S, 6)
        deps_future  = torch.from_numpy(self.deps[i_spec, f_start:f_end]).float()   # (H+S, 6)
        sig_y        = torch.from_numpy(self.sig[i_spec,  f_start:f_end]).float()   # (H+S, 6)
        epsp_y       = torch.from_numpy(self.epsp[i_spec, f_start:f_end]).float()   # (H+S, 6)

        # Sanity check
        assert eps_hist.shape[-1] == 6 and sig_y.shape[-1] == 6, \
            f"Expected last dim=6, got eps_hist {eps_hist.shape}, sig_y {sig_y.shape}"

        t0       = torch.tensor(t, dtype=torch.long)       # central time index
        traj_idx = torch.tensor(i_spec, dtype=torch.long)  # specimen index

        # Order matches SingleRelaxationWindowDataset (+ traj_idx extra)
        return eps_hist, epsp_hist, deps_future, sig_y, epsp_y, t0, traj_idx
