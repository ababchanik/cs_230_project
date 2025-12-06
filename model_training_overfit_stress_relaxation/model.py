import torch
import torch.nn as nn
import torch.nn.functional as F

# Use shared tensor utilities
from tensor_utils import TensorUtils  # provides to_voigt6, from_voigt6, invariants_voigt6

# ---------- Model ----------
class Encoder(nn.Module):
    def __init__(self, in_dim, hid=128, layers=1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid, layers, batch_first=True)
    def forward(self, x):
        # x: (batch, L, in_dim) or (L, in_dim)
        if x.dim() == 2:                 # (L, in_dim) → (1, L, in_dim)
            x = x.unsqueeze(0)

        _, h = self.gru(x)               # h: (layers, batch, hid)
        return h[-1]                     # (batch, hid) from last layer

class Decoder(nn.Module):
    def __init__(self, hid, H, out_per_step):
        super().__init__()
        self.H = H
        self.net = nn.Sequential(
            nn.Linear(hid, 2*hid), nn.ReLU(),
            nn.Linear(2*hid, H*out_per_step)
        )
    def forward(self, h):
        # h: (m, hid)
        y = self.net(h)                  # (m, H*out_per_step)
        m = h.size(0)
        return y.view(m, self.H, -1)     # (m, H, out_per_step)

class SeqModel(nn.Module):
    """
    Per-step inputs (features): total strain eps(6) + elastic strain epse(6) + increment deps(6) = 18
    Per-step outputs (forecast): stress sig(6) + elastic strain epse(6) = 12
    """
    def __init__(self, H=10, hid=128):
        super().__init__()
        self.H = H
        self.in_dim = 18
        self.out_dim = 12
        self.enc = Encoder(self.in_dim, hid=hid, layers=1)
        self.dec = Decoder(hid, H, self.out_dim)

    def _step_features(self, eps, epse, deps):
        # eps, epse, deps: (m, L, 6)
        return torch.cat([eps, epse, deps], dim=-1)  # (B, L, 18)

    def forward(self, eps_seq, epse_seq, deps_seq):
        # eps_seq, epse_seq, deps_seq: (B, L, 6)
        # X from (t - L + 1) to (t)
        X = self._step_features(eps_seq, epse_seq, deps_seq)  # (B, L, 18)
        # h at (t)
        h = self.enc(X)                                       # (B, hid)
        # Y from (t + 1) to (t + H)
        Y = self.dec(h)                                       # (B, H, 12)
        sig, epse = Y[..., :6], Y[..., 6:12]                  # (B, H, 6) each
        # Create new epse_seq based on predicted sig, epse
        # Run prediction again

        return sig, epse
