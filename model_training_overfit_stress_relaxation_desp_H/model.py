import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, hid, H, out_per_step, deps_dim=6):
        """
        hid        : latent dim from encoder
        H          : prediction horizon (number of future steps actually used)
        out_per_step : dim of [sig, epsp] = 12
        deps_dim   : dim of deps per step (6)
        """
        super().__init__()
        self.H = H
        self.fc1 = nn.Linear(hid + deps_dim, 2 * hid)
        self.fc2 = nn.Linear(2 * hid, out_per_step)

    def forward(self, h, deps_future):
        """
        h           : (B, hid)
        deps_future : (B, H, deps_dim)   # deps for t+1..t+H
        returns     : (B, H, out_per_step)
        """
        B, H, deps_dim = deps_future.shape
        assert H == self.H, f"Expected H={self.H}, got {H}"

        # Repeat latent h across H future steps
        h_rep = h.unsqueeze(1).expand(-1, H, -1)      # (B, H, hid)

        # Per-step features [h_t, deps_t+k], k=1,...,H
        z = torch.cat([h_rep, deps_future], dim=-1)   # (B, H, hid+deps_dim)

        z = F.relu(self.fc1(z))                       # (B, H, 2*hid)
        y = self.fc2(z)                               # (B, H, out_per_step)

        return y                                      # (B, H, out_per_step)

class SeqModel(nn.Module):
    """
    Per-step inputs to the ENCODER (history/features):
        total strain eps(6) + plastic strain epsp(6) + history stress sig_hist(6) = 18

    Per-step outputs from the DECODER (future/forecast):
        stress sig(6) + plastic strain epsp(6) = 12

    Decoder additionally conditions on future deps(t+1..t+H) with shape (B, H, 6).
    """
    def __init__(self, H=10, hid=128):
        super().__init__()
        self.H = H
        self.in_dim = 18        # 6 (eps) + 6 (epsp) + 6 (sig_hist)
        self.out_dim = 12       # 6 (sig) + 6 (epsp)
        self.enc = Encoder(self.in_dim, hid=hid, layers=1)
        self.dec = Decoder(hid, H, self.out_dim, deps_dim=6)

    def _step_features(self, eps, epsp, sig_hist):
        """
        eps, epsp, sig_hist: (B, L, 6)
        returns X          : (B, L, 18)
        """
        return torch.cat([eps, epsp, sig_hist], dim=-1)  # (B, L, 18)

    def forward(self, eps_hist, epsp_hist, sig_hist, deps_future):
        """
        Encode history [eps, epsp, sig_hist] over L steps,
        then decode H future steps conditioned on deps_future.
        """
        # Build encoder sequence over history
        X = self._step_features(eps_hist, epsp_hist, sig_hist)  # (B, L, 18)

        # Encode history to latent at time t
        h = self.enc(X)                                         # (B, hid)

        # Decode conditioned on future deps(t+1..t+H)
        Y = self.dec(h, deps_future)                            # (B, H, 12)

        sig, epsp = Y[..., :6], Y[..., 6:12]                    # (B, H, 6) each
        return sig, epsp
