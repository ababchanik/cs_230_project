import torch
import torch.nn as nn
import torch.nn.functional as F 

# ---------- Encoder ----------
class Encoder(nn.Module):
    def __init__(self, in_dim, hid=128, layers=1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid, layers, batch_first=True)

    def forward(self, x):
        """
        x: (batch, L, in_dim) or (L, in_dim)
        Returns: (batch, hid)
        """
        if x.dim() == 2:        # (L, in_dim) → (1, L, in_dim)
            x = x.unsqueeze(0)

        _, h = self.gru(x)      # h: (layers, batch, hid)
        return h[-1]            # last layer hidden state: (batch, hid)


# ---------- Decoder ----------
class Decoder(nn.Module):
    def __init__(self, hid, H, out_per_step, deps_dim=6):
        """
        hid         : latent dim from encoder
        H           : prediction horizon (number of future steps actually used)
        out_per_step: dim of [sig, epsp] per step = 12
        deps_dim    : dim of deps per step (6)
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
        assert H == self.H, f"Decoder expected H={self.H}, got H={H}"

        # Repeat latent h across H future steps
        h_rep = h.unsqueeze(1).expand(-1, H, -1)      # (B, H, hid)

        # Per-step features [h_t, deps_t+k], k=1,...,H
        z = torch.cat([h_rep, deps_future], dim=-1)   # (B, H, hid+deps_dim)

        z = F.relu(self.fc1(z))                       # (B, H, 2*hid)
        y = self.fc2(z)                               # (B, H, out_per_step)

        return y                                      # (B, H, out_per_step)


# ---------- SeqModel ----------
class SeqModel(nn.Module):
    """
    Multi-trajectory sequence model with the SAME structure as the single-trajectory model.

    Per-step encoder inputs (history features):
        - total strain  eps(6)
        - plastic strain epsp(6)
        - history stress sig_hist(6)   [computed via elastic_update outside the model]

        => 18 features per time step

    Per-step decoder outputs (future forecast):
        - stress sig(6)
        - plastic strain epsp(6)

        => 12 outputs per time step

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
        return torch.cat([eps, epsp, sig_hist], dim=-1)

    def forward(self, eps_hist, epsp_hist, sig_hist, deps_future):
        """
        Encode history [eps, epsp, sig_hist] over L steps,
        then decode H future steps conditioned on deps_future.

        Inputs
        ------
        eps_hist    : (B, L, 6)  total strain history
        epsp_hist   : (B, L, 6)  plastic strain history
        sig_hist    : (B, L, 6)  history stress (typically elastic predictor)
        deps_future : (B, H, 6)  future strain increments (t+1..t+H)

        Outputs
        -------
        sig_pred  : (B, H, 6)  predicted stress over horizon
        epsp_pred : (B, H, 6)  predicted plastic strain over horizon
        """
        # Build encoder sequence over history
        X = self._step_features(eps_hist, epsp_hist, sig_hist)  # (B, L, 18)

        # Encode history to latent at time t
        h = self.enc(X)                                         # (B, hid)

        # Decode conditioned on future deps(t+1..t+H)
        Y = self.dec(h, deps_future)                            # (B, H, 12)

        sig_pred, epsp_pred = Y[..., :6], Y[..., 6:12]          # (B, H, 6) each
        return sig_pred, epsp_pred
