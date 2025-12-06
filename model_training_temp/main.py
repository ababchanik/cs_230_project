#!/usr/bin/env python3
import argparse
import datetime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb


from model import SeqModel
from data_loader import TrajWindowDataset
from loss_functions import mse_loss
from constitutive_update import assemble_elastic_tensor_voigt, elastic_update

# ============================================================
# Logging utility
# ============================================================
def log_device_info(log_path="train_log.txt"):
    """
    Append a timestamped entry describing which device is used.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)
        msg = f"[{now}] Using GPU: {gpu_name} (device id: {gpu_id})"
    else:
        msg = f"[{now}] Using CPU"

    with open(log_path, "a") as f:
        f.write(msg + "\n")
    print(msg)
    return msg

def log_epoch_loss(epoch, loss_fit, loss_overlap, loss_self, total_loss, log_path="train_log.txt"):
    """
    Append timestamped loss info to the log file.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = (
        f"[{now}] [TRAIN] Epoch {epoch:03d} | "
        f"fit={loss_fit:.6e}, "
        f"overlap(unscaled)={loss_overlap:.6e}, "
        f"self(unscaled)={loss_self:.6e}, "
        f"total={total_loss:.6e}"
    )
    with open(log_path, "a") as f:
        f.write(msg + "\n")
    print(msg)
    return msg

def log_eval_loss(epoch, mse_sig, mse_epsp, log_path="train_log.txt"):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = (
        f"[{now}] [EVAL] Epoch {epoch:03d} | "
        f"MSE: sig={mse_sig:.6e}, epsp={mse_epsp:.6e}"
    )
    with open(log_path, "a") as f:
        f.write(msg + "\n")
    print(msg)
    return msg

# ============================================================
# Overlap consistency loss
# ============================================================
def overlap_consistency_loss(global_key, pred, eps=1e-12):
    """
    Penalize disagreement among predictions with the same (traj_id, time).

    pred: (B, H, D)
    global_key: (B, H) encoded as traj_id * SCALE + time
    """
    B, H, D = pred.shape
    keys_flat = global_key.reshape(-1)   # (B*H,)
    pred_flat = pred.reshape(-1, D)      # (B*H, D)

    # Group by unique keys
    uniq, inv, counts = keys_flat.unique(
        return_inverse=True, return_counts=True
    )
    mask_multi = counts > 1

    if not mask_multi.any():
        # return zero that still participates in autograd
        return torch.zeros((), device=pred.device, dtype=pred.dtype, requires_grad=True)

    K = uniq.size(0)
    group_sum = torch.zeros(K, D, device=pred.device)
    group_sum.index_add_(0, inv, pred_flat)   # sum over groups

    counts_f = counts.float().unsqueeze(1)    # (K, 1)
    group_mean = group_sum / (counts_f + eps) # (K, D)
    mean_flat = group_mean[inv]               # (B*H, D)

    valid = mask_multi[inv]                   # (B*H,)
    return mse_loss(pred_flat[valid], mean_flat[valid])

# ============================================================
# Full-dev eval: MSE over ALL windows in a loader
# ============================================================
def eval_full_mse(model, dataloader, args, device, CCe_voigt):
    """
    Compute global MSE(sig) and MSE(epsp) over *all* windows in `dataloader`.

    Uses the same slicing logic as training:
      - history: first L steps
      - horizon: first H steps
    No overlap/self/transform here.
    """
    model.eval()
    L, H, S = args.L, args.H, args.S

    sse_sig = 0.0   # sum of squared errors for sig
    sse_epsp = 0.0  # sum of squared errors for epsp
    n_sig = 0       # total number of sig components
    n_epsp = 0      # total number of epsp components

    with torch.no_grad():
        for eps_hist, epsp_hist, deps_future, sig_y, epsp_y, t0, traj_idx in dataloader:
            eps_hist    = eps_hist.to(device)      # (B, L+S, 6)
            epsp_hist   = epsp_hist.to(device)     # (B, L+S, 6)
            deps_future = deps_future.to(device)   # (B, H+S, 6)
            sig_y       = sig_y.to(device)         # (B, H+S, 6)
            epsp_y      = epsp_y.to(device)        # (B, H+S, 6)

            # Same slicing as training
            eps_hist_L    = eps_hist[:, :L, :]     # (B, L, 6)
            epsp_hist_L   = epsp_hist[:, :L, :]    # (B, L, 6)
            deps_future_H = deps_future[:, :H, :]  # (B, H, 6)
            sig_y_H       = sig_y[:, :H, :]        # (B, H, 6)
            epsp_y_H      = epsp_y[:, :H, :]       # (B, H, 6)

            # Elastic predictor on the history only
            sig_hist = elastic_update(eps_hist_L - epsp_hist_L, CCe_voigt)  # (B, L, 6)

            # Forward model
            sig_pred, epsp_pred = model(eps_hist_L, epsp_hist_L, sig_hist, deps_future_H)

            # Flatten
            sig_pred_flat  = sig_pred.reshape(-1, 6)
            epsp_pred_flat = epsp_pred.reshape(-1, 6)
            sig_y_flat     = sig_y_H.reshape(-1, 6)
            epsp_y_flat    = epsp_y_H.reshape(-1, 6)

            # Accumulate SSE and counts
            diff_sig  = sig_pred_flat - sig_y_flat
            diff_epsp = epsp_pred_flat - epsp_y_flat

            sse_sig  += (diff_sig ** 2).sum().item()
            sse_epsp += (diff_epsp ** 2).sum().item()

            n_sig   += diff_sig.numel()
            n_epsp  += diff_epsp.numel()

    mse_sig  = sse_sig / n_sig   if n_sig  > 0 else 0.0
    mse_epsp = sse_epsp / n_epsp if n_epsp > 0 else 0.0
    return mse_sig, mse_epsp

# ============================================================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_device_info()

    # Assemble elastic tensor in Voigt form (used to rebuild sig_hist)
    CCe_voigt_np = assemble_elastic_tensor_voigt("material_parameters.csv")
    CCe_voigt = torch.from_numpy(CCe_voigt_np).float().to(device)

    # >>> Init wandb <<<
    wandb.init(
        project="cs230-ti-mcc-small-latest",
        entity="hemiaozh-stanford-university",
        name=(
            f"L{args.L:02d}_"
            f"H{args.H:02d}_"
            f"S{args.S:02d}_"
            f"lp{args.lambda_pstrain:g}_"
            f"lo{args.lambda_overlap:g}_"
            f"ls{args.lambda_self:g}"
        ),
        config={
            "L": args.L,
            "H": args.H,
            "S": args.S,
            "batch_size": args.bs,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "lambda_pstrain": args.lambda_pstrain,
            "lambda_overlap": args.lambda_overlap,
            "lambda_self": args.lambda_self,
        },
    )

    # Use labeled dataset: train/dev, with (L, H, S) windowing
    ds_tr = TrajWindowDataset(args.data, L=args.L, H=args.H, S=args.S, split='train')
    ds_va = TrajWindowDataset(args.data, L=args.L, H=args.H, S=args.S, split='dev')

    tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=False,  num_workers=8)
    va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, num_workers=8)

    model = SeqModel(H=args.H, hid=args.hid).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    KEY_SCALE = 1_000_000  # for encoding (traj_idx, time) into a single integer

    # --------------------------------------------------------
    # Initial eval (epoch 0): full-dev MSE over all dev windows
    # --------------------------------------------------------
    mse_sig_va, mse_epsp_va = eval_full_mse(model, va, args, device, CCe_voigt)
    print(
        f"[EVAL] INITIAL | "
        f"dev MSE: sig={mse_sig_va:.3e}, epsp={mse_epsp_va:.3e}"
    )
    wandb.log({
        "epoch": 0,
        "eval_dev_mse_sig": mse_sig_va,
        "eval_dev_mse_epsp": mse_epsp_va,
    }, step=0)

    # =======================================================
    # Training loop
    # =======================================================
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_fit_sum = loss_overlap_sum = loss_self_sum = total_loss_sum = 0.0

        for eps_hist, epsp_hist, deps_future, sig_y, epsp_y, t0, traj_idx in tr:
            # Move to device
            eps_hist    = eps_hist.to(device)      # (B, L+S, 6)
            epsp_hist   = epsp_hist.to(device)     # (B, L+S, 6)
            deps_future = deps_future.to(device)   # (B, H+S, 6)
            sig_y       = sig_y.to(device)         # (B, H+S, 6)
            epsp_y      = epsp_y.to(device)        # (B, H+S, 6)
            t0          = t0.to(device).long()     # (B,)
            traj_idx    = traj_idx.to(device).long()

            L, H, S = args.L, args.H, args.S
            # -------------------------------------------------
            # Split windows:
            #   history:        first L steps     (t-L+1 ... t)
            #   horizon (fit):  first H steps     (t+1   ... t+H)
            #   extra S steps:  additional overlap, unused in fit
            # -------------------------------------------------
            eps_hist_L   = eps_hist[:, :L, :]          # (B, L, 6)
            epsp_hist_L  = epsp_hist[:, :L, :]         # (B, L, 6)
            deps_future_H = deps_future[:, :H, :]      # (B, H, 6)

            sig_y_H   = sig_y[:, :H, :]                # (B, H, 6)
            epsp_y_H  = epsp_y[:, :H, :]               # (B, H, 6)

            # Elastic predictor on the history only
            sig_hist = elastic_update(eps_hist_L - epsp_hist_L, CCe_voigt)  # (B, L, 6)

            # Forward model: outputs (B, H, 6) each
            sig_pred, epsp_pred = model(eps_hist_L, epsp_hist_L, sig_hist, deps_future_H)

            B, H_model, _ = sig_pred.shape
            assert H_model == H, "Model H and args.H mismatch"

            # Flatten to (B*H, 6) for MSE
            sig_pred_flat   = sig_pred.reshape(B * H, 6)
            epsp_pred_flat  = epsp_pred.reshape(B * H, 6)
            sig_y_flat      = sig_y_H.reshape(B * H, 6)
            epsp_y_flat     = epsp_y_H.reshape(B * H, 6)

            # -------------------------------------------------
            # 1. Fit loss (teacher forcing) with plastic-strain weighting
            #    and nonlinear transform: -exp(-L) + L
            # -------------------------------------------------
            loss_fit = mse_loss(sig_pred_flat, sig_y_flat) \
                       + args.lambda_pstrain * mse_loss(epsp_pred_flat, epsp_y_flat)

            loss_fit = - torch.exp(-loss_fit) + loss_fit

            # -------------------------------------------------
            # 2. Overlap consistency loss across traj/time
            # -------------------------------------------------
            step_idx = torch.arange(1, H + 1, device=device).unsqueeze(0)  # (1, H)
            global_t = t0.unsqueeze(1) + step_idx                          # (B, H)
            global_key = traj_idx.unsqueeze(1) * KEY_SCALE + global_t      # (B, H)

            loss_overlap_sig  = overlap_consistency_loss(global_key, sig_pred)
            loss_overlap_epsp = overlap_consistency_loss(global_key, epsp_pred)
            loss_overlap = loss_overlap_sig \
                           + args.lambda_pstrain * loss_overlap_epsp

            # -------------------------------------------------
            # 3. Self-fed loss over first S steps (if enabled)
            # -------------------------------------------------
            loss_self = torch.tensor(0.0, device=device)
            if (S > 0) and (args.lambda_self > 0.0):
                # history clone (L steps)
                epsp_self = epsp_hist_L.clone()               # (B, L, 6)

                # shift history left by S and append first S predicted epsp
                epsp_self[:, :-S, :] = epsp_self[:, S:, :]    # drop earliest S
                epsp_self[:, -S:, :] = epsp_pred[:, :S, :]    # append predicted epsp(1:S)

                # history shift for eps: use full (L+S) window to produce L shifted steps
                eps_self = eps_hist[:, S:L+S, :]              # (B, L, 6)

                # rebuild sig_self from (eps_self - epsp_self)
                sig_self = elastic_update(eps_self - epsp_self, CCe_voigt)  # (B, L, 6)

                # future shift for deps: (t+1+S ... t+H+S)
                deps_self = deps_future[:, S:H+S, :]          # (B, H, 6)

                # model re-prediction on shifted history/future
                sig_self_pred, epsp_self_pred = model(eps_self, epsp_self, sig_self, deps_self)

                # shifted ground truth for comparison
                sig_y_S  = sig_y[:, S:S+H, :]                 # (B, H, 6)
                epsp_y_S = epsp_y[:, S:S+H, :]                # (B, H, 6)

                sig_self_flat       = sig_self_pred.reshape(B * H, 6)
                sig_y_S_flat        = sig_y_S.reshape(B * H, 6)
                epsp_self_pred_flat = epsp_self_pred.reshape(B * H, 6)
                epsp_y_S_flat       = epsp_y_S.reshape(B * H, 6)

                loss_self = mse_loss(sig_self_flat, sig_y_S_flat) \
                            + args.lambda_pstrain * mse_loss(epsp_self_pred_flat, epsp_y_S_flat)

            # -------------------------------------------------
            # Total loss
            # -------------------------------------------------
            loss = (
                loss_fit
                + args.lambda_overlap * loss_overlap
                + args.lambda_self * loss_self
            )

            # Optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Record losses
            loss_fit_sum     += loss_fit.item()
            loss_overlap_sum += loss_overlap.item()
            loss_self_sum    += loss_self.item()
            total_loss_sum   += loss.item()

        num_batches = len(tr)
        avg_fit     = loss_fit_sum / num_batches
        avg_overlap = loss_overlap_sum / num_batches
        avg_self    = loss_self_sum / num_batches
        avg_total   = total_loss_sum / num_batches
        # Only print/log every 100 epochs
        if (epoch % 100 == 0):
            log_epoch_loss(epoch, avg_fit, avg_overlap, avg_self, avg_total)
        # Log every epoch to wandb
        wandb.log({
            "epoch": epoch,
            "fit_loss": avg_fit,
            "overlap_loss": avg_overlap,
            "self_loss": avg_self,
            "total_loss": avg_total,
        }, step=epoch)

        # --------------------------------------------------------
        # Periodic eval: plain MSE(sig), MSE(epsp) on DEV every 100 epochs
        # --------------------------------------------------------
        if epoch % 100 == 0:
            mse_sig_dev, mse_epsp_dev = eval_full_mse(model, va, args, device, CCe_voigt)
            log_eval_loss(epoch, mse_sig_dev, mse_epsp_dev)
            print(f"[EVAL] Epoch {epoch:03d} | MSE: sig={mse_sig_dev:.3e}, epsp={mse_epsp_dev:.3e}")
            wandb.log({
                "eval_dev_mse_sig": mse_sig_dev,
                "eval_dev_mse_epsp": mse_epsp_dev,
            }, step=epoch)

            # go back to train mode (eval_full_mse sets model.eval())
            model.train()

    # Save trained model
    torch.save(model.state_dict(), args.out)
    print(f"[OK] Training completed. Model saved to {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./data_stress_strain_labeled.npz")
    ap.add_argument("--out", type=str, default="./model_multitraj.pt")
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--S", type=int, default=1, help="Number of self-fed prediction steps / extra overlap (0 ≤ S ≤ min(L, H))")
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--lambda_pstrain", type=float, default=1e4, help="Weight for plastic strain prediction loss")
    ap.add_argument("--lambda_overlap", type=float, default=0.1, help="Weight for overlap consistency loss")
    ap.add_argument("--lambda_self", type=float, default=0.1, help="Weight for one-step self-fed consistency loss")
    args = ap.parse_args()
    assert 0 <= args.S <= min(args.L, args.H), "S must satisfy 0 ≤ S ≤ min(L, H)"
    train(args)


# # above works without extra lie loss and other
