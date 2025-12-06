#!/usr/bin/env python3
import argparse
import datetime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from model import SeqModel
from data_loader import SingleRelaxationWindowDataset
from loss_functions import mse_loss, overlap_consistency_loss

# ============================================================
# Logging utility (kept from your original style)
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

def log_epoch_loss(epoch, train_loss, log_path="train_log.txt"):
    """
    Append timestamped train loss to the log file.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{now}] Epoch {epoch:03d} | train {train_loss:.6f}"
    with open(log_path, "a") as f:
        f.write(msg + "\n")
    print(msg)
    return msg

# ============================================================
# Training loop: overfit on a single trajectory
# ============================================================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_device_info()

    # Dataset: single stress–relaxation path, all windows used for training
    ds_tr = SingleRelaxationWindowDataset(args.data, L=args.L, H=args.H)
    tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=0)
    print(f"[INFO] Number of training windows: {len(ds_tr)}")

    model = SeqModel(H=args.H, hid=args.hid).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0

        for eps, epse, deps, sig_y, epse_y, t0 in tr:
            # Input and ground truth
            eps, epse, deps = [x.to(device) for x in (eps, epse, deps)]      # (B,L,6)
            sig_y, epse_y   = [x.to(device) for x in (sig_y, epse_y)]        # (B,H,6)
            t0 = t0.to(device).long()                                        # (B,)

            # Forward
            sig_pred, epse_pred = model(eps, epse, deps)                     # (B,H,6) each

            B, H, _ = sig_pred.shape

            # Flatten (B,H,6)->(B*H,6) for MSE to avoid broadcasting
            sig_pred_flat   = sig_pred.reshape(B * H, 6)
            epse_pred_flat  = epse_pred.reshape(B * H, 6)
            sig_y_flat      = sig_y.reshape(B * H, 6)
            epse_y_flat     = epse_y.reshape(B * H, 6)

            # Simple fit loss (no overlap consistency)
            loss_fit = mse_loss(sig_pred_flat, sig_y_flat) + args.lambda_estrain * mse_loss(epse_pred_flat, epse_y_flat)

            # Overlap consistency loss
            # global time indices t = t0 + 1, ..., t0 + H
            h_idx = torch.arange(1, H + 1, device=device).view(1, H)  # (1, H)
            global_time = t0.view(B, 1) + h_idx                       # (B, H)

            # single trajectory, use global_time as key
            global_key = global_time.long()                           # (B, H)

            loss_overlap = overlap_consistency_loss(global_key, sig_pred)

            # total loss
            loss = loss_fit + args.lambda_overlap * loss_overlap

            # Step
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()

        train_avg = loss_sum / len(tr)
        log_epoch_loss(epoch, train_avg)

    # Save overfit model
    torch.save(model.state_dict(), args.out)
    print(f"[OK] Training completed. Model saved to {args.out}")

    # Quick sanity check: evaluate MSE on a single batch
    model.eval()
    with torch.no_grad():
        eps, epse, deps, sig_y, epse_y, t0 = next(iter(tr))
        eps, epse, deps = [x.to(device) for x in (eps, epse, deps)]
        sig_y, epse_y   = [x.to(device) for x in (sig_y, epse_y)]

        sig_pred, epse_pred = model(eps, epse, deps)

        B, H, _ = sig_pred.shape
        sig_pred_flat   = sig_pred.reshape(B * H, 6)
        epse_pred_flat  = epse_pred.reshape(B * H, 6)
        sig_y_flat      = sig_y.reshape(B * H, 6)
        epse_y_flat     = epse_y.reshape(B * H, 6)

        mse_sig  = mse_loss(sig_pred_flat, sig_y_flat).item()
        mse_epse = mse_loss(epse_pred_flat, epse_y_flat).item()
        print(f"[FINAL] single-batch MSE: sig={mse_sig:.3e}, epse={mse_epse:.3e}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Overfit SeqModel on a single stress–relaxation trajectory"
    )
    ap.add_argument(
        "--data",
        type=str,
        default="../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz",
        help="Path to single stress–relaxation .npz file",
    )
    ap.add_argument("--out", type=str, default="./model_overfit.pt")
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument(
        "--lambda_estrain",
        type=float,
        default=1e4,
        help="Weight for elastic strain prediction loss",
    )
    ap.add_argument(
        "--lambda_overlap",
        type=float,
        default=1.0,
        help="Weight for overlap consistency loss",
    )
    args = ap.parse_args()
    train(args)
