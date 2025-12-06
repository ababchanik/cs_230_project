#!/usr/bin/env python3
import argparse
import datetime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb

from model import SeqModel
from data_loader import SingleRelaxationWindowDataset
from loss_functions import mse_loss, overlap_consistency_loss
from constitutive_update import assemble_elastic_tensor_voigt, elastic_update

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

def log_epoch_loss(epoch, loss_fit, loss_overlap, loss_self, total_loss, log_path="train_log.txt"):
    """
    Append timestamped loss info to the log file.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    msg = (
        f"[{now}] Epoch {epoch:03d} | "
        f"fit={loss_fit:.6e}, "
        f"overlap(unscaled)={loss_overlap:.6e}, "
        f"self(unscaled)={loss_self:.6e}, "
        f"total={total_loss:.6e}"
    )

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

    wandb.init(
        project="ti-mcc-single-trajectory",
        entity="hemiaozh-stanford-university",
        name=(
            f"L{args.L:02d}_"
            f"H{args.H:02d}_"
            f"S{args.S:02d}_"
            f"le{args.lambda_estrain:g}_"
            f"lo{args.lambda_overlap:g}_"
            f"ls{args.lambda_self:g}"
        ),
        config={
            "L": args.L,
            "H": args.H,
            "lambda_estrain": args.lambda_estrain,
            "lambda_overlap": args.lambda_overlap,
            "lambda_self": args.lambda_self,
            "S": args.S,
            "epochs": args.epochs,
            "lr": args.lr,
        },
    )

    # Dataset: single stress–relaxation path, all windows used for training
    ds_tr = SingleRelaxationWindowDataset(args.data, L=args.L, H=args.H, S=args.S)
    tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=0)
    print(f"[INFO] Number of training windows: {len(ds_tr)}")

    # Assemble elastic tensor in Voigt form and move to device
    CCe_voigt_np = assemble_elastic_tensor_voigt("material_parameters.csv")
    CCe_voigt = torch.from_numpy(CCe_voigt_np).float().to(device)

    model = SeqModel(H=args.H, hid=args.hid).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_fit_sum = loss_overlap_sum = loss_self_sum = total_loss_sum = 0.0

        for eps, epse, deps, sig_y, epse_y, t0 in tr:
            # Input and ground truth
            eps, epse, deps = [x.to(device) for x in (eps, epse, deps)]      # (B,L+S,6)
            sig_y, epse_y   = [x.to(device) for x in (sig_y, epse_y)]        # (B,H+S,6)
            t0 = t0.to(device).long()                                        # (B,)

            L, H, S = args.L, args.H, args.S
            L_total = L + S
            H_total = H + S

            # -------------------------------------------------
            # Split windows:
            #   history:        first L steps  (t-L+1 ... t)
            #   horizon (fit):  first H steps  (t+1   ... t+H)
            #   extra S steps:  additional overlap, unused in fit
            # -------------------------------------------------
            eps_hist  = eps[:, :L, :]        # (B,L,6)
            epse_hist = epse[:, :L, :]       # (B,L,6)
            deps_hist = deps[:, :L, :]       # (B,L,6)

            sig_y_H   = sig_y[:, :H, :]      # (B,H,6)
            epse_y_H  = epse_y[:, :H, :]     # (B,H,6)

            # -------------------------------------------------
            # Elastic predictor step on the history only
            # -------------------------------------------------
            sig_trial = elastic_update(epse_hist + deps_hist, CCe_voigt)     # (B,L,6)

            # Forward: model still takes L and outputs H
            sig_pred, epse_pred = model(eps_hist, epse_hist, deps_hist, sig_trial)  # (B,H,6)

            B, H_model, _ = sig_pred.shape
            assert H_model == H, "Model H and args.H mismatch"

            # Flatten (B,H,6)->(B*H,6) for MSE to avoid broadcasting
            sig_pred_flat   = sig_pred.reshape(B * H, 6)
            epse_pred_flat  = epse_pred.reshape(B * H, 6)
            sig_y_flat      = sig_y_H.reshape(B * H, 6)
            epse_y_flat     = epse_y_H.reshape(B * H, 6)

            # - - - - - - - - - - - - - - - - - - - - - - - -
            # 1. Simple fit loss from teacher forcing
            # - - - - - - - - - - - - - - - - - - - - - - - -
            loss_fit = mse_loss(sig_pred_flat, sig_y_flat) \
                        + args.lambda_estrain * mse_loss(epse_pred_flat, epse_y_flat)
            loss_fit = - torch.exp(-loss_fit) \
                        + loss_fit
                        
            # - - - - - - - - - - - - - - - - - - - - - - - -
            # 2. Overlap consistency loss
            # - - - - - - - - - - - - - - - - - - - - - - - -
            # global time indices t = t0 + 1, ..., t0 + H
            h_idx = torch.arange(1, H + 1, device=device).view(1, H)  # (1, H)
            global_time = t0.view(B, 1) + h_idx                       # (B, H)
            # single trajectory, use global_time as key
            global_key = global_time.long()                           # (B, H)

            loss_overlap_sig  = overlap_consistency_loss(global_key, sig_pred)
            loss_overlap_epse = overlap_consistency_loss(global_key, epse_pred)
            loss_overlap = loss_overlap_sig \
                            + args.lambda_estrain * loss_overlap_epse

            # - - - - - - - - - - - - - - - - - - - - - - - -
            # 3. Self-fed loss over first S_eff steps
            # - - - - - - - - - - - - - - - - - - - - - - - -
            loss_self = torch.tensor(0.0, device=device)

            if args.lambda_self > 0.0:
                # history clone
                epse_self = epse_hist.clone()     # (B,L,6)
                eps_self  = eps_hist.clone()      # (B,L,6)
                deps_self = deps_hist.clone()     # (B,L,6)

                # shift history left by S and append first S predicted epse
                epse_self[:, :-S, :] = epse_self[:, S:, :]      # shift left by S
                epse_self[:, -S:, :] = epse_pred[:, :S, :]      # append predicted epse(1:S)

                # eps_self and deps_self shifts accordingly
                eps_self[:, :-S, :]  = eps_self[:, S:, :]       # shift left by S
                eps_self[:, -S:, :]  = eps[:, L:, :]            # append true eps(1:S)
                deps_self[:, :-S, :] = deps_self[:, S:, :]      # shift left by S
                deps_self[:, -S:, :] = deps[:, L:, :]           # append true deps(1:S)

                sig_self_trial = elastic_update(epse_self + deps_self, CCe_voigt)    # (B,L,6)
                sig_self, epse_self_pred = model(eps_self, epse_self, deps_self, sig_self_trial)  # (B,H,6)

                # shift ground truth for comparison
                sig_y_S  = sig_y_H.clone()
                epse_y_S = epse_y_H.clone()
                sig_y_S[:, :-S, :]  = sig_y_S[:, S:, :]          # shift left by S
                sig_y_S[:, -S:, :]  = sig_y[:, H:, :]            # append ground truth sig_y(H+1:H+S)
                epse_y_S[:, :-S, :] = epse_y_S[:, S:, :]         # shift left by S
                epse_y_S[:, -S:, :] = epse_y[:, H:, :]           # append ground truth epse_y(H+1:H+S)
                
                # compare from input  (t - L + 1 + S) to (t + S)
                #              output (t + 1 + S) to (t + H + S)
                loss_self = mse_loss(sig_self, sig_y_S) \
                            + args.lambda_estrain * mse_loss(epse_self_pred, epse_y_S)

            # - - - - - - - - - - - - - - - - - - - - - - - -
            # Total loss
            # - - - - - - - - - - - - - - - - - - - - - - - -
            loss = (
                loss_fit
                + args.lambda_overlap * loss_overlap
                + args.lambda_self * loss_self
            )

            # Step
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
        if epoch % 100 == 0:
            log_epoch_loss(epoch, avg_fit, avg_overlap, avg_self, avg_total)
        # Log every epoch to wandb
        wandb.log({
            "epoch": epoch,
            "fit_loss": avg_fit,
            "overlap_loss": avg_overlap,
            "self_loss": avg_self,
            "total_loss": avg_total,
        }, step=epoch)

    # Save overfit model
    torch.save(model.state_dict(), args.out)
    print(f"[OK] Training completed. Model saved to {args.out}")

    # Quick sanity check: evaluate MSE on a single batch
        # Quick sanity check: evaluate MSE on a single batch
    model.eval()
    with torch.no_grad():
        eps, epse, deps, sig_y, epse_y, t0 = next(iter(tr))
        eps, epse, deps = [x.to(device) for x in (eps, epse, deps)]    # (B,L+S,6)
        sig_y, epse_y   = [x.to(device) for x in (sig_y, epse_y)]      # (B,H+S,6)

        L, H, S = args.L, args.H, args.S

        # Use the same slices as during training
        eps_hist  = eps[:, :L, :]        # (B,L,6)
        epse_hist = epse[:, :L, :]       # (B,L,6)
        deps_hist = deps[:, :L, :]       # (B,L,6)

        sig_y_H   = sig_y[:, :H, :]      # (B,H,6)
        epse_y_H  = epse_y[:, :H, :]     # (B,H,6)

        # Rebuild σ_trial using history only
        sig_trial = elastic_update(epse_hist + deps_hist, CCe_voigt)   # (B,L,6)

        # Forward
        sig_pred, epse_pred = model(eps_hist, epse_hist, deps_hist, sig_trial)

        B, H_model, _ = sig_pred.shape
        assert H_model == H

        sig_pred_flat   = sig_pred.reshape(B * H, 6)
        epse_pred_flat  = epse_pred.reshape(B * H, 6)
        sig_y_flat      = sig_y_H.reshape(B * H, 6)
        epse_y_flat     = epse_y_H.reshape(B * H, 6)

        mse_sig  = mse_loss(sig_pred_flat, sig_y_flat).item()
        mse_epse = mse_loss(epse_pred_flat, epse_y_flat).item()
        print(f"[FINAL] single-batch MSE: sig={mse_sig:.3e}, epse={mse_epse:.3e}")
        wandb.log({
            "final_mse_sig": mse_sig,
            "final_mse_epse": mse_epse,
        })

    wandb.finish()

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
    ap.add_argument(
        "--lambda_self",
        type=float,
        default=1.0,
        help="Weight for one-step self-fed consistency loss",
    )
    ap.add_argument(
        "--S",
        type=int,
        default=1,
        help="Number of self-fed prediction steps used in loss (0 ≤ S ≤ min(L, H))",
    )
    args = ap.parse_args()
    assert 0 <= args.S <= min(args.L, args.H), "S must satisfy 0 ≤ S ≤ min(L, H)"
    train(args)
