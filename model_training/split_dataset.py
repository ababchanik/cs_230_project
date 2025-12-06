#!/usr/bin/env python3
import argparse, os
import numpy as np

def compute_counts_from_ratios(m, ratios=(0.70, 0.15, 0.15)):
    r_train, r_dev, r_eval = ratios
    # initial floor allocation
    n_train = int(np.floor(r_train * m))
    n_dev   = int(np.floor(r_dev   * m))
    n_eval  = m - n_train - n_dev  # remainder goes to eval
    # ensure at least 1 for dev/eval if possible (when m >= 3)
    if m >= 3:
        if n_dev == 0:  n_dev  = 1
        if n_eval == 0: n_eval = 1
        # re-adjust train if we bumped dev/eval
        adjust = n_train + n_dev + n_eval - m
        if adjust > 0:
            n_train = max(0, n_train - adjust)
    return n_train, n_dev, n_eval

def main(args):
    data = np.load(args.data, allow_pickle=True)
    required = ["eps","epse","deps","sig"]
    if not all(k in data.files for k in required):
        raise RuntimeError(f"Missing arrays; need: {required}")

    m, T, _ = data["sig"].shape
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(m)

    n_train, n_dev, n_eval = compute_counts_from_ratios(
        m, (args.r_train, args.r_dev, args.r_eval)
    )
    train_idx = perm[:n_train]
    dev_idx   = perm[n_train:n_train+n_dev]
    eval_idx  = perm[n_train+n_dev:]

    split_arr = np.empty((m,), dtype="U8")
    split_arr[train_idx] = "train"
    split_arr[dev_idx]   = "dev"
    split_arr[eval_idx]  = "eval"

    base, ext = os.path.splitext(os.path.basename(args.data))  
    out_path = base + "_labeled" + ext

    # save in current folder instead of data_generation
    out_path = os.path.join(os.getcwd(), out_path)

    save_dict = {k: data[k] for k in data.files}
    save_dict["split"] = split_arr
    save_dict["split_seed"] = np.int64(args.seed)
    save_dict["split_ratios"] = np.array([args.r_train, args.r_dev, args.r_eval], dtype=float)

    np.savez(out_path, **save_dict)
    print(f"Labeled dataset saved to: {out_path}")
    print(f"Counts train: {np.sum(split_arr=='train')} | dev: {np.sum(split_arr=='dev')} | eval: {np.sum(split_arr=='eval')}")
    print(f"Ratios train: {args.r_train:.2f} | dev: {args.r_dev:.2f} | eval: {args.r_eval:.2f} | seed={args.seed}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--r_train", type=float, default=0.70)
    ap.add_argument("--r_dev",   type=float, default=0.15)
    ap.add_argument("--r_eval",  type=float, default=0.15)
    ap.add_argument("--seed",    type=int,   default=0)
    args = ap.parse_args()
    main(args)
