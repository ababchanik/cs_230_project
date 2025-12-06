#!/usr/bin/env bash

# ------------------------------------------------------------------
# Usage:
#   ./run.sh
#
#   i_spec : which trajectory index to use for rollout & plots (default 0)
#   H_TEST : rollout horizon in steps (default 80)
# ------------------------------------------------------------------

I_SPEC=0
H_TEST=80

L=4
H=4
S=1
HID=128
P=4

DATA_RAW="../data_generation/protocol_2_differential_random_them_fixed/data_stress_strain_differential_random_then_fixed.npz"
DATA_LABELED="./data_stress_strain_differential_random_then_fixed_labeled.npz"
CKPT="./model.pt"
ROLLOUT_NPZ="./rollout_results_L${L}_H${H}_Htest${H_TEST}_spec${I_SPEC}.npz"
FIG_DIR="./figs"

echo "[INFO] Using i_spec=${I_SPEC}, H_TEST=${H_TEST}"
echo "[INFO] L=${L}, H=${H}, HID=${HID}"

# ------------------------------------------------------------------
# Activate training and testing environment
# ------------------------------------------------------------------
activate() {
  local d="$1"
  [[ -d "$d" ]] || return 1
  [[ -f "$d/etc/profile.d/conda.sh" ]] && source "$d/etc/profile.d/conda.sh"
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate seqml >/dev/null 2>&1 && return 0
  fi
  return 1
}

activate "/home/groups/borja/hemiao/miniforge" || activate "$HOME/miniforge" || {
  echo "[error] Could not activate 'seqml' conda env."; exit 1; }

# ------------------------------------------------------------------
# Verify torch
# ------------------------------------------------------------------
python - <<'PY'
try:
    import torch, sys
    print(f"[OK] torch {torch.__version__}, CUDA={torch.cuda.is_available()}")
except Exception:
    sys.exit("[error] torch not found. Try:\n  conda install -n seqml pytorch pytorch-cuda=12.1 -c pytorch -c nvidia\n  # or CPU-only:\n  conda install -n seqml pytorch cpuonly -c pytorch")
PY

# ------------------------------------------------------------------
# Label the dataset by train/dev/test
# ------------------------------------------------------------------
python split_dataset.py \
  --data  "${DATA_RAW}" \
  --r_train  0.70 \
  --r_dev    0.15 \
  --r_eval   0.15 \
  --seed 0

echo "[OK] Dataset randomly split to train/dev/test -> ${DATA_LABELED}"

# ------------------------------------------------------------------
# Run training (multi-trajectory training script main.py)
# ------------------------------------------------------------------

# python main_lie.py \ (for lie loss)
python main.py \
  --data "${DATA_LABELED}" \
  --out  "${CKPT}" \
  --L    ${L} \
  --H    ${H} \
  --S    ${S} \
  --bs   1024 \
  --epochs 500 \
  # epochs was 500
  --lr   1e-3 \
  --hid  ${HID} \
  --lambda_pstrain 1e3

echo "[OK] Training completed. Model saved to ${CKPT}"

# ------------------------------------------------------------------
# Rollout on a chosen trajectory i_spec (multi-traj test_rollout.py)
# ------------------------------------------------------------------
python test_rollout.py \
  --data "${DATA_LABELED}" \
  --ckpt "${CKPT}" \
  --L      ${L} \
  --H      ${H} \
  --H_test ${H_TEST} \
  --P      ${P} \
  --hid  ${HID} \
  --i_spec ${I_SPEC} \
  --out_npz "${ROLLOUT_NPZ}"

echo "[OK] Rollout completed for i_spec=${I_SPEC}, results -> ${ROLLOUT_NPZ}"

# ------------------------------------------------------------------
# Plot stress–strain for that trajectory (multi-traj plot script)
# ------------------------------------------------------------------
python plot_rollout_stress_strain.py \
  --data_truth "${DATA_LABELED}" \
  --data_pred  "${ROLLOUT_NPZ}" \
  --out_dir    "${FIG_DIR}"

echo "[OK] Stress–strain rollout plots saved to ${FIG_DIR}"

# ------------------------------------------------------------------
# Plot p–q invariants for that trajectory (multi-traj plot script)
# ------------------------------------------------------------------
python plot_rollout_stress_invariants.py \
  --data_truth "${DATA_LABELED}" \
  --data_pred  "${ROLLOUT_NPZ}" \
  --out_dir    "${FIG_DIR}" \
  --M  2.0 \
  --pc -6.0

echo "[OK] Stress invariant rollout plots saved to ${FIG_DIR}"
echo "[OK] All done."
