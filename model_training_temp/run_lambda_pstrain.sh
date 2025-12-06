#!/usr/bin/env bash

# ============================================================
# Activate training and testing environment
# ============================================================
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

# ------------------------------------------------------------
# Verify torch
# ------------------------------------------------------------
python - <<'PY'
try:
    import torch, sys
    print(f"[OK] torch {torch.__version__}, CUDA={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)} (device id: 0)")
except Exception:
    sys.exit("[error] torch not found. Try:\n  conda install -n seqml pytorch pytorch-cuda=12.1 -c pytorch -c nvidia\n  # or CPU-only:\n  conda install -n seqml pytorch cpuonly -c pytorch")
PY

# ============================================================
# Data paths
# ============================================================
# Multi-trajectory dataset (random-walk then fixed strain)
DATA_RAW="../data_generation/protocol_2_differential_random_them_fixed/data_stress_strain_differential_random_then_fixed.npz"
DATA_LABELED="./data_stress_strain_differential_random_then_fixed_labeled.npz"

# Single relaxation trajectory dataset
DATA_RELAX="../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz"

# ============================================================
# Hyperparameters (shared)
# ============================================================
L=4
H=4
S=1
HID=128
BS=1024
EPOCHS=500
LR=1e-4

# Multi-traj rollout horizon
H_TEST_MULTI=80

# Single relaxation path: T_relax ≈ 20, max H_test = T - L
H_TEST_RELAX=$((20 - L))

# Rollout stride (P ≤ H)
P=4

# Sweep over these lambda_pstrain values (strings for filenames)
LAMBDAS=("1e2" "1e3" "1e4" "1e5" "1e6" "1e7" "1e8")

# Zero-padded strings
L_STR=$(printf "%02d" "$L")
H_STR=$(printf "%02d" "$H")
S_STR=$(printf "%02d" "$S")
P_STR=$(printf "%02d" "$P")
H_TEST_MULTI_STR=$(printf "%02d" "$H_TEST_MULTI")
H_TEST_RELAX_STR=$(printf "%02d" "$H_TEST_RELAX")

echo "[INFO] Using L=${L}, H=${H}, S=${S}, hid=${HID}, epochs=${EPOCHS}, lr=${LR}"
echo "[INFO] H_TEST_MULTI=${H_TEST_MULTI}, H_TEST_RELAX=${H_TEST_RELAX}, P=${P}"
echo "[INFO] Sweeping over lambda_pstrain values: ${LAMBDAS[*]}"

# ============================================================
# 1) Split multi-trajectory dataset once
# ============================================================
python split_dataset.py \
  --data  "${DATA_RAW}" \
  --r_train  0.70 \
  --r_dev    0.15 \
  --r_eval   0.15 \
  --seed 0

echo "[OK] Dataset randomly split to train/dev/test -> ${DATA_LABELED}"

# ============================================================
# 2) Lambda sweep
# ============================================================
for LAMBDA_P in "${LAMBDAS[@]}"; do
  echo
  echo "======================================================"
  echo "[INFO] Starting run with lambda_pstrain=${LAMBDA_P}"
  echo "======================================================"

  # ----------------------------
  # Model checkpoint (one .pt per lambda)
  # ----------------------------
  TAG_MODEL="lamP_${LAMBDA_P}_L${L_STR}_H${H_STR}_S${S_STR}"
  MODEL_OUT="model_${TAG_MODEL}.pt"

  echo "[INFO] Output model checkpoint: ${MODEL_OUT}"

  # ----------------------------
  # 2a) Training on multi-trajectory dataset
  # ----------------------------
  python main.py \
      --data "${DATA_LABELED}" \
      --L "${L}" \
      --H "${H}" \
      --bs "${BS}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --hid "${HID}" \
      --lambda_pstrain "${LAMBDA_P}" \
      --lambda_overlap 0.1 \
      --lambda_self 0.1 \
      --S "${S}" \
      --out "${MODEL_OUT}"

  echo "[OK] Training completed for lambda_pstrain=${LAMBDA_P}. Model -> ${MODEL_OUT}"

  # ----------------------------
  # 2b) Random trajectory rollout on multi dataset
  # ----------------------------
  # Choose a random valid specimen index from DATA_LABELED
  I_SPEC_RANDOM=$(python - "${DATA_LABELED}" <<'PY'
import sys, numpy as np, random
path = sys.argv[1]
data = np.load(path)
eps = data["eps"]
if eps.ndim == 3:
    m = eps.shape[0]
else:
    m = 1
print(random.randrange(m))
PY
)

  echo "[INFO] Random specimen index chosen: i_spec=${I_SPEC_RANDOM}"

  TAG_RANDOM="${TAG_MODEL}_Htest${H_TEST_MULTI_STR}_P${P_STR}_random${I_SPEC_RANDOM}"
  NPZ_RANDOM="rollout_${TAG_RANDOM}.npz"
  FIG_DIR_RANDOM="figs_${TAG_RANDOM}"

  echo "[INFO] Random rollout NPZ: ${NPZ_RANDOM}"
  echo "[INFO] Random figs dir   : ${FIG_DIR_RANDOM}"

  python test_rollout.py \
      --data "${DATA_LABELED}" \
      --ckpt "${MODEL_OUT}" \
      --L "${L}" \
      --H "${H}" \
      --hid "${HID}" \
      --H_test "${H_TEST_MULTI}" \
      --P "${P}" \
      --i_spec "${I_SPEC_RANDOM}" \
      --out_npz "${NPZ_RANDOM}"

  echo "[OK] Random rollout completed for lambda_pstrain=${LAMBDA_P}, i_spec=${I_SPEC_RANDOM}."

  # Stress–strain plots (random)
  python plot_rollout_stress_strain.py \
      --data_truth "${DATA_LABELED}" \
      --data_pred  "./${NPZ_RANDOM}" \
      --out_dir    "./${FIG_DIR_RANDOM}"

  echo "[OK] Stress–strain rollout plots (random) saved to ${FIG_DIR_RANDOM}."

  # p–q invariant plots (random)
  python plot_rollout_stress_invariants.py \
      --data_truth "${DATA_LABELED}" \
      --data_pred  "./${NPZ_RANDOM}" \
      --M 2.0 \
      --pc -6.0 \
      --out_dir "./${FIG_DIR_RANDOM}"

  echo "[OK] Stress invariant rollout plots (random) saved to ${FIG_DIR_RANDOM}."

  # ----------------------------
  # 2c) Single relaxation trajectory rollout
  # ----------------------------
  TAG_RELAX="${TAG_MODEL}_Htest${H_TEST_RELAX_STR}_P${P_STR}_relaxation"
  NPZ_RELAX="rollout_${TAG_RELAX}.npz"
  FIG_DIR_RELAX="figs_${TAG_RELAX}"

  echo "[INFO] Relaxation rollout NPZ: ${NPZ_RELAX}"
  echo "[INFO] Relaxation figs dir   : ${FIG_DIR_RELAX}"

  python test_rollout.py \
      --data "${DATA_RELAX}" \
      --ckpt "${MODEL_OUT}" \
      --L "${L}" \
      --H "${H}" \
      --hid "${HID}" \
      --H_test "${H_TEST_RELAX}" \
      --P "${P}" \
      --i_spec 0 \
      --out_npz "${NPZ_RELAX}"

  echo "[OK] Relaxation rollout completed for lambda_pstrain=${LAMBDA_P} (single traj i_spec=0)."

  # Stress–strain plots (relaxation)
  python plot_rollout_stress_strain.py \
      --data_truth "${DATA_RELAX}" \
      --data_pred  "./${NPZ_RELAX}" \
      --out_dir    "./${FIG_DIR_RELAX}"

  echo "[OK] Stress–strain rollout plots (relaxation) saved to ${FIG_DIR_RELAX}."

  # p–q invariant plots (relaxation)
  python plot_rollout_stress_invariants.py \
      --data_truth "${DATA_RELAX}" \
      --data_pred  "./${NPZ_RELAX}" \
      --M 2.0 \
      --pc -6.0 \
      --out_dir "./${FIG_DIR_RELAX}"

  echo "[OK] Stress invariant rollout plots (relaxation) saved to ${FIG_DIR_RELAX}."
done

echo "[INFO] All lambda_pstrain sweeps completed."
