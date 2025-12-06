#!/usr/bin/env bash

# Activate training and testing environment
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

# Verify torch
python - <<'PY'
try:
    import torch, sys
    print(f"[OK] torch {torch.__version__}, CUDA={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)} (device id: 0)")
except Exception:
    sys.exit("[error] torch not found. Try:\n  conda install -n seqml pytorch pytorch-cuda=12.1 -c pytorch -c nvidia\n  # or CPU-only:\n  conda install -n seqml pytorch cpuonly -c pytorch")
PY

# Hyperparameters (shared)
# ----------------------------
L=4
H=4
S=1
H_TEST=$((20 - L))  # max: T - (L - 1) - 1, here T=20
HID=128
BS=128 # more than enough
EPOCHS=50000
LR=5e-5
P=4

# Sweep over these lambda_pstrain values (strings so we can use them in filenames)
LAMBDAS=("1e3" "2e3" "4e3" "1e4")

L_STR=$(printf "%02d" "$L")
H_STR=$(printf "%02d" "$H")
S_STR=$(printf "%02d" "$S")
H_TEST_STR=$(printf "%02d" "$H_TEST")
P_STR=$(printf "%02d" "$P")

echo "[INFO] Using L=${L}, H=${H}, hid=${HID}, epochs=${EPOCHS}, lr=${LR}"
echo "[INFO] Sweeping over lambda_pstrain values: ${LAMBDAS[*]}"

for LAMBDA_P in "${LAMBDAS[@]}"; do
  echo
  echo "==============================================="
  echo "[INFO] Starting run with lambda_pstrain=${LAMBDA_P}"
  echo "==============================================="

  # Unique tags for this lambda
  TAG="lamP_${LAMBDA_P}_L${L_STR}_H${H_STR}_S${S_STR}"
  MODEL_OUT="model_overfit_${TAG}_Htest16_P01.pt"
  NPZ_OUT="test_rollout_${TAG}_Htest${H_TEST_STR}_P${P_STR}.npz"
  FIG_DIR="figs_${TAG}_Htest${H_TEST_STR}_P${P_STR}"

  echo "[INFO] Output model: ${MODEL_OUT}"
  echo "[INFO] Rollout NPZ: ${NPZ_OUT}"
  echo "[INFO] Figure dir : ${FIG_DIR}"

  # ----------------------------
  # Predict stress and plastic strain
  # ----------------------------
  python test_rollout.py \
      --data ../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz \
      --ckpt "${MODEL_OUT}" \
      --L "${L}" \
      --H "${H}" \
      --P "${P}" \
      --hid "${HID}" \
      --H_test "${H_TEST}" \
      --out_npz "${NPZ_OUT}"

  echo "[OK] Rollout completed for lambda_pstrain=${LAMBDA_P}."

  # ----------------------------
  # Plotting: stress–strain
  # ----------------------------
  python plot_rollout_stress_strain.py \
      --data_truth ../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz \
      --data_pred  "./${NPZ_OUT}" \
      --out_dir    "./${FIG_DIR}"

  echo "[OK] Rolling Stress Strain Plot completed for lambda_pstrain=${LAMBDA_P}."

  # ----------------------------
  # Plotting: invariants (p–q)
  # ----------------------------
  python plot_rollout_stress_invariants.py \
      --data_truth ../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz \
      --data_pred  "./${NPZ_OUT}" \
      --M 2.0 \
      --pc -6.0 \
      --out_dir "./${FIG_DIR}"

  echo "[OK] Rolling Stress Invariants Plot completed for lambda_pstrain=${LAMBDA_P}."
done

echo "[INFO] All lambda_pstrain sweeps completed."
