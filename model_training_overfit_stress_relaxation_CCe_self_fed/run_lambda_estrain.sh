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
EPOCHS=20000
LR=5e-4

# Sweep over these lambda_estrain values (strings so we can use them in filenames)
LAMBDAS=("1e4" "1e6" "1e7")

L_STR=$(printf "%02d" "$L")
H_STR=$(printf "%02d" "$H")
S_STR=$(printf "%02d" "$S")
H_TEST_STR=$(printf "%02d" "$H_TEST")

echo "[INFO] Using L=${L}, H=${H}, hid=${HID}, epochs=${EPOCHS}, lr=${LR}"
echo "[INFO] Sweeping over lambda_estrain values: ${LAMBDAS[*]}"

for LAMBDA_E in "${LAMBDAS[@]}"; do
  echo
  echo "==============================================="
  echo "[INFO] Starting run with lambda_estrain=${LAMBDA_E}"
  echo "==============================================="

  # Unique tags for this lambda
  TAG="lamE_${LAMBDA_E}_L${L_STR}_H${H_STR}_S${S_STR}_Htest${H_TEST_STR}"
  MODEL_OUT="model_overfit_${TAG}.pt"
  NPZ_OUT="test_rollout_${TAG}.npz"
  FIG_DIR="figs_${TAG}"

  echo "[INFO] Output model: ${MODEL_OUT}"
  echo "[INFO] Rollout NPZ: ${NPZ_OUT}"
  echo "[INFO] Figure dir : ${FIG_DIR}"

  # ----------------------------
  # Run training
  # ----------------------------
  python main.py \
      --data ../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz \
      --L "${L}" \
      --H "${H}" \
      --bs "${BS}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --hid "${HID}" \
      --lambda_estrain "${LAMBDA_E}" \
      --lambda_overlap 0.1 \
      --lambda_self 0.1 \
      --S "${S}" \
      --out "${MODEL_OUT}"

  echo "[OK] Training completed for lambda_estrain=${LAMBDA_E}."

  # ----------------------------
  # Predict stress and elastic strain
  # ----------------------------
  python test_rollout.py \
      --data ../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz \
      --ckpt "${MODEL_OUT}" \
      --L "${L}" \
      --H "${H}" \
      --hid "${HID}" \
      --H_test "${H_TEST}" \
      --out_npz "${NPZ_OUT}"

  echo "[OK] Rollout completed for lambda_estrain=${LAMBDA_E}."

  # ----------------------------
  # Plotting: stress–strain
  # ----------------------------
  python plot_rollout_stress_strain.py \
      --data_truth ../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz \
      --data_pred  "./${NPZ_OUT}" \
      --out_dir    "./${FIG_DIR}"

  echo "[OK] Rolling Stress Strain Plot completed for lambda_estrain=${LAMBDA_E}."

  # ----------------------------
  # Plotting: invariants (p–q)
  # ----------------------------
  python plot_rollout_stress_invariants.py \
      --data_truth ../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz \
      --data_pred  "./${NPZ_OUT}" \
      --M 2.0 \
      --pc -6.0 \
      --out_dir "./${FIG_DIR}"

  echo "[OK] Rolling Stress Invariants Plot completed for lambda_estrain=${LAMBDA_E}."
done

echo "[INFO] All lambda_estrain sweeps completed."
