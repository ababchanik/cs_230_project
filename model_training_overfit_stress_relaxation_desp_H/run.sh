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
except Exception:
    sys.exit("[error] torch not found. Try:\n  conda install -n seqml pytorch pytorch-cuda=12.1 -c pytorch -c nvidia\n  # or CPU-only:\n  conda install -n seqml pytorch cpuonly -c pytorch")
PY

# Hyperparameters
# ----------------------------
L=4
H=4
S=1
H_TEST=$((20 - L)) # max: T - (L - 1) - 1
HID=128
BS=128
EPOCHS=50000
LR=1e-4

echo "[INFO] Using L=${L}, H=${H}, hid=${HID}, epochs=${EPOCHS}, lr=${LR}"

# Run training
python main.py \
        --data ../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz \
        --L "${L}" \
        --H "${H}" \
        --bs "${BS}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --hid "${HID}" \
        --lambda_pstrain 1e7 \
        --lambda_overlap 0.1 \
        --lambda_self 0.1 \
        --S "${S}" \
        --out model_overfit.pt
echo "[OK] Training completed."

# Run testing
L_STR=$(printf "%02d" "$L")
H_STR=$(printf "%02d" "$H")
S_STR=$(printf "%02d" "$S")
H_TEST_STR=$(printf "%02d" "$H_TEST")

# Predict stress and elastic strain
python test_rollout.py \
        --data ../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz \
        --ckpt model_overfit.pt \
        --L "${L}" \
        --H "${H}" \
        --hid "${HID}" \
        --H_test "${H_TEST}" \
        --out_npz test_rollout_L${L_STR}_H${H_STR}_S${S_STR}_Htest${H_TEST_STR}.npz

#echo "[INFO] Cleaning old figures in ./figs ..."
#mkdir -p ./figs
#rm -f ./figs/* 2>/dev/null || true

# Plotting
python plot_rollout_stress_strain.py \
        --data_truth ../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz \
        --data_pred ./test_rollout_L${L_STR}_H${H_STR}_S${S_STR}_Htest${H_TEST_STR}.npz \
        --out_dir ./figs_L${L_STR}_H${H_STR}_S${S_STR}_Htest${H_TEST_STR}
echo "[OK] Rolling Stress Strain Plot completed."

python plot_rollout_stress_invariants.py \
        --data_truth ../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz \
        --data_pred ./test_rollout_L${L_STR}_H${H_STR}_S${S_STR}_Htest${H_TEST_STR}.npz \
        --M 2.0 \
        --pc -6.0 \
        --out_dir ./figs_L${L_STR}_H${H_STR}_S${S_STR}_Htest${H_TEST_STR}
echo "[OK] Rolling Stress Invariants Plot completed."
