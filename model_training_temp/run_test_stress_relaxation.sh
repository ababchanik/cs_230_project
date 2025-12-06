#!/usr/bin/env bash

# ------------------------------------------------------------------
# Usage:
#   ./run.sh
#
#   i_spec : which trajectory index to use for rollout & plots (default 0)
#   H_TEST : rollout horizon in steps (default 80)
# ------------------------------------------------------------------

H_TEST=16

L=4
H=4
S=1
HID=128
P=4

DATA="../data_generation/test_stress_relaxation/stress_relaxation_strain_rate_dt_1e-1.npz"
CKPT="./model_lamP_1e4_L04_H04_S01.pt"
ROLLOUT_NPZ="./rollout_results_L${L}_H${H}_S${S}_Htest${H_TEST}_relaxation.npz"
FIG_DIR="./figs_lamP_1e4_L04_H04_S01_H_test16_P04"

echo "[INFO] Using relaxation, H_TEST=${H_TEST}"
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
# Rollout on a chosen trajectory i_spec (multi-traj test_rollout.py)
# ------------------------------------------------------------------
python test_rollout.py \
  --data "${DATA}" \
  --ckpt "${CKPT}" \
  --L      ${L} \
  --H      ${H} \
  --H_test ${H_TEST} \
  --P      ${P} \
  --hid  ${HID} \
  --i_spec 0 \
  --out_npz "${ROLLOUT_NPZ}"

echo "[OK] Rollout completed for i_spec=${I_SPEC}, results -> ${ROLLOUT_NPZ}"

# ------------------------------------------------------------------
# Plot stress–strain for that trajectory (multi-traj plot script)
# ------------------------------------------------------------------
python plot_rollout_stress_strain.py \
  --data_truth "${DATA}" \
  --data_pred  "${ROLLOUT_NPZ}" \
  --out_dir    "${FIG_DIR}_relaxation"

echo "[OK] Stress–strain rollout plots saved to ${FIG_DIR}"

# ------------------------------------------------------------------
# Plot p–q invariants for that trajectory (multi-traj plot script)
# ------------------------------------------------------------------
python plot_rollout_stress_invariants.py \
  --data_truth "${DATA}" \
  --data_pred  "${ROLLOUT_NPZ}" \
  --out_dir    "${FIG_DIR}_relaxation" \
  --M  2.0 \
  --pc -6.0

echo "[OK] Stress invariant rollout plots saved to ${FIG_DIR}"
echo "[OK] All done."
