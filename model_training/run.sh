# #!/usr/bin/env bash
# ################################
# ####### RUN myrun.sh not this file ##########
# ##############################
# # Activate training and testing environment
# activate() {
#   local d="$1"
#   [[ -d "$d" ]] || return 1
#   [[ -f "$d/etc/profile.d/conda.sh" ]] && source "$d/etc/profile.d/conda.sh"
#   if command -v conda >/dev/null 2>&1; then
#     eval "$(conda shell.bash hook)"
#     conda activate seqml >/dev/null 2>&1 && return 0
#   fi
#   return 1
# }
# activate "/home/groups/borja/hemiao/miniforge" || activate "$HOME/miniforge" || {
#   echo "[error] Could not activate 'seqml' conda env."; exit 1; }

# # Verify torch
# python - <<'PY'
# try:
#     import torch, sys
#     print(f"[OK] torch {torch.__version__}, CUDA={torch.cuda.is_available()}")
# except Exception:
#     sys.exit("[error] torch not found. Try:\n  conda install -n seqml pytorch pytorch-cuda=12.1 -c pytorch -c nvidia\n  # or CPU-only:\n  conda install -n seqml pytorch cpuonly -c pytorch")
# PY

# # Label the dataset by train/dev/test
# python split_dataset.py \
#         --data  ../data_generation/protocol_2_differential_random_them_fixed/data_stress_strain_differential_random_then_fixed.npz\
#         --r_train  0.70\
#         --r_dev 0.15\
#         --r_eval 0.15 \
#         --seed 0
# echo "[OK] Dataset randomly splitted to train/dev/test."

# # Run training
# python main.py \
#         --data ./data_stress_strain_differential_random_then_fixed_labeled.npz \
#         --out ./model.pt \
#         --L 10 \
#         --H 10 \
#         --bs 16 \
#         --epochs 20 \
#         --lr 1e-5 \
#         --hid 128 
# echo "[OK] Training completed."

# # Run testing
# python test_stress_strain.py \
#         --data ./data_stress_strain_differential_random_then_fixed_labeled.npz \
#         --ckpt model.pt \
#         --L 50 \
#         --H 50 \
#         --hid 128 \
#         --n_samples 5 \
#         --seed 0 \
#         --out_dir ./figs
# echo "[OK] Testing completed."

# python test_stress_invariants_rolling.py \
#     --data ./data_stress_strain_differential_random_then_fixed_labeled.npz \
#     --ckpt model_voigt.pt \
#     --L 10 \
#     --H 10 \
#     --H_test 45 \
#     --hid 128 \
#     --n_samples 10 \
#     --seed 0 \
#     --out_dir ./figs \
#     --M 2.0 \
#     --pc -6



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

# Label the dataset by train/dev/test
python split_dataset.py \
        --data  ../data_generation/protocol_2_differential_random_them_fixed/data_stress_strain_differential_random_then_fixed.npz\
        --r_train  0.70\
        --r_dev 0.15\
        --r_eval 0.15 \
        --seed 0
echo "[OK] Dataset randomly splitted to train/dev/test."

# Run training
python main.py \
        --data ./data_stress_strain_differential_random_then_fixed_labeled.npz \
        --out ./model.pt \
        --L 10 \
        --H 10 \
        --bs 16 \
        --epochs 20 \
        --lr 1e-3 \
        --hid 128 \
        --lambda_overlap 1
echo "[OK] Training completed."

# Run testing
python test_stress_strain.py \
        --data ./data_stress_strain_differential_random_then_fixed_labeled.npz \
        --ckpt model.pt \
        --L 10 \
        --H 10 \
        --hid 128 \
        --n_samples 5 \
        --seed 0 \
        --out_dir ./figs
echo "[OK] Stress Strain Plot completed."

python test_stress_strain_rolling.py \
        --data ./data_stress_strain_differential_random_then_fixed_labeled.npz \
        --ckpt model.pt \
        --L 10 \
        --H 10 \
        --H_test 80 \
        --hid 128 \
        --n_samples 5 \
        --seed 0 \
        --out_dir ./figs
echo "[OK] Rolling Stress Strain Plot completed."

python test_stress_invariants.py \
    --data ./data_stress_strain_differential_random_then_fixed_labeled.npz \
    --ckpt model.pt \
    --L 10 \
    --H 10 \
    --hid 128 \
    --n_samples 10 \
    --seed 0 \
    --out_dir ./figs \
    --M 2.0 \
    --pc -6
echo "[OK] Stress Invariants Plot completed."

python test_stress_invariants_rolling.py \
    --data ./data_stress_strain_differential_random_then_fixed_labeled.npz \
    --ckpt model.pt \
    --L 10 \
    --H 10 \
    --H_test 45 \
    --hid 128 \
    --n_samples 10 \
    --seed 0 \
    --out_dir ./figs \
    --M 2.0 \
    --pc -6
echo "[OK] Rolling Stress Invariants Plot completed."
