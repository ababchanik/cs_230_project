# # # myrun.sh
# # #!/usr/bin/env bash

# # # Activate environment
# # activate() {
# #   local d="$1"
# #   [[ -d "$d" ]] || return 1
# #   [[ -f "$d/etc/profile.d/conda.sh" ]] && source "$d/etc/profile.d/conda.sh"
# #   if command -v conda >/dev/null 2>&1; then
# #     eval "$(conda shell.bash hook)"
# #     conda activate seqml >/dev/null 2>&1 && return 0
# #   fi
# #   return 1
# # }
# # activate "/home/groups/borja/hemiao/miniforge" || activate "$HOME/miniforge" || {
# #   echo "[error] Could not activate 'seqml' conda env."; exit 1; }

# # echo "[OK] Environment activated."

# # # Paths
# # ORIGINAL_DATA="../data_generation/protocol_1_random_then_fixed/data_stress_strain_random_then_fixed.npz"
# # LABELED_DATA="./data_stress_strain_random_then_fixed_labeled.npz"  # Created in current dir

# # # Check if original data exists
# # if [ ! -f "$ORIGINAL_DATA" ]; then
# #     echo "[ERROR] Original data not found: $ORIGINAL_DATA"
# #     echo "Current directory: $(pwd)"
# #     echo "Files in data_generation:"
# #     ls -la ../data_generation/protocol_1_random_then_fixed/ 2>/dev/null || echo "Cannot list"
# #     exit 1
# # fi

# # echo "Original data found: $ORIGINAL_DATA"

# # # Label dataset if needed
# # if [ ! -f "$LABELED_DATA" ]; then
# #     echo "Creating labeled dataset..."
# #     python split_dataset.py \
# #         --data "$ORIGINAL_DATA" \
# #         --r_train 0.70 \
# #         --r_dev 0.15 \
# #         --r_eval 0.15 \
# #         --seed 0
# # fi

# # if [ ! -f "$LABELED_DATA" ]; then
# #     echo "[ERROR] Failed to create labeled data!"
# #     exit 1
# # fi

# # echo "Using labeled data: $LABELED_DATA"

# # # Verify the labeled data
# # echo "Verifying labeled data..."
# # python -c "
# # import numpy as np
# # data = np.load('$LABELED_DATA', allow_pickle=True)
# # print('Keys:', list(data.keys()))
# # print('Shapes:')
# # for k in ['eps', 'epse', 'deps', 'sig', 'split']:
# #     if k in data:
# #         print(f'  {k}: {data[k].shape}')
# # print('Split distribution:')
# # if 'split' in data:
# #     unique, counts = np.unique(data['split'], return_counts=True)
# #     for u, c in zip(unique, counts):
# #         print(f'  {u}: {c} ({100*c/len(data[\"split\"]):.1f}%)')
# # "

# # # Training parameters - adjusted for your data
# # L=1  # Lookback window
# # H=1  # Prediction horizon
# # BS=64  # Batch size
# # EPOCHS=30
# # HID=128

# # # "fro"
# # # Train with each loss function
# # for loss_fn in "lie" "voigt" ; do
# #     echo "========================================="
# #     echo "Training with loss: $loss_fn"
# #     echo "========================================="
    
# #     model_out="model_${loss_fn}.pt"
# #     figs_dir="figs_${loss_fn}"
# #     mkdir -p "$figs_dir"
    
# #     # Set learning rate
# #     if [ "$loss_fn" = "lie" ]; then
# #         lr=0.0003
# #     else
# #         lr=0.001
# #     fi
    
# #     # Run training
# #     echo "Starting training: L=$L, H=$H, bs=$BS, lr=$lr, epochs=$EPOCHS"
# #     python main.py \
# #         --data "$LABELED_DATA" \
# #         --out "$model_out" \
# #         --L $L \
# #         --H $H \
# #         --bs $BS \
# #         --epochs $EPOCHS \
# #         --lr $lr \
# #         --hid $HID \
# #         --loss "$loss_fn"
    
# #     if [ $? -eq 0 ]; then
# #         echo "[OK] Training completed for $loss_fn"
# #     else
# #         echo "[WARNING] Training failed for $loss_fn"
# #         continue
# #     fi
# #     echo "Creating debug plots..."
# #     python debug_plots.py \
# #         --data "$LABELED_DATA" \
# #         --ckpt "$model_out" \
# #         --L $L \
# #         --H $H \
# #         --hid $HID \
# #         --out_dir "$figs_dir"
# #     echo "[OK] Debug plots created for $loss_fn"

# #     # Test plots
# #     echo "Creating test plots..."
# #     python test_stress_strain.py \
# #         --data "$LABELED_DATA" \
# #         --ckpt "$model_out" \
# #         --L $L \
# #         --H $H \
# #         --hid $HID \
# #         --n_samples 3 \
# #         --seed 0 \
# #         --out_dir "$figs_dir"
    
# #     echo "[OK] Completed $loss_fn"

# #     echo "Creating rolling plots..."
# #     python test_stress_invariants_rolling.py \
# #         --data "$LABELED_DATA" \
# #         --ckpt "$model_out" \
# #         --L $L \
# #         --H $H \
# #         --hid $HID \
# #         --n_samples 10 \
# #         --seed 0 \
# #         --out_dir "$figs_dir"
# #         --H_test 45 \
# #         --hid 128 \
# #         --M 2.0 \
# #         --pc -6
# #     echo "[OK] Rolling Stress Invariants Plot completed."
# # done

# # echo "========================================="
# # echo "All training completed!"
# # echo "========================================="


# # myrun.sh
# #!/usr/bin/env bash

# # Activate environment
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

# echo "[OK] Environment activated."

# # Paths
# ORIGINAL_DATA="../data_generation/protocol_2_differential_random_them_fixed/data_stress_strain_differential_random_then_fixed.npz"
# LABELED_DATA="./data_stress_strain_differential_random_then_fixed_labeled.npz"  # Created in current dir

# # Check if original data exists
# if [ ! -f "$ORIGINAL_DATA" ]; then
#     echo "[ERROR] Original data not found: $ORIGINAL_DATA"
#     echo "Current directory: $(pwd)"
#     echo "Files in data_generation:"
#     ls -la ../data_generation/protocol_2_differential_random_them_fixed/ 2>/dev/null || echo "Cannot list"
#     exit 1
# fi

# echo "Original data found: $ORIGINAL_DATA"

# # Label dataset if needed
# if [ ! -f "$LABELED_DATA" ]; then
#     echo "Creating labeled dataset..."
#     python split_dataset.py \
#         --data "$ORIGINAL_DATA" \
#         --r_train 0.70 \
#         --r_dev 0.15 \
#         --r_eval 0.15 \
#         --seed 0
# fi

# if [ ! -f "$LABELED_DATA" ]; then
#     echo "[ERROR] Failed to create labeled data!"
#     exit 1
# fi

# echo "Using labeled data: $LABELED_DATA"

# # Verify the labeled data
# echo "Verifying labeled data..."
# python -c "
# import numpy as np
# data = np.load('$LABELED_DATA', allow_pickle=True)
# print('Keys:', list(data.keys()))
# print('Shapes:')
# for k in ['eps', 'epse', 'deps', 'sig', 'split']:
#     if k in data:
#         print(f'  {k}: {data[k].shape}')
# print('Split distribution:')
# if 'split' in data:
#     unique, counts = np.unique(data['split'], return_counts=True)
#     for u, c in zip(unique, counts):
#         print(f'  {u}: {c} ({100*c/len(data[\"split\"]):.1f}%)')
# "

# # Training parameters 
# L=4  # Lookback window
# H=4  # Prediction horizon
# BS=128  # Batch size
# EPOCHS=20
# HID=128

# # Train with each loss function
# for loss_fn in "lie" "voigt"; do
#     echo "========================================="
#     echo "Training with loss: $loss_fn"
#     echo "========================================="
    
#     model_out="model_${loss_fn}.pt"
#     figs_dir="figs_${loss_fn}"
#     mkdir -p "$figs_dir"
    
#     # Set learning rate
#     if [ "$loss_fn" = "lie" ]; then
#         lr=0.0004
#     else
#         lr=0.001
#     fi
    
#     # Run training
#     echo "Starting training: L=$L, H=$H, bs=$BS, lr=$lr, epochs=$EPOCHS"
#     python main.py \
#         --data "$LABELED_DATA" \
#         --out "$model_out" \
#         --L $L \
#         --H $H \
#         --bs $BS \
#         --epochs $EPOCHS \
#         --lr $lr \
#         --hid $HID \
#         --loss "$loss_fn"
    
#     if [ $? -eq 0 ]; then
#         echo "[OK] Training completed for $loss_fn"
#     else
#         echo "[WARNING] Training failed for $loss_fn"
#         continue
#     fi
    
#     # =========================================
#     # SINGLE PREDICTION EVALUATION 
#     # =========================================
#     echo "Creating single prediction plots..."
#     python test_stress_strain.py \
#         --data "$LABELED_DATA" \
#         --ckpt "$model_out" \
#         --L $L \
#         --H $H \
#         --hid $HID \
#         --n_samples 3 \
#         --out_dir "${figs_dir}_single"
#     echo "[OK] Single prediction plots completed"
    
#     # =========================================
#     # ROLLING PREDICTION EVALUATION 
#     # =========================================
#     echo "Creating rolling prediction plots..."
#     python test_stress_strain.py \
#         --data "$LABELED_DATA" \
#         --ckpt "$model_out" \
#         --L $L \
#         --H $H \
#         --H_test 45 \
#         --rolling \
#         --hid $HID \
#         --n_samples 3 \
#         --out_dir "${figs_dir}_rolling"
#     echo "[OK] Rolling prediction plots completed"
    
#     # =========================================
#     # STRESS INVARIANTS 
#     # =========================================
#     echo "Creating stress invariants plots..."
#     python test_stress_invariants_rolling.py \
#         --data "$LABELED_DATA" \
#         --ckpt "$model_out" \
#         --L $L \
#         --H $H \
#         --H_test 45 \
#         --hid $HID \
#         --n_samples 3 \
#         --seed 0 \
#         --out_dir "${figs_dir}_invariants" \
#         --M 2.0 \
#         --pc -6.0
#     echo "[OK] Stress invariants plots completed"
    
#     echo "[OK] Completed all evaluations for $loss_fn"
# done

# echo "========================================="
# echo "All training and evaluation completed!"
# echo "========================================="



## above is tha tlast that worked

####### HEmiao
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
python main.py \
  --data "${DATA_LABELED}" \
  --out  "${CKPT}" \
  --L    ${L} \
  --H    ${H} \
  --S    ${S} \
  --bs   1024 \
  --epochs 500 \
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

