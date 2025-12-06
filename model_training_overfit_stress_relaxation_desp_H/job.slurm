#!/bin/bash
#SBATCH --job-name=ti_mcc_single_trajectory_deps_H_P1
#SBATCH --output=ti_mcc_rollout.out
#SBATCH --error=ti_mcc_rollout.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1          
#SBATCH --partition=serc
#SBATCH --time=0:30:00
#SBATCH --mem=4G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hemiaozh@stanford.edu

echo "[INFO] Job started on $(hostname) at $(date)"

# Call your run script (the one you pasted earlier)
#bash run.sh     
bash run_lambda_pstrain.sh

echo "[INFO] Job finished at $(date)"
