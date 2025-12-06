#!/bin/bash
#SBATCH --job-name=ti_mcc_multi_trajectory
#SBATCH --output=ti_mcc_rollout.out
#SBATCH --error=ti_mcc_rollout.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1          
#SBATCH --partition=serc
#SBATCH --cpus-per-task=10
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hemiaozh@stanford.edu

echo "[INFO] Job started on $(hostname) at $(date)"

# Call your run script (the one you pasted earlier)
#bash run.sh     
bash run_lambda_pstrain.sh

echo "[INFO] Job finished at $(date)"
