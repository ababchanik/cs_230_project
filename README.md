# TI-Visco-MCC Small — Sequential Model (CS230)

Windowed Sequence-to-Sequence Learning for Transversely-Isotropic Modified Cam-Clay (TI-MCC)

Note: model_training_temp is the working dir

End-to-end pipeline to train and evaluate a sequential model on synthetic
transversely-isotropic elasto-viscoplastic Cam-Clay stress–strain trajectories.

This repository contains:

Synthetic data generation pipelines for multiple strain-increment protocols tailored to TI-MCC small-strain constitutive modeling.
Physics-guided neural network training code (GRU-based seq-to-seq model).
Rolling prediction and evaluation tools reproducing stress relaxation, p–q paths, and stress–strain curves.


## Repo layout

```
.
├── data_generation/
│   ├── protocol_0_random_walk/
│   │   ├── generate_data_random_walk.py
│   │   ├── prepare_strain_random_walk.py
│   │
│   ├── protocol_1_random_then_fixed/
│   │   ├── compute_invariants.py
│   │   ├── generate_data_random_then_fixed.py
│   │   ├── inspect_stress_invariants_fictitious.py
│   │
│   ├── protocol_2_differential_random_then_fixed/
│   │   ├── compute_invariants.py
│   │   ├── generate_data_differential_random_then_fixed.py
│   │   ├── inspect_stress_invariants_fictitious.py
│   │
│   ├── test_stress_relaxation/
│   ├── test_stress_update_implementation/
│   ├── inspect_stress_invariants.py
│   ├── material_parameters.csv
│   ├── tensor_utils.py
│   ├── update_stress_ti_mcc_small.py
│
├── model_training/
│   ├── constitutive_update.py
│   ├── data_loader.py
│   ├── job.slurm
│   ├── loss_functions.py
│   ├── main.py
│   ├── model.py
│   ├── split_dataset.py
│   ├── tensor_utils.py
│   ├── test_rollout.py
│   ├── plot_rollout_stress_invariants.py
│   ├── plot_rollout_stress_strain.py
│   ├── run.sh
│   ├── run_rollout_only.sh
│   ├── run_test_stress_relaxation.sh
│   ├── run_lambda_pstrain.sh
│
├── .gitignore
└── README.md
```

## Quick start

### 1) Create/activate the `seqml` conda env

Prefer Miniforge at `/home/groups/borja/hemiao/miniforge`.

Alternatively, you can use any environment containing the essential packages with similar versions.

| Package        | Recommended Version | Notes |
|----------------|--------------------|--------|
| Python         | 3.10.x             | Required for compatibility with Sherlock system Python and PyTorch builds |
| PyTorch        | 2.4.0              | GPU (CUDA 12.1) or CPU-only variant supported |
| NumPy          | 1.26.4             | Core array/tensor utilities |
| Matplotlib     | 3.9.2              | For figure generation in `test_stress_strain.py` |
| tqdm           | 4.66.5             | For progress bar logging |
| pandas         | 2.2.2              | (optional) For data preprocessing scripts |
| torchinfo      | 1.8.0              | (optional) For model summary display |

### 2) Run the full pipeline

From `./model_training`:

```bash
$ bash run.sh
```

This will:
- **split** `../data_generation/data_stress_strain.npz` into `data_stress_strain_labeled.npz`
- **train** and save `model.pt`
- **test** and save figures to `./figs/`

### 3) Run steps manually (optional)

```bash
# Split
(seqml) $ python split_dataset.py   --data ../data_generation/data_stress_strain.npz   --r_train 0.70 --r_dev 0.15 --r_eval 0.15 --seed 0

# Train
(seqml) $ python main.py   --data ./data_stress_strain_labeled.npz   --out ./model.pt   --L 10 --H 10 --bs 16 --epochs 20 --lr 1e-3 --hid 128

# Test
(seqml) $ python test_stress_strain.py   --data ./data_stress_strain_labeled.npz   --ckpt ./model.pt   --L 10 --H 10 --hid 128   --n_samples 5 --seed 0   --out_dir ./figs
```

## Notes on `run.sh`

The script:
- Activates `seqml` from `/home/groups/borja/hemiao/miniforge` (falls back to `~/miniforge`)
- Verifies `torch` is importable before running
- Writes figures to `model_training/figs/`

If activation fails, install/activate the env, then re-run.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'`**  
  You’re not in `seqml` or PyTorch isn’t installed there.  
  Activate and install:
  ```bash
  $ /home/groups/borja/hemiao/miniforge/bin/conda activate seqml
  $ conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia   # or cpuonly
  ```

## Reproducibility

- Dataset split is controlled by `--seed` in `split_dataset.py`
- Training seeds and dataloaders are set in `main.py` (adjust if you need full determinism)

%## Citation / Acknowledgment

%If you use this code or figures in reports for CS230 or related manuscripts, please acknowledge the TI-MCC sequential modeling pipeline by Borja Group (Stanford CEE).

---
*Maintainer:* Hemiao Zhang, Anna Babchanik (CEE, Stanford).
