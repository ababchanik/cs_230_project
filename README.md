# TI-Visco-MCC Small — Sequential Model (CS230)

Note: model_training_temp is the working dir

End-to-end pipeline to train and evaluate a sequential model on synthetic
transversely-isotropic elasto-viscoplastic Cam-Clay stress–strain trajectories.

## Repo layout

```
./
├─ data_generation/
│  ├─ data_stress_strain.npz         # input dataset (eps, epse, deps, sig, …)
│  ├─ generate_data_random_walk.py    # (optional) data generator
│  └─ inspect_stress_invariants.py    # (optional) p–q inspectors
├─ model_training/
│  ├─ split_dataset.py                # adds train/dev/eval labels
│  ├─ model.py                        # SeqModel and losses
│  ├─ main.py                         # training entry
│  ├─ test_stress_strain.py           # evaluation/plotting
│  ├─ tensor_utils.py                 # Voigt helpers
│  ├─ run.sh                          # one-shot split + train + test
│  ├─ figs/                           # output figures
│  └─ train_log.txt                   # device/loss logs
└─ README.md
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
