# TF-STNet: Wind-Speed Forecasting

This repository contains the code and example files for the **TF-STNet**
wind-speed forecasting model described in the accompanying manuscript. The
model combines three views of the input signal:

- gated temporal convolutions for recent observations and numerical weather
  prediction (NWP) sequences;
- bidirectional graph convolution for spatial interactions between stations;
- frequency-aware attention over the nearest NWP grid points.

The provided checkpoint maps **24 hours of history to the following 24 hours**
at **60 stations**.

## What is included

| Component | Description |
| --- | --- |
| `model/` | TF-STNet architecture and temporal, graph, and frequency-aware layers |
| `checkpoints/tf_stnet_wind_speed.pth` | PyTorch checkpoint |
| `data/example_test_batch.npz` | One anonymized test batch with model inputs, targets, coordinates, and normalization statistics |
| `data/adj_mat.pkl` | Spatial adjacency matrix used by the model |
| `configs/model.json` | Key architecture and experiment settings |
| `run.py` | Deterministic command-line inference entry point |
| `trainer.py` | Self-contained training loop, inverse scaling, and regression metrics |
| `run_wind_speed.py` | Training/validation/test command-line entry point |
| `configs/train.json` | Default training configuration |

The complete meteorological dataset and geographic labels are not included
because they are not publicly redistributable. The training code expects a
prepared `.npz` archive, described below.

## Quick start

Python 3.9 or newer is recommended. Create an isolated environment and install
the pinned dependencies:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the example on CPU:

```bash
python run.py
```

To use a CUDA device, provide a valid PyTorch device string:

```bash
python run.py --device cuda:0
```

All paths are configurable, which makes it possible to evaluate a different
compatible batch or checkpoint without modifying the source code:

```bash
python run.py \
  --data path/to/test_batch.npz \
  --checkpoint path/to/model.pth \
  --adjacency path/to/adj_mat.pkl \
  --output-dir outputs \
  --device cpu
```

## Outputs and reference result

The command creates the output directory (if needed) and writes:

- `prediction.npy`: inverse-transformed predictions with shape
  `[batch, station, horizon]`;
- `metrics.json`: batch size, prediction shape, device, and overall MAE.

For the included 64-sample example batch, `outputs/metrics.json` reports:

```text
prediction shape: [64, 60, 24]
CPU MAE:          0.899071
```

The exact value can vary slightly with a different PyTorch/CUDA runtime. The
script validates required array names, tensor shapes, finite values, graph
dimensions, and target normalization before loading the checkpoint, so input
format errors are reported early.

## Reproduce training and evaluation

`run_wind_speed.py` is the training entry point for TF-STNet. It uses Adam
optimization, physical-unit MAE as the training objective, deterministic
seeding, validation-MAE model selection, and a final held-out test evaluation.
The best model is saved as
`best_model.pth`; `history.json`, `train_config.json`, and `test_metrics.json`
record the settings and metrics needed to audit a run.

The example batch is intended to check the inference path; it is not a
training set. To reproduce the paper experiment, supply the complete prepared
archive (or an equivalent archive with the format below):

```bash
python run_wind_speed.py \
  --data path/to/wind_speed_train.npz \
  --config configs/train.json \
  --device cuda:0 \
  --epochs 50
```

Before a long run, validate the archive, graph, and model wiring without
updating weights:

```bash
python run_wind_speed.py \
  --data path/to/wind_speed_train.npz \
  --dry-run
```

For an archive containing unsplit arrays, the script makes a seeded random
split using `val_fraction` and `test_fraction` from the config. For a paper
reproduction, explicit split arrays are preferred: use the prefixes
`train_`, `val_`, and `test_` (for example, `train_obs_his`). This prevents
accidental changes to the train/validation/test membership when comparing
runs.

### Training archive format

Each split uses the same arrays as the inference batch. The first dimension is
the number of samples `B`; station and temporal dimensions are fixed by the
provided checkpoint:

```text
obs_his                 [B, 60, 24]              normalized station history
obs_fut                 [B, 60, 24]              future station targets (physical units)
nwp_his, nwp_fut       [B, latitude, longitude, 24]
his_mark, fut_mark     [B, 24, 5]                temporal covariates
station_coordinates     [60, 2]                  station coordinates
grid_coordinates        [latitude, longitude, 2] NWP grid coordinates
target_mean, target_std scalar                   target inverse-scaling statistics
```

`target_mean` and `target_std` are shared scalar statistics from the training
set. The trainer compares inverse-transformed predictions with `obs_fut` in
physical units, while the model checkpoint stores the same normalized output
parameterization used by `run.py`. The adjacency file is the tuple
`(sensor_ids, id_map, adjacency_matrix)` documented above.

### Reported metrics

Every epoch and the final test evaluation report MAE, RMSE, symmetric MAPE,
and Pearson correlation over all samples, stations, and forecast horizons.
These definitions are implemented in `trainer.py`, so the reported values do
not depend on an external metrics package.

## Input format

`run.py` expects an `.npz` archive containing the following arrays:

| Key | Expected shape | Role |
| --- | --- | --- |
| `obs_his` | `[B, 60, 24]` | Historical station wind speed (normalized) |
| `obs_fut` | `[B, 60, 24]` | Ground-truth future station wind speed |
| `nwp_his` | `[B, latitude, longitude, 24]` | Historical NWP field |
| `nwp_fut` | `[B, latitude, longitude, 24]` | Future NWP field |
| `his_mark`, `fut_mark` | `[B, 24, 5]` | Temporal covariates |
| `station_coordinates` | `[60, 2]` | Anonymized station coordinates |
| `grid_coordinates` | `[latitude, longitude, 2]` | NWP grid coordinates |
| `target_mean`, `target_std` | scalar | Statistics for inverse scaling |

The adjacency pickle follows the original experiment format:
`(sensor_ids, id_map, adjacency_matrix)`.

## Repository layout

```text
.
├── checkpoints/
│   └── tf_stnet_wind_speed.pth
├── configs/
│   ├── model.json
│   └── train.json
├── data/
│   ├── adj_mat.pkl
│   └── example_test_batch.npz
├── model/
│   ├── layers/
│   │   ├── frequency.py
│   │   ├── gcn.py
│   │   └── tcn.py
│   └── tf_stnet.py
├── trainer.py
├── run_wind_speed.py
├── requirements.txt
└── run.py
```

## Reproducibility notes

- Inference runs in evaluation mode with `torch.inference_mode()`.
- The random seed is fixed to `2026` for consistent execution.
- CPU is the default device; CUDA is optional.
- Generated files under `outputs/` and `runs/`, IDE metadata, Python caches,
  logs, and virtual environments are excluded by `.gitignore`.

## Citation

If you use this code, please cite the accompanying TF-STNet manuscript.
