# TF-STNet: Reproducible Inference Release

This repository provides the official inference implementation and a compact
release package for the **TF-STNet** wind-speed forecasting model described in
the accompanying manuscript. TF-STNet combines three complementary views of
the input signal:

- gated temporal convolutions for recent observations and numerical weather
  prediction (NWP) sequences;
- bidirectional graph convolution for spatial interactions between stations;
- frequency-aware attention over the nearest NWP grid points.

The released checkpoint maps **24 hours of history to the following 24 hours**
at **60 stations**. The package is designed for reviewers and researchers who
want to inspect the implementation and reproduce the released test-batch
inference without access to the complete, non-public meteorological dataset.

## What is included

| Component | Description |
| --- | --- |
| `model/` | TF-STNet architecture and temporal, graph, and frequency-aware layers |
| `checkpoints/tf_stnet_wind_speed.pth` | Released PyTorch checkpoint |
| `data/example_test_batch.npz` | One anonymized test batch with model inputs, targets, coordinates, and normalization statistics |
| `data/adj_mat.pkl` | Spatial adjacency matrix used by the released model |
| `configs/model.json` | Key architecture and experiment settings |
| `run.py` | Deterministic command-line inference entry point |

The complete training dataset, geographic labels, and training script are not
included because the underlying meteorological data are not publicly
redistributable. Training is therefore outside the scope of this release.

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

Run the released example on CPU:

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

For the included 64-sample release batch, the packaged reference metadata
reports:

```text
prediction shape: [64, 60, 24]
CPU MAE:          0.899071
```

The exact value can vary slightly with a different PyTorch/CUDA runtime. The
script validates required array names, tensor shapes, finite values, graph
dimensions, and target normalization before loading the checkpoint, so input
format errors are reported early.

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
│   └── model.json
├── data/
│   ├── adj_mat.pkl
│   └── example_test_batch.npz
├── model/
│   ├── layers/
│   │   ├── frequency.py
│   │   ├── gcn.py
│   │   └── tcn.py
│   └── tf_stnet.py
├── requirements.txt
└── run.py
```

## Reproducibility notes

- Inference runs in evaluation mode with `torch.inference_mode()`.
- The random seed is fixed to `2026` for consistent execution.
- CPU is the default device; CUDA is optional.
- Generated files under `outputs/`, IDE metadata, Python caches, and virtual
  environments are excluded by `.gitignore`.

## Citation

If you use this release, please cite the accompanying TF-STNet manuscript and
retain the release version when reporting results.
