"""Train, select, and evaluate TF-STNet on a prepared wind-speed archive.

The formal experiment data are not part of this repository.  Pass an ``.npz``
archive containing either unsplit arrays (the script makes a deterministic
split) or arrays prefixed with ``train_``, ``val_``, and ``test_``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from model import TFSTNet, TFSTNetConfig
from run import load_adjacency
from trainer import INPUT_KEYS, TargetScaler, Trainer


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "train.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the released TF-STNet model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, required=False, help="Prepared training .npz archive.")
    parser.add_argument("--adjacency", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--test-fraction", type=float)
    parser.add_argument("--dry-run", action="store_true", help="Validate the archive and model, then exit.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Training config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Training config must be a JSON object.")
    return config


def merge_args(args: argparse.Namespace, config: dict) -> dict:
    values = dict(config)
    cli_names = {
        "data": "data",
        "adjacency": "adjacency",
        "output_dir": "output_dir",
        "device": "device",
        "epochs": "epochs",
        "batch_size": "batch_size",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "seed": "seed",
        "val_fraction": "val_fraction",
        "test_fraction": "test_fraction",
    }
    for namespace_name, config_name in cli_names.items():
        value = getattr(args, namespace_name)
        if value is not None:
            values[config_name] = str(value) if isinstance(value, Path) else value
    defaults = {
        "data": None,
        "adjacency": "data/adj_mat.pkl",
        "output_dir": "runs/tf_stnet",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "seed": 2026,
        "val_fraction": 0.1,
        "test_fraction": 0.1,
    }
    for key, value in defaults.items():
        values.setdefault(key, value)
    if values["data"] is None:
        raise ValueError("A training archive is required. Pass --data path/to/train.npz.")
    if int(values["epochs"]) <= 0 or int(values["batch_size"]) <= 0:
        raise ValueError("epochs and batch_size must be positive integers.")
    if float(values["learning_rate"]) <= 0 or float(values["weight_decay"]) < 0:
        raise ValueError("learning_rate must be positive and weight_decay cannot be negative.")
    if not 0 <= float(values["val_fraction"]) < 1 or not 0 <= float(values["test_fraction"]) < 1:
        raise ValueError("val_fraction and test_fraction must be in [0, 1).")
    if float(values["val_fraction"]) + float(values["test_fraction"]) >= 1:
        raise ValueError("val_fraction + test_fraction must be less than 1.")
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _find_key(files: set[str], split: str, key: str) -> Optional[str]:
    prefixed = f"{split}_{key}"
    if prefixed in files:
        return prefixed
    return key if split == "all" and key in files else None


def canonicalize_layout(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Accept both loader-style time-first arrays and model-style arrays."""

    config = TFSTNetConfig()
    result = dict(arrays)
    for key in ("obs_his", "obs_fut"):
        if key not in result:
            continue
        value = result[key]
        if value.ndim == 3 and value.shape[1:] == (config.history_len, config.num_nodes):
            result[key] = value.transpose(0, 2, 1)
    for key in ("nwp_his", "nwp_fut"):
        if key not in result:
            continue
        value = result[key]
        if value.ndim == 4 and value.shape[1] == config.history_len:
            result[key] = value.transpose(0, 2, 3, 1)
    return result


def load_archive(
    path: Path,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], TargetScaler]:
    """Load explicit splits or create a deterministic split from base arrays."""

    if not path.is_file():
        raise FileNotFoundError(f"Training archive not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    arrays = canonicalize_layout(arrays)
    files = set(arrays)
    mean_key = "target_mean" if "target_mean" in files else "train_target_mean"
    std_key = "target_std" if "target_std" in files else "train_target_std"
    if mean_key not in files or std_key not in files:
        raise ValueError("The archive must contain scalar target_mean and target_std arrays.")
    scaler = TargetScaler(float(np.asarray(arrays[mean_key]).reshape(())), float(np.asarray(arrays[std_key]).reshape(())))

    explicit = all(_find_key(files, split, key) for split in ("train", "val", "test") for key in INPUT_KEYS)
    splits: dict[str, dict[str, np.ndarray]] = {}
    if explicit:
        for split in ("train", "val", "test"):
            split_arrays = {key: arrays[_find_key(files, split, key)] for key in INPUT_KEYS}  # type: ignore[index]
            splits[split] = canonicalize_layout(split_arrays)
    else:
        missing = [key for key in INPUT_KEYS if key not in files]
        if missing:
            raise ValueError(f"Archive is missing arrays: {missing}")
        n_samples = int(arrays["obs_his"].shape[0])
        if n_samples < 3:
            raise ValueError("At least three samples are required for train/validation/test splits.")
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)
        n_test = max(1, int(round(n_samples * test_fraction)))
        n_val = max(1, int(round(n_samples * val_fraction)))
        if n_test + n_val >= n_samples:
            raise ValueError("Fractions leave no samples for training.")
        index_map = {
            "test": indices[:n_test],
            "val": indices[n_test : n_test + n_val],
            "train": indices[n_test + n_val :],
        }
        splits = {
            split: {key: arrays[key][split_indices] if arrays[key].ndim > 0 and arrays[key].shape[0] == n_samples else arrays[key] for key in INPUT_KEYS}
            for split, split_indices in index_map.items()
        }
    validate_splits(splits)
    return splits, scaler


def validate_splits(splits: dict[str, dict[str, np.ndarray]]) -> None:
    config = TFSTNetConfig()
    for split, batch in splits.items():
        required = set(INPUT_KEYS)
        if set(batch) != required:
            raise ValueError(f"{split} split must contain exactly {sorted(required)}.")
        n_samples = batch["obs_his"].shape[0]
        if n_samples == 0:
            raise ValueError(f"{split} split is empty.")
        expected = {
            "obs_his": (n_samples, config.num_nodes, config.history_len),
            "obs_fut": (n_samples, config.num_nodes, config.prediction_len),
            "his_mark": (n_samples, config.history_len, 5),
            "fut_mark": (n_samples, config.prediction_len, 5),
            "station_coordinates": (config.num_nodes, 2),
        }
        for key, shape in expected.items():
            if batch[key].shape != shape:
                raise ValueError(f"{split}/{key} has shape {batch[key].shape}; expected {shape}.")
        for key in ("nwp_his", "nwp_fut"):
            value = batch[key]
            if value.ndim != 4 or value.shape[0] != n_samples or value.shape[-1] != config.history_len:
                raise ValueError(f"{split}/{key} must have shape [B, latitude, longitude, 24], got {value.shape}.")
        grid_shape = batch["nwp_fut"].shape[1:3] + (2,)
        if batch["grid_coordinates"].shape != grid_shape:
            raise ValueError(f"{split}/grid_coordinates has shape {batch['grid_coordinates'].shape}; expected {grid_shape}.")
        if not all(np.isfinite(value).all() for value in batch.values()):
            raise ValueError(f"{split} split contains NaN or infinite values.")


def iter_batches(
    split: dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> Iterator[dict[str, np.ndarray]]:
    n_samples = split["obs_his"].shape[0]
    indices = rng.permutation(n_samples) if shuffle else np.arange(n_samples)
    for start in range(0, n_samples, batch_size):
        batch_indices = indices[start : start + batch_size]
        yield {
            key: value[batch_indices] if value.ndim > 0 and value.shape[0] == n_samples else value
            for key, value in split.items()
        }


def main() -> None:
    args = parse_args()
    settings = merge_args(args, load_config(args.config))
    set_seed(int(settings["seed"]))
    device = torch.device(settings["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")

    data_path = Path(settings["data"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    splits, target_scaler = load_archive(
        data_path,
        float(settings["val_fraction"]),
        float(settings["test_fraction"]),
        int(settings["seed"]),
    )
    adjacency_path = Path(settings["adjacency"])
    if not adjacency_path.is_absolute():
        adjacency_path = ROOT / adjacency_path
    adjacency_forward, adjacency_backward = load_adjacency(adjacency_path, TFSTNetConfig.num_nodes)
    supports = (
        torch.from_numpy(adjacency_forward).to(device),
        torch.from_numpy(adjacency_backward).to(device),
    )
    model = TFSTNet(supports, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    trainer = Trainer(model, optimizer, target_scaler, device)

    sizes = {name: int(value["obs_his"].shape[0]) for name, value in splits.items()}
    print(json.dumps({"device": str(device), "samples": sizes, "target_mean": target_scaler.mean, "target_std": target_scaler.std}, indent=2))
    if args.dry_run:
        return

    output_dir = Path(settings["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    rng = np.random.default_rng(int(settings["seed"]))
    history = []
    best_value = float("inf")
    best_epoch = -1
    for epoch in range(1, int(settings["epochs"]) + 1):
        train_metrics = trainer.run_epoch(iter_batches(splits["train"], int(settings["batch_size"]), True, rng), True)
        val_metrics = trainer.run_epoch(iter_batches(splits["val"], int(settings["batch_size"]), False, rng), False)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if val_metrics["mae"] < best_value:
            best_value = val_metrics["mae"]
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pth")

    if best_epoch < 0:
        raise RuntimeError("No checkpoint was selected; validation split was empty.")
    model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=device))
    test_metrics = trainer.run_epoch(iter_batches(splits["test"], int(settings["batch_size"]), False, rng), False)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    result = {"best_epoch": best_epoch, "best_val_mae": best_value, "test": test_metrics, "device": str(device)}
    (output_dir / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
