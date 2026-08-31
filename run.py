"""Reproduce TF-STNet inference on the released real test batch."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import scipy.sparse as sp

from model import TFSTNet, TFSTNetConfig


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "example_test_batch.npz"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "tf_stnet_wind_speed.pth"
DEFAULT_ADJACENCY = ROOT / "data" / "adj_mat.pkl"
REQUIRED_KEYS = {
    "obs_his",
    "obs_fut",
    "nwp_his",
    "nwp_fut",
    "his_mark",
    "fut_mark",
    "station_coordinates",
    "grid_coordinates",
    "target_mean",
    "target_std",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the released TF-STNet inference example.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--adjacency", type=Path, default=DEFAULT_ADJACENCY)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", default="cpu", help="PyTorch device, for example cpu or cuda:0.")
    return parser.parse_args()


def load_batch(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Released test batch not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"Batch archive is missing keys: {sorted(missing)}")
        return {key: np.asarray(archive[key]) for key in REQUIRED_KEYS}


def validate_batch(batch: dict[str, np.ndarray]) -> None:
    config = TFSTNetConfig()
    expected = {
        "obs_his": (64, config.num_nodes, config.history_len),
        "obs_fut": (64, config.num_nodes, config.prediction_len),
        "his_mark": (64, config.history_len, 5),
        "fut_mark": (64, config.prediction_len, 5),
        "station_coordinates": (config.num_nodes, 2),
    }
    for name, shape in expected.items():
        if batch[name].shape != shape:
            raise ValueError(f"{name} has shape {batch[name].shape}; expected {shape}.")
    for name in ("nwp_his", "nwp_fut"):
        array = batch[name]
        if array.ndim != 4 or array.shape[0] != 64 or array.shape[-1] != config.history_len:
            raise ValueError(f"{name} must have shape [64, latitude, longitude, 24], got {array.shape}.")
    grid_shape = batch["nwp_fut"].shape[1:3] + (2,)
    if batch["grid_coordinates"].shape != grid_shape:
        raise ValueError(
            f"grid_coordinates has shape {batch['grid_coordinates'].shape}; expected {grid_shape}."
        )
    if not all(np.isfinite(value).all() for value in batch.values()):
        raise ValueError("The released batch contains NaN or infinite values.")
    target_std = float(np.asarray(batch["target_std"]).reshape(()))
    if target_std <= 0:
        raise ValueError("target_std must be positive.")


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Released checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a PyTorch state dictionary.")
    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)


def asym_adj(adjacency: np.ndarray) -> np.ndarray:
    """Match the asymmetric normalization used by the training pipeline."""
    adjacency = sp.coo_matrix(adjacency)
    row_sum = np.asarray(adjacency.sum(1)).flatten()
    with np.errstate(divide="ignore"):
        inverse_degree = np.power(row_sum, -1).flatten()
    inverse_degree[np.isinf(inverse_degree)] = 0.0
    inverse_degree_matrix = sp.diags(inverse_degree)
    return np.asarray(inverse_degree_matrix.dot(adjacency).astype(np.float32).todense())


def load_adjacency(path: Path, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Load the released graph exactly as in the original experiment."""
    if not path.is_file():
        raise FileNotFoundError(f"Released adjacency file not found: {path}")
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    except UnicodeDecodeError:
        with path.open("rb") as stream:
            payload = pickle.load(stream, encoding="latin1")
    if not isinstance(payload, tuple) or len(payload) != 3:
        raise ValueError("Adjacency pickle must contain (sensor_ids, id_map, adjacency_matrix).")
    adjacency = np.asarray(payload[2], dtype=np.float32)
    expected_shape = (num_nodes, num_nodes)
    if adjacency.shape != expected_shape:
        raise ValueError(f"Adjacency has shape {adjacency.shape}; expected {expected_shape}.")
    return asym_adj(adjacency), asym_adj(adjacency.T)


def tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")

    torch.manual_seed(2026)
    batch = load_batch(args.data)
    validate_batch(batch)
    config = TFSTNetConfig()
    adjacency_forward, adjacency_backward = load_adjacency(args.adjacency, config.num_nodes)
    supports = (
        tensor(adjacency_forward, device),
        tensor(adjacency_backward, device),
    )
    model = TFSTNet(supports, device).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    with torch.inference_mode():
        prediction_scaled = model(
            tensor(batch["obs_his"], device),
            tensor(batch["nwp_his"], device),
            tensor(batch["nwp_fut"], device),
            tensor(batch["his_mark"], device),
            tensor(batch["fut_mark"], device),
            np.asarray(batch["station_coordinates"], dtype=np.float32),
            np.asarray(batch["grid_coordinates"], dtype=np.float32),
        )

    prediction_scaled = prediction_scaled.cpu().numpy()
    target_mean = float(np.asarray(batch["target_mean"]).reshape(()))
    target_std = float(np.asarray(batch["target_std"]).reshape(()))
    prediction = prediction_scaled * target_std + target_mean
    truth = np.asarray(batch["obs_fut"], dtype=np.float32)
    mae = float(np.mean(np.abs(prediction - truth)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "prediction.npy", prediction.astype(np.float32))
    metrics = {
        "model": "TF-STNet",
        "batch_size": int(prediction.shape[0]),
        "prediction_shape": list(prediction.shape),
        "mae": round(mae, 6),
        "device": str(device),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
