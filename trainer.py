"""Training and evaluation utilities for the released TF-STNet model.

The trainer deliberately depends only on NumPy, SciPy, and PyTorch.  It uses
the same tensor layout and target inverse-scaling convention as ``run.py`` so
that a checkpoint selected during training can be evaluated by the inference
entry point without conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import torch


INPUT_KEYS = (
    "obs_his",
    "obs_fut",
    "nwp_his",
    "nwp_fut",
    "his_mark",
    "fut_mark",
    "station_coordinates",
    "grid_coordinates",
)


@dataclass(frozen=True)
class TargetScaler:
    """Scalar standardization statistics used by the released experiment."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean) or not np.isfinite(self.std) or self.std <= 0:
            raise ValueError("target_mean must be finite and target_std must be positive.")

    def inverse_transform(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std + self.mean


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Return scalar metrics over all samples, stations, and horizons."""

    pred = prediction.detach().float().reshape(-1)
    truth = target.detach().float().reshape(-1)
    error = pred - truth
    mae = torch.mean(torch.abs(error))
    rmse = torch.sqrt(torch.mean(error.square()))
    denominator = torch.abs(pred) + torch.abs(truth)
    smape = torch.mean(2.0 * torch.abs(error) / torch.clamp(denominator, min=1e-6))
    pred_centered = pred - pred.mean()
    truth_centered = truth - truth.mean()
    correlation_denominator = torch.sqrt(pred_centered.square().sum() * truth_centered.square().sum())
    pearson = torch.where(
        correlation_denominator > 1e-12,
        (pred_centered * truth_centered).sum() / correlation_denominator,
        torch.zeros((), device=pred.device),
    )
    return {
        "mae": float(mae.cpu()),
        "rmse": float(rmse.cpu()),
        "smape": float(smape.cpu()),
        "pearson": float(pearson.cpu()),
    }


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        raise ValueError("Cannot average metrics from an empty split.")
    return {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}


class Trainer:
    """One-model training loop with validation and checkpoint-friendly metrics."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        target_scaler: TargetScaler,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.target_scaler = target_scaler
        self.device = device

    def _forward(self, batch: Mapping[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        def tensor(name: str) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(batch[name], dtype=np.float32)).to(self.device)

        prediction_scaled = self.model(
            tensor("obs_his"),
            tensor("nwp_his"),
            tensor("nwp_fut"),
            tensor("his_mark"),
            tensor("fut_mark"),
            np.asarray(batch["station_coordinates"], dtype=np.float32),
            np.asarray(batch["grid_coordinates"], dtype=np.float32),
        )
        prediction = self.target_scaler.inverse_transform(prediction_scaled)
        target = tensor("obs_fut")
        return prediction, target

    def train_batch(self, batch: Mapping[str, np.ndarray]) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction, target = self._forward(batch)
        loss = torch.mean(torch.abs(prediction - target))
        loss.backward()
        self.optimizer.step()
        result = regression_metrics(prediction, target)
        result["loss"] = float(loss.detach().cpu())
        return result

    @torch.no_grad()
    def eval_batch(self, batch: Mapping[str, np.ndarray]) -> dict[str, float]:
        self.model.eval()
        prediction, target = self._forward(batch)
        result = regression_metrics(prediction, target)
        result["loss"] = result["mae"]
        return result

    def run_epoch(
        self,
        batches: Iterable[Mapping[str, np.ndarray]],
        training: bool,
    ) -> dict[str, float]:
        batch_metrics = []
        for batch in batches:
            batch_metrics.append(self.train_batch(batch) if training else self.eval_batch(batch))
        return _mean_metrics(batch_metrics)
