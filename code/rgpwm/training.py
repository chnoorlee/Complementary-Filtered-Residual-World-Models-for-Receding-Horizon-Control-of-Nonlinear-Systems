"""Training and evaluation routines shared by pilot and full experiments."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import Normalizer, WindowBatch
from .metrics import (
    calibration_error,
    catastrophic_rollout_rate,
    gaussian_nll,
    horizon_metrics,
    interval_metrics,
)
from .models import WorldModelBase


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    min_delta: float = 1e-5
    gradient_clip: float = 5.0
    reliability_weight: float = 0.2
    num_workers: int = 0


@dataclass
class TrainResult:
    best_epoch: int
    best_validation_nll: float
    training_seconds: float
    epochs_completed: int
    history: list[dict[str, float]]
    stopped_for_budget: bool


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def windows_to_loader(
    windows: WindowBatch,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    tensors = [
        torch.from_numpy(windows.context_values),
        torch.from_numpy(windows.context_mask),
        torch.from_numpy(windows.context_delta),
        torch.from_numpy(windows.context_actions),
        torch.from_numpy(windows.future_actions),
        torch.from_numpy(windows.targets),
    ]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
        generator=generator, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )


def rollout_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    reliability_weight: float = 0.0,
    one_step: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    horizon = 1 if one_step else target.shape[1]
    selected_target = target[:, :horizon]
    mean = output["mean"][:, :horizon]
    variance = output["variance"][:, :horizon].clamp_min(1e-6)
    nll = 0.5 * (torch.log(2.0 * math.pi * variance) + (selected_target - mean).square() / variance).mean()
    reliability = torch.zeros((), device=target.device)
    required = {"physics_mean", "data_mean", "physics_reliability", "data_reliability"}
    if reliability_weight > 0.0 and required.issubset(output):
        physics_target = (output["physics_mean"][:, :horizon] - selected_target).square().mean(-1, keepdim=True).sqrt().detach()
        data_target = (output["data_mean"][:, :horizon] - selected_target).square().mean(-1, keepdim=True).sqrt().detach()
        physics_prediction = output["physics_reliability"][:, :horizon]
        data_prediction = output["data_reliability"][:, :horizon]
        reliability = 0.5 * (
            F.smooth_l1_loss(torch.log1p(physics_prediction), torch.log1p(physics_target))
            + F.smooth_l1_loss(torch.log1p(data_prediction), torch.log1p(data_target))
        )
    loss = nll + reliability_weight * reliability
    return loss, {"nll": float(nll.detach()), "reliability": float(reliability.detach())}


def _model_call(model: nn.Module, batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    values, mask, delta, context_actions, future_actions, target = (
        value.to(device, non_blocking=True) for value in batch
    )
    return model(values, mask, delta, context_actions, future_actions), target


@torch.no_grad()
def validation_nll(model: nn.Module, loader: DataLoader, device: torch.device, one_step: bool = False) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        output, target = _model_call(model, batch, device)
        _, parts = rollout_loss(output, target, one_step=one_step)
        batch_size = target.shape[0]
        total += parts["nll"] * batch_size
        count += batch_size
    return total / max(count, 1)


def train_model(
    model: nn.Module,
    train_windows: WindowBatch,
    validation_windows: WindowBatch,
    config: TrainConfig,
    device: torch.device,
    seed: int,
    wall_clock_deadline: float | None = None,
    one_step: bool = False,
) -> TrainResult:
    seed_everything(seed)
    model.to(device)
    train_loader = windows_to_loader(train_windows, config.batch_size, True, seed, config.num_workers)
    val_loader = windows_to_loader(validation_windows, config.batch_size * 2, False, seed, config.num_workers)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_nll = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    start = time.perf_counter()
    stopped_for_budget = False
    for epoch in range(1, config.epochs + 1):
        if wall_clock_deadline is not None and time.perf_counter() >= wall_clock_deadline:
            stopped_for_budget = True
            break
        model.train()
        running_loss = running_nll = running_rel = 0.0
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output, target = _model_call(model, batch, device)
            reliability_weight = config.reliability_weight if getattr(model, "reliability_supervision", False) else 0.0
            loss, parts = rollout_loss(
                output, target, reliability_weight=reliability_weight,
                one_step=one_step,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            batch_size = target.shape[0]
            running_loss += float(loss.detach()) * batch_size
            running_nll += parts["nll"] * batch_size
            running_rel += parts["reliability"] * batch_size
            count += batch_size
        val_nll = validation_nll(model, val_loader, device, one_step=one_step)
        history.append({
            "epoch": float(epoch),
            "train_loss": running_loss / max(count, 1),
            "train_nll": running_nll / max(count, 1),
            "train_reliability": running_rel / max(count, 1),
            "validation_nll": val_nll,
        })
        if val_nll < best_nll - config.min_delta:
            best_nll = val_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    model.load_state_dict(best_state)
    return TrainResult(
        best_epoch=best_epoch,
        best_validation_nll=best_nll,
        training_seconds=time.perf_counter() - start,
        epochs_completed=len(history),
        history=history,
        stopped_for_budget=stopped_for_budget,
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    if np.any(counts > 1):
        for group in np.flatnonzero(counts > 1):
            positions = np.flatnonzero(inverse == group)
            ranks[positions] = ranks[positions].mean()
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    first, second = first.ravel(), second.ravel()
    if len(first) < 2 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return None
    return float(np.corrcoef(_rankdata(first), _rankdata(second))[0, 1])


@torch.no_grad()
def _pooled_filter_gain_metrics(
    gains: np.ndarray,
    masks: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Summarize gains over observed entries, retaining trajectory diagnostics."""
    gains = np.asarray(gains, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    if gains.shape != masks.shape or gains.ndim < 2:
        raise ValueError("filter gains and masks must have the same batched shape")
    pooled = gains[masks]
    if pooled.size == 0:
        raise ValueError("filter-gain diagnostics require at least one observed entry")

    rows: list[dict[str, float]] = []
    nonempty_stds: list[float] = []
    for sample_gain, sample_mask in zip(gains, masks, strict=True):
        observed = sample_gain[sample_mask]
        if observed.size == 0:
            rows.append({
                "mean_observed_filter_gain": 0.0,
                "filter_gain_std": 0.0,
                "filter_gain_near_zero_fraction": 0.0,
                "filter_gain_near_one_fraction": 0.0,
                "observed_filter_gain_count": 0.0,
            })
            continue
        within_std = float(observed.std())
        nonempty_stds.append(within_std)
        rows.append({
            "mean_observed_filter_gain": float(observed.mean()),
            "filter_gain_std": within_std,
            "filter_gain_near_zero_fraction": float((observed <= 0.05).mean()),
            "filter_gain_near_one_fraction": float((observed >= 0.95).mean()),
            "observed_filter_gain_count": float(observed.size),
        })

    aggregate = {
        "mean_observed_filter_gain": float(pooled.mean()),
        "pooled_observed_filter_gain_std": float(pooled.std()),
        "mean_within_trajectory_filter_gain_std": float(np.mean(nonempty_stds)),
        "observed_filter_gain_near_zero_fraction": float((pooled <= 0.05).mean()),
        "observed_filter_gain_near_one_fraction": float((pooled >= 0.95).mean()),
        "observed_filter_gain_count": float(pooled.size),
    }
    return aggregate, rows


@torch.no_grad()
def evaluate_model(
    model: WorldModelBase,
    windows: WindowBatch,
    normalizer: Normalizer,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    """Evaluate and retain trajectory-level rows for paired inference."""
    loader = windows_to_loader(windows, batch_size, False, 0)
    model.eval().to(device)
    means, variances, targets = [], [], []
    nominal_defects = []
    weights, physics_means, data_means = [], [], []
    filter_gains, filter_masks = [], []
    elapsed = 0.0
    for batch in loader:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_start = time.perf_counter()
        output, target = _model_call(model, batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - batch_start
        values, _, _, _, future_actions, _ = (value.to(device, non_blocking=True) for value in batch)
        initial_state = output.get("filtered_state", values[:, -1])
        previous = torch.cat((initial_state[:, None], output["mean"][:, :-1]), dim=1)
        nominal = torch.stack([
            model.normalized_nominal_step(previous[:, step], future_actions[:, step])
            for step in range(future_actions.shape[1])
        ], dim=1)
        nominal_defects.append((output["mean"] - nominal).square().mean(dim=(1, 2)).sqrt().cpu().numpy())
        means.append(output["mean"].cpu().numpy())
        variances.append(output["variance"].cpu().numpy())
        targets.append(target.cpu().numpy())
        if "data_weight" in output:
            weights.append(output["data_weight"].cpu().numpy())
            physics_means.append(output["physics_mean"].cpu().numpy())
            data_means.append(output["data_mean"].cpu().numpy())
        if "filter_gain" in output:
            filter_gains.append(output["filter_gain"].cpu().numpy())
            filter_masks.append(batch[1][:, 1:].cpu().numpy().astype(bool))
    mean = np.concatenate(means)
    variance = np.concatenate(variances)
    target = np.concatenate(targets)
    nominal_defect = np.concatenate(nominal_defects)
    # Windows are expressed in training-normalized state coordinates, so one
    # unit is exactly one training-set standard deviation for every channel.
    fixed_scale = np.ones(target.shape[-1], dtype=np.float64)
    aggregate = horizon_metrics(target, mean, fixed_scale)
    aggregate.update({
        "gaussian_nll": gaussian_nll(target, mean, variance),
        **interval_metrics(target, mean, variance),
        "catastrophic_rollout_rate": catastrophic_rollout_rate(target, mean, fixed_scale),
        "rollout_latency_ms_per_trajectory": 1000.0 * elapsed / len(target),
        "normalized_nominal_dynamics_defect": float(nominal_defect.mean()),
    })
    if weights:
        weight = np.concatenate(weights)
        physics = np.concatenate(physics_means)
        data = np.concatenate(data_means)
        physics_error = np.mean((physics - target) ** 2, axis=-1)
        data_error = np.mean((data - target) ** 2, axis=-1)
        oracle_advantage = physics_error - data_error
        aggregate["mean_data_gate"] = float(weight.mean())
        aggregate["gate_oracle_spearman"] = _spearman(weight[..., 0], oracle_advantage)
    filter_rows: list[dict[str, float]] | None = None
    if filter_gains:
        filter_aggregate, filter_rows = _pooled_filter_gain_metrics(
            np.concatenate(filter_gains), np.concatenate(filter_masks),
        )
        aggregate.update(filter_aggregate)
    rows: list[dict[str, float | str]] = []
    for index, trajectory_id in enumerate(windows.trajectory_ids):
        row: dict[str, float | str] = {"trajectory_id": str(trajectory_id)}
        row.update(horizon_metrics(target[index:index + 1], mean[index:index + 1], fixed_scale))
        row["gaussian_nll"] = gaussian_nll(target[index], mean[index], variance[index])
        row["calibration_error"] = calibration_error(target[index], mean[index], variance[index])
        row["catastrophic"] = catastrophic_rollout_rate(
            target[index:index + 1], mean[index:index + 1], fixed_scale,
        )
        row["nominal_dynamics_defect"] = float(nominal_defect[index])
        if weights:
            row["mean_data_gate"] = float(weight[index].mean())
        if filter_rows is not None:
            row.update(filter_rows[index])
        rows.append(row)
    return aggregate, rows


def save_checkpoint(
    path: Path,
    model: nn.Module,
    train_result: TrainResult,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "train_result": asdict(train_result),
        "metadata": metadata,
    }, path)
