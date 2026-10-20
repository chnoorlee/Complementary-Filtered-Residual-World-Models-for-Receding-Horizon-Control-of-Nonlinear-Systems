"""Pre-registered trajectory accuracy and probabilistic calibration metrics."""

from __future__ import annotations

import hashlib
from pathlib import Path
from statistics import NormalDist

import numpy as np


EVALUATION_METRIC_VERSION = "fixed_training_scale_nrmse_v2"


def evaluation_code_sha256() -> str:
    """Hash every local module that can change generated or evaluated runs."""
    digest = hashlib.sha256()
    package = Path(__file__).parent
    for name in (
        "data.py",
        "dynamics.py",
        "models.py",
        "training.py",
        "metrics.py",
        "experiment.py",
        "confirm_protocol.py",
        "confirm_run.py",
    ):
        path = package / name
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def gaussian_nll(target: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 1e-9)
    return float(0.5 * (np.log(2.0 * np.pi * variance) + (target - mean) ** 2 / variance).mean())


def calibration_error(target: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> float:
    """Mean absolute central-interval calibration error over 10--90% levels."""
    target = np.asarray(target, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    standard_deviation = np.sqrt(np.maximum(np.asarray(variance, dtype=np.float64), 1e-9))
    errors: list[float] = []
    normal = NormalDist()
    for level in np.linspace(0.1, 0.9, 9):
        quantile = normal.inv_cdf((1.0 + float(level)) / 2.0)
        covered = np.abs(target - mean) <= quantile * standard_deviation
        errors.append(abs(float(covered.mean()) - float(level)))
    return float(np.mean(errors))


def _validated_scale(normalization_scale: np.ndarray, state_dim: int) -> np.ndarray:
    scale = np.asarray(normalization_scale, dtype=np.float64)
    if scale.shape != (state_dim,):
        raise ValueError("normalization_scale must have one positive value per state dimension")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("normalization_scale must contain finite positive values")
    return scale


def horizon_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    normalization_scale: np.ndarray,
) -> dict[str, float]:
    """Calculate cumulative NRMSE using a scale fixed from training data.

    ``target`` and ``prediction`` may already be in normalized coordinates; in
    that case callers pass a vector of ones.  Re-estimating a denominator from
    each test trajectory would leak test-set dynamics into the metric and is
    undefined at horizon one.
    """
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 3:
        raise ValueError("target and prediction must have matching [batch, horizon, state] shapes")
    scale = _validated_scale(normalization_scale, target.shape[-1])
    output: dict[str, float] = {}
    for horizon in (1, 5, 10, 25):
        if horizon > target.shape[1]:
            continue
        residual = prediction[:, :horizon] - target[:, :horizon]
        rmse_per_dimension = np.sqrt(np.mean(residual**2, axis=(0, 1)))
        output[f"nrmse_h{horizon}"] = float(np.mean(rmse_per_dimension / scale))
    final_horizon = min(25, target.shape[1])
    output[f"mae_h{final_horizon}"] = float(
        np.mean(np.abs(prediction[:, :final_horizon] - target[:, :final_horizon]))
    )
    return output


def interval_metrics(
    target: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    level: float = 0.9,
) -> dict[str, float]:
    quantile = NormalDist().inv_cdf((1.0 + level) / 2.0)
    half_width = quantile * np.sqrt(np.maximum(variance, 1e-9))
    covered = np.abs(target - mean) <= half_width
    return {
        "coverage_90": float(covered.mean()),
        "interval_width_90": float((2.0 * half_width).mean()),
        "calibration_error": calibration_error(target, mean, variance),
    }


def catastrophic_rollout_rate(
    target: np.ndarray,
    prediction: np.ndarray,
    normalization_scale: np.ndarray,
    threshold: float = 3.0,
) -> float:
    """Fraction of trajectories with error above a fixed training-data scale."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 3:
        raise ValueError("target and prediction must have matching [batch, horizon, state] shapes")
    scale = _validated_scale(normalization_scale, target.shape[-1]).reshape(1, 1, -1)
    normalized_error = np.abs(prediction - target) / scale
    return float((normalized_error.max(axis=(1, 2)) > threshold).mean())
