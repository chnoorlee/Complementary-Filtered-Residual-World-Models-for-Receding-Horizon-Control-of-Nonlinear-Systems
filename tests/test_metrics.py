import numpy as np

from rgpwm.metrics import (
    calibration_error,
    catastrophic_rollout_rate,
    evaluation_code_sha256,
    gaussian_nll,
    horizon_metrics,
)


def test_evaluation_fingerprint_is_sha256() -> None:
    value = evaluation_code_sha256()
    assert len(value) == 64
    int(value, 16)


def test_horizon_metrics_exact_zero_for_perfect_prediction() -> None:
    target = np.arange(2 * 25 * 3, dtype=np.float64).reshape(2, 25, 3) / 10.0
    metrics = horizon_metrics(target, target.copy(), normalization_scale=np.ones(3))
    for horizon in (1, 5, 10, 25):
        assert metrics[f"nrmse_h{horizon}"] == 0.0
    assert metrics["mae_h25"] == 0.0


def test_horizon_metrics_use_fixed_training_scale_for_single_trajectory() -> None:
    target = np.zeros((1, 25, 2), dtype=np.float64)
    prediction = np.ones_like(target)
    metrics = horizon_metrics(
        target,
        prediction,
        normalization_scale=np.array([2.0, 4.0]),
    )
    assert metrics["nrmse_h1"] == 0.375
    assert metrics["nrmse_h25"] == 0.375


def test_catastrophic_rate_uses_fixed_training_scale() -> None:
    target = np.zeros((2, 25, 1), dtype=np.float64)
    prediction = np.zeros_like(target)
    prediction[0, 0, 0] = 3.1
    assert catastrophic_rollout_rate(
        target,
        prediction,
        normalization_scale=np.ones(1),
    ) == 0.5


def test_gaussian_nll_matches_closed_form_at_unit_variance() -> None:
    target = np.array([0.0, 1.0])
    mean = np.array([0.0, 0.0])
    variance = np.ones(2)
    expected = 0.5 * (np.log(2.0 * np.pi) + np.array([0.0, 1.0])).mean()
    assert abs(gaussian_nll(target, mean, variance) - expected) < 1e-12


def test_calibration_error_is_bounded() -> None:
    rng = np.random.default_rng(1)
    target = rng.normal(size=(2000, 1))
    mean = np.zeros_like(target)
    variance = np.ones_like(target)
    value = calibration_error(target, mean, variance)
    assert 0.0 <= value <= 1.0
    assert value < 0.04
