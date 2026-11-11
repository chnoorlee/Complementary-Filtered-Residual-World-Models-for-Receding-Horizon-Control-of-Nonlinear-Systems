"""Scientific aggregation and fail-closed checks for measured-flight assets."""
import numpy as np
import pandas as pd
import pytest

from make_external_assets import (
    FLIGHTS, METRICS, SEEDS, compare_metrics, expected_grid, metrics_from_arrays,
    sha256, summarize, validate_grid, verify_hashes,
)


def complete_grid():
    rows = []
    for method, seed, regime, horizon, flight in sorted(expected_grid()):
        index = FLIGHTS.index(flight)
        rows.append({"method": method, "seed": seed, "regime": regime,
                     "horizon": horizon, "flight": flight, "windows": (1, 10, 100)[index],
                     **{metric: float(index + 1) for metric in METRICS}})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("alteration", ["missing", "duplicate", "unexpected", "nonfinite"])
def test_partial_or_invalid_grid_cannot_be_summarized(alteration):
    frame = complete_grid()
    if alteration == "missing":
        frame = frame.iloc[:-1]
    elif alteration == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif alteration == "unexpected":
        frame.loc[0, "seed"] = 999
    else:
        frame.loc[0, "nrmse"] = np.nan
    with pytest.raises(ValueError):
        summarize(frame)


def test_each_flight_receives_equal_weight_despite_window_count():
    frame = complete_grid()
    seed_scores, summary, contrasts = summarize(frame)
    assert len(frame) == 384
    assert len(summary) == 32
    assert len(contrasts) == 16
    # Flight errors1,2,3 have sample counts1,10,100; the descriptive estimand
    # is2.0, not the heavily window-weighted estimate321/111.
    np.testing.assert_allclose(seed_scores["nrmse"], 2.0)
    np.testing.assert_allclose(summary["nrmse_mean"], 2.0)
    np.testing.assert_allclose(summary["nrmse_sd"], 0.0)


def test_contrasts_average_paired_ratios_and_sd_uses_training_seeds():
    frame = complete_grid()
    baseline = np.array([1., 2., 4., 8., 16.])
    candidate = np.array([.5, 1.8, 3.0, 4.0, 15.0])
    for seed, b, c in zip(SEEDS, baseline, candidate):
        frame.loc[(frame.method == "unfiltered") & (frame.seed == seed), "nrmse"] = b
        frame.loc[(frame.method == "cfr") & (frame.seed == seed), "nrmse"] = c
    _, summary, contrasts = summarize(frame)
    rows = contrasts[contrasts.comparator == "unfiltered"]
    expected = 100 * (baseline - candidate) / baseline
    np.testing.assert_allclose(rows.relative_improvement_mean_pct, expected.mean())
    assert expected.mean() != pytest.approx(100 * (baseline.mean() - candidate.mean()) / baseline.mean())
    np.testing.assert_allclose(summary[summary.method == "cfr"].nrmse_sd, candidate.std(ddof=1))


def test_prediction_recomputation_uses_pooled_coordinates_and_si_units():
    target = np.zeros((3, 50, 12), dtype=np.float32)
    prediction = target.copy()
    prediction[:, :25, 0] = 1.0
    prediction[:, 25:, 0] = 3.0
    prediction[..., 3] = 2.0
    scales = np.ones(12)
    scales[0], scales[3] = 2., 3.
    rows, windows = metrics_from_arrays(prediction, target, np.asarray(FLIGHTS),
                                        np.array([7, 7, 7]), scales, "cfr", 131, "recorded")
    frame = pd.DataFrame(rows)
    short = frame[frame.horizon == 25]
    long = frame[frame.horizon == 50]
    np.testing.assert_allclose(short.nrmse, np.sqrt(5 / 12))
    np.testing.assert_allclose(long.nrmse, np.sqrt(9 / 12))
    np.testing.assert_allclose(short.position_rmse_m, 2.)
    np.testing.assert_allclose(long.position_rmse_m, np.sqrt(20))
    np.testing.assert_allclose(frame.velocity_rmse_m_s, 6.)
    assert len(windows) == 6
    # Reject a changed metric even if someone has updated its CSV hash.
    altered = frame.copy()
    altered.loc[0, "nrmse"] *= .5
    with pytest.raises(ValueError, match="disagree"):
        compare_metrics(altered, frame, ("method", "seed", "regime", "horizon", "flight"), METRICS)


def test_prediction_receipt_detects_changed_arrays(tmp_path):
    path = tmp_path / "prediction.npz"
    np.savez_compressed(path, prediction=np.zeros((1, 50, 12)))
    hashes = {path.name: sha256(path)}
    verify_hashes(tmp_path, hashes)
    np.savez_compressed(path, prediction=np.ones((1, 50, 12)))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_hashes(tmp_path, hashes)


def test_forecast_origins_must_be_unique_within_each_flight():
    values = np.zeros((2, 50, 12))
    with pytest.raises(ValueError, match="duplicate forecast origin"):
        metrics_from_arrays(values, values, np.asarray([FLIGHTS[0], FLIGHTS[0]]),
                            np.array([7, 7]), np.ones(12), "cfr", 131, "recorded")


def test_metric_comparison_rejects_different_identity_even_with_equal_values():
    frame = complete_grid()
    altered = frame.copy()
    altered.loc[0, "flight"] = "another_flight"
    with pytest.raises(ValueError, match="identities differ"):
        compare_metrics(altered, frame, ("method", "seed", "regime", "horizon", "flight"), METRICS)
