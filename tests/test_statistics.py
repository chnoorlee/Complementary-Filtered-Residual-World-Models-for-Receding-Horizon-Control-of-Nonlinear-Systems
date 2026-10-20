import numpy as np
import pandas as pd
import pytest

from rgpwm.analyze import require_evaluation_metric_version
from rgpwm.metrics import EVALUATION_METRIC_VERSION
from rgpwm.statistics import (
    holm_adjust,
    paired_fixed_cell_bootstrap,
    paired_hierarchical_bootstrap,
)


def test_hierarchical_bootstrap_preserves_constant_paired_effect() -> None:
    rows = []
    for seed in (11, 23, 37):
        for system in ("a", "b"):
            for regime in ("clean", "shift"):
                for trajectory in range(5):
                    rows.append({
                        "seed": seed, "system": system, "regime": regime,
                        "trajectory_id": str(trajectory), "method": "baseline", "metric": 2.0,
                    })
                    rows.append({
                        "seed": seed, "system": system, "regime": regime,
                        "trajectory_id": str(trajectory), "method": "proposed", "metric": 1.0,
                    })
    frame = pd.DataFrame(rows)
    result = paired_hierarchical_bootstrap(
        frame, "baseline", "proposed", "metric", replicates=200, seed=9,
    )
    assert result["difference_proposed_minus_baseline"] == -1.0
    np.testing.assert_allclose(result["ci_95"], [-1.0, -1.0])


def test_fixed_cell_bootstrap_preserves_benchmark_cells() -> None:
    rows = []
    for seed in (41, 53, 67):
        for system in ("a", "b"):
            for regime in ("clean", "shift"):
                for trajectory in range(4):
                    rows.extend([
                        {"seed": seed, "system": system, "regime": regime,
                         "trajectory_id": str(trajectory), "method": "base", "metric": 3.0},
                        {"seed": seed, "system": system, "regime": regime,
                         "trajectory_id": str(trajectory), "method": "new", "metric": 2.0},
                    ])
    result = paired_fixed_cell_bootstrap(
        pd.DataFrame(rows), "base", "new", "metric", replicates=100, seed=7,
    )
    assert result["difference_proposed_minus_baseline"] == -1.0
    np.testing.assert_allclose(result["ci_95"], [-1.0, -1.0])
    assert result["relative_improvement_percent"] == pytest.approx(100.0 / 3.0)
    np.testing.assert_allclose(
        result["relative_improvement_ci_95"], [100.0 / 3.0, 100.0 / 3.0],
    )
    assert result["p_two_sided"] > 0.0
    assert result["fixed_benchmark_cells"] == 4


def test_fixed_cell_bootstrap_bundles_repeated_trajectory_regimes() -> None:
    rows = []
    effects = {
        "clean": (1.0, -1.0),
        "corrupt": (-1.0, 1.0),
    }
    for seed in (41, 53, 67):
        for regime, regime_effects in effects.items():
            for trajectory, effect in enumerate(regime_effects):
                rows.extend([
                    {"seed": seed, "system": "a", "regime": regime,
                     "trajectory_id": str(trajectory), "method": "base", "metric": 2.0},
                    {"seed": seed, "system": "a", "regime": regime,
                     "trajectory_id": str(trajectory), "method": "new", "metric": 2.0 + effect},
                ])
    result = paired_fixed_cell_bootstrap(
        pd.DataFrame(rows), "base", "new", "metric", replicates=200, seed=19,
    )
    assert result["difference_proposed_minus_baseline"] == 0.0
    np.testing.assert_allclose(result["ci_95"], [0.0, 0.0], atol=1e-12)


def test_fixed_cell_bootstrap_recomputes_relative_ratio_per_seed() -> None:
    rows = []
    for seed, baseline in ((41, 1.0), (53, 3.0), (67, 9.0)):
        for trajectory in range(4):
            rows.extend([
                {"seed": seed, "system": "a", "regime": "clean",
                 "trajectory_id": str(trajectory), "method": "base", "metric": baseline},
                {"seed": seed, "system": "a", "regime": "clean",
                 "trajectory_id": str(trajectory), "method": "new", "metric": baseline / 2.0},
            ])
    result = paired_fixed_cell_bootstrap(
        pd.DataFrame(rows), "base", "new", "metric", replicates=200, seed=23,
        relative_margin_percent=5.0,
    )
    assert result["relative_improvement_percent"] == pytest.approx(50.0)
    np.testing.assert_allclose(result["relative_improvement_ci_95"], [50.0, 50.0])
    assert result["relative_margin_percent"] == 5.0
    assert 0.0 < result["p_one_sided_relative_margin"] <= 1.0


def test_fixed_cell_bootstrap_rejects_unpaired_method_rows() -> None:
    frame = pd.DataFrame([
        {"seed": 41, "system": "a", "regime": "clean", "trajectory_id": "0",
         "method": "base", "metric": 1.0},
        {"seed": 41, "system": "a", "regime": "clean", "trajectory_id": "0",
         "method": "new", "metric": 0.5},
        {"seed": 41, "system": "a", "regime": "clean", "trajectory_id": "1",
         "method": "base", "metric": 1.0},
    ])
    with pytest.raises(ValueError, match="complete paired observations"):
        paired_fixed_cell_bootstrap(frame, "base", "new", "metric", replicates=10)


def test_fixed_cell_bootstrap_rejects_ood_identifier_reuse() -> None:
    rows = []
    for seed in (41, 53, 67):
        for regime in ("id_clean", "id_corrupt", "ood_parameter_corrupt"):
            for trajectory in range(2):
                rows.extend([
                    {"seed": seed, "system": "a", "regime": regime,
                     "trajectory_id": str(trajectory), "method": "base", "metric": 1.0},
                    {"seed": seed, "system": "a", "regime": regime,
                     "trajectory_id": str(trajectory), "method": "new", "metric": 0.9},
                ])
    with pytest.raises(ValueError, match="OOD trajectory identifiers must be disjoint"):
        paired_fixed_cell_bootstrap(
            pd.DataFrame(rows), "base", "new", "metric", replicates=10,
        )


def test_holm_adjust_is_bounded_and_order_preserving() -> None:
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
    assert np.all((0.0 <= adjusted) & (adjusted <= 1.0))
    assert adjusted[0] <= adjusted[2] <= adjusted[1]


def test_analysis_rejects_missing_or_mixed_metric_versions() -> None:
    require_evaluation_metric_version([
        {"evaluation_metric_version": EVALUATION_METRIC_VERSION, "evaluation_code_sha256": "abc"},
        {"evaluation_metric_version": EVALUATION_METRIC_VERSION, "evaluation_code_sha256": "abc"},
    ])
    with pytest.raises(RuntimeError, match="metric version"):
        require_evaluation_metric_version([
            {"evaluation_metric_version": EVALUATION_METRIC_VERSION, "evaluation_code_sha256": "abc"},
            {"evaluation_metric_version": "trajectory_local_v1", "evaluation_code_sha256": "abc"},
        ])
    with pytest.raises(RuntimeError, match="metric version"):
        require_evaluation_metric_version([{}])
    with pytest.raises(RuntimeError, match="code hash"):
        require_evaluation_metric_version([
            {"evaluation_metric_version": EVALUATION_METRIC_VERSION, "evaluation_code_sha256": "abc"},
            {"evaluation_metric_version": EVALUATION_METRIC_VERSION, "evaluation_code_sha256": "def"},
        ])
