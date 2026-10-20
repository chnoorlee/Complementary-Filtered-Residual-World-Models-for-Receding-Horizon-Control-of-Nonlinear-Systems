import pandas as pd
import pytest

from rgpwm.dynamics import SYSTEMS
from rgpwm.pivot_analyze import pfr_kill_decision


def test_pfr_kill_decision_passes_only_all_frozen_thresholds() -> None:
    aggregate_rows = []
    trajectory_rows = []
    for system in SYSTEMS:
        for regime in ("id_clean", "id_corrupt", "ood_parameter_corrupt"):
            for method, value in (("pgnn", 1.0), ("pfr_fixed", 0.96), ("pfrwm", 0.90)):
                aggregate_rows.append({
                    "method": method, "seed": 11, "system": system, "regime": regime,
                    "mean_observed_filter_gain": 0.5 if method == "pfrwm" else float("nan"),
                    "observed_filter_gain_near_zero_fraction": 0.0,
                    "observed_filter_gain_near_one_fraction": 0.0,
                })
                for trajectory in range(96):
                    trajectory_rows.append({
                        "method": method, "seed": 11, "system": system, "regime": regime,
                        "trajectory_id": str(trajectory), "nrmse_h25": value,
                        "gaussian_nll": 0.0, "calibration_error": 0.1,
                    })
    result = pfr_kill_decision(
        pd.DataFrame(aggregate_rows), pd.DataFrame(trajectory_rows),
    )
    assert result["kill_rule_passed"] is True
    assert result["pfr_improvement_vs_pgnn_percent"] == pytest.approx(10.0)


def test_pfr_kill_decision_rejects_unpaired_or_wrong_seed_grid() -> None:
    aggregate = pd.DataFrame([{
        "method": "pfrwm", "seed": 999, "system": "damped_oscillator",
        "regime": "id_clean", "mean_observed_filter_gain": 0.5,
        "observed_filter_gain_near_zero_fraction": 0.0,
        "observed_filter_gain_near_one_fraction": 0.0,
    }])
    trajectories = pd.DataFrame([{
        "method": "pfrwm", "seed": 999, "system": "damped_oscillator",
        "regime": "id_clean", "trajectory_id": "wrong", "nrmse_h25": 0.1,
        "gaussian_nll": 0.0, "calibration_error": 0.1,
    }])
    with pytest.raises(ValueError, match="frozen"):
        pfr_kill_decision(aggregate, trajectories)
