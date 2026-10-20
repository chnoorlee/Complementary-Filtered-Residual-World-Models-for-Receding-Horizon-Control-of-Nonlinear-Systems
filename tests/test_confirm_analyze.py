import pandas as pd
import pytest

from rgpwm.confirm_analyze import (
    CONFIRM_METHODS,
    CONFIRM_SEEDS,
    _validate_confirmatory_grid,
    _validate_payload_data_pairing,
    analyze_confirmatory,
)
from rgpwm.dynamics import SYSTEMS


def _frozen_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows = []
    trajectory_rows = []
    for method in CONFIRM_METHODS:
        for seed in CONFIRM_SEEDS:
            for system in SYSTEMS:
                for regime in ("id_clean", "id_corrupt", "ood_parameter_corrupt"):
                    aggregate_rows.append({
                        "method": method,
                        "seed": seed,
                        "system": system,
                        "regime": regime,
                    })
                    prefix = "id" if regime != "ood_parameter_corrupt" else "ood"
                    for trajectory in range(96):
                        trajectory_rows.append({
                            "method": method,
                            "seed": seed,
                            "system": system,
                            "regime": regime,
                            "trajectory_id": f"{prefix}-trajectory-{trajectory}",
                        })
    return pd.DataFrame(aggregate_rows), pd.DataFrame(trajectory_rows)


def test_confirmatory_grid_requires_all_methods_seeds_cells_and_pairs() -> None:
    aggregate, trajectories = _frozen_grid()
    _validate_confirmatory_grid(aggregate, trajectories)

    with pytest.raises(ValueError, match="frozen method/seed/system/regime grid"):
        _validate_confirmatory_grid(aggregate.iloc[:-1], trajectories)

    broken = trajectories.copy()
    broken.loc[0, "trajectory_id"] = "unpaired"
    with pytest.raises(ValueError, match="paired across methods"):
        _validate_confirmatory_grid(aggregate, broken)

    extra = pd.concat([trajectories, trajectories.iloc[[0]].assign(method="not_frozen")])
    with pytest.raises(ValueError, match="exact frozen trajectory grid"):
        _validate_confirmatory_grid(aggregate, extra)

    mismatched_regime = trajectories.copy()
    clean_indices = mismatched_regime[
        (mismatched_regime["seed"] == CONFIRM_SEEDS[0])
        & (mismatched_regime["system"] == next(iter(SYSTEMS)))
        & (mismatched_regime["regime"] == "id_clean")
        & (mismatched_regime["trajectory_id"] == "id-trajectory-0")
    ].index
    mismatched_regime.loc[clean_indices, "trajectory_id"] = "clean-only-id"
    with pytest.raises(ValueError, match="clean/corrupt trajectory identifiers"):
        _validate_confirmatory_grid(aggregate, mismatched_regime)

    reused_ood = trajectories.copy()
    reused_ood.loc[
        reused_ood["regime"] == "ood_parameter_corrupt", "trajectory_id"
    ] = reused_ood.loc[
        reused_ood["regime"] == "ood_parameter_corrupt", "trajectory_id"
    ].str.replace("ood-trajectory-", "id-trajectory-", regex=False)
    with pytest.raises(ValueError, match="OOD trajectory identifiers must be disjoint"):
        _validate_confirmatory_grid(aggregate, reused_ood)


def test_confirmatory_payloads_require_one_shared_dataset_per_seed_system() -> None:
    payloads = [
        {"method": method, "seed": seed, "system": system,
         "data_sha256": f"{seed}-{system}"}
        for method in CONFIRM_METHODS
        for seed in CONFIRM_SEEDS
        for system in SYSTEMS
    ]
    assert _validate_payload_data_pairing(payloads) == len(CONFIRM_SEEDS) * len(SYSTEMS)

    payloads[0]["data_sha256"] = "mismatched"
    with pytest.raises(RuntimeError, match="data fingerprint"):
        _validate_payload_data_pairing(payloads)

    duplicated = [
        {"method": method, "seed": seed, "system": system,
         "data_sha256": f"{seed}-{system}"}
        for method in CONFIRM_METHODS
        for seed in CONFIRM_SEEDS
        for system in SYSTEMS
    ]
    duplicated_hash = duplicated[0]["data_sha256"]
    target_seed = CONFIRM_SEEDS[1]
    target_system = list(SYSTEMS)[0]
    for payload in duplicated:
        if payload["seed"] == target_seed and payload["system"] == target_system:
            payload["data_sha256"] = duplicated_hash
    with pytest.raises(RuntimeError, match="distinct data fingerprint"):
        _validate_payload_data_pairing(duplicated)


def test_confirmatory_decision_tests_the_five_percent_relative_margin() -> None:
    aggregate_rows = []
    trajectory_rows = []
    method_nrmse = {method: 1.1 for method in CONFIRM_METHODS}
    method_nrmse["pgnn"] = 1.0
    method_nrmse["cfrwm"] = 0.9
    for method in CONFIRM_METHODS:
        for seed in CONFIRM_SEEDS:
            for system in SYSTEMS:
                for regime in ("id_clean", "id_corrupt", "ood_parameter_corrupt"):
                    aggregate_rows.append({
                        "method": method, "seed": seed, "system": system,
                        "regime": regime, "parameter_count": 1000,
                    })
                    prefix = "id" if regime != "ood_parameter_corrupt" else "ood"
                    for trajectory in range(96):
                        nrmse = method_nrmse[method]
                        trajectory_rows.append({
                            "method": method, "seed": seed, "system": system,
                            "regime": regime, "trajectory_id": f"{prefix}-{trajectory}",
                            "nrmse_h1": nrmse, "nrmse_h5": nrmse,
                            "nrmse_h10": nrmse, "nrmse_h25": nrmse,
                            "mae_h25": nrmse, "gaussian_nll": nrmse,
                            "calibration_error": 0.1, "catastrophic": 0.0,
                        })
    decision, _, cell = analyze_confirmatory(
        pd.DataFrame(aggregate_rows), pd.DataFrame(trajectory_rows), replicates=20,
    )
    assert decision["primary_h1"]["relative_improvement_percent"] == pytest.approx(10.0)
    assert decision["primary_h1"]["relative_improvement_ci_95"][0] > 5.0
    assert decision["primary_h1"]["p_one_sided_relative_margin"] == pytest.approx(
        1 / (2 ** len(CONFIRM_SEEDS))
    )
    assert decision["primary_h1"]["h1_supported"] is True
    assert decision["robustness_h2"]["supported"] is True
    assert decision["confirmation_passed"] is True
    assert "p_raw" not in cell and "p_holm" not in cell
