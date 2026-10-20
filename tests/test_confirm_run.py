import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from rgpwm.confirm_analyze import (
    ABLATIONS,
    COMPARATORS,
    CONFIRM_METHODS,
    CONFIRM_REGIMES,
    CONFIRM_SEEDS,
    _analysis_sha256,
)
from rgpwm.confirm_run import (
    _expected_train_config,
    _validate_contract,
    _validate_existing_payload,
)
from rgpwm.dynamics import SYSTEMS
from rgpwm.experiment import _reject_confirmatory_cli_overlap
from rgpwm.metrics import EVALUATION_METRIC_VERSION, evaluation_code_sha256
from rgpwm.models import FIXED_FILTER_GAINS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(tuning_path: Path) -> dict[str, object]:
    return {
        "contract_version": "cfr-confirmatory-v3.0",
        "status": "frozen_before_confirmation",
        "methods": list(CONFIRM_METHODS),
        "seeds": list(CONFIRM_SEEDS),
        "systems": list(SYSTEMS),
        "regimes": list(CONFIRM_REGIMES),
        "comparators": list(COMPARATORS),
        "ablations": list(ABLATIONS),
        "candidate": {"method": "cfrwm", "fixed_filter_gain": 0.5},
        "data_design": {
            "train_count": 256, "validation_count": 64, "test_count": 96,
            "trajectory_steps": 72, "trajectories_per_cell": 96,
            "expected_runs": len(CONFIRM_METHODS) * len(CONFIRM_SEEDS) * len(SYSTEMS),
            "expected_aggregate_rows": (
                len(CONFIRM_METHODS) * len(CONFIRM_SEEDS)
                * len(SYSTEMS) * len(CONFIRM_REGIMES)
            ),
            "expected_trajectory_rows": (
                len(CONFIRM_METHODS) * len(CONFIRM_SEEDS) * len(SYSTEMS)
                * len(CONFIRM_REGIMES) * 96
            ),
        },
        "training": {
            "optimizer": "AdamW", "epochs": 60, "patience": 10,
            "batch_size": 128, "min_delta": 1e-5, "gradient_clip": 5.0,
            "num_workers": 0, "tuning_file": str(tuning_path),
            "tuning_file_sha256": _sha256(tuning_path),
            "baseline_tuning_source": "own_frozen_system_method",
            "cfr_tuning_source": "pgnn_same_system",
            "output_directory": "results/confirmatory",
            "wall_clock_hours": 8.0, "deadline_fraction": 0.8,
        },
        "statistics": {
            "bootstrap_replicates": 5000,
            "primary_comparator": "pgnn",
            "h1_relative_margin_percent": 5.0,
            "h2_clean_noninferiority_margin_percent": -5.0,
            "h2_hierarchically_gated_by_h1": True,
        },
        "hashes": {
            "training_code_sha256": evaluation_code_sha256(),
            "analysis_code_sha256": _analysis_sha256(),
        },
    }


def _evaluations() -> dict[str, object]:
    output = {}
    for regime in CONFIRM_REGIMES:
        prefix = "id" if regime in {"id_clean", "id_corrupt"} else "ood"
        output[regime] = {
            "aggregate": {"nrmse_h25": 1.0},
            "trajectories": [
                {"trajectory_id": f"{prefix}-{index}", "nrmse_h25": 1.0}
                for index in range(96)
            ],
        }
    return output


def test_confirmatory_contract_preflight_fails_closed_on_grid_or_hash_change() -> None:
    tuning_path = Path("results/tuning/best_configs.json")
    contract = _contract(tuning_path)
    tuning = _validate_contract(contract)
    assert set(tuning) == set(SYSTEMS)

    wrong_grid = json.loads(json.dumps(contract))
    wrong_grid["methods"].append("pfrwm")
    with pytest.raises(RuntimeError, match="methods"):
        _validate_contract(wrong_grid)

    wrong_hash = json.loads(json.dumps(contract))
    wrong_hash["training"]["tuning_file_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="tuning file hash"):
        _validate_contract(wrong_hash)


def test_generic_experiment_cli_cannot_consume_frozen_confirmation() -> None:
    _reject_confirmatory_cli_overlap([11, 23, 37], Path("results/primary"))
    with pytest.raises(RuntimeError, match="dedicated frozen runner"):
        _reject_confirmatory_cli_overlap([11, CONFIRM_SEEDS[0]], Path("results/other"))
    with pytest.raises(RuntimeError, match="reserved"):
        _reject_confirmatory_cli_overlap([11], Path("results/confirmatory"))


def test_existing_confirmatory_payload_requires_exact_provenance_and_config() -> None:
    tuning_path = Path("results/tuning/best_configs.json")
    contract = _contract(tuning_path)
    tuning = _validate_contract(contract)
    method, seed, system = "cfrwm", CONFIRM_SEEDS[0], next(iter(SYSTEMS))
    config = _expected_train_config(method, system, contract, tuning)
    provenance = {
        "confirmatory_contract_sha256": "a" * 64,
        "tuning_file_sha256": contract["training"]["tuning_file_sha256"],
        "analysis_code_sha256": contract["hashes"]["analysis_code_sha256"],
    }
    payload = {
        "method": method, "seed": seed, "system": system,
        "run_code_sha256": evaluation_code_sha256(),
        "evaluation_code_sha256": evaluation_code_sha256(),
        "evaluation_metric_version": EVALUATION_METRIC_VERSION,
        "hidden_dim": 72, "train_config": asdict(config),
        "one_step_training": False,
        "fixed_filter_selection": {
            "selected": FIXED_FILTER_GAINS[method],
            "selection": "prespecified_before_training",
        },
        "data_sha256": "d" * 64,
        "train_result": {"stopped_for_budget": False},
        "evaluations": _evaluations(),
        **provenance,
    }
    _validate_existing_payload(
        payload, method, seed, system, config, provenance,
    )
    stale = dict(payload, analysis_code_sha256="b" * 64)
    with pytest.raises(RuntimeError, match="stale confirmatory payload"):
        _validate_existing_payload(
            stale, method, seed, system, config, provenance,
        )
