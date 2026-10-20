"""Fail-closed runner for the frozen CFR-WM confirmatory experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from .confirm_protocol import (
    ABLATIONS,
    COMPARATORS,
    CONFIRM_METHODS,
    CONFIRM_REGIMES,
    CONFIRM_SEEDS,
    analysis_code_sha256 as _analysis_sha256,
)
from .dynamics import SYSTEMS
from .experiment import HIDDEN_DIMENSIONS, prepare_windows, run_condition
from .metrics import EVALUATION_METRIC_VERSION, evaluation_code_sha256
from .models import FIXED_FILTER_GAINS
from .training import TrainConfig


CONTRACT_PATH = Path("docs/12_cfr_confirmatory_contract.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_equal(label: str, found: object, expected: object) -> None:
    if found != expected:
        raise RuntimeError(f"frozen contract {label} mismatch: expected {expected!r}, found {found!r}")


def _validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Validate every executable contract field before any data are generated."""
    _require_equal("version", contract.get("contract_version"), "cfr-confirmatory-v3.0")
    _require_equal("status", contract.get("status"), "frozen_before_confirmation")
    _require_equal("methods", contract.get("methods"), list(CONFIRM_METHODS))
    _require_equal("seeds", contract.get("seeds"), list(CONFIRM_SEEDS))
    _require_equal("systems", contract.get("systems"), list(SYSTEMS))
    _require_equal("regimes", contract.get("regimes"), list(CONFIRM_REGIMES))
    _require_equal("comparators", contract.get("comparators"), list(COMPARATORS))
    _require_equal("ablations", contract.get("ablations"), list(ABLATIONS))
    if "pfrwm" in contract.get("methods", []):
        raise RuntimeError("failed learned-gain PFR-WM is prohibited from confirmation")

    candidate = contract.get("candidate", {})
    _require_equal("candidate method", candidate.get("method"), "cfrwm")
    _require_equal(
        "candidate fixed filter gain",
        candidate.get("fixed_filter_gain"),
        FIXED_FILTER_GAINS["cfrwm"],
    )

    data = contract.get("data_design", {})
    expected_runs = len(CONFIRM_METHODS) * len(CONFIRM_SEEDS) * len(SYSTEMS)
    expected_aggregate = expected_runs * len(CONFIRM_REGIMES)
    expected_trajectories = expected_aggregate * 96
    for key, expected in {
        "train_count": 256,
        "validation_count": 64,
        "test_count": 96,
        "trajectory_steps": 72,
        "trajectories_per_cell": 96,
        "expected_runs": expected_runs,
        "expected_aggregate_rows": expected_aggregate,
        "expected_trajectory_rows": expected_trajectories,
    }.items():
        _require_equal(f"data_design.{key}", data.get(key), expected)

    training = contract.get("training", {})
    expected_training = {
        "optimizer": "AdamW",
        "epochs": 60,
        "patience": 10,
        "batch_size": 128,
        "min_delta": 1e-5,
        "gradient_clip": 5.0,
        "num_workers": 0,
        "baseline_tuning_source": "own_frozen_system_method",
        "cfr_tuning_source": "pgnn_same_system",
        "output_directory": "results/confirmatory",
        "wall_clock_hours": 8.0,
        "deadline_fraction": 0.8,
    }
    for key, expected in expected_training.items():
        _require_equal(f"training.{key}", training.get(key), expected)
    tuning_path = Path(str(training.get("tuning_file", "")))
    if not tuning_path.is_file():
        raise RuntimeError(f"frozen tuning file is missing: {tuning_path}")
    if _sha256(tuning_path) != training.get("tuning_file_sha256"):
        raise RuntimeError("frozen tuning file hash mismatch")
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    _require_equal("tuning systems", set(tuning), set(SYSTEMS))
    for system in SYSTEMS:
        missing = set(COMPARATORS) - set(tuning[system])
        if missing:
            raise RuntimeError(f"frozen tuning lacks {system} methods: {sorted(missing)}")

    statistics = contract.get("statistics", {})
    for key, expected in {
        "bootstrap_replicates": 5000,
        "primary_comparator": "pgnn",
        "h1_relative_margin_percent": 5.0,
        "h2_clean_noninferiority_margin_percent": -5.0,
        "h2_hierarchically_gated_by_h1": True,
    }.items():
        _require_equal(f"statistics.{key}", statistics.get(key), expected)

    hashes = contract.get("hashes", {})
    _require_equal(
        "training code hash", hashes.get("training_code_sha256"), evaluation_code_sha256(),
    )
    _require_equal(
        "analysis code hash", hashes.get("analysis_code_sha256"), _analysis_sha256(),
    )
    return tuning


def _expected_train_config(
    method: str,
    system: str,
    contract: dict[str, Any],
    tuning: dict[str, Any],
) -> TrainConfig:
    if method not in CONFIRM_METHODS or system not in SYSTEMS:
        raise RuntimeError(f"condition is outside the frozen grid: {method}/{system}")
    training = contract["training"]
    base = TrainConfig(
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        patience=int(training["patience"]),
        min_delta=float(training["min_delta"]),
        gradient_clip=float(training["gradient_clip"]),
        num_workers=int(training["num_workers"]),
    )
    tuning_method = method if method in COMPARATORS else "pgnn"
    selected = tuning[system][tuning_method]
    return replace(
        base,
        learning_rate=float(selected["learning_rate"]),
        weight_decay=float(selected["weight_decay"]),
        reliability_weight=float(selected["reliability_weight"]),
    )


def _expected_fixed_filter_selection(method: str) -> dict[str, object] | None:
    if method not in FIXED_FILTER_GAINS:
        return None
    return {
        "selected": FIXED_FILTER_GAINS[method],
        "selection": "prespecified_before_training",
    }


def _validate_existing_payload(
    payload: dict[str, Any],
    method: str,
    seed: int,
    system: str,
    config: TrainConfig,
    provenance: dict[str, object],
) -> None:
    expected = {
        "method": method,
        "seed": seed,
        "system": system,
        "run_code_sha256": evaluation_code_sha256(),
        "evaluation_code_sha256": evaluation_code_sha256(),
        "evaluation_metric_version": EVALUATION_METRIC_VERSION,
        "hidden_dim": HIDDEN_DIMENSIONS[method],
        "train_config": asdict(config),
        "one_step_training": method == "cfr_1step",
        "fixed_filter_selection": _expected_fixed_filter_selection(method),
        **provenance,
    }
    mismatches = {
        key: {"expected": value, "found": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if payload.get("train_result", {}).get("stopped_for_budget") is not False:
        mismatches["train_result.stopped_for_budget"] = {
            "expected": False,
            "found": payload.get("train_result", {}).get("stopped_for_budget"),
        }
    data_hash = payload.get("data_sha256")
    if not isinstance(data_hash, str) or len(data_hash) != 64:
        mismatches["data_sha256"] = {"expected": "64-character SHA-256", "found": data_hash}
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, dict) or set(evaluations) != set(CONFIRM_REGIMES):
        mismatches["evaluations"] = {
            "expected": list(CONFIRM_REGIMES),
            "found": list(evaluations) if isinstance(evaluations, dict) else type(evaluations).__name__,
        }
    else:
        identifier_sets: dict[str, set[str]] = {}
        for regime in CONFIRM_REGIMES:
            evaluation = evaluations[regime]
            aggregate = evaluation.get("aggregate") if isinstance(evaluation, dict) else None
            trajectories = evaluation.get("trajectories") if isinstance(evaluation, dict) else None
            if not isinstance(aggregate, dict) or not aggregate:
                mismatches[f"evaluations.{regime}.aggregate"] = {
                    "expected": "nonempty metric dictionary", "found": aggregate,
                }
            elif any(
                isinstance(value, (int, float)) and not math.isfinite(float(value))
                for value in aggregate.values()
            ):
                mismatches[f"evaluations.{regime}.aggregate"] = {
                    "expected": "finite metrics", "found": aggregate,
                }
            if not isinstance(trajectories, list) or len(trajectories) != 96:
                mismatches[f"evaluations.{regime}.trajectories"] = {
                    "expected": 96,
                    "found": len(trajectories) if isinstance(trajectories, list) else None,
                }
                continue
            identifiers = [str(row.get("trajectory_id")) for row in trajectories]
            if len(set(identifiers)) != 96 or "None" in identifiers:
                mismatches[f"evaluations.{regime}.trajectory_ids"] = {
                    "expected": "96 unique nonmissing identifiers", "found": len(set(identifiers)),
                }
            for row in trajectories:
                if any(
                    isinstance(value, (int, float)) and not math.isfinite(float(value))
                    for key, value in row.items() if key != "trajectory_id"
                ):
                    mismatches[f"evaluations.{regime}.finite"] = {
                        "expected": True, "found": False,
                    }
                    break
            identifier_sets[regime] = set(identifiers)
        if (
            "id_clean" in identifier_sets
            and "id_corrupt" in identifier_sets
            and identifier_sets["id_clean"] != identifier_sets["id_corrupt"]
        ):
            mismatches["evaluations.id_repeated_measurements"] = {
                "expected": "matching clean/corrupt identifiers", "found": "mismatch",
            }
        if (
            "id_clean" in identifier_sets
            and "ood_parameter_corrupt" in identifier_sets
            and identifier_sets["id_clean"] & identifier_sets["ood_parameter_corrupt"]
        ):
            mismatches["evaluations.ood_independence"] = {
                "expected": "disjoint ID and OOD identifiers", "found": "overlap",
            }
    if mismatches:
        raise RuntimeError(f"stale confirmatory payload for {system}/{seed}/{method}: {mismatches}")


def _validate_existing_output(
    output: Path,
    contract: dict[str, Any],
    tuning: dict[str, Any],
    provenance: dict[str, object],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    expected_paths = {
        output / system / f"seed_{seed}" / f"{method}.json"
        for system in SYSTEMS for seed in CONFIRM_SEEDS for method in CONFIRM_METHODS
    }
    found_json = set(output.rglob("*.json")) if output.exists() else set()
    unexpected = found_json - expected_paths
    if unexpected:
        raise RuntimeError(f"unexpected files in frozen confirmatory output: {sorted(map(str, unexpected))}")
    found_checkpoints = set(output.rglob("*.pt")) if output.exists() else set()
    expected_checkpoints = {path.with_suffix(".pt") for path in found_json}
    if found_checkpoints != expected_checkpoints:
        raise RuntimeError("confirmatory JSON/checkpoint files are not an exact pairwise subset")
    temporary = list(output.rglob("*.tmp")) if output.exists() else []
    if temporary:
        raise RuntimeError(f"incomplete temporary confirmatory files exist: {temporary}")

    payloads: dict[tuple[str, int, str], dict[str, Any]] = {}
    for path in sorted(found_json):
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = str(payload.get("method"))
        seed = int(payload.get("seed"))
        system = str(payload.get("system"))
        config = _expected_train_config(method, system, contract, tuning)
        _validate_existing_payload(payload, method, seed, system, config, provenance)
        expected_path = output / system / f"seed_{seed}" / f"{method}.json"
        if path != expected_path:
            raise RuntimeError(f"confirmatory payload is stored at the wrong path: {path}")
        payloads[(method, seed, system)] = payload
    data_groups: dict[tuple[int, str], set[str]] = {}
    for (_method, seed, system), payload in payloads.items():
        data_groups.setdefault((seed, system), set()).add(str(payload["data_sha256"]))
    if any(len(values) != 1 for values in data_groups.values()):
        raise RuntimeError("partial confirmatory methods have mismatched data fingerprints")
    distinct_hashes = {next(iter(values)) for values in data_groups.values()}
    if len(distinct_hashes) != len(data_groups):
        raise RuntimeError("partial confirmatory seed/system datasets are not distinct")
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the immutable CFR-WM confirmatory grid; no overrides are accepted.",
    )
    parser.parse_args()

    if not CONTRACT_PATH.is_file():
        raise RuntimeError(f"frozen confirmatory contract is missing: {CONTRACT_PATH}")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    tuning = _validate_contract(contract)
    contract_hash = _sha256(CONTRACT_PATH)
    training = contract["training"]
    provenance: dict[str, object] = {
        "confirmatory_contract_sha256": contract_hash,
        "tuning_file_sha256": training["tuning_file_sha256"],
        "analysis_code_sha256": contract["hashes"]["analysis_code_sha256"],
    }
    output = Path(training["output_directory"])
    existing = _validate_existing_output(output, contract, tuning, provenance)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    deadline = start + float(training["wall_clock_hours"]) * 3600.0 * float(
        training["deadline_fraction"]
    )
    print(json.dumps({
        "event": "confirmatory_start",
        "contract_sha256": contract_hash,
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "existing_conditions": len(existing),
        "expected_conditions": contract["data_design"]["expected_runs"],
    }))

    completed = dict(existing)
    for system_name, spec in SYSTEMS.items():
        for seed in CONFIRM_SEEDS:
            missing_methods = [
                method for method in CONFIRM_METHODS
                if (method, seed, system_name) not in completed
            ]
            if not missing_methods:
                continue
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    f"confirmatory time guard reached after {len(completed)} conditions"
                )
            normalizer, train_windows, validation_windows, tests = prepare_windows(
                spec,
                seed,
                train_count=int(contract["data_design"]["train_count"]),
                validation_count=int(contract["data_design"]["validation_count"]),
                test_count=int(contract["data_design"]["test_count"]),
                trajectory_steps=int(contract["data_design"]["trajectory_steps"]),
            )
            for method in missing_methods:
                if time.perf_counter() >= deadline:
                    raise RuntimeError(
                        f"confirmatory time guard reached after {len(completed)} conditions"
                    )
                config = _expected_train_config(method, system_name, contract, tuning)
                condition_start = time.perf_counter()
                payload = run_condition(
                    method,
                    spec,
                    seed,
                    normalizer,
                    train_windows,
                    validation_windows,
                    tests,
                    config,
                    device,
                    output,
                    deadline,
                    overwrite=False,
                    provenance=provenance,
                )
                _validate_existing_payload(
                    payload, method, seed, system_name, config, provenance,
                )
                completed[(method, seed, system_name)] = payload
                print(json.dumps({
                    "event": "condition_complete",
                    "system": system_name,
                    "seed": seed,
                    "method": method,
                    "seconds": time.perf_counter() - condition_start,
                    "completed": len(completed),
                }))

    final_payloads = _validate_existing_output(output, contract, tuning, provenance)
    _require_equal(
        "completed run count",
        len(final_payloads),
        contract["data_design"]["expected_runs"],
    )
    elapsed = time.perf_counter() - start
    manifest_path = Path("results/confirmatory_run_manifest.json")
    manifest = {
        "contract_sha256": contract_hash,
        "training_code_sha256": evaluation_code_sha256(),
        "analysis_code_sha256": _analysis_sha256(),
        "tuning_file_sha256": training["tuning_file_sha256"],
        "conditions": len(final_payloads),
        "elapsed_seconds_this_invocation": elapsed,
        "device": str(device),
        "complete": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps({"event": "confirmatory_complete", **manifest}))


if __name__ == "__main__":
    main()
