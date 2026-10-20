"""Apply the frozen development-only kill rule for the PFR-WM pivot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import load_run_directory
from .dynamics import SYSTEMS


EXPECTED_METHODS = {"pgnn", "pfrwm", "pfr_fixed"}
EXPECTED_REGIMES = {"id_clean", "id_corrupt", "ood_parameter_corrupt"}


def _validate_development_grid(
    aggregate: pd.DataFrame,
    trajectories: pd.DataFrame,
) -> None:
    expected_cells = {
        (method, 11, system, regime)
        for method in EXPECTED_METHODS
        for system in SYSTEMS
        for regime in EXPECTED_REGIMES
    }
    aggregate_keys = ["method", "seed", "system", "regime"]
    if aggregate.duplicated(aggregate_keys).any():
        raise ValueError("development aggregate contains duplicate method-cell rows")
    found_cells = set(aggregate[aggregate_keys].itertuples(index=False, name=None))
    if found_cells != expected_cells:
        raise ValueError("development aggregate does not match the frozen seed/system/regime grid")
    trajectory_keys = [*aggregate_keys, "trajectory_id"]
    if trajectories.duplicated(trajectory_keys).any():
        raise ValueError("development trajectories contain duplicate paired keys")
    for system in SYSTEMS:
        for regime in EXPECTED_REGIMES:
            identifier_sets = []
            for method in EXPECTED_METHODS:
                selected = trajectories[
                    (trajectories["method"] == method)
                    & (trajectories["seed"] == 11)
                    & (trajectories["system"] == system)
                    & (trajectories["regime"] == regime)
                ]
                identifiers = set(selected["trajectory_id"].astype(str))
                if len(identifiers) != 96:
                    raise ValueError("each frozen method-cell must contain 96 trajectories")
                identifier_sets.append(identifiers)
            if not all(value == identifier_sets[0] for value in identifier_sets[1:]):
                raise ValueError("trajectory identifiers are not paired across methods")


def pfr_kill_decision(
    aggregate: pd.DataFrame,
    trajectories: pd.DataFrame,
) -> dict[str, object]:
    _validate_development_grid(aggregate, trajectories)
    cell = (
        trajectories.groupby(["method", "system", "regime"])["nrmse_h25"]
        .mean()
        .unstack("method")
        .reset_index()
    )
    mean_by_method = cell[sorted(EXPECTED_METHODS)].mean()
    pgnn_improvement = 100.0 * (
        mean_by_method["pgnn"] - mean_by_method["pfrwm"]
    ) / mean_by_method["pgnn"]
    fixed_improvement = 100.0 * (
        mean_by_method["pfr_fixed"] - mean_by_method["pfrwm"]
    ) / mean_by_method["pfr_fixed"]
    cell["relative_change_vs_pgnn_percent"] = 100.0 * (
        cell["pfrwm"] - cell["pgnn"]
    ) / cell["pgnn"]
    cells_worse_gt_10 = int((cell["relative_change_vs_pgnn_percent"] > 10.0).sum())
    proposed = aggregate[aggregate["method"] == "pfrwm"]
    gains = proposed.groupby("system")["mean_observed_filter_gain"].mean()
    degenerate_rows = (
        (proposed["observed_filter_gain_near_zero_fraction"] > 0.95)
        | (proposed["observed_filter_gain_near_one_fraction"] > 0.95)
    )
    degenerate_systems = int(proposed.loc[degenerate_rows, "system"].nunique())
    finite = bool(np.isfinite(
        trajectories[["nrmse_h25", "gaussian_nll", "calibration_error"]]
        .to_numpy(dtype=float)
    ).all())
    passed = bool(
        pgnn_improvement >= 5.0
        and fixed_improvement >= 2.0
        and cells_worse_gt_10 <= 2
        and degenerate_systems <= 1
        and finite
    )
    return {
        "development_only": True,
        "cell_balanced_nrmse_h25": {
            method: float(mean_by_method[method]) for method in sorted(EXPECTED_METHODS)
        },
        "pfr_improvement_vs_pgnn_percent": float(pgnn_improvement),
        "pfr_improvement_vs_fixed_filter_percent": float(fixed_improvement),
        "cells_worse_than_pgnn_by_more_than_10_percent": cells_worse_gt_10,
        "degenerate_gain_systems": degenerate_systems,
        "mean_observed_gain_by_system": {
            str(system): float(value) for system, value in gains.items()
        },
        "all_metrics_finite": finite,
        "kill_rule_passed": passed,
        "cell_details": cell.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="results/pfr_development")
    parser.add_argument("--output", default="results/pfr_development/kill_test.json")
    args = parser.parse_args()
    aggregate, trajectories, payloads = load_run_directory(Path(args.directory))
    if len(payloads) != 15:
        raise RuntimeError(f"expected 15 development runs, found {len(payloads)}")
    decision = pfr_kill_decision(aggregate, trajectories)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(decision, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
