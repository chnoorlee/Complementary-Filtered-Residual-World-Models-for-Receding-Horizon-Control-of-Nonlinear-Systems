"""Analyze the frozen CFR-WM confirmatory benchmark on untouched seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import load_run_directory
from .confirm_protocol import (
    ABLATIONS,
    COMPARATORS,
    CONFIRM_METHODS,
    CONFIRM_REGIMES,
    CONFIRM_SEEDS,
    METHOD_LABELS,
    analysis_code_sha256 as _analysis_sha256,
)
from .confirm_run import (
    CONTRACT_PATH,
    _expected_train_config,
    _sha256,
    _validate_contract,
    _validate_existing_payload,
)
from .dynamics import SYSTEMS
from .statistics import holm_adjust, paired_cell_bootstrap, paired_fixed_cell_bootstrap


def _validate_confirmatory_grid(
    aggregate: pd.DataFrame,
    trajectories: pd.DataFrame,
) -> None:
    expected_cells = {
        (method, seed, system, regime)
        for method in CONFIRM_METHODS
        for seed in CONFIRM_SEEDS
        for system in SYSTEMS
        for regime in CONFIRM_REGIMES
    }
    aggregate_keys = ["method", "seed", "system", "regime"]
    if aggregate.duplicated(aggregate_keys).any():
        raise ValueError("confirmatory aggregate contains duplicate method-cell rows")
    found_cells = set(aggregate[aggregate_keys].itertuples(index=False, name=None))
    if found_cells != expected_cells:
        raise ValueError(
            "confirmatory aggregate does not match the frozen method/seed/system/regime grid"
        )

    trajectory_keys = [*aggregate_keys, "trajectory_id"]
    if trajectories.duplicated(trajectory_keys).any():
        raise ValueError("confirmatory trajectories contain duplicate paired keys")
    found_trajectory_cells = set(
        trajectories[aggregate_keys].itertuples(index=False, name=None)
    )
    expected_trajectory_rows = len(expected_cells) * 96
    if (
        found_trajectory_cells != expected_cells
        or len(trajectories) != expected_trajectory_rows
    ):
        raise ValueError(
            "confirmatory trajectories do not match the exact frozen trajectory grid"
        )
    for seed in CONFIRM_SEEDS:
        for system in SYSTEMS:
            regime_identifier_sets: dict[str, set[str]] = {}
            for regime in CONFIRM_REGIMES:
                identifier_sets = []
                for method in CONFIRM_METHODS:
                    selected = trajectories[
                        (trajectories["method"] == method)
                        & (trajectories["seed"] == seed)
                        & (trajectories["system"] == system)
                        & (trajectories["regime"] == regime)
                    ]
                    identifiers = set(selected["trajectory_id"].astype(str))
                    if len(identifiers) != 96:
                        raise ValueError(
                            "each frozen confirmatory method-cell must contain 96 trajectories"
                        )
                    identifier_sets.append(identifiers)
                if not all(value == identifier_sets[0] for value in identifier_sets[1:]):
                    raise ValueError("confirmatory trajectory identifiers are not paired across methods")
                regime_identifier_sets[regime] = identifier_sets[0]
            if regime_identifier_sets["id_clean"] != regime_identifier_sets["id_corrupt"]:
                raise ValueError(
                    "ID-clean/corrupt trajectory identifiers must match as repeated measurements"
                )
            if (
                regime_identifier_sets["id_clean"]
                & regime_identifier_sets["ood_parameter_corrupt"]
            ):
                raise ValueError(
                    "OOD trajectory identifiers must be disjoint from ID trajectories"
                )


def _validate_payload_data_pairing(payloads: list[dict[str, object]]) -> int:
    payload_keys = [
        (str(payload["method"]), int(payload["seed"]), str(payload["system"]))
        for payload in payloads
    ]
    expected_payload_keys = {
        (method, seed, system)
        for method in CONFIRM_METHODS
        for seed in CONFIRM_SEEDS
        for system in SYSTEMS
    }
    if len(payload_keys) != len(set(payload_keys)) or set(payload_keys) != expected_payload_keys:
        raise RuntimeError("confirmatory payloads do not match the exact frozen run grid")
    groups: dict[tuple[int, str], set[str]] = {}
    for payload in payloads:
        key = (int(payload["seed"]), str(payload["system"]))
        groups.setdefault(key, set()).add(str(payload["data_sha256"]))
    expected_groups = {
        (seed, system) for seed in CONFIRM_SEEDS for system in SYSTEMS
    }
    if set(groups) != expected_groups or any(len(values) != 1 for values in groups.values()):
        raise RuntimeError(
            "confirmatory methods do not share one data fingerprint per seed/system"
        )
    unique_count = len({next(iter(values)) for values in groups.values()})
    if unique_count != len(expected_groups):
        raise RuntimeError(
            "every confirmatory seed/system requires a distinct data fingerprint"
        )
    return unique_count


def _summary(aggregate: pd.DataFrame, trajectories: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "nrmse_h1", "nrmse_h5", "nrmse_h10", "nrmse_h25", "mae_h25",
        "gaussian_nll", "calibration_error", "catastrophic",
    )
    cells = trajectories.groupby(
        ["method", "seed", "system", "regime"], as_index=False,
    )[list(metrics)].mean()
    seed_means = cells.groupby(["method", "seed"])[list(metrics)].mean()
    mean = seed_means.groupby("method").mean()
    std = seed_means.groupby("method").std(ddof=1)
    parameter_counts = aggregate.groupby("method")["parameter_count"].mean()
    rows: list[dict[str, object]] = []
    for method in CONFIRM_METHODS:
        row: dict[str, object] = {
            "method": method,
            "label": METHOD_LABELS[method],
            "parameter_count_mean": float(parameter_counts.loc[method]),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(mean.loc[method, metric])
            row[f"{metric}_sd"] = float(std.loc[method, metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("nrmse_h25_mean")


def _contrast_family(
    trajectories: pd.DataFrame,
    baselines: tuple[str, ...],
    replicates: int,
    seed: int,
) -> list[dict[str, object]]:
    results = [
        paired_fixed_cell_bootstrap(
            trajectories, baseline, "cfrwm", "nrmse_h25", replicates, seed,
        )
        for baseline in baselines
    ]
    adjusted = holm_adjust(np.asarray([item["p_two_sided"] for item in results]))
    for item, p_holm in zip(results, adjusted, strict=True):
        item["p_holm"] = float(p_holm)
    return results


def _cell_contrasts(trajectories: pd.DataFrame, replicates: int) -> pd.DataFrame:
    table = paired_cell_bootstrap(
        trajectories, "pgnn", "cfrwm", "nrmse_h25", replicates, 20260805,
    )
    baseline = (
        trajectories[trajectories["method"] == "pgnn"]
        .groupby(["system", "regime"])["nrmse_h25"]
        .mean()
        .rename("pgnn_nrmse_h25")
        .reset_index()
    )
    table = table.merge(baseline, on=["system", "regime"], validate="one_to_one")
    table = table.drop(columns=["p_raw", "p_holm"])
    table["relative_change_cfr_minus_pgnn_percent"] = (
        100.0 * table["difference_proposed_minus_baseline"]
        / table["pgnn_nrmse_h25"]
    )
    return table


def analyze_confirmatory(
    aggregate: pd.DataFrame,
    trajectories: pd.DataFrame,
    replicates: int = 5000,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    _validate_confirmatory_grid(aggregate, trajectories)
    finite_columns = [
        "nrmse_h1", "nrmse_h5", "nrmse_h10", "nrmse_h25", "mae_h25",
        "gaussian_nll", "calibration_error", "catastrophic",
    ]
    all_finite = bool(np.isfinite(trajectories[finite_columns].to_numpy(float)).all())
    summary = _summary(aggregate, trajectories)

    primary = paired_fixed_cell_bootstrap(
        trajectories, "pgnn", "cfrwm", "nrmse_h25", replicates, 20260805,
        relative_margin_percent=5.0,
    )
    primary["minimum_relative_improvement_percent"] = 5.0
    primary["preselected_comparator"] = True
    primary["h1_supported"] = bool(
        primary["relative_improvement_ci_95"][0] >= 5.0
        and primary["p_one_sided_relative_margin"] < 0.05
        and all_finite
    )

    corrupt = trajectories[trajectories["regime"] != "id_clean"]
    clean = trajectories[trajectories["regime"] == "id_clean"]
    corrupt_contrast = paired_fixed_cell_bootstrap(
        corrupt, "pgnn", "cfrwm", "nrmse_h25", replicates, 20260806,
        relative_margin_percent=5.0,
    )
    clean_contrast = paired_fixed_cell_bootstrap(
        clean, "pgnn", "cfrwm", "nrmse_h25", replicates, 20260807,
        relative_margin_percent=-5.0,
    )
    h2_supported = bool(
        primary["h1_supported"]
        and corrupt_contrast["relative_improvement_ci_95"][0] >= 5.0
        and corrupt_contrast["p_one_sided_relative_margin"] < 0.05
        and clean_contrast["relative_improvement_ci_95"][0] >= -5.0
        and clean_contrast["p_one_sided_relative_margin"] < 0.05
        and all_finite
    )

    comparator_contrasts = _contrast_family(
        trajectories, COMPARATORS, replicates, 20260808,
    )
    ablation_contrasts = _contrast_family(
        trajectories, ABLATIONS, replicates, 20260809,
    )
    cell = _cell_contrasts(trajectories, replicates)
    cells_worse_gt_10 = int(
        (cell["relative_change_cfr_minus_pgnn_percent"] > 10.0).sum()
    )
    one_step = next(
        item for item in ablation_contrasts if item["baseline"] == "cfr_1step"
    )

    decision: dict[str, object] = {
        "confirmatory_only": True,
        "methods": list(CONFIRM_METHODS),
        "seeds": list(CONFIRM_SEEDS),
        "systems": list(SYSTEMS),
        "regimes": list(CONFIRM_REGIMES),
        "all_metrics_finite": all_finite,
        "primary_h1": primary,
        "robustness_h2": {
            "hierarchically_gated_by_h1": True,
            "corrupt_and_shifted": corrupt_contrast,
            "clean_signed_relative_margin_percent": -5.0,
            "clean_maximum_degradation_percent": 5.0,
            "clean": clean_contrast,
            "supported": h2_supported,
        },
        "comparator_contrasts": comparator_contrasts,
        "ablation_contrasts": ablation_contrasts,
        "multi_step_ablation_supported": bool(
            one_step["ci_95"][1] < 0.0 and one_step["p_holm"] < 0.05
        ),
        "cells_worse_than_pgnn_by_more_than_10_percent": cells_worse_gt_10,
        "cell_degradation_guard_is_descriptive": True,
        "cell_contrasts_are_descriptive": True,
        "confirmation_passed": bool(primary["h1_supported"]),
    }
    return decision, summary, cell


def _write_latex_summary(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & NRMSE-1 & NRMSE-10 & NRMSE-25 \\",
        r"\midrule",
    ]
    best = float(summary["nrmse_h25_mean"].min())
    for _, row in summary.iterrows():
        values = []
        for metric in ("nrmse_h1", "nrmse_h10", "nrmse_h25"):
            value = f"{row[f'{metric}_mean']:.3f} $\\pm$ {row[f'{metric}_sd']:.3f}"
            if metric == "nrmse_h25" and np.isclose(row[f"{metric}_mean"], best):
                value = r"\textbf{" + value + "}"
            values.append(value)
        lines.append(f"{row['label']} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the immutable CFR-WM confirmatory grid; no overrides are accepted.",
    )
    parser.parse_args()
    if not CONTRACT_PATH.is_file():
        raise RuntimeError(f"frozen confirmatory contract is missing: {CONTRACT_PATH}")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    tuning = _validate_contract(contract)
    contract_hash = _sha256(CONTRACT_PATH)
    provenance: dict[str, object] = {
        "confirmatory_contract_sha256": contract_hash,
        "tuning_file_sha256": contract["training"]["tuning_file_sha256"],
        "analysis_code_sha256": contract["hashes"]["analysis_code_sha256"],
    }
    directory = Path(contract["training"]["output_directory"])
    aggregate, trajectories, payloads = load_run_directory(directory)
    expected_runs = len(CONFIRM_METHODS) * len(CONFIRM_SEEDS) * len(SYSTEMS)
    if len(payloads) != expected_runs:
        raise RuntimeError(f"expected {expected_runs} confirmatory runs, found {len(payloads)}")
    for payload in payloads:
        method = str(payload["method"])
        seed = int(payload["seed"])
        system = str(payload["system"])
        config = _expected_train_config(method, system, contract, tuning)
        _validate_existing_payload(
            payload, method, seed, system, config, provenance,
        )
    unique_data_count = _validate_payload_data_pairing(payloads)
    decision, summary, cell = analyze_confirmatory(
        aggregate, trajectories, int(contract["statistics"]["bootstrap_replicates"]),
    )
    decision["analysis_code_sha256"] = _analysis_sha256()
    decision["training_code_sha256"] = contract["hashes"]["training_code_sha256"]
    decision["evaluation_code_sha256"] = contract["hashes"]["training_code_sha256"]
    decision["confirmatory_contract_sha256"] = contract_hash
    decision["tuning_file_sha256"] = contract["training"]["tuning_file_sha256"]
    decision["unique_data_sha256_count"] = unique_data_count

    output = Path("results/confirmatory_analysis")
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "confirmatory_summary.csv", index=False)
    aggregate.to_csv(output / "aggregate_cells.csv", index=False)
    trajectories.to_csv(output / "trajectory_metrics.csv", index=False)
    cell.to_csv(output / "cell_contrasts.csv", index=False)
    (output / "confirmatory_inference.json").write_text(
        json.dumps(decision, indent=2, allow_nan=False), encoding="utf-8",
    )
    _write_latex_summary(summary, output / "table_confirmatory.tex")
    print(json.dumps(decision, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
