"""Paired hierarchical bootstrap and multiplicity control."""

from __future__ import annotations

import numpy as np
import pandas as pd


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _paired_wide(
    frame: pd.DataFrame,
    baseline: str,
    proposed: str,
    metric: str,
) -> pd.DataFrame:
    keys = ["seed", "system", "regime", "trajectory_id"]
    selected = frame[frame["method"].isin([baseline, proposed])]
    if selected.duplicated([*keys, "method"]).any():
        raise ValueError("paired comparison contains duplicate method observations")
    wide = selected.pivot(index=keys, columns="method", values=metric)
    if baseline not in wide or proposed not in wide:
        raise ValueError("both paired methods are required")
    if wide[[baseline, proposed]].isna().any().any():
        raise ValueError("both methods require complete paired observations")
    wide["difference"] = wide[proposed] - wide[baseline]
    return wide.reset_index()


def _exact_sign_flip_p(values: np.ndarray, alternative: str) -> float:
    """Exact paired sign-flip p-value under a symmetric cluster-effect null."""
    effects = np.asarray(values, dtype=np.float64)
    if effects.ndim != 1 or len(effects) == 0 or not np.isfinite(effects).all():
        raise ValueError("sign-flip effects must be a nonempty finite vector")
    if len(effects) > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 clusters")
    configurations = np.arange(1 << len(effects), dtype=np.uint64)[:, None]
    bits = (configurations >> np.arange(len(effects), dtype=np.uint64)) & 1
    signs = np.where(bits == 0, 1.0, -1.0)
    permuted = np.mean(signs * effects[None, :], axis=1)
    observed = float(np.mean(effects))
    tolerance = 1e-12 * max(1.0, abs(observed))
    if alternative == "two-sided":
        extreme = np.abs(permuted) >= abs(observed) - tolerance
    elif alternative == "greater":
        extreme = permuted >= observed - tolerance
    else:
        raise ValueError(f"unsupported sign-flip alternative: {alternative}")
    return float(np.mean(extreme))


def _fixed_grid_structure(
    wide: pd.DataFrame,
    baseline: str,
    proposed: str,
) -> tuple[
    np.ndarray,
    list[tuple[str, str]],
    dict[tuple[object, str], list[tuple[tuple[str, ...], dict[str, np.ndarray], dict[str, np.ndarray]]]],
]:
    """Build trajectory bundles shared by repeated-measurement regimes."""
    seeds = np.sort(wide["seed"].unique())
    reference_cells: set[tuple[str, str]] | None = None
    for seed_value in seeds:
        seed_cells = set(
            wide.loc[wide["seed"] == seed_value, ["system", "regime"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if reference_cells is None:
            reference_cells = seed_cells
        elif seed_cells != reference_cells:
            raise ValueError("every seed must contain the same fixed benchmark cells")
    cells = sorted(reference_cells or set())
    if not cells:
        raise ValueError("fixed-cell comparison contains no benchmark cells")

    bundles: dict[
        tuple[object, str],
        list[tuple[tuple[str, ...], dict[str, np.ndarray], dict[str, np.ndarray]]],
    ] = {}
    systems = sorted({system for system, _ in cells})
    for seed_value in seeds:
        for system in systems:
            system_frame = wide[
                (wide["seed"] == seed_value) & (wide["system"] == system)
            ]
            regimes = sorted(system_frame["regime"].unique())
            by_identifier_set: dict[tuple[str, ...], list[str]] = {}
            regime_tables: dict[str, pd.DataFrame] = {}
            for regime in regimes:
                table = system_frame[system_frame["regime"] == regime].copy()
                table["trajectory_id"] = table["trajectory_id"].astype(str)
                table = table.sort_values("trajectory_id").set_index("trajectory_id")
                identifiers = tuple(table.index)
                by_identifier_set.setdefault(identifiers, []).append(regime)
                regime_tables[regime] = table
            if {"id_clean", "id_corrupt"}.issubset(regime_tables):
                clean_ids = set(regime_tables["id_clean"].index)
                corrupt_ids = set(regime_tables["id_corrupt"].index)
                if clean_ids != corrupt_ids:
                    raise ValueError(
                        "ID-clean/corrupt trajectory identifiers must match"
                    )
                if "ood_parameter_corrupt" in regime_tables:
                    ood_ids = set(regime_tables["ood_parameter_corrupt"].index)
                    if clean_ids & ood_ids:
                        raise ValueError(
                            "OOD trajectory identifiers must be disjoint from ID trajectories"
                        )
            system_bundles = []
            for identifiers, bundled_regimes in by_identifier_set.items():
                baseline_arrays = {
                    regime: regime_tables[regime][baseline].to_numpy(dtype=np.float64)
                    for regime in bundled_regimes
                }
                proposed_arrays = {
                    regime: regime_tables[regime][proposed].to_numpy(dtype=np.float64)
                    for regime in bundled_regimes
                }
                system_bundles.append(
                    (tuple(bundled_regimes), baseline_arrays, proposed_arrays)
                )
            bundles[(seed_value, system)] = system_bundles
    return seeds, cells, bundles


def paired_hierarchical_bootstrap(
    frame: pd.DataFrame,
    baseline: str,
    proposed: str,
    metric: str,
    replicates: int = 5000,
    seed: int = 20260805,
) -> dict[str, object]:
    """Bootstrap seeds, system-regime cells, and paired trajectories."""
    wide = _paired_wide(frame, baseline, proposed, metric)
    seeds = np.sort(wide["seed"].unique())
    cells = wide[["system", "regime"]].drop_duplicates().itertuples(index=False, name=None)
    cells = list(cells)
    lookup = {
        (seed_value, system, regime): group["difference"].to_numpy()
        for (seed_value, system, regime), group in wide.groupby(["seed", "system", "regime"])
    }
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
        sampled_cells = [cells[index] for index in rng.integers(0, len(cells), len(cells))]
        cluster_means: list[float] = []
        for seed_value in sampled_seeds:
            for system, regime in sampled_cells:
                differences = lookup[(seed_value, system, regime)]
                cluster_means.append(float(rng.choice(differences, len(differences), replace=True).mean()))
        samples[replicate] = np.mean(cluster_means)
    point = float(
        wide.groupby(["seed", "system", "regime"])["difference"].mean().mean()
    )
    baseline_point = float(
        wide.groupby(["seed", "system", "regime"])[baseline].mean().mean()
    )
    return {
        "baseline": baseline,
        "proposed": proposed,
        "metric": metric,
        "difference_proposed_minus_baseline": point,
        "relative_improvement_percent": None if abs(baseline_point) < 1e-12 else float(-100.0 * point / baseline_point),
        "ci_95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "p_two_sided": float(min(1.0, 2.0 * min(np.mean(samples >= 0.0), np.mean(samples <= 0.0)))),
        "replicates": replicates,
        "paired_trajectories": len(wide),
    }


def paired_fixed_cell_bootstrap(
    frame: pd.DataFrame,
    baseline: str,
    proposed: str,
    metric: str,
    replicates: int = 5000,
    seed: int = 20260805,
    relative_margin_percent: float = 0.0,
) -> dict[str, object]:
    """Bootstrap seeds and trajectories while retaining every benchmark cell.

    This targets performance on the prespecified finite benchmark rather than
    treating the five systems and three regimes as a random sample of cells.
    """
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    wide = _paired_wide(frame, baseline, proposed, metric)
    seeds, cells, bundles = _fixed_grid_structure(wide, baseline, proposed)
    systems = sorted({system for system, _ in cells})

    cell_means = wide.groupby(["seed", "system", "regime"])[
        [baseline, proposed]
    ].mean()
    seed_means = cell_means.groupby("seed")[[baseline, proposed]].mean()
    if (np.abs(seed_means[baseline].to_numpy()) < 1e-12).any():
        raise ValueError("relative improvement requires a nonzero baseline for every seed")
    seed_differences = (
        seed_means[proposed] - seed_means[baseline]
    ).to_numpy(dtype=np.float64)
    seed_relative = (
        100.0 * (seed_means[baseline] - seed_means[proposed]) / seed_means[baseline]
    ).to_numpy(dtype=np.float64)

    rng = np.random.default_rng(seed)
    difference_samples = np.empty(replicates, dtype=np.float64)
    relative_samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
        sampled_seed_differences: list[float] = []
        sampled_seed_relative: list[float] = []
        for seed_value in sampled_seeds:
            sampled_baseline_cells: list[float] = []
            sampled_proposed_cells: list[float] = []
            for system in systems:
                for regimes, baseline_arrays, proposed_arrays in bundles[(seed_value, system)]:
                    count = len(next(iter(baseline_arrays.values())))
                    sampled_indices = rng.integers(0, count, count)
                    for regime in regimes:
                        sampled_baseline_cells.append(float(
                            baseline_arrays[regime][sampled_indices].mean()
                        ))
                        sampled_proposed_cells.append(float(
                            proposed_arrays[regime][sampled_indices].mean()
                        ))
            sampled_baseline = float(np.mean(sampled_baseline_cells))
            sampled_proposed = float(np.mean(sampled_proposed_cells))
            if abs(sampled_baseline) < 1e-12:
                raise ValueError("bootstrap sampled a zero baseline mean")
            sampled_seed_differences.append(sampled_proposed - sampled_baseline)
            sampled_seed_relative.append(
                100.0 * (sampled_baseline - sampled_proposed) / sampled_baseline
            )
        difference_samples[replicate] = float(np.mean(sampled_seed_differences))
        relative_samples[replicate] = float(np.mean(sampled_seed_relative))

    point = float(seed_differences.mean())
    baseline_point = float(seed_means[baseline].mean())
    proposed_point = float(seed_means[proposed].mean())
    return {
        "baseline": baseline,
        "proposed": proposed,
        "metric": metric,
        "baseline_cell_balanced_mean": baseline_point,
        "proposed_cell_balanced_mean": proposed_point,
        "difference_proposed_minus_baseline": point,
        "relative_improvement_percent": float(seed_relative.mean()),
        "ci_95": [
            float(value) for value in np.quantile(difference_samples, [0.025, 0.975])
        ],
        "relative_improvement_ci_95": [
            float(value) for value in np.quantile(relative_samples, [0.025, 0.975])
        ],
        "p_two_sided": _exact_sign_flip_p(seed_differences, "two-sided"),
        "relative_margin_percent": float(relative_margin_percent),
        "p_one_sided_relative_margin": _exact_sign_flip_p(
            seed_relative - relative_margin_percent, "greater",
        ),
        "seed_level_relative_improvements_percent": [
            float(value) for value in seed_relative
        ],
        "sign_flip_clusters": len(seeds),
        "sign_flip_symmetry_assumption": True,
        "replicates": replicates,
        "paired_trajectories": len(wide),
        "fixed_benchmark_cells": len(cells),
        "trajectory_resampling": (
            "bundled_when trajectory identifiers repeat across regimes within seed/system"
        ),
    }


def paired_cell_bootstrap(
    frame: pd.DataFrame,
    baseline: str,
    proposed: str,
    metric: str,
    replicates: int = 5000,
    seed: int = 20260805,
) -> pd.DataFrame:
    wide = _paired_wide(frame, baseline, proposed, metric)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for (system, regime), cell in wide.groupby(["system", "regime"]):
        seeds = np.sort(cell["seed"].unique())
        samples = np.empty(replicates)
        for replicate in range(replicates):
            seed_means = []
            for sampled_seed in rng.choice(seeds, len(seeds), replace=True):
                differences = cell.loc[cell["seed"] == sampled_seed, "difference"].to_numpy()
                seed_means.append(float(rng.choice(differences, len(differences), replace=True).mean()))
            samples[replicate] = np.mean(seed_means)
        point = float(cell.groupby("seed")["difference"].mean().mean())
        seed_effects = cell.groupby("seed")["difference"].mean().to_numpy(dtype=np.float64)
        p_value = _exact_sign_flip_p(seed_effects, "two-sided")
        rows.append({
            "system": system, "regime": regime,
            "difference_proposed_minus_baseline": point,
            "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)),
            "p_raw": p_value,
        })
    output = pd.DataFrame(rows)
    output["p_holm"] = holm_adjust(output["p_raw"].to_numpy())
    return output
