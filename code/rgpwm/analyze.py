"""Generate machine-readable inference, manuscript tables, and figures."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import EVALUATION_METRIC_VERSION
from .statistics import paired_cell_bootstrap, paired_hierarchical_bootstrap


METHOD_LABELS = {
    "rssm": "RSSM",
    "simnorm": "SimNorm-WM",
    "transformer": "TransWM",
    "pgnn": "PGNN",
    "pgdmm": "PgDMM",
    "rgpwm": "RGP-WM",
    "rgp_no_rel": "RGP-no-rel",
    "rgp_fixed": "RGP-fixed",
    "rgp_no_phys": "RGP-no-phys",
    "rgp_1step": "RGP-1step",
}
SYSTEM_LABELS = {
    "damped_oscillator": "Oscillator",
    "forced_duffing": "Duffing",
    "controlled_van_der_pol": "Van der Pol",
    "torque_pendulum": "Pendulum",
    "cartpole": "Cart-pole",
}
REGIME_LABELS = {
    "id_clean": "ID clean",
    "id_corrupt": "ID corrupt",
    "ood_parameter_corrupt": "OOD + corrupt",
}


def require_evaluation_metric_version(payloads: list[dict[str, object]]) -> None:
    versions = {payload.get("evaluation_metric_version") for payload in payloads}
    hashes = {payload.get("evaluation_code_sha256") for payload in payloads}
    if versions != {EVALUATION_METRIC_VERSION}:
        raise RuntimeError(
            f"evaluation metric version mismatch: expected {EVALUATION_METRIC_VERSION}, found {versions}"
        )
    if None in hashes or "" in hashes or len(hashes) != 1:
        raise RuntimeError(f"evaluation metric code hash mismatch: found {hashes}")
    run_hashes = {payload.get("run_code_sha256") for payload in payloads}
    if run_hashes != {None} and (None in run_hashes or len(run_hashes) != 1):
        raise RuntimeError(f"training run code hash mismatch: found {run_hashes}")


def load_run_directory(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    aggregate_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    payloads: list[dict[str, object]] = []
    for path in sorted(directory.glob("*/seed_*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads.append(payload)
        identity = {
            "system": payload["system"], "seed": payload["seed"],
            "method": payload["method"],
        }
        for regime, evaluation in payload["evaluations"].items():
            aggregate_rows.append({**identity, "regime": regime, **evaluation["aggregate"]})
            trajectory_rows.extend(
                {**identity, "regime": regime, **row}
                for row in evaluation["trajectories"]
            )
    require_evaluation_metric_version(payloads)
    return pd.DataFrame(aggregate_rows), pd.DataFrame(trajectory_rows), payloads


def _seed_balanced_summary(aggregate: pd.DataFrame, trajectories: pd.DataFrame) -> pd.DataFrame:
    trajectory_metrics = [
        "nrmse_h1", "nrmse_h5", "nrmse_h10", "nrmse_h25", "mae_h25",
        "gaussian_nll", "calibration_error", "catastrophic",
    ]
    aggregate_metrics = ["coverage_90", "interval_width_90", "rollout_latency_ms_per_trajectory"]
    trajectory_cells = trajectories.groupby(["method", "seed", "system", "regime"])[trajectory_metrics].mean().reset_index()
    seed_trajectory = trajectory_cells.groupby(["method", "seed"])[trajectory_metrics].mean()
    seed_aggregate = aggregate.groupby(["method", "seed"])[aggregate_metrics].mean()
    seed_means = seed_trajectory.join(seed_aggregate).reset_index()
    metrics = trajectory_metrics + aggregate_metrics
    mean = seed_means.groupby("method")[metrics].mean()
    std = seed_means.groupby("method")[metrics].std(ddof=1)
    rows = []
    for method in mean.index:
        row: dict[str, object] = {"method": method, "label": METHOD_LABELS[method]}
        for metric in metrics:
            row[f"{metric}_mean"] = float(mean.loc[method, metric])
            row[f"{metric}_sd"] = float(std.loc[method, metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("nrmse_h25_mean")


def _write_result_macros(primary: dict[str, object], path: Path) -> None:
    decision = "supported" if primary["h1_supported"] else "not supported"
    baseline = METHOD_LABELS[str(primary["baseline"])]
    low, high = primary["ci_95"]
    lines = [
        "% Generated from results/analysis/primary_inference.json.",
        f"\\newcommand{{\\PrimaryBaseline}}{{{baseline}}}",
        f"\\newcommand{{\\PrimaryDifference}}{{{primary['difference_proposed_minus_baseline']:.3f}}}",
        f"\\newcommand{{\\PrimaryCILow}}{{{low:.3f}}}",
        f"\\newcommand{{\\PrimaryCIHigh}}{{{high:.3f}}}",
        f"\\newcommand{{\\PrimaryImprovement}}{{{primary['relative_improvement_percent']:.1f}}}",
        f"\\newcommand{{\\PrimaryPValue}}{{{primary['p_two_sided']:.4f}}}",
        f"\\newcommand{{\\PrimaryDecision}}{{{decision}}}",
        f"\\newcommand{{\\GateCorrelation}}{{{primary['mean_gate_oracle_spearman']:.3f}}}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex_summary(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Method & NRMSE-1 & NRMSE-5 & NRMSE-10 & NRMSE-25 & Cal. error \\",
        r"\midrule",
    ]
    best = summary["nrmse_h25_mean"].min()
    for _, row in summary.iterrows():
        values = []
        for metric in ("nrmse_h1", "nrmse_h5", "nrmse_h10", "nrmse_h25"):
            text = f"{row[f'{metric}_mean']:.3f} $\\pm$ {row[f'{metric}_sd']:.3f}"
            if metric == "nrmse_h25" and np.isclose(row[f"{metric}_mean"], best):
                text = r"\textbf{" + text + "}"
            values.append(text)
        lines.append(f"{row['label']} & " + " & ".join(values) + f" & {row['calibration_error_mean']:.3f} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_heatmap(cell: pd.DataFrame, output: Path) -> None:
    pivot = cell.pivot(index="system", columns="regime", values="difference_proposed_minus_baseline")
    order_systems = list(SYSTEM_LABELS)
    order_regimes = list(REGIME_LABELS)
    values = pivot.reindex(index=order_systems, columns=order_regimes).to_numpy()
    figure, axis = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    limit = max(abs(np.nanmin(values)), abs(np.nanmax(values)), 1e-6)
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:+.3f}", ha="center", va="center", fontsize=8)
    axis.set_xticks(range(len(order_regimes)), [REGIME_LABELS[value] for value in order_regimes])
    axis.set_yticks(range(len(order_systems)), [SYSTEM_LABELS[value] for value in order_systems])
    axis.set_title("RGP-WM minus best comparator: trajectory NRMSE-25")
    figure.colorbar(image, ax=axis, label="Difference (lower favors RGP-WM)")
    figure.savefig(output, dpi=300)
    plt.close(figure)


def _plot_accuracy_calibration(aggregate: pd.DataFrame, output: Path) -> None:
    seed = aggregate.groupby(["method", "seed"])[["nrmse_h25", "calibration_error"]].mean().reset_index()
    figure, axis = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)
    for method, group in seed.groupby("method"):
        axis.errorbar(
            group["nrmse_h25"].mean(), group["calibration_error"].mean(),
            xerr=group["nrmse_h25"].std(ddof=1), yerr=group["calibration_error"].std(ddof=1),
            marker="o", capsize=3, label=METHOD_LABELS[method],
        )
    axis.set_xlabel("NRMSE-25 (lower is better)")
    axis.set_ylabel("Calibration error (lower is better)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def _plot_gate(aggregate: pd.DataFrame, output: Path) -> None:
    rgp = aggregate[aggregate["method"] == "rgpwm"]
    if "mean_data_gate" not in rgp or rgp.empty:
        return
    table = rgp.groupby(["system", "regime"])["mean_data_gate"].mean().unstack()
    table = table.reindex(index=SYSTEM_LABELS, columns=REGIME_LABELS)
    axis = table.rename(index=SYSTEM_LABELS, columns=REGIME_LABELS).plot.bar(
        figsize=(7.0, 4.0), ylim=(0.0, 1.0), width=0.78,
    )
    axis.set_ylabel("Mean learned data-expert weight")
    axis.set_xlabel("")
    axis.legend(title="", fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(output, dpi=300)
    plt.close(axis.figure)


def analyze_primary(input_dir: Path, output_dir: Path, replicates: int) -> dict[str, object]:
    aggregate, trajectories, payloads = load_run_directory(input_dir)
    expected = 5 * 3 * 6
    if len(payloads) != expected:
        raise RuntimeError(f"expected {expected} primary runs, found {len(payloads)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _seed_balanced_summary(aggregate, trajectories)
    comparator_rows = summary[summary["method"] != "rgpwm"]
    best_comparator = str(comparator_rows.iloc[0]["method"])
    primary = paired_hierarchical_bootstrap(
        trajectories, best_comparator, "rgpwm", "nrmse_h25", replicates,
    )
    cell = paired_cell_bootstrap(
        trajectories, best_comparator, "rgpwm", "nrmse_h25", replicates,
    )
    primary["h1_threshold_percent"] = 10.0
    primary["h1_supported"] = bool(
        primary["relative_improvement_percent"] >= 10.0 and primary["ci_95"][1] < 0.0
    )
    gate = aggregate[aggregate["method"] == "rgpwm"]
    primary["mean_gate_oracle_spearman"] = float(gate["gate_oracle_spearman"].mean())
    corrupt = aggregate[aggregate["regime"].isin(["id_corrupt", "ood_parameter_corrupt"])]
    corrupt_means = corrupt.groupby("method")[["calibration_error", "gaussian_nll", "catastrophic_rollout_rate"]].mean()
    primary["h2_corrupt_calibration_better"] = bool(
        corrupt_means.loc["rgpwm", "calibration_error"] < corrupt_means.loc[best_comparator, "calibration_error"]
        and corrupt_means.loc["rgpwm", "gaussian_nll"] < corrupt_means.loc[best_comparator, "gaussian_nll"]
    )
    one_step = trajectories.groupby("method")["nrmse_h1"].mean()
    primary["h2_one_step_relative_change_percent"] = float(
        100.0 * (one_step["rgpwm"] - one_step[best_comparator]) / one_step[best_comparator]
    )
    primary["h2_one_step_noninferior"] = bool(primary["h2_one_step_relative_change_percent"] <= 5.0)
    primary["h2_supported"] = bool(primary["h2_corrupt_calibration_better"] and primary["h2_one_step_noninferior"])
    primary["h3_positive_gate_alignment_point"] = bool(primary["mean_gate_oracle_spearman"] > 0.0)
    primary["h3_catastrophic_rate_lower"] = bool(
        corrupt_means.loc["rgpwm", "catastrophic_rollout_rate"]
        < corrupt_means.loc[best_comparator, "catastrophic_rollout_rate"]
    )
    primary["h3_preliminary_without_ablation_comparators"] = bool(
        primary["h3_positive_gate_alignment_point"] and primary["h3_catastrophic_rate_lower"]
    )
    summary.to_csv(output_dir / "primary_summary.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_cells.csv", index=False)
    trajectories.to_csv(output_dir / "trajectory_metrics.csv", index=False)
    cell.to_csv(output_dir / "cell_contrasts.csv", index=False)
    (output_dir / "primary_inference.json").write_text(json.dumps(primary, indent=2, allow_nan=False), encoding="utf-8")
    _write_result_macros(primary, Path("paper/results_macros.tex"))
    _write_latex_summary(summary, output_dir / "table_primary.tex")
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    _plot_heatmap(cell, figure_dir / "cell_difference_heatmap.png")
    _plot_accuracy_calibration(aggregate, figure_dir / "accuracy_calibration.png")
    _plot_gate(aggregate, figure_dir / "gate_by_regime.png")
    manuscript_figures = Path("paper/mypaper/figures")
    manuscript_figures.mkdir(parents=True, exist_ok=True)
    for figure_path in figure_dir.glob("*.png"):
        shutil.copy2(figure_path, manuscript_figures / figure_path.name)
    return primary


def analyze_ablations(primary_dir: Path, ablation_dir: Path, output_dir: Path, replicates: int) -> dict[str, object]:
    primary_aggregate, primary_trajectories, primary_payloads = load_run_directory(primary_dir)
    ablation_aggregate, ablation_trajectories, ablation_payloads = load_run_directory(ablation_dir)
    if len(primary_payloads) != 90 or len(ablation_payloads) != 60:
        raise RuntimeError("primary and ablation grids must be complete")
    combined = pd.concat([
        primary_trajectories[primary_trajectories["method"] == "rgpwm"],
        ablation_trajectories,
    ], ignore_index=True)
    results: dict[str, object] = {}
    for ablation in ("rgp_fixed", "rgp_no_phys", "rgp_1step"):
        result = paired_hierarchical_bootstrap(combined, ablation, "rgpwm", "nrmse_h25", replicates)
        cell = paired_cell_bootstrap(combined, ablation, "rgpwm", "nrmse_h25", replicates)
        worsened_systems = int(
            (cell.groupby("system")["difference_proposed_minus_baseline"].mean() < 0.0).sum()
        )
        result["systems_where_full_rgp_is_better"] = worsened_systems
        result["h4_component_supported"] = worsened_systems >= 4
        results[ablation] = result
    gate_frame = pd.concat([
        primary_aggregate[primary_aggregate["method"] == "rgpwm"],
        ablation_aggregate[ablation_aggregate["method"] == "rgp_no_rel"],
    ], ignore_index=True)
    gate_wide = gate_frame.pivot(
        index=["seed", "system", "regime"], columns="method", values="gate_oracle_spearman",
    ).dropna().reset_index()
    gate_wide["full_minus_no_rel"] = gate_wide["rgpwm"] - gate_wide["rgp_no_rel"]
    gate_system = gate_wide.groupby("system")["full_minus_no_rel"].mean()
    results["rgp_no_rel"] = {
        "target_metric": "gate_oracle_spearman",
        "difference_full_minus_ablation": float(gate_wide["full_minus_no_rel"].mean()),
        "systems_where_full_rgp_is_better": int((gate_system > 0.0).sum()),
        "h4_component_supported": bool((gate_system > 0.0).sum() >= 4),
    }
    results["h4_supported_for_all_components"] = all(
        results[name]["h4_component_supported"]
        for name in ("rgp_no_rel", "rgp_fixed", "rgp_no_phys", "rgp_1step")
    )
    rgp_gate = primary_aggregate.loc[
        primary_aggregate["method"] == "rgpwm", "gate_oracle_spearman"
    ].dropna().to_numpy()
    rng = np.random.default_rng(20260805)
    gate_bootstrap = np.array([
        rng.choice(rgp_gate, len(rgp_gate), replace=True).mean()
        for _ in range(replicates)
    ])
    gate_ci = [float(value) for value in np.quantile(gate_bootstrap, [0.025, 0.975])]
    corrupt_combined = combined[
        combined["regime"].isin(["id_corrupt", "ood_parameter_corrupt"])
    ]
    catastrophe_contrasts = {}
    for ablation in ("rgp_fixed", "rgp_no_phys"):
        catastrophe_contrasts[ablation] = paired_hierarchical_bootstrap(
            corrupt_combined, ablation, "rgpwm", "catastrophic", replicates,
        )
    results["h3"] = {
        "gate_spearman_mean": float(rgp_gate.mean()),
        "gate_spearman_ci_95": gate_ci,
        "catastrophic_contrasts": catastrophe_contrasts,
        "supported": bool(
            gate_ci[0] > 0.0
            and all(
                value["difference_proposed_minus_baseline"] < 0.0
                for value in catastrophe_contrasts.values()
            )
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_inference.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", default="results/primary")
    parser.add_argument("--ablations", default="results/ablations")
    parser.add_argument("--output", default="results/analysis")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--include-ablations", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    primary = analyze_primary(Path(args.primary), output, args.bootstrap_replicates)
    print(json.dumps({"event": "primary_analysis_complete", **primary}))
    if args.include_ablations:
        ablations = analyze_ablations(Path(args.primary), Path(args.ablations), output, args.bootstrap_replicates)
        print(json.dumps({"event": "ablation_analysis_complete", **ablations}))


if __name__ == "__main__":
    main()
