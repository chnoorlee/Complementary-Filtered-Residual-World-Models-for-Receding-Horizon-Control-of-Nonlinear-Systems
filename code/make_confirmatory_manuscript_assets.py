"""Create manuscript-only figures/macros from locked confirmatory analysis."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from rgpwm.analyze import load_run_directory
from rgpwm.confirm_analyze import (
    _validate_payload_data_pairing,
    analyze_confirmatory,
)
from rgpwm.confirm_analyze_amended import (
    AMENDMENT_PATH,
    amendment_code_sha256,
    attach_parameter_counts,
    validate_analysis_amendment,
)
from rgpwm.confirm_protocol import (
    CONFIRM_METHODS,
    CONFIRM_REGIMES,
    CONFIRM_SEEDS,
    analysis_code_sha256,
)
from rgpwm.confirm_run import (
    CONTRACT_PATH,
    _sha256,
    _validate_contract,
    _validate_existing_output,
)
from rgpwm.dynamics import SYSTEMS


ANALYSIS = Path("results/confirmatory_analysis")
FIGURES = Path("paper/mypaper/figures")
MACROS = Path("paper/results_macros.tex")
TABLES = Path("paper")
MAIN_TABLE = ANALYSIS / "table_confirmatory.tex"
ASSET_MANIFEST = Path("paper/manuscript_assets_manifest.json")
ASSET_GENERATOR = Path("code/make_confirmatory_manuscript_assets.py")
RUN_MANIFEST = Path("results/confirmatory_run_manifest.json")
MAIN_METHODS = ("rssm", "simnorm", "transformer", "pgnn", "pgdmm", "cfrwm")
GAINS = (
    ("cfr_gain0", 0.0),
    ("cfr_gain025", 0.25),
    ("cfrwm", 0.5),
    ("cfr_gain075", 0.75),
    ("cfr_gain1", 1.0),
)
REGIME_LABELS = {
    "id_clean": "ID clean",
    "id_corrupt": "ID corrupt",
    "ood_parameter_corrupt": "OOD + corrupt",
}
METHOD_LABELS = {
    "rssm": "RSSM-style",
    "simnorm": "SimNorm-style",
    "transformer": "Transformer-style",
    "pgnn": "PGNN-style",
    "pgdmm": "PG-DMM-style",
}
SENSITIVITY_LABELS = {
    "cfr_gain0": "Gain 0",
    "cfr_gain025": "Gain 0.25",
    "cfr_gain075": "Gain 0.75",
    "cfr_gain1": "Gain 1",
    "cfr_1step": "One-step training",
}


def _require_outputs() -> None:
    required = (
        "confirmatory_inference.json",
        "confirmatory_summary.csv",
        "aggregate_cells.csv",
        "trajectory_metrics.csv",
        "cell_contrasts.csv",
    )
    missing = [name for name in required if not (ANALYSIS / name).is_file()]
    if missing:
        raise RuntimeError(f"locked confirmatory analysis is incomplete: {missing}")


def _assert_frame_matches(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    try:
        assert_frame_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=1e-13, atol=1e-13,
        )
    except AssertionError as error:
        raise RuntimeError(f"locked {label} does not match recomputed raw results") from error


def _load_locked_outputs() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Authenticate analysis outputs against the frozen contract and raw payloads."""
    _require_outputs()
    if not CONTRACT_PATH.is_file() or not RUN_MANIFEST.is_file():
        raise RuntimeError("frozen contract or completed-run manifest is missing")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    amendment = validate_analysis_amendment()
    tuning = _validate_contract(contract)
    contract_hash = _sha256(CONTRACT_PATH)
    provenance: dict[str, object] = {
        "confirmatory_contract_sha256": contract_hash,
        "tuning_file_sha256": contract["training"]["tuning_file_sha256"],
        "analysis_code_sha256": contract["hashes"]["analysis_code_sha256"],
    }
    manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    expected_manifest = {
        "contract_sha256": contract_hash,
        "training_code_sha256": contract["hashes"]["training_code_sha256"],
        "analysis_code_sha256": contract["hashes"]["analysis_code_sha256"],
        "tuning_file_sha256": contract["training"]["tuning_file_sha256"],
        "conditions": contract["data_design"]["expected_runs"],
        "complete": True,
    }
    mismatched_manifest = {
        key: {"expected": value, "found": manifest.get(key)}
        for key, value in expected_manifest.items() if manifest.get(key) != value
    }
    if mismatched_manifest:
        raise RuntimeError(f"completed-run manifest mismatch: {mismatched_manifest}")

    raw_directory = Path(contract["training"]["output_directory"])
    payload_map = _validate_existing_output(raw_directory, contract, tuning, provenance)
    if len(payload_map) != int(contract["data_design"]["expected_runs"]):
        raise RuntimeError("raw confirmatory payload grid is incomplete")
    raw_aggregate, raw_trajectories, payloads = load_run_directory(raw_directory)
    reloaded_payload_map = {
        (str(payload["method"]), int(payload["seed"]), str(payload["system"])): payload
        for payload in payloads
    }
    if reloaded_payload_map != payload_map:
        raise RuntimeError("validated and manuscript-loaded payloads differ")
    unique_data_hashes = _validate_payload_data_pairing(payloads)
    raw_aggregate = attach_parameter_counts(raw_aggregate, payloads)

    inference = json.loads((ANALYSIS / "confirmatory_inference.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(ANALYSIS / "confirmatory_summary.csv")
    aggregate = pd.read_csv(ANALYSIS / "aggregate_cells.csv")
    trajectories = pd.read_csv(ANALYSIS / "trajectory_metrics.csv")
    cell = pd.read_csv(ANALYSIS / "cell_contrasts.csv")
    _assert_frame_matches(aggregate, raw_aggregate, "aggregate grid")
    _assert_frame_matches(trajectories, raw_trajectories, "trajectory grid")

    expected_inference, expected_summary, expected_cell = analyze_confirmatory(
        raw_aggregate,
        raw_trajectories,
        int(contract["statistics"]["bootstrap_replicates"]),
    )
    expected_inference.update({
        "analysis_code_sha256": analysis_code_sha256(),
        "training_code_sha256": contract["hashes"]["training_code_sha256"],
        "evaluation_code_sha256": contract["hashes"]["training_code_sha256"],
        "confirmatory_contract_sha256": contract_hash,
        "tuning_file_sha256": contract["training"]["tuning_file_sha256"],
        "unique_data_sha256_count": unique_data_hashes,
        "analysis_amendment_version": amendment["amendment_version"],
        "analysis_amendment_contract_sha256": _sha256(AMENDMENT_PATH),
        "analysis_amendment_code_sha256": amendment_code_sha256(),
        "analysis_amendment_inference_change": "none",
    })
    if inference != expected_inference:
        raise RuntimeError("locked inference JSON does not match deterministic recomputation")
    _assert_frame_matches(summary, expected_summary, "summary")
    _assert_frame_matches(cell, expected_cell, "cell contrasts")

    expected_methods = list(CONFIRM_METHODS)
    if (
        inference.get("methods") != expected_methods
        or inference.get("seeds") != list(CONFIRM_SEEDS)
        or inference.get("systems") != list(SYSTEMS)
        or inference.get("regimes") != list(CONFIRM_REGIMES)
        or len(trajectories) != int(contract["data_design"]["expected_trajectory_rows"])
        or len(aggregate) != int(contract["data_design"]["expected_aggregate_rows"])
        or len(cell) != len(SYSTEMS) * len(CONFIRM_REGIMES)
    ):
        raise RuntimeError("authenticated analysis does not match the exact frozen grid")
    # Downstream presentation assets are rendered from the deterministic
    # recomputation, not from the serialized copies that were merely checked.
    return expected_inference, expected_summary, raw_trajectories, expected_cell


def _write_asset_manifest() -> None:
    """Bind every result-dependent manuscript asset to its authenticated sources."""
    assets = (
        MAIN_TABLE,
        FIGURES / "confirmatory_overview.png",
        FIGURES / "cell_difference_heatmap.png",
        MACROS,
        TABLES / "table_comparator_contrasts.tex",
        TABLES / "table_sensitivity_contrasts.tex",
        TABLES / "table_cell_contrasts.tex",
        TABLES / "table_secondary_metrics.tex",
        TABLES / "table_seed_effects.tex",
    )
    sources = (
        CONTRACT_PATH,
        AMENDMENT_PATH,
        Path("code/rgpwm/confirm_analyze_amended.py"),
        RUN_MANIFEST,
        ANALYSIS / "confirmatory_inference.json",
        ANALYSIS / "confirmatory_summary.csv",
        ANALYSIS / "aggregate_cells.csv",
        ANALYSIS / "trajectory_metrics.csv",
        ANALYSIS / "cell_contrasts.csv",
    )
    missing = [
        str(path) for path in (*assets, *sources, ASSET_GENERATOR)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"cannot complete manuscript asset manifest: {missing}")
    payload = {
        "complete": True,
        "contract_sha256": _sha256(CONTRACT_PATH),
        "analysis_code_sha256": analysis_code_sha256(),
        "analysis_amendment_contract_sha256": _sha256(AMENDMENT_PATH),
        "analysis_amendment_code_sha256": amendment_code_sha256(),
        "generator": {
            ASSET_GENERATOR.as_posix(): _sha256(ASSET_GENERATOR),
        },
        "assets": {path.as_posix(): _sha256(path) for path in assets},
        "authenticated_sources": {path.as_posix(): _sha256(path) for path in sources},
    }
    ASSET_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = ASSET_MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(ASSET_MANIFEST)


def _tex_escape(value: object) -> str:
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return re.sub(
        r"[&%$#_{}~^\\]", lambda match: replacements[match.group(0)], str(value),
    )


def _confirmatory_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Mean parameters & NRMSE-1 & NRMSE-10 & NRMSE-25 \\",
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
        lines.append(
            f"{_tex_escape(METHOD_LABELS.get(row['method'], row['label']))} & "
            f"{row['parameter_count_mean']:.0f} & "
            + " & ".join(values) + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _overview(
    inference: dict[str, object],
    summary: pd.DataFrame,
    trajectories: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)
    palette = "#3A78A1"

    main = summary.set_index("method").loc[list(MAIN_METHODS)]
    axes[0, 0].bar(
        np.arange(len(main)), main["nrmse_h25_mean"],
        yerr=main["nrmse_h25_sd"], capsize=3, color=palette, alpha=0.86,
    )
    main_labels = [METHOD_LABELS.get(method, label) for method, label in zip(
        main.index, main["label"], strict=True,
    )]
    axes[0, 0].set_xticks(np.arange(len(main)), main_labels, rotation=28, ha="right")
    axes[0, 0].set_ylabel("NRMSE-25 (mean $\\pm$ seed SD)")
    axes[0, 0].set_title("A  Main transition comparison", loc="left", weight="bold")

    cells = trajectories.groupby(["method", "seed", "regime"])["nrmse_h25"].mean()
    regime_rows = []
    for regime in REGIME_LABELS:
        baseline = cells.xs(("pgnn", regime), level=("method", "regime"))
        proposed = cells.xs(("cfrwm", regime), level=("method", "regime"))
        improvement = 100.0 * (baseline - proposed) / baseline
        regime_rows.append((REGIME_LABELS[regime], improvement.mean(), improvement.std(ddof=1)))
    labels, means, standard_deviations = zip(*regime_rows, strict=True)
    axes[0, 1].bar(
        np.arange(3), means, yerr=standard_deviations, capsize=4,
        color=["#88B04B", "#E69F00", "#C44E52"], alpha=0.88,
    )
    axes[0, 1].axhline(5.0, color="#333333", linestyle="--", linewidth=1.1, label="5% margin")
    axes[0, 1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[0, 1].set_xticks(np.arange(3), labels, rotation=20, ha="right")
    axes[0, 1].set_ylabel("CFR-WM improvement over PGNN-style (%)")
    axes[0, 1].set_title("B  Regime-specific paired effects", loc="left", weight="bold")
    axes[0, 1].legend(frameon=False)

    gain_summary = summary.set_index("method")
    gain_x = np.array([gain for _, gain in GAINS])
    gain_y = np.array([gain_summary.loc[method, "nrmse_h25_mean"] for method, _ in GAINS])
    gain_sd = np.array([gain_summary.loc[method, "nrmse_h25_sd"] for method, _ in GAINS])
    axes[1, 0].errorbar(gain_x, gain_y, yerr=gain_sd, marker="o", capsize=3, color="#3A8754")
    axes[1, 0].axvline(0.5, color="#333333", linestyle="--", linewidth=1.0)
    axes[1, 0].set_xlabel("Fixed observed-channel gain")
    axes[1, 0].set_ylabel("NRMSE-25 (mean $\\pm$ seed SD)")
    axes[1, 0].set_title("C  Prespecified gain sensitivity", loc="left", weight="bold")

    contrasts = inference["comparator_contrasts"]
    labels = [METHOD_LABELS[item["baseline"]] for item in contrasts]
    points = np.array([item["relative_improvement_percent"] for item in contrasts])
    intervals = np.array([item["relative_improvement_ci_95"] for item in contrasts])
    positions = np.arange(len(labels))
    axes[1, 1].errorbar(
        points, positions,
        xerr=np.maximum(
            0.0,
            np.vstack((points - intervals[:, 0], intervals[:, 1] - points)),
        ),
        fmt="o", capsize=3, color="#76519C",
    )
    axes[1, 1].axvline(0.0, color="#777777", linewidth=0.8)
    axes[1, 1].axvline(5.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[1, 1].set_yticks(positions, labels)
    axes[1, 1].set_xlabel("CFR-WM relative improvement (%)")
    axes[1, 1].set_title("D  Fixed-cell paired bootstrap intervals", loc="left", weight="bold")

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "confirmatory_overview.png", dpi=300, facecolor="white")
    plt.close(figure)


def _heatmap(cell: pd.DataFrame) -> None:
    systems = list(dict.fromkeys(cell["system"]))
    regimes = list(REGIME_LABELS)
    pivot = cell.pivot(
        index="system", columns="regime",
        values="relative_change_cfr_minus_pgnn_percent",
    ).loc[systems, regimes]
    values = pivot.to_numpy()
    bound = max(5.0, float(np.nanmax(np.abs(values))))
    figure, axis = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    image = axis.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:+.1f}%", ha="center", va="center", fontsize=9)
    axis.set_xticks(np.arange(len(regimes)), [REGIME_LABELS[value] for value in regimes])
    axis.set_yticks(np.arange(len(systems)), [value.replace("_", " ").title() for value in systems])
    axis.set_title("Descriptive CFR-WM change relative to PGNN-style (negative is better)")
    figure.colorbar(image, ax=axis, label="CFR-WM minus PGNN-style (%)")
    figure.savefig(FIGURES / "cell_difference_heatmap.png", dpi=300, facecolor="white")
    plt.close(figure)


def _p_text(value: float) -> str:
    return f"{value:.4f}" if value >= 0.0001 else "$<10^{-4}$"


def _contrast_table(rows: list[dict[str, object]], labels: dict[str, str], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Reference & Relative effect [95\% CI], \% & Exact $p$ & Holm $p$ \\",
        r"\midrule",
    ]
    for row in rows:
        low, high = row["relative_improvement_ci_95"]
        effect = f"{row['relative_improvement_percent']:+.2f} [{low:+.2f}, {high:+.2f}]"
        lines.append(
            f"{_tex_escape(labels[row['baseline']])} & {effect} & "
            f"{_p_text(row['p_two_sided'])} & {_p_text(row['p_holm'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cell_table(cell: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"System & Regime & CFR--PGNN-style [95\% CI] & Relative change, \% \\",
        r"\midrule",
    ]
    for _, row in cell.iterrows():
        system = _tex_escape(str(row["system"]).replace("_", " ").title())
        regime = _tex_escape(REGIME_LABELS[str(row["regime"])])
        difference = (
            f"{row['difference_proposed_minus_baseline']:+.4f} "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
        )
        relative = f"{row['relative_change_cfr_minus_pgnn_percent']:+.2f}"
        lines.append(f"{system} & {regime} & {difference} & {relative} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _secondary_table(summary: pd.DataFrame, path: Path) -> None:
    selected = summary.set_index("method").loc[list(MAIN_METHODS)]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Parameters & Gaussian NLL & Calibration error & Catastrophic rate \\",
        r"\midrule",
    ]
    for method, row in selected.iterrows():
        lines.append(
            f"{_tex_escape(METHOD_LABELS.get(method, row['label']))} & "
            f"{row['parameter_count_mean']:.0f} & "
            f"{row['gaussian_nll_mean']:.3f} & {row['calibration_error_mean']:.3f} & "
            f"{100.0 * row['catastrophic_mean']:.2f}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seed_effect_table(inference: dict[str, object], path: Path) -> None:
    effects = inference["primary_h1"]["seed_level_relative_improvements_percent"]
    if len(effects) != len(CONFIRM_SEEDS):
        raise RuntimeError("primary inference lacks one relative effect per confirmatory seed")
    lines = [
        r"\begin{tabular}{rr}",
        r"\toprule",
        r"Seed & CFR-WM improvement over PGNN-style, \% \\",
        r"\midrule",
    ]
    lines.extend(
        f"{seed} & {float(effect):+.2f} \\\\"
        for seed, effect in zip(CONFIRM_SEEDS, effects, strict=True)
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_macros(
    inference: dict[str, object],
    summary: pd.DataFrame,
    cell: pd.DataFrame,
) -> None:
    primary = inference["primary_h1"]
    robustness = inference["robustness_h2"]
    corrupt = robustness["corrupt_and_shifted"]
    clean = robustness["clean"]
    method_summary = summary.set_index("method")
    decision = "supported" if primary["h1_supported"] else "not supported"
    h2_decision = "supported" if robustness["supported"] else "not supported"
    interpretation = (
        "supports the prespecified practical-gain claim"
        if primary["h1_supported"]
        else "does not support the prespecified practical-gain claim"
    )
    h2_interpretation = (
        "supports the gated robustness claim"
        if robustness["supported"]
        else "does not support the gated robustness claim"
    )
    best_row = summary.loc[summary["nrmse_h25_mean"].idxmin()]
    gain_methods = [name for name, _ in GAINS]
    best_gain_method = min(gain_methods, key=lambda name: method_summary.loc[name, "nrmse_h25_mean"])
    gain_lookup = dict(GAINS)
    one_step = next(
        item for item in inference["ablation_contrasts"]
        if item["baseline"] == "cfr_1step"
    )
    indexed_cells = cell.set_index(["system", "regime"])

    def relative_change(system: str, regime: str) -> float:
        try:
            return float(indexed_cells.loc[(system, regime), "relative_change_cfr_minus_pgnn_percent"])
        except KeyError as error:
            raise RuntimeError(f"locked cell contrast is missing: {system}/{regime}") from error

    id_corrupt_changes = cell.loc[
        cell["regime"] == "id_corrupt", "relative_change_cfr_minus_pgnn_percent",
    ].astype(float)
    if len(id_corrupt_changes) != len(SYSTEMS):
        raise RuntimeError("locked cell contrasts do not contain one ID-corrupt row per system")
    id_corrupt_favored = int((id_corrupt_changes < 0.0).sum())
    lines = [
        "% Generated only from locked results/confirmatory_analysis artifacts.",
        r"\newcommand{\PrimaryBaseline}{PGNN-style}",
        f"\\newcommand{{\\PrimaryDifference}}{{{primary['difference_proposed_minus_baseline']:.4f}}}",
        f"\\newcommand{{\\PrimaryCILow}}{{{primary['ci_95'][0]:.4f}}}",
        f"\\newcommand{{\\PrimaryCIHigh}}{{{primary['ci_95'][1]:.4f}}}",
        f"\\newcommand{{\\PrimaryImprovement}}{{{primary['relative_improvement_percent']:.2f}}}",
        f"\\newcommand{{\\PrimaryRelativeCILow}}{{{primary['relative_improvement_ci_95'][0]:.2f}}}",
        f"\\newcommand{{\\PrimaryRelativeCIHigh}}{{{primary['relative_improvement_ci_95'][1]:.2f}}}",
        f"\\newcommand{{\\PrimaryPValue}}{{{primary['p_one_sided_relative_margin']:.5f}}}",
        f"\\newcommand{{\\PrimaryDecision}}{{{decision}}}",
        f"\\newcommand{{\\PrimaryInterpretation}}{{{interpretation}}}",
        f"\\newcommand{{\\RobustnessDecision}}{{{h2_decision}}}",
        f"\\newcommand{{\\RobustnessInterpretation}}{{{h2_interpretation}}}",
        f"\\newcommand{{\\CorruptImprovement}}{{{corrupt['relative_improvement_percent']:.2f}}}",
        f"\\newcommand{{\\CorruptRelativeCILow}}{{{corrupt['relative_improvement_ci_95'][0]:.2f}}}",
        f"\\newcommand{{\\CorruptRelativeCIHigh}}{{{corrupt['relative_improvement_ci_95'][1]:.2f}}}",
        f"\\newcommand{{\\CorruptPValue}}{{{corrupt['p_one_sided_relative_margin']:.5f}}}",
        f"\\newcommand{{\\CleanImprovement}}{{{clean['relative_improvement_percent']:.2f}}}",
        f"\\newcommand{{\\CleanRelativeCILow}}{{{clean['relative_improvement_ci_95'][0]:.2f}}}",
        f"\\newcommand{{\\CleanRelativeCIHigh}}{{{clean['relative_improvement_ci_95'][1]:.2f}}}",
        f"\\newcommand{{\\CleanPValue}}{{{clean['p_one_sided_relative_margin']:.5f}}}",
        f"\\newcommand{{\\CFRNRMSE}}{{{method_summary.loc['cfrwm', 'nrmse_h25_mean']:.4f}}}",
        f"\\newcommand{{\\PGNNNRMSE}}{{{method_summary.loc['pgnn', 'nrmse_h25_mean']:.4f}}}",
        f"\\newcommand{{\\BestMethod}}{{{_tex_escape(best_row['label'])}}}",
        f"\\newcommand{{\\BestNRMSE}}{{{best_row['nrmse_h25_mean']:.4f}}}",
        f"\\newcommand{{\\BestGain}}{{{gain_lookup[best_gain_method]:.2f}}}",
        f"\\newcommand{{\\OneStepImprovement}}{{{one_step['relative_improvement_percent']:.2f}}}",
        f"\\newcommand{{\\OneStepRelativeCILow}}{{{one_step['relative_improvement_ci_95'][0]:.2f}}}",
        f"\\newcommand{{\\OneStepRelativeCIHigh}}{{{one_step['relative_improvement_ci_95'][1]:.2f}}}",
        f"\\newcommand{{\\OneStepHolmP}}{{{one_step['p_holm']:.5f}}}",
        f"\\newcommand{{\\CellsWorseTen}}{{{inference['cells_worse_than_pgnn_by_more_than_10_percent']}}}",
        f"\\newcommand{{\\CartpoleCleanChange}}{{{relative_change('cartpole', 'id_clean'):+.1f}}}",
        f"\\newcommand{{\\VanDerPolCleanChange}}{{{relative_change('controlled_van_der_pol', 'id_clean'):+.1f}}}",
        f"\\newcommand{{\\DuffingCleanChange}}{{{relative_change('forced_duffing', 'id_clean'):+.1f}}}",
        f"\\newcommand{{\\IDCorruptFavoredCount}}{{{id_corrupt_favored}}}",
        f"\\newcommand{{\\IDCorruptCellCount}}{{{len(id_corrupt_changes)}}}",
        f"\\newcommand{{\\IDCorruptLeastChange}}{{{id_corrupt_changes.max():+.1f}}}",
        f"\\newcommand{{\\IDCorruptMostChange}}{{{id_corrupt_changes.min():+.1f}}}",
        f"\\newcommand{{\\AllMetricsFinite}}{{{'yes' if inference['all_metrics_finite'] else 'no'}}}",
        f"\\newcommand{{\\UniqueDataHashes}}{{{inference['unique_data_sha256_count']}}}",
        r"\newcommand{\ConfirmatoryRuns}{440}",
        r"\newcommand{\ConfirmatoryTrajectories}{126{,}720}",
    ]
    MACROS.parent.mkdir(parents=True, exist_ok=True)
    MACROS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    inference, summary, trajectories, cell = _load_locked_outputs()
    MAIN_TABLE.parent.mkdir(parents=True, exist_ok=True)
    _confirmatory_table(summary, MAIN_TABLE)
    _overview(inference, summary, trajectories)
    _heatmap(cell)
    _latex_macros(inference, summary, cell)
    TABLES.mkdir(parents=True, exist_ok=True)
    _contrast_table(
        inference["comparator_contrasts"], METHOD_LABELS,
        TABLES / "table_comparator_contrasts.tex",
    )
    _contrast_table(
        inference["ablation_contrasts"], SENSITIVITY_LABELS,
        TABLES / "table_sensitivity_contrasts.tex",
    )
    _cell_table(cell, TABLES / "table_cell_contrasts.tex")
    _secondary_table(summary, TABLES / "table_secondary_metrics.tex")
    _seed_effect_table(inference, TABLES / "table_seed_effects.tex")
    _write_asset_manifest()
    print(MAIN_TABLE)
    print(FIGURES / "confirmatory_overview.png")
    print(FIGURES / "cell_difference_heatmap.png")
    print(MACROS)
    print(TABLES / "table_comparator_contrasts.tex")
    print(TABLES / "table_sensitivity_contrasts.tex")
    print(TABLES / "table_cell_contrasts.tex")
    print(TABLES / "table_secondary_metrics.tex")
    print(TABLES / "table_seed_effects.tex")
    print(ASSET_MANIFEST)


if __name__ == "__main__":
    main()
