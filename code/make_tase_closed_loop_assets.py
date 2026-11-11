"""Validate the frozen T-ASE closed-loop study and build paper assets."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tase_closed_loop as study


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tase_closed_loop"
PAPER = ROOT / "paper" / "tase"
FIGURES = PAPER / "figures"
CONTROLLER_LABELS = {
    "nominal_last": "Nominal, last state",
    "complementary_nominal": "Filter + nominal",
    "pgnn": "PGNN-style",
    "cfrwm": "CFR-WM",
}
CONDITION_LABELS = {
    "id_clean": "ID clean",
    "id_corrupt": "ID corrupt",
    "ood_parameter_corrupt": "OOD corrupt",
}
SYSTEM_LABELS = {
    "damped_oscillator": "Damped osc.",
    "forced_duffing": "Duffing",
    "controlled_van_der_pol": "Van der Pol",
    "torque_pendulum": "Pendulum",
    "cartpole": "Cart--pole",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inference_matches(recorded: object, recomputed: object) -> bool:
    """Compare an inference receipt while tolerating CSV round-trip noise."""
    if isinstance(recorded, dict) and isinstance(recomputed, dict):
        recorded_by_text = {str(key): value for key, value in recorded.items()}
        recomputed_by_text = {str(key): value for key, value in recomputed.items()}
        return set(recorded_by_text) == set(recomputed_by_text) and all(
            inference_matches(recorded_by_text[key], recomputed_by_text[key])
            for key in recorded_by_text
        )
    if isinstance(recorded, list) and isinstance(recomputed, list):
        return len(recorded) == len(recomputed) and all(
            inference_matches(left, right) for left, right in zip(recorded, recomputed)
        )
    if isinstance(recorded, (int, float)) and isinstance(recomputed, (int, float)):
        return bool(np.isclose(recorded, recomputed, rtol=1e-12, atol=1e-12))
    return recorded == recomputed


def validate_manifest() -> tuple[dict[str, object], pd.DataFrame]:
    manifest_path = RESULTS / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"closed-loop manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("closed-loop study is not marked complete")
    expected_protocol = json.loads(json.dumps(study.asdict(study.Protocol())))
    if manifest.get("protocol") != expected_protocol:
        raise ValueError("closed-loop protocol differs from the frozen code")
    if manifest.get("source_sha256") != study.source_fingerprint():
        raise ValueError("closed-loop source files changed after execution")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {
        "episode_results.csv", "overall_summary.csv", "condition_summary.csv"
    }:
        raise ValueError("closed-loop output inventory is incomplete or unexpected")
    for name, digest in outputs.items():
        path = RESULTS / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"closed-loop output hash mismatch: {name}")

    expected_checkpoints = {
        f"{system}/seed_{seed}/{method}.pt"
        for system in study.SYSTEMS
        for seed in study.CONFIRM_SEEDS
        for method in ("cfrwm", "pgnn")
    }
    checkpoint_hashes = manifest.get("checkpoint_sha256")
    if not isinstance(checkpoint_hashes, dict) or set(checkpoint_hashes) != expected_checkpoints:
        raise ValueError("closed-loop checkpoint inventory is incomplete or unexpected")
    for name, digest in checkpoint_hashes.items():
        path = study.CHECKPOINT_DIR / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"checkpoint hash mismatch: {name}")

    frame = pd.read_csv(RESULTS / "episode_results.csv")
    required = {
        "controller", "seed", "system", "condition", "episode",
        "mean_state_loss", "mean_action_effort", "mean_combined_cost",
        "terminal_normalized_rmse", "initial_state", "plant_parameters",
        "planner_seed",
    }
    if not required.issubset(frame.columns):
        raise ValueError("episode results lack required columns")
    keys = ["controller", "seed", "system", "condition", "episode"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate closed-loop episode identity")
    expected = {
        (controller, seed, system, condition, episode)
        for controller in study.CONTROLLERS
        for seed in study.CONFIRM_SEEDS
        for system in study.SYSTEMS
        for condition in study.CONDITIONS
        for episode in range(study.EPISODES_PER_CELL)
    }
    actual = set(frame.loc[:, keys].itertuples(index=False, name=None))
    if actual != expected or len(frame) != manifest.get("row_count"):
        raise ValueError("closed-loop episode grid is incomplete or unexpected")
    metrics = frame.loc[:, [
        "mean_state_loss", "mean_action_effort", "mean_combined_cost",
        "terminal_normalized_rmse",
    ]].to_numpy(float)
    if not np.isfinite(metrics).all() or (metrics < 0).any():
        raise ValueError("closed-loop metrics must be finite and nonnegative")

    pair_keys = ["seed", "system", "condition", "episode"]
    for _, group in frame.groupby(pair_keys, sort=False):
        if len(group) != len(study.CONTROLLERS):
            raise ValueError("controller pairing is incomplete")
        for column in ("initial_state", "plant_parameters", "planner_seed"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"controllers do not share {column}")
    plant = frame[frame.controller == study.CONTROLLERS[0]].copy()
    paired_id = plant[plant.condition.isin(("id_clean", "id_corrupt"))].pivot(
        index=["seed", "system", "episode"], columns="condition",
        values=["initial_state", "plant_parameters"],
    )
    for column in ("initial_state", "plant_parameters"):
        if not (paired_id[(column, "id_clean")] == paired_id[(column, "id_corrupt")]).all():
            raise ValueError(f"ID clean/corrupt plant draws differ: {column}")

    # Parse receipts so malformed list/dict strings cannot pass merely by matching.
    for value in frame.initial_state.unique():
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list) or not np.isfinite(np.asarray(parsed, float)).all():
            raise ValueError("invalid initial-state receipt")
    for value in frame.plant_parameters.unique():
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, dict) or not np.isfinite(np.asarray(list(parsed.values()), float)).all():
            raise ValueError("invalid plant-parameter receipt")

    overall, conditions, inference = study.summarize(frame)
    saved_overall = pd.read_csv(RESULTS / "overall_summary.csv")
    saved_conditions = pd.read_csv(RESULTS / "condition_summary.csv")
    pd.testing.assert_frame_equal(saved_overall, overall, check_exact=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(saved_conditions, conditions, check_exact=False, rtol=1e-12, atol=1e-12)
    recorded_inference = manifest.get("inference")
    if not isinstance(recorded_inference, dict):
        raise ValueError("closed-loop inference receipt is missing")
    if not inference_matches(recorded_inference, inference):
        raise ValueError("closed-loop inference differs from recomputation")
    return manifest, frame


def aggregate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_condition = frame.groupby(
        ["controller", "condition", "seed"], as_index=False
    )[["mean_state_loss", "mean_action_effort"]].mean()
    seed_overall = frame.groupby(
        ["controller", "seed"], as_index=False
    )[["mean_state_loss", "mean_action_effort"]].mean()
    overall = seed_overall.groupby("controller", as_index=False).agg(
        state_loss_mean=("mean_state_loss", "mean"),
        state_loss_sd=("mean_state_loss", lambda values: values.std(ddof=1)),
        action_effort_mean=("mean_action_effort", "mean"),
        action_effort_sd=("mean_action_effort", lambda values: values.std(ddof=1)),
    )
    condition = seed_condition.groupby(["controller", "condition"], as_index=False).agg(
        state_loss_mean=("mean_state_loss", "mean"),
        state_loss_sd=("mean_state_loss", lambda values: values.std(ddof=1)),
    )
    cell_seed = frame.groupby(
        ["controller", "system", "condition", "seed"], as_index=False
    ).mean(numeric_only=True)
    pivot = cell_seed.pivot(
        index=["system", "condition", "seed"], columns="controller",
        values="mean_state_loss",
    )
    relative = 100.0 * (pivot["pgnn"] - pivot["cfrwm"]) / pivot["pgnn"]
    cells = relative.rename("relative_improvement_pct").reset_index().groupby(
        ["system", "condition"], as_index=False
    ).agg(
        relative_improvement_mean_pct=("relative_improvement_pct", "mean"),
        relative_improvement_sd_pct=("relative_improvement_pct", lambda values: values.std(ddof=1)),
    )
    return overall, condition, cells


def write_macros(manifest: dict[str, object], overall: pd.DataFrame) -> Path:
    inference = manifest["inference"]
    assert isinstance(inference, dict)
    interval = inference["bootstrap_95_percent_interval"]
    assert isinstance(interval, list)
    indexed = overall.set_index("controller")
    lines = [
        "% Generated from the validated frozen T-ASE closed-loop study.",
        f"\\newcommand{{\\ClosedLoopEpisodes}}{{{manifest['row_count']}}}",
        f"\\newcommand{{\\ClosedLoopImprovement}}{{{float(inference['mean_percent']):.2f}}}",
        f"\\newcommand{{\\ClosedLoopCILow}}{{{float(interval[0]):.2f}}}",
        f"\\newcommand{{\\ClosedLoopCIHigh}}{{{float(interval[1]):.2f}}}",
        f"\\newcommand{{\\ClosedLoopPValue}}{{{float(inference['exact_one_sided_sign_flip_p_above_zero']):.5f}}}",
        f"\\newcommand{{\\ClosedLoopCFRLoss}}{{{indexed.loc['cfrwm', 'state_loss_mean']:.4f}}}",
        f"\\newcommand{{\\ClosedLoopPGNNLoss}}{{{indexed.loc['pgnn', 'state_loss_mean']:.4f}}}",
        f"\\newcommand{{\\ClosedLoopFilterNominalLoss}}{{{indexed.loc['complementary_nominal', 'state_loss_mean']:.4f}}}",
        f"\\newcommand{{\\ClosedLoopNominalLoss}}{{{indexed.loc['nominal_last', 'state_loss_mean']:.4f}}}",
    ]
    path = PAPER / "closed_loop_macros.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_table(overall: pd.DataFrame, condition: pd.DataFrame) -> Path:
    overall_indexed = overall.set_index("controller")
    condition_indexed = condition.set_index(["controller", "condition"])
    rows = []
    for controller in study.CONTROLLERS:
        values = overall_indexed.loc[controller]
        condition_cells = []
        for condition_name in study.CONDITIONS:
            cell = condition_indexed.loc[(controller, condition_name)]
            condition_cells.append(f"{cell.state_loss_mean:.3f} $\\pm$ {cell.state_loss_sd:.3f}")
        rows.append(
            f"{CONTROLLER_LABELS[controller]} & {values.state_loss_mean:.3f} $\\pm$ {values.state_loss_sd:.3f} "
            f"& {values.action_effort_mean:.3f} $\\pm$ {values.action_effort_sd:.3f} & "
            + " & ".join(condition_cells) + r" \\"
        )
    text = "\n".join([
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Controller & State loss & Action effort & ID clean & ID corrupt & OOD corrupt \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    path = PAPER / "table_closed_loop.tex"
    path.write_text(text, encoding="utf-8")
    return path


def plot(overall: pd.DataFrame, cells: pd.DataFrame) -> tuple[Path, Path]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), gridspec_kw={"width_ratios": [1.05, 1.45]})
    ordered = overall.set_index("controller").loc[list(study.CONTROLLERS)]
    colors = ["#7A7A7A", "#2A9D8F", "#D55E00", "#0072B2"]
    x = np.arange(len(ordered))
    axes[0].bar(x, ordered.state_loss_mean, yerr=ordered.state_loss_sd,
                color=colors, edgecolor="black", linewidth=0.45, capsize=2)
    axes[0].set_xticks(x, ["Nominal\nlast", "Filter +\nnominal", "PGNN", "CFR-WM"])
    axes[0].set_ylabel("Realized normalized state loss")
    axes[0].set_title("(a) Seed-balanced closed-loop loss")
    axes[0].grid(axis="y", color="#D0D0D0", linewidth=0.45)
    axes[0].set_axisbelow(True)

    matrix = np.empty((len(study.SYSTEMS), len(study.CONDITIONS)))
    indexed = cells.set_index(["system", "condition"])
    for row, system in enumerate(study.SYSTEMS):
        for column, condition in enumerate(study.CONDITIONS):
            matrix[row, column] = indexed.loc[(system, condition), "relative_improvement_mean_pct"]
    limit = max(10.0, float(np.nanmax(np.abs(matrix))))
    image = axes[1].imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axes[1].set_xticks(range(len(study.CONDITIONS)),
                       [CONDITION_LABELS[item] for item in study.CONDITIONS])
    axes[1].set_yticks(range(len(study.SYSTEMS)), [SYSTEM_LABELS[item] for item in study.SYSTEMS])
    axes[1].set_title("(b) CFR-WM improvement over PGNN (%)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if abs(matrix[row, column]) > 0.58 * limit else "black"
            axes[1].text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center", color=color, fontsize=7)
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.03)
    colorbar.ax.set_ylabel("Positive favors CFR-WM", rotation=90, va="bottom")
    fig.tight_layout(pad=0.7, w_pad=1.1)
    pdf = FIGURES / "FigClosedLoop.pdf"
    png = FIGURES / "FigClosedLoop.png"
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(png, dpi=350, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    manifest, frame = validate_manifest()
    overall, condition, cells = aggregate(frame)
    cells_path = PAPER / "closed_loop_system_condition.csv"
    cells.to_csv(cells_path, index=False)
    outputs = [write_macros(manifest, overall), write_table(overall, condition), cells_path]
    outputs.extend(plot(overall, cells))
    asset_manifest = {
        "complete": True,
        "closed_loop_manifest_sha256": sha256(RESULTS / "manifest.json"),
        "inputs": {name: digest for name, digest in manifest["outputs"].items()},
        "outputs": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in outputs},
    }
    output_path = PAPER / "closed_loop_assets_manifest.json"
    output_path.write_text(json.dumps(asset_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asset_manifest, indent=2))


if __name__ == "__main__":
    main()
