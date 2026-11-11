"""Post-review analyses from frozen CFR-WM checkpoints and deterministic data.

These analyses are explicitly secondary.  They do not change the locked H1/H2
decisions or overwrite any confirmatory or external-evaluation artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from external_validity.run import (  # noqa: E402
    METHODS as EXTERNAL_METHODS,
    OUT as EXTERNAL_OUT,
    REGIMES as EXTERNAL_REGIMES,
    SEEDS as EXTERNAL_SEEDS,
    ForecastModel,
    digest as external_digest,
    make_windows as make_external_windows,
    preparation as external_preparation,
    provenance as external_provenance,
)
from rgpwm.confirm_protocol import (  # noqa: E402
    CONFIRM_REGIMES,
    CONFIRM_SEEDS,
)
from rgpwm.data import Normalizer, WindowBatch  # noqa: E402
from rgpwm.dynamics import SYSTEMS, SystemSpec  # noqa: E402
from rgpwm.experiment import (  # noqa: E402
    _condition_data_sha256,
    _model_for,
    prepare_windows,
)
from rgpwm.metrics import (  # noqa: E402
    gaussian_nll,
    horizon_metrics,
    interval_metrics,
)
from rgpwm.training import windows_to_loader  # noqa: E402


OUT = ROOT / "results" / "reviewer_additional_analysis"
CONFIRM_OUT = ROOT / "results" / "confirmatory"
PAPER = ROOT / "paper"
RESIDUAL_SCALES = (0.0, 0.125, 0.25, 0.5)
OUTAGE_STARTS = (2, 4, 6)
MAIN_METHODS = ("rssm", "simnorm", "transformer", "pgnn", "pgdmm", "cfrwm")
METHOD_LABELS = {
    "rssm": "RSSM-style",
    "simnorm": "SimNorm-WM",
    "transformer": "Transformer-style",
    "pgnn": "PGNN-style",
    "pgdmm": "PG-DMM-style",
    "cfrwm": "CFR-WM",
    "nominal_last": "Nominal, final imputation",
    "complementary_nominal": "Complementary filter + nominal",
    "cfr": "CFR-WM",
    "unfiltered": "Unfiltered residual",
    "recurrent": "Data-only recurrent",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def copy_windows(windows: WindowBatch) -> WindowBatch:
    return WindowBatch(**{
        item.name: np.array(getattr(windows, item.name), copy=True)
        for item in fields(WindowBatch)
    })


def structured_outage(windows: WindowBatch, start: int, length: int = 2) -> WindowBatch:
    """Insert a synchronized all-channel context outage with causal imputation."""
    if start < 1 or length < 1 or start + length > windows.context_values.shape[1]:
        raise ValueError("outage must lie after the first and within the context")
    modified = copy_windows(windows)
    last_observed = modified.context_values[:, start - 1:start, :]
    for offset in range(length):
        index = start + offset
        modified.context_values[:, index:index + 1, :] = last_observed
        modified.context_mask[:, index, :] = 0.0
        modified.context_delta[:, index, :] = float(offset + 1)
    return modified


def optimal_scalar_gain(
    prediction_variance: float,
    measurement_variance: float,
    covariance: float = 0.0,
) -> float:
    """MSE-optimal scalar gain for correlated prediction/measurement errors."""
    denominator = prediction_variance + measurement_variance - 2.0 * covariance
    if denominator <= 0.0:
        raise ValueError("gain denominator must be positive")
    return float(np.clip((prediction_variance - covariance) / denominator, 0.0, 1.0))


def checkpoint_model(
    method: str,
    spec: SystemSpec,
    seed: int,
    normalizer: Normalizer,
    device: torch.device,
) -> torch.nn.Module:
    path = CONFIRM_OUT / spec.name / f"seed_{seed}" / f"{method}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = _model_for(method, spec, normalizer)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval().to(device)


def validate_condition_data(
    spec: SystemSpec,
    seed: int,
    normalizer: Normalizer,
    train: WindowBatch,
    validation: WindowBatch,
    tests: dict[str, WindowBatch],
) -> None:
    current = _condition_data_sha256(normalizer, train, validation, tests)
    for method in ("cfrwm", "pgnn"):
        record = json.loads(
            (CONFIRM_OUT / spec.name / f"seed_{seed}" / f"{method}.json").read_text(
                encoding="utf-8"
            )
        )
        if record["data_sha256"] != current:
            raise RuntimeError(f"data fingerprint mismatch for {spec.name} seed {seed}")


@torch.no_grad()
def predict_standard(
    model: torch.nn.Module,
    windows: WindowBatch,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    for batch in windows_to_loader(windows, batch_size, False, 0):
        values, mask, delta, context_actions, future_actions = (
            value.to(device, non_blocking=True) for value in batch[:5]
        )
        output = model(values, mask, delta, context_actions, future_actions)
        means.append(output["mean"].cpu().numpy())
        variances.append(output["variance"].cpu().numpy())
    return np.concatenate(means), np.concatenate(variances)


@torch.no_grad()
def predict_cfr_scale(
    model: torch.nn.Module,
    windows: WindowBatch,
    residual_scale: float,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay CFR-WM while changing only the inference-time residual multiplier."""
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    for batch in windows_to_loader(windows, batch_size, False, 0):
        values, mask, delta, context_actions, future_actions = (
            value.to(device, non_blocking=True) for value in batch[:5]
        )
        encoder_features = torch.cat(
            (values, mask, torch.log1p(delta.clamp_min(0.0)), context_actions), dim=-1
        )
        context_hidden, _ = model.encoder(encoder_features)
        hidden = context_hidden[:, -1]
        state, _ = model._filter_context(  # type: ignore[attr-defined]
            values, mask, delta, context_actions, context_hidden
        )
        batch_means: list[torch.Tensor] = []
        batch_variances: list[torch.Tensor] = []
        for action in future_actions.unbind(1):
            physics = model.normalized_nominal_step(state, action)
            raw = model.residual(torch.cat((state, action, hidden), dim=-1))
            correction, raw_log_variance = raw.chunk(2, dim=-1)
            state = model._project_state(  # type: ignore[attr-defined]
                physics + residual_scale * torch.tanh(correction)
            )
            variance = model._variance(raw_log_variance)
            hidden = model._update_hidden(state, action, hidden)
            batch_means.append(state)
            batch_variances.append(variance)
        means.append(torch.stack(batch_means, 1).cpu().numpy())
        variances.append(torch.stack(batch_variances, 1).cpu().numpy())
    return np.concatenate(means), np.concatenate(variances)


@torch.no_grad()
def predict_classical(
    model: torch.nn.Module,
    windows: WindowBatch,
    use_complementary_filter: bool,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for batch in windows_to_loader(windows, batch_size, False, 0):
        values, mask, _, context_actions, future_actions = (
            value.to(device, non_blocking=True) for value in batch[:5]
        )
        if use_complementary_filter:
            state = values[:, 0]
            for index in range(1, values.shape[1]):
                predicted = model.normalized_nominal_step(state, context_actions[:, index])
                gain = 0.5 * mask[:, index]
                state = model._project_state(  # type: ignore[attr-defined]
                    predicted + gain * (values[:, index] - predicted)
                )
        else:
            state = values[:, -1]
        rollout: list[torch.Tensor] = []
        for action in future_actions.unbind(1):
            state = model._project_state(  # type: ignore[attr-defined]
                model.normalized_nominal_step(state, action)
            )
            rollout.append(state)
        predictions.append(torch.stack(rollout, 1).cpu().numpy())
    return np.concatenate(predictions)


@torch.no_grad()
def pendulum_projection_changes(
    model: torch.nn.Module,
    windows: WindowBatch,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    changes: list[np.ndarray] = []
    for batch in windows_to_loader(windows, batch_size, False, 0):
        values, mask, _, actions = (
            value.to(device, non_blocking=True) for value in batch[:4]
        )
        state = values[:, 0]
        for index in range(1, values.shape[1]):
            predicted = model.normalized_nominal_step(state, actions[:, index])
            pre_projection = predicted + 0.5 * mask[:, index] * (
                values[:, index] - predicted
            )
            state = model._project_state(pre_projection)  # type: ignore[attr-defined]
            missing_sincos = mask[:, index, :2] == 0
            selected = (state[:, :2] - pre_projection[:, :2]).abs()[missing_sincos]
            if selected.numel():
                changes.append(selected.cpu().numpy())
    if not changes:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(changes).astype(np.float64)


def nrmse25(target: np.ndarray, prediction: np.ndarray) -> float:
    return horizon_metrics(target, prediction, np.ones(target.shape[-1]))["nrmse_h25"]


def seed_balanced_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
) -> pd.DataFrame:
    seed_rows = (
        frame.groupby([*group_columns, "seed"], as_index=False)[value_column]
        .mean()
        .rename(columns={value_column: "seed_value"})
    )
    return (
        seed_rows.groupby(group_columns, as_index=False)["seed_value"]
        .agg(mean="mean", sd=lambda values: values.std(ddof=1), seeds="count")
    )


def synthetic_analyses(device: torch.device) -> dict[str, object]:
    baseline_rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    outage_rows: list[dict[str, object]] = []
    projection_values: list[np.ndarray] = []
    reconstruction_differences: list[float] = []
    validated_conditions = 0

    for spec in SYSTEMS.values():
        for seed in CONFIRM_SEEDS:
            normalizer, train, validation, tests = prepare_windows(spec, seed)
            validate_condition_data(spec, seed, normalizer, train, validation, tests)
            validated_conditions += 1
            cfr = checkpoint_model("cfrwm", spec, seed, normalizer, device)
            pgnn = checkpoint_model("pgnn", spec, seed, normalizer, device)

            for regime in CONFIRM_REGIMES:
                windows = tests[regime]
                for name, filtered in (
                    ("nominal_last", False),
                    ("complementary_nominal", True),
                ):
                    prediction = predict_classical(cfr, windows, filtered, device)
                    baseline_rows.append({
                        "method": name,
                        "seed": seed,
                        "system": spec.name,
                        "regime": regime,
                        "nrmse_h25": nrmse25(windows.targets, prediction),
                    })
                for method in ("pgnn", "cfrwm"):
                    stored = json.loads(
                        (CONFIRM_OUT / spec.name / f"seed_{seed}" / f"{method}.json")
                        .read_text(encoding="utf-8")
                    )
                    baseline_rows.append({
                        "method": method,
                        "seed": seed,
                        "system": spec.name,
                        "regime": regime,
                        "nrmse_h25": stored["evaluations"][regime]["aggregate"]["nrmse_h25"],
                    })

                for scale in RESIDUAL_SCALES:
                    prediction, _ = predict_cfr_scale(cfr, windows, scale, device)
                    score = nrmse25(windows.targets, prediction)
                    scale_rows.append({
                        "scale": scale,
                        "seed": seed,
                        "system": spec.name,
                        "regime": regime,
                        "nrmse_h25": score,
                    })
                    if scale == 0.25:
                        stored = json.loads(
                            (CONFIRM_OUT / spec.name / f"seed_{seed}" / "cfrwm.json")
                            .read_text(encoding="utf-8")
                        )["evaluations"][regime]["aggregate"]["nrmse_h25"]
                        reconstruction_differences.append(abs(score - stored))

            clean = tests["id_clean"]
            for start in OUTAGE_STARTS:
                outage = structured_outage(clean, start=start, length=2)
                for method, model in (("pgnn", pgnn), ("cfrwm", cfr)):
                    prediction, _ = predict_standard(model, outage, device)
                    outage_rows.append({
                        "method": method,
                        "seed": seed,
                        "system": spec.name,
                        "regime": "synchronous_two_step_outage",
                        "outage_start": start,
                        "nrmse_h25": nrmse25(outage.targets, prediction),
                    })
            if spec.name == "torque_pendulum":
                for regime in ("id_corrupt", "ood_parameter_corrupt"):
                    projection_values.append(
                        pendulum_projection_changes(cfr, tests[regime], device)
                    )

    baseline = pd.DataFrame(baseline_rows)
    scales = pd.DataFrame(scale_rows)
    outages = pd.DataFrame(outage_rows)
    baseline.to_csv(OUT / "classical_baseline_cells.csv", index=False)
    scales.to_csv(OUT / "residual_scale_cells.csv", index=False)
    outages.to_csv(OUT / "structured_outage_cells.csv", index=False)

    baseline_summary = seed_balanced_summary(baseline, ["method"], "nrmse_h25")
    baseline_regime_summary = seed_balanced_summary(
        baseline, ["method", "regime"], "nrmse_h25"
    )
    scale_summary = seed_balanced_summary(scales, ["scale"], "nrmse_h25")
    scale_regime_summary = seed_balanced_summary(
        scales, ["scale", "regime"], "nrmse_h25"
    )
    outage_summary = seed_balanced_summary(outages, ["method"], "nrmse_h25")

    clean_reference = baseline[baseline["method"].isin(("pgnn", "cfrwm"))]
    clean_reference = clean_reference[clean_reference["regime"] == "id_clean"]
    clean_by_seed = clean_reference.groupby(["method", "seed"], as_index=False)["nrmse_h25"].mean()
    outage_by_seed = outages.groupby(["method", "seed"], as_index=False)["nrmse_h25"].mean()
    degradation = outage_by_seed.merge(
        clean_by_seed, on=["method", "seed"], suffixes=("_outage", "_clean")
    )
    degradation["relative_degradation_percent"] = 100.0 * (
        degradation["nrmse_h25_outage"] / degradation["nrmse_h25_clean"] - 1.0
    )
    degradation_summary = seed_balanced_summary(
        degradation, ["method"], "relative_degradation_percent"
    )

    baseline_summary.to_csv(OUT / "classical_baseline_summary.csv", index=False)
    baseline_regime_summary.to_csv(
        OUT / "classical_baseline_regime_summary.csv", index=False
    )
    scale_summary.to_csv(OUT / "residual_scale_summary.csv", index=False)
    scale_regime_summary.to_csv(
        OUT / "residual_scale_regime_summary.csv", index=False
    )
    outage_summary.to_csv(OUT / "structured_outage_summary.csv", index=False)
    degradation_summary.to_csv(OUT / "structured_outage_degradation.csv", index=False)

    projection = np.concatenate([item for item in projection_values if item.size])
    projection_summary = {
        "missing_sine_cosine_entries": int(projection.size),
        "fraction_changed_above_1e-6": float(np.mean(projection > 1e-6)),
        "mean_absolute_normalized_change": float(np.mean(projection)),
        "p95_absolute_normalized_change": float(np.quantile(projection, 0.95)),
        "max_absolute_normalized_change": float(np.max(projection)),
    }
    write_json(OUT / "projection_summary.json", projection_summary)
    return {
        "validated_seed_system_conditions": validated_conditions,
        "maximum_scale_025_reconstruction_difference": float(
            max(reconstruction_differences)
        ),
        "projection": projection_summary,
        "baseline_summary": baseline_summary.to_dict("records"),
        "baseline_regime_summary": baseline_regime_summary.to_dict("records"),
        "scale_summary": scale_summary.to_dict("records"),
        "scale_regime_summary": scale_regime_summary.to_dict("records"),
        "outage_summary": outage_summary.to_dict("records"),
        "outage_degradation": degradation_summary.to_dict("records"),
    }


def latency_analysis() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in MAIN_METHODS:
        for spec in SYSTEMS.values():
            for seed in CONFIRM_SEEDS:
                record = json.loads(
                    (CONFIRM_OUT / spec.name / f"seed_{seed}" / f"{method}.json")
                    .read_text(encoding="utf-8")
                )
                for regime in CONFIRM_REGIMES:
                    rows.append({
                        "method": method,
                        "seed": seed,
                        "system": spec.name,
                        "regime": regime,
                        "latency_ms_per_trajectory": record["evaluations"][regime]["aggregate"][
                            "rollout_latency_ms_per_trajectory"
                        ],
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "latency_cells.csv", index=False)
    summary = seed_balanced_summary(frame, ["method"], "latency_ms_per_trajectory")
    summary.to_csv(OUT / "latency_summary.csv", index=False)
    return summary.to_dict("records")


def validate_external_artifacts() -> dict[str, object]:
    training = json.loads((EXTERNAL_OUT / "training_manifest.json").read_text())
    evaluation = json.loads((EXTERNAL_OUT / "evaluation_manifest.json").read_text())
    if training.get("complete") is not True or training["sources"] != external_provenance():
        raise RuntimeError("external training manifest is incomplete or stale")
    for relative, expected in training["training_outputs"].items():
        if external_digest(ROOT / relative) != expected:
            raise RuntimeError(f"changed external training artifact: {relative}")
    if evaluation.get("complete") is not True:
        raise RuntimeError("external evaluation manifest is incomplete")
    if evaluation["training_manifest_sha256"] != external_digest(
        EXTERNAL_OUT / "training_manifest.json"
    ):
        raise RuntimeError("external evaluation does not match training manifest")
    for relative, expected in evaluation["files"].items():
        if external_digest(ROOT / relative) != expected:
            raise RuntimeError(f"changed external evaluation artifact: {relative}")
    return training


def flight_metric_rows(
    target: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    flight_ids: np.ndarray,
    horizon: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for flight in np.unique(flight_ids):
        selected = flight_ids == flight
        flight_target = target[selected, :horizon]
        flight_mean = mean[selected, :horizon]
        flight_variance = variance[selected, :horizon]
        rows.append({
            "flight": str(flight),
            "gaussian_nll": gaussian_nll(flight_target, flight_mean, flight_variance),
            **interval_metrics(flight_target, flight_mean, flight_variance),
        })
    return rows


@torch.no_grad()
def external_uncertainty_analysis(device: torch.device) -> dict[str, object]:
    training = validate_external_artifacts()
    records, normalizer, prior, _, _ = external_preparation()
    rows: list[dict[str, object]] = []
    maximum_mean_difference = 0.0

    for regime, (noise, missing) in EXTERNAL_REGIMES.items():
        windows, _ = make_external_windows(
            records["test"], normalizer, horizon=50, stride=50,
            noise=noise, missing=missing,
        )
        for method in EXTERNAL_METHODS:
            for seed in EXTERNAL_SEEDS:
                model = ForecastModel(method, normalizer, prior)
                model.load_state_dict(
                    torch.load(
                        EXTERNAL_OUT / "final" / f"{method}_{seed}.pt",
                        weights_only=True,
                        map_location="cpu",
                    ),
                    strict=True,
                )
                model.eval().to(device)
                mean, variance = predict_standard(model, windows, device)
                archived = np.load(
                    EXTERNAL_OUT / "predictions" / f"{method}_{seed}_{regime}.npz"
                )["prediction"]
                maximum_mean_difference = max(
                    maximum_mean_difference,
                    float(np.max(np.abs(mean - archived))),
                )
                for horizon in (25, 50):
                    for flight_row in flight_metric_rows(
                        windows.targets, mean, variance, windows.trajectory_ids, horizon
                    ):
                        rows.append({
                            "method": method,
                            "seed": seed,
                            "regime": regime,
                            "horizon": horizon,
                            **flight_row,
                        })

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "external_uncertainty_flight_rows.csv", index=False)
    per_seed = (
        frame.groupby(["method", "seed", "regime", "horizon"], as_index=False)[
            ["gaussian_nll", "coverage_90", "interval_width_90", "calibration_error"]
        ]
        .mean()
    )
    summary = (
        per_seed.groupby(["method", "regime", "horizon"], as_index=False)
        .agg(
            nll_mean=("gaussian_nll", "mean"),
            nll_sd=("gaussian_nll", lambda x: x.std(ddof=1)),
            coverage_mean=("coverage_90", "mean"),
            coverage_sd=("coverage_90", lambda x: x.std(ddof=1)),
            width_mean=("interval_width_90", "mean"),
            width_sd=("interval_width_90", lambda x: x.std(ddof=1)),
            calibration_mean=("calibration_error", "mean"),
            calibration_sd=("calibration_error", lambda x: x.std(ddof=1)),
        )
    )
    summary.to_csv(OUT / "external_uncertainty_summary.csv", index=False)
    return {
        "training_manifest_complete": bool(training["complete"]),
        "maximum_recomputed_mean_difference": maximum_mean_difference,
        "rows": summary.to_dict("records"),
    }


def latex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def selected_row(records: list[dict[str, object]], **match: object) -> dict[str, object]:
    selected = [row for row in records if all(row[key] == value for key, value in match.items())]
    if len(selected) != 1:
        raise RuntimeError(f"expected one row for {match}, found {len(selected)}")
    return selected[0]


def render_paper_assets(summary: dict[str, object]) -> list[Path]:
    synthetic = summary["synthetic"]
    assert isinstance(synthetic, dict)
    baseline = synthetic["baseline_summary"]
    baseline_regime = synthetic["baseline_regime_summary"]
    scales = synthetic["scale_summary"]
    scale_regime = synthetic["scale_regime_summary"]
    outages = synthetic["outage_summary"]
    degradation = synthetic["outage_degradation"]
    latency = summary["latency"]
    external = summary["external_uncertainty"]
    assert isinstance(baseline, list) and isinstance(baseline_regime, list)
    assert isinstance(scales, list) and isinstance(scale_regime, list)
    assert isinstance(outages, list) and isinstance(degradation, list)
    assert isinstance(latency, list) and isinstance(external, dict)

    posthoc_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Analysis and variant & All & Clean & ID corrupt & OOD corrupt \\",
        r"\midrule",
    ]
    for method in ("nominal_last", "complementary_nominal", "pgnn", "cfrwm"):
        row = selected_row(baseline, method=method)
        regimes = [
            selected_row(baseline_regime, method=method, regime=regime)
            for regime in CONFIRM_REGIMES
        ]
        posthoc_lines.append(
            f"{METHOD_LABELS[method]} & {float(row['mean']):.3f} "
            + "".join(f"& {float(item['mean']):.3f} " for item in regimes)
            + r"\\"
        )
    posthoc_lines.append(r"\midrule")
    for scale in RESIDUAL_SCALES:
        row = selected_row(scales, scale=scale)
        regimes = [
            selected_row(scale_regime, scale=scale, regime=regime)
            for regime in CONFIRM_REGIMES
        ]
        posthoc_lines.append(
            f"CFR-WM, inference scale {scale:g} & {float(row['mean']):.3f} "
            + "".join(f"& {float(item['mean']):.3f} " for item in regimes)
            + r"\\"
        )
    posthoc_lines.extend((r"\bottomrule", r"\end{tabular}"))
    posthoc_path = PAPER / "table_reviewer_posthoc.tex"
    write_text(posthoc_path, "\n".join(posthoc_lines) + "\n")

    outage_lines = [
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Method & Outage NRMSE-25 & Change from clean (\%) \\", r"\midrule",
    ]
    for method in ("pgnn", "cfrwm"):
        row = selected_row(outages, method=method)
        change = selected_row(degradation, method=method)
        outage_lines.append(
            f"{METHOD_LABELS[method]} & {float(row['mean']):.3f} $\\pm$ {float(row['sd']):.3f} "
            f"& {float(change['mean']):+.1f} \\\\"
        )
    outage_lines.extend((r"\bottomrule", r"\end{tabular}"))
    outage_path = PAPER / "table_structured_outage.tex"
    write_text(outage_path, "\n".join(outage_lines) + "\n")

    latency_lines = [
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Method & ms/trajectory & SD \\", r"\midrule",
    ]
    for method in MAIN_METHODS:
        row = selected_row(latency, method=method)
        latency_lines.append(
            f"{METHOD_LABELS[method]} & {float(row['mean']):.3f} & {float(row['sd']):.3f} \\\\"
        )
    latency_lines.extend((r"\bottomrule", r"\end{tabular}"))
    latency_path = PAPER / "table_latency.tex"
    write_text(latency_path, "\n".join(latency_lines) + "\n")

    external_rows = external["rows"]
    assert isinstance(external_rows, list)
    uncertainty_lines = [
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Condition & Method & NLL & Cov. & Width & Cal. error \\", r"\midrule",
    ]
    for regime in EXTERNAL_REGIMES:
        for method in ("cfr", "unfiltered"):
            row = selected_row(external_rows, method=method, regime=regime, horizon=50)
            uncertainty_lines.append(
                f"{regime.title()} & {METHOD_LABELS[method]} & {float(row['nll_mean']):.3f} "
                f"& {float(row['coverage_mean']):.3f} & {float(row['width_mean']):.3f} "
                f"& {float(row['calibration_mean']):.3f} \\\\"
            )
    uncertainty_lines.extend((r"\bottomrule", r"\end{tabular}"))
    uncertainty_path = PAPER / "table_external_uncertainty.tex"
    write_text(uncertainty_path, "\n".join(uncertainty_lines) + "\n")

    projection = synthetic["projection"]
    assert isinstance(projection, dict)
    cfr_latency = selected_row(latency, method="cfrwm")
    best_scale = min(scales, key=lambda row: float(row["mean"]))
    macros = "\n".join((
        f"\\newcommand{{\\PosthocBestResidualScale}}{{{float(best_scale['scale']):g}}}",
        f"\\newcommand{{\\ProjectionMissingEntries}}{{{int(projection['missing_sine_cosine_entries'])}}}",
        f"\\newcommand{{\\ProjectionChangedFraction}}{{{100.0 * float(projection['fraction_changed_above_1e-6']):.1f}}}",
        f"\\newcommand{{\\ProjectionMeanChange}}{{{float(projection['mean_absolute_normalized_change']):.4f}}}",
        f"\\newcommand{{\\ProjectionPChange}}{{{float(projection['p95_absolute_normalized_change']):.4f}}}",
        f"\\newcommand{{\\ProjectionMaxChange}}{{{float(projection['max_absolute_normalized_change']):.4f}}}",
        f"\\newcommand{{\\CFRLatencyMean}}{{{float(cfr_latency['mean']):.3f}}}",
        f"\\newcommand{{\\CFRLatencySD}}{{{float(cfr_latency['sd']):.3f}}}",
        f"\\newcommand{{\\ResidualReconstructionDifference}}{{{float(synthetic['maximum_scale_025_reconstruction_difference']):.2e}}}",
        f"\\newcommand{{\\ExternalMeanReconstructionDifference}}{{{float(external['maximum_recomputed_mean_difference']):.2e}}}",
    )) + "\n"
    macros_path = PAPER / "reviewer_additional_macros.tex"
    write_text(macros_path, macros)
    return [posthoc_path, outage_path, latency_path, uncertainty_path, macros_path]


def output_inventory() -> list[Path]:
    return [
        OUT / "classical_baseline_cells.csv",
        OUT / "classical_baseline_summary.csv",
        OUT / "classical_baseline_regime_summary.csv",
        OUT / "residual_scale_cells.csv",
        OUT / "residual_scale_summary.csv",
        OUT / "residual_scale_regime_summary.csv",
        OUT / "structured_outage_cells.csv",
        OUT / "structured_outage_summary.csv",
        OUT / "structured_outage_degradation.csv",
        OUT / "projection_summary.json",
        OUT / "latency_cells.csv",
        OUT / "latency_summary.csv",
        OUT / "external_uncertainty_flight_rows.csv",
        OUT / "external_uncertainty_summary.csv",
        OUT / "summary.json",
        PAPER / "table_reviewer_posthoc.tex",
        PAPER / "table_structured_outage.tex",
        PAPER / "table_latency.tex",
        PAPER / "table_external_uncertainty.tex",
        PAPER / "reviewer_additional_macros.tex",
    ]


def build_manifest(outputs: list[Path]) -> dict[str, object]:
    confirm_inputs = []
    for spec in SYSTEMS.values():
        for seed in CONFIRM_SEEDS:
            for method in ("cfrwm", "pgnn"):
                confirm_inputs.extend((
                    CONFIRM_OUT / spec.name / f"seed_{seed}" / f"{method}.json",
                    CONFIRM_OUT / spec.name / f"seed_{seed}" / f"{method}.pt",
                ))
    external_inputs = [
        EXTERNAL_OUT / "training_manifest.json",
        EXTERNAL_OUT / "evaluation_manifest.json",
        *(EXTERNAL_OUT / "final" / f"{method}_{seed}.pt"
          for method in EXTERNAL_METHODS for seed in EXTERNAL_SEEDS),
    ]
    inputs = [Path(__file__), *confirm_inputs, *external_inputs]
    return {
        "schema_version": 1,
        "complete": True,
        "analysis_scope": "post_review_secondary_no_confirmatory_decision_change",
        "inputs": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in inputs
        },
        "outputs": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in outputs
        },
    }


def run(device: torch.device) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    summary: dict[str, object] = {
        "scope": "post-review secondary analyses; locked H1/H2 unchanged",
        "device": str(device),
        "synthetic": synthetic_analyses(device),
        "latency": latency_analysis(),
        "external_uncertainty": external_uncertainty_analysis(device),
    }
    write_json(OUT / "summary.json", summary)
    render_paper_assets(summary)
    outputs = output_inventory()
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing analysis outputs: {missing}")
    write_json(OUT / "manifest.json", build_manifest(outputs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    run(device)


if __name__ == "__main__":
    main()
