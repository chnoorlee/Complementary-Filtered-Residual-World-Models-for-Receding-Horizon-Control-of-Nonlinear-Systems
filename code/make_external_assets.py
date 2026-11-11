"""Validate and summarize the separate NanoDrone extension from saved forecasts.

NRMSE here is the square root of the mean squared normalized error over all
times and coordinates within a flight. Flights receive equal weight, followed
by equal weight across training seeds. SD describes optimizer variability;
neither windows nor training seeds are counted as independent platforms.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from external_validity import run as runner
from external_validity.data import expected_records, fit_normalizer, load_records

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/external_nanodrone"
PAPER = ROOT / "paper"
METHODS = ("cfr", "unfiltered", "recurrent", "nominal")
SEEDS = (131, 149, 167, 181, 199)
REGIMES = ("recorded", "noise", "missing", "combined")
HORIZONS = (25, 50)
FLIGHTS = tuple(record for split, record, _ in expected_records() if split == "test")
METRICS = ("nrmse", "position_rmse_m", "velocity_rmse_m_s")
KEYS = ("method", "seed", "regime", "horizon", "flight")
LABELS = {"cfr": "CFR-WM", "unfiltered": "Unfiltered residual",
          "recurrent": "Recurrent", "nominal": "Nominal prior"}
CONDITIONS = {"recorded": "As recorded", "noise": "Noise",
              "missing": "Missing", "combined": "Noise + missing"}
TOKENS = {"cfr": "CFR", "unfiltered": "Unfiltered", "recurrent": "Recurrent",
          "nominal": "Nominal", "recorded": "Recorded", "noise": "Noise",
          "missing": "Missing", "combined": "Combined", 25: "Short", 50: "Long",
          "nrmse": "NRMSE", "position_rmse_m": "Position", "velocity_rmse_m_s": "Velocity"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def verify_hashes(root: Path, entries: dict[str, str]) -> None:
    """Require every recorded file to exist within the project with its old hash."""
    if not isinstance(entries, dict) or not entries:
        raise ValueError("empty or invalid evidence hashes")
    for name, recorded in entries.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"evidence path escapes project: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(recorded)):
            raise ValueError(f"invalid evidence hash: {name}")
        if not path.is_file() or sha256(path) != recorded:
            raise ValueError(f"evidence hash mismatch: {name}")


def expected_grid() -> set[tuple]:
    return {(method, seed, regime, horizon, flight)
            for method in METHODS for seed in (SEEDS if method != "nominal" else (0,))
            for regime, horizon, flight in itertools.product(REGIMES, HORIZONS, FLIGHTS)}


def validate_grid(frame: pd.DataFrame) -> None:
    required = {*KEYS, "windows", *METRICS}
    if not required.issubset(frame.columns):
        raise ValueError("flight metric columns are incomplete")
    keys = list(frame.loc[:, list(KEYS)].itertuples(index=False, name=None))
    if len(keys) != len(set(keys)) or set(keys) != expected_grid():
        raise ValueError("flight metric grid is incomplete, duplicated, or unexpected")
    numeric = frame.loc[:, ["windows", *METRICS]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric[:, 1:] < 0).any():
        raise ValueError("flight metric values must be finite and nonnegative")
    if (numeric[:, 0] <= 0).any() or not np.equal(numeric[:, 0], np.floor(numeric[:, 0])).all():
        raise ValueError("window counts must be positive integers")


def metrics_from_arrays(prediction: np.ndarray, target: np.ndarray,
                        flight_ids: np.ndarray, origins: np.ndarray,
                        state_std: np.ndarray, method: str, seed: int,
                        regime: str) -> tuple[list[dict], list[dict]]:
    """Independently recompute the saved per-flight and per-window score rows."""
    if (prediction.shape != target.shape or prediction.ndim != 3 or
            prediction.shape[1:] != (50, 12) or len(prediction) == 0):
        raise ValueError("forecast arrays require matching (windows,50,12) shapes")
    if flight_ids.shape != (len(prediction),) or origins.shape != (len(prediction),):
        raise ValueError("forecast identities have wrong dimensions")
    if (not np.isfinite(prediction).all() or not np.isfinite(target).all() or
            state_std.shape != (12,) or not np.isfinite(state_std).all() or (state_std <= 0).any()):
        raise ValueError("forecast values and training scales must be finite")
    identities = list(zip(flight_ids.tolist(), origins.tolist()))
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate forecast origin within a flight")
    flight_rows, window_rows = [], []
    for horizon in HORIZONS:
        # Subtract in the recorded prediction dtype, exactly as in evaluation;
        # accumulate squared errors in double precision.
        error = (prediction[:, :horizon] - target[:, :horizon]).astype(np.float64)
        squared = error**2
        position = (error[..., :3] * state_std[:3])**2
        velocity = (error[..., 3:6] * state_std[3:6])**2
        identity = {"method": method, "seed": seed, "regime": regime, "horizon": horizon}
        for flight in np.unique(flight_ids):
            select = flight_ids == flight
            flight_rows.append({**identity, "flight": flight, "windows": int(select.sum()),
                                "nrmse": float(np.sqrt(squared[select].mean())),
                                "position_rmse_m": float(np.sqrt(position[select].sum(-1).mean())),
                                "velocity_rmse_m_s": float(np.sqrt(velocity[select].sum(-1).mean()))})
        for index, flight in enumerate(flight_ids):
            window_rows.append({**identity, "flight": flight, "origin": int(origins[index]),
                                "mse": float(squared[index].mean())})
    return flight_rows, window_rows


def compare_metrics(saved: pd.DataFrame, computed: pd.DataFrame,
                    keys: tuple[str, ...], metrics: tuple[str, ...]) -> None:
    """A new CSV receipt cannot conceal disagreement with saved predictions."""
    required = {*keys, *metrics}
    if not required.issubset(saved.columns) or not required.issubset(computed.columns):
        raise ValueError("metric comparison lacks required columns")
    if saved.duplicated(list(keys)).any() or computed.duplicated(list(keys)).any():
        raise ValueError("duplicate metric identity")
    left = saved.set_index(list(keys)).sort_index()
    right = computed.set_index(list(keys)).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("saved and recomputed metric identities differ")
    a, b = left.loc[:, list(metrics)].to_numpy(float), right.loc[:, list(metrics)].to_numpy(float)
    if not np.isfinite(a).all() or not np.allclose(a, b, rtol=1e-10, atol=1e-12):
        raise ValueError("saved metrics disagree with recomputed predictions")


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_grid(frame)
    # Do not weight flights by their number of sliding windows.
    seed_scores = frame.groupby(["method", "seed", "regime", "horizon"], sort=True)[list(METRICS)].mean().reset_index()
    rows = []
    for (method, regime, horizon), group in seed_scores.groupby(["method", "regime", "horizon"], sort=True):
        row = {"method": method, "regime": regime, "horizon": int(horizon), "training_seeds": len(group)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    contrasts = []
    indexed = seed_scores.set_index(["method", "regime", "horizon", "seed"]).sort_index()
    for comparator, regime, horizon in itertools.product(("unfiltered", "recurrent"), REGIMES, HORIZONS):
        cfr = indexed.loc[("cfr", regime, horizon), "nrmse"].reindex(SEEDS)
        other = indexed.loc[(comparator, regime, horizon), "nrmse"].reindex(SEEDS)
        if (other <= 0).any():
            raise ValueError("relative contrast requires positive comparator error")
        relative = 100.0 * (other - cfr) / other
        contrasts.append({"comparator": comparator, "regime": regime, "horizon": horizon,
                          "relative_improvement_mean_pct": float(relative.mean()),
                          "relative_improvement_sd_pct": float(relative.std(ddof=1)),
                          "relative_improvement_min_pct": float(relative.min()),
                          "relative_improvement_max_pct": float(relative.max()),
                          "seeds_favoring_cfr": int((relative > 0).sum()), "training_seeds": len(SEEDS)})
    return seed_scores, pd.DataFrame(rows), pd.DataFrame(contrasts)


def validate_training_receipts(training: dict) -> None:
    recorded = training["training_outputs"]
    for method in METHODS[:-1]:
        trials = []
        for lr in (.001, .003):
            path = OUT / "tuning" / f"{method}_17_lr{int(lr * 1_000_000)}.json"
            checkpoint = path.with_suffix(".pt")
            if key(path) not in recorded or key(checkpoint) not in recorded:
                raise ValueError("development receipt missing from training manifest")
            trial = json.loads(path.read_text())
            configuration = runner.asdict(runner.TrainConfig(epochs=40, batch_size=256,
                                          learning_rate=lr, patience=8, reliability_weight=0.))
            if (trial["method"] != method or trial["seed"] != 17 or trial["learning_rate"] != lr or
                    trial["config"] != configuration or trial["result"]["stopped_for_budget"] or
                    trial["checkpoint_sha256"] != sha256(checkpoint) or
                    not np.isfinite(trial["result"]["best_validation_nll"])):
                raise ValueError("development receipt does not match trial")
            trials.append(trial)
        selected = min(trials, key=lambda trial: trial["result"]["best_validation_nll"])["learning_rate"]
        if training["selection"][method] != selected:
            raise ValueError("learning-rate selection disagrees with validation winner")
        for seed in SEEDS:
            path = OUT / "final" / f"{method}_{seed}.json"
            checkpoint = path.with_suffix(".pt")
            if key(path) not in recorded or key(checkpoint) not in recorded:
                raise ValueError("final receipt missing from training manifest")
            receipt = json.loads(path.read_text())
            configuration = runner.asdict(runner.TrainConfig(epochs=40, batch_size=256,
                                          learning_rate=selected, patience=8, reliability_weight=0.))
            if (receipt["method"] != method or receipt["seed"] != seed or
                    receipt["config"] != configuration or
                    receipt["learning_rate"] != selected or receipt["parameter_count"] != 47000 or
                    receipt["checkpoint_sha256"] != sha256(checkpoint) or
                    receipt["result"]["stopped_for_budget"] or
                    not np.isfinite(receipt["result"]["best_validation_nll"])):
                raise ValueError("final receipt does not match declared method/seed/budget")


def load_verified_evidence() -> tuple[pd.DataFrame, dict, int]:
    training_path, evaluation_path = OUT / "training_manifest.json", OUT / "evaluation_manifest.json"
    training = json.loads(training_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    if training.get("complete") is not True or evaluation.get("complete") is not True:
        raise ValueError("external experiment is incomplete")
    sources = runner.provenance()
    if training.get("sources") != sources or evaluation.get("sources") != sources:
        raise ValueError("experiment sources changed or have incomplete coverage")
    if evaluation.get("training_manifest_sha256") != sha256(training_path):
        raise ValueError("evaluation refers to another training manifest")
    for entries in (sources, training["training_outputs"], evaluation["files"]):
        verify_hashes(ROOT, entries)
    validate_training_receipts(training)
    records = load_records(ROOT / "data/external/nanodrone")
    norm = fit_normalizer(records["train"])
    for name in ("state_mean", "state_std", "action_mean", "action_std"):
        if not np.array_equal(getattr(norm, name), np.asarray(training["norm"][name])):
            raise ValueError("training normalization receipt disagrees with training records")
    expected_target, expected_ids, expected_origins = [], [], []
    for record in records["test"]:
        states = (record.states - norm.state_mean) / norm.state_std
        for anchor in range(8, len(states) - 50 + 1, 50):
            expected_target.append(states[anchor:anchor+50])
            expected_ids.append(record.record_id)
            expected_origins.append(anchor - 1)
    expected_target = np.asarray(expected_target, dtype=np.float32)
    expected_ids, expected_origins = np.asarray(expected_ids), np.asarray(expected_origins)
    flight_rows, window_rows = [], []
    prediction_paths = []
    for method in METHODS:
        for seed, regime in itertools.product(SEEDS if method != "nominal" else (0,), REGIMES):
            path = OUT / "predictions" / f"{method}_{seed}_{regime}.npz"
            prediction_paths.append(path)
            if key(path) not in evaluation["files"]:
                raise ValueError("prediction missing from evaluation manifest")
            with np.load(path, allow_pickle=False) as values:
                for name, wanted in (("target", expected_target), ("record", expected_ids), ("origin", expected_origins)):
                    if not np.array_equal(values[name], wanted):
                        raise ValueError(f"forecast {name} differs from source-derived test windows: {path.name}")
                flights, windows = metrics_from_arrays(values["prediction"], values["target"], values["record"],
                                                       values["origin"], norm.state_std, method, seed, regime)
            flight_rows.extend(flights)
            window_rows.extend(windows)
    required = {key(path) for path in [*prediction_paths, OUT / "flight_metrics.csv", OUT / "window_metrics.csv"]}
    if set(evaluation["files"]) != required:
        raise ValueError("evaluation file inventory differs from the exact protocol grid")
    frame, window_frame = pd.DataFrame(flight_rows), pd.DataFrame(window_rows)
    validate_grid(frame)
    if evaluation.get("flight_rows") != len(frame) or evaluation.get("window_rows") != len(window_frame):
        raise ValueError("evaluation row counts disagree with forecasts")
    compare_metrics(pd.read_csv(OUT / "flight_metrics.csv"), frame, KEYS, ("windows", *METRICS))
    compare_metrics(pd.read_csv(OUT / "window_metrics.csv"), window_frame, (*KEYS, "origin"), ("mse",))
    authenticated = {**sources, **training["training_outputs"], **evaluation["files"],
                     key(training_path): sha256(training_path), key(evaluation_path): sha256(evaluation_path)}
    return frame, authenticated, len(expected_origins)


def format_value(row: pd.Series, metric: str = "nrmse", decimals: int = 3) -> str:
    mean = f"{row[f'{metric}_mean']:.{decimals}f}"
    if row["method"] == "nominal":
        return f"${mean}$"
    return f"${mean} \\pm {row[f'{metric}_sd']:.{decimals}f}$"


def make_tables(summary: pd.DataFrame) -> tuple[str, str]:
    indexed = summary.set_index(["regime", "horizon", "method"], drop=False)
    rows = [r"\begin{tabular}{llrrrr}", r"\toprule",
            r"Context & Horizon & CFR-WM & Unfiltered & Recurrent & Nominal \\", r"\midrule"]
    for regime, horizon in itertools.product(REGIMES, HORIZONS):
        values = [format_value(indexed.loc[(regime, horizon, method)]) for method in METHODS]
        rows.append(f"{CONDITIONS[regime]} & {horizon} & " + " & ".join(values) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular}", ""])
    physical = [r"\begin{tabular}{lrrrr}", r"\toprule",
                r"& \multicolumn{2}{c}{Position RMSE (m)} & \multicolumn{2}{c}{Velocity RMSE (m/s)} \\",
                r"Context & CFR-WM & Unfiltered & CFR-WM & Unfiltered \\", r"\midrule"]
    for regime in REGIMES:
        values = [format_value(indexed.loc[(regime, 50, method)], metric)
                  for metric in METRICS[1:] for method in METHODS[:2]]
        physical.append(CONDITIONS[regime] + " & " + " & ".join(values) + r" \\")
    physical.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(rows), "\n".join(physical)


def make_macros(summary: pd.DataFrame, contrasts: pd.DataFrame, window_count: int) -> str:
    values = {"ExternalTestFlights": len(FLIGHTS), "ExternalFinalSeeds": len(SEEDS),
              "ExternalLearnedParameters": 47000, "ExternalForecastWindows": window_count}
    for row in summary.to_dict("records"):
        prefix = "External" + TOKENS[row["method"]] + TOKENS[row["regime"]] + TOKENS[row["horizon"]]
        for metric in METRICS:
            for statistic in ("Mean", "SD"):
                values[prefix + TOKENS[metric] + statistic] = f"{row[f'{metric}_{statistic.lower()}']:.3f}"
    for row in contrasts.to_dict("records"):
        name = "ExternalVs" + TOKENS[row["comparator"]] + TOKENS[row["regime"]] + TOKENS[row["horizon"]] + "Improvement"
        values[name] = f"{row['relative_improvement_mean_pct']:.2f}"
    return "% Generated from verified saved external forecasts.\n" + "".join(
        rf"\newcommand{{\{name}}}{{{value}}}" + "\n" for name, value in values.items())


def make_figure(summary: pd.DataFrame, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = ("#0072B2", "#D55E00", "#009E73", "#777777")
    markers = ("o", "s", "^", "D")
    panels = ((25, "nrmse", "A   25-step forecast", "Normalized RMSE"),
              (50, "nrmse", "B   50-step forecast", "Normalized RMSE"),
              (50, "position_rmse_m", "C   Position, 50 steps", "Position RMSE (m)"),
              (50, "velocity_rmse_m_s", "D   Velocity, 50 steps", "Velocity RMSE (m/s)"))
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.6), constrained_layout=True)
        handles = []
        for ax, (horizon, metric, title, ylabel) in zip(axes.flat, panels):
            for i, method in enumerate(METHODS):
                frame = summary[(summary.method == method) & (summary.horizon == horizon)].set_index("regime").reindex(REGIMES)
                # Small horizontal offsets retain readable error bars without implying
                # a continuous relation between the four perturbation conditions.
                artist = ax.errorbar(np.arange(4) + (i - 1.5) * .13, frame[f"{metric}_mean"],
                                     yerr=frame[f"{metric}_sd"], fmt=markers[i], linestyle="none",
                                     color=palette[i], label=LABELS[method], capsize=3, markersize=5)
                if ax is axes[0, 0]:
                    handles.append(artist)
            ax.set_xticks(np.arange(4), ["Recorded", "Noise", "Missing", "Combined"])
            ax.set_ylabel(ylabel)
            ax.set_title(title, loc="left", fontsize=11)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=.22)
        axes[0, 0].legend(handles, [LABELS[method] for method in METHODS], fontsize=8, frameon=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=300, facecolor="white")
        fig.savefig(destination.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)


def main() -> None:
    frame, sources, window_count = load_verified_evidence()
    seed_scores, summary, contrasts = summarize(frame)
    assets = [OUT / "seed_scores.csv", OUT / "summary.csv", OUT / "contrasts.csv",
              PAPER / "table_external_results.tex", PAPER / "table_external_physical.tex",
              PAPER / "external_results_macros.tex", PAPER / "mypaper/figures/external_flight.png",
              PAPER / "mypaper/figures/external_flight.pdf"]
    for table, destination in zip((seed_scores, summary, contrasts), assets[:3]):
        table.to_csv(destination, index=False)
    first, second = make_tables(summary)
    assets[3].write_text(first, encoding="utf-8")
    assets[4].write_text(second, encoding="utf-8")
    assets[5].write_text(make_macros(summary, contrasts, window_count), encoding="utf-8")
    make_figure(summary, assets[6])
    manifest = {"complete": True, "generator": {key(Path(__file__)): sha256(Path(__file__))},
                "authenticated_sources": sources, "assets": {key(path): sha256(path) for path in assets},
                "counts": {"flight_rows": len(frame), "summary_rows": len(summary),
                           "contrast_rows": len(contrasts), "prediction_files": 64},
                "aggregation": "Root mean squared normalized coordinate/time error per flight; equal flight mean per seed; mean and sample SD across five training seeds. Nominal prior has one deterministic realization. Contrasts average paired seed relative effects. No new inferential tests."}
    destination = PAPER / "external_assets_manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({"complete": True, **manifest["counts"], "forecast_windows": window_count}))


if __name__ == "__main__":
    main()
