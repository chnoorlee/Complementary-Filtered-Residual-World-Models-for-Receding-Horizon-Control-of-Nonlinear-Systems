"""Command-line runner for the frozen pilot, primary grid, and ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .confirm_protocol import CONFIRM_SEEDS
from .data import Normalizer, WindowBatch, build_normalizer, generate_trajectory_split, make_windows
from .dynamics import SYSTEMS, SystemSpec
from .metrics import EVALUATION_METRIC_VERSION, evaluation_code_sha256
from .models import FIXED_FILTER_GAINS, build_model
from .training import (
    TrainConfig,
    evaluate_model,
    save_checkpoint,
    seed_everything,
    train_model,
    validation_nll,
    windows_to_loader,
)


PRIMARY_METHODS = ("rssm", "simnorm", "transformer", "pgnn", "pgdmm", "rgpwm")
ABLATION_METHODS = ("rgp_no_rel", "rgp_fixed", "rgp_no_phys", "rgp_1step")
PIVOT_METHODS = (
    "pfrwm", "pfr_fixed", "cfrwm", "cfr_gain0", "cfr_gain025",
    "cfr_gain075", "cfr_gain1", "cfr_1step",
)
HIDDEN_DIMENSIONS = {
    "rssm": 80,
    "simnorm": 80,
    "transformer": 48,
    "pgnn": 84,
    "pgdmm": 84,
    "rgpwm": 64,
    "rgp_no_rel": 64,
    "rgp_fixed": 64,
    "rgp_no_phys": 64,
    "rgp_1step": 64,
    "pfrwm": 72,
    "pfr_fixed": 72,
    "cfrwm": 72,
    "cfr_gain0": 72,
    "cfr_gain025": 72,
    "cfr_gain075": 72,
    "cfr_gain1": 72,
    "cfr_1step": 72,
}


def _noise_scale(normalizer: Normalizer, fraction: float) -> float:
    return fraction * float(np.mean(normalizer.state_std))


def _condition_data_sha256(
    normalizer: Normalizer,
    train_windows: WindowBatch,
    validation_windows: WindowBatch,
    test_windows: dict[str, WindowBatch],
) -> str:
    digest = hashlib.sha256()
    for name in ("state_mean", "state_std", "action_mean", "action_std"):
        array = np.ascontiguousarray(getattr(normalizer, name))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    for split_name, windows in (
        ("train", train_windows),
        ("validation", validation_windows),
        *sorted(test_windows.items()),
    ):
        digest.update(split_name.encode("utf-8"))
        for field in (
            "context_values", "context_mask", "context_delta", "context_actions",
            "future_actions", "targets",
        ):
            array = np.ascontiguousarray(getattr(windows, field))
            digest.update(field.encode("utf-8"))
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(array.tobytes())
        digest.update("\n".join(map(str, windows.trajectory_ids)).encode("utf-8"))
    return digest.hexdigest()


def prepare_windows(
    spec: SystemSpec,
    seed: int,
    train_count: int = 256,
    validation_count: int = 64,
    test_count: int = 96,
    trajectory_steps: int = 72,
) -> tuple[Normalizer, WindowBatch, WindowBatch, dict[str, WindowBatch]]:
    split = generate_trajectory_split(
        spec,
        counts={"train": train_count, "val": validation_count, "test": test_count},
        steps=trajectory_steps,
        seed=seed * 1009 + 17,
        parameter_regime="train",
    )
    ood = generate_trajectory_split(
        spec,
        counts={"test": test_count},
        steps=trajectory_steps,
        seed=seed * 1009 + 7919,
        parameter_regime="ood",
    )["test"]
    normalizer = build_normalizer(split["train"])
    train_windows = make_windows(
        split["train"], normalizer, 8, 25, 8,
        _noise_scale(normalizer, 0.03), 0.10, seed * 100 + 1,
    )
    validation_windows = make_windows(
        split["val"], normalizer, 8, 25, 16,
        _noise_scale(normalizer, 0.03), 0.10, seed * 100 + 2,
    )
    tests = {
        "id_clean": make_windows(
            split["test"], normalizer, 8, 25, trajectory_steps,
            0.0, 0.0, seed * 100 + 3,
        ),
        "id_corrupt": make_windows(
            split["test"], normalizer, 8, 25, trajectory_steps,
            _noise_scale(normalizer, 0.08), 0.25, seed * 100 + 4,
        ),
        "ood_parameter_corrupt": make_windows(
            ood, normalizer, 8, 25, trajectory_steps,
            _noise_scale(normalizer, 0.08), 0.25, seed * 100 + 5,
        ),
    }
    return normalizer, train_windows, validation_windows, tests


def _model_for(method: str, spec: SystemSpec, normalizer: Normalizer) -> torch.nn.Module:
    return build_model(
        method,
        spec,
        state_mean=torch.from_numpy(normalizer.state_mean),
        state_std=torch.from_numpy(normalizer.state_std),
        action_mean=torch.from_numpy(normalizer.action_mean),
        action_std=torch.from_numpy(normalizer.action_std),
        hidden_dim=HIDDEN_DIMENSIONS[method],
    )


def _select_fixed_gate(
    model: torch.nn.Module,
    validation_windows: WindowBatch,
    device: torch.device,
    seed: int,
) -> tuple[float, dict[str, float]]:
    loader = windows_to_loader(validation_windows, 256, False, seed)
    scores: dict[str, float] = {}
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        model.set_fixed_weight(weight)  # type: ignore[attr-defined]
        scores[str(weight)] = validation_nll(model, loader, device)
    best = min(scores, key=scores.get)
    model.set_fixed_weight(float(best))  # type: ignore[attr-defined]
    return float(best), scores


def run_condition(
    method: str,
    spec: SystemSpec,
    seed: int,
    normalizer: Normalizer,
    train_windows: WindowBatch,
    validation_windows: WindowBatch,
    test_windows: dict[str, WindowBatch],
    config: TrainConfig,
    device: torch.device,
    output_dir: Path,
    wall_clock_deadline: float | None,
    overwrite: bool = False,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    provenance = dict(provenance or {})
    reserved_metadata = {
        "method", "system", "seed", "parameter_count", "hidden_dim",
        "train_config", "one_step_training", "fixed_gate_selection",
        "fixed_filter_selection", "device", "run_code_sha256", "data_sha256",
        "train_result", "evaluations", "evaluation_metric_version",
        "evaluation_code_sha256",
    }
    conflicts = reserved_metadata & set(provenance)
    if conflicts:
        raise ValueError(f"provenance keys conflict with run metadata: {sorted(conflicts)}")
    result_path = output_dir / spec.name / f"seed_{seed}" / f"{method}.json"
    checkpoint_path = result_path.with_suffix(".pt")
    code_fingerprint = evaluation_code_sha256()
    data_fingerprint = _condition_data_sha256(
        normalizer, train_windows, validation_windows, test_windows,
    )
    one_step = method in {"rgp_1step", "cfr_1step"}
    fixed_filter_selection = (
        {
            "selected": FIXED_FILTER_GAINS[method],
            "selection": "prespecified_before_training",
        }
        if method in FIXED_FILTER_GAINS else None
    )
    if result_path.exists() and not overwrite:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        expected = {
            "run_code_sha256": code_fingerprint,
            "data_sha256": data_fingerprint,
            "hidden_dim": HIDDEN_DIMENSIONS[method],
            "train_config": asdict(config),
            "one_step_training": one_step,
            "fixed_filter_selection": fixed_filter_selection,
            "evaluation_metric_version": EVALUATION_METRIC_VERSION,
            **provenance,
        }
        mismatches = {
            key: {"expected": value, "found": existing.get(key)}
            for key, value in expected.items() if existing.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"refusing to reuse stale result {result_path}: {mismatches}; use a new directory"
            )
        return existing
    seed_everything(seed)
    model = _model_for(method, spec, normalizer)
    if method in FIXED_FILTER_GAINS:
        model.set_fixed_filter_gain(FIXED_FILTER_GAINS[method])  # type: ignore[attr-defined]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_result = train_model(
        model, train_windows, validation_windows, config, device,
        seed=seed, wall_clock_deadline=wall_clock_deadline, one_step=one_step,
    )
    fixed_gate_selection = None
    if method == "rgp_fixed":
        selected, scores = _select_fixed_gate(model, validation_windows, device, seed)
        fixed_gate_selection = {"selected": selected, "validation_nll_by_weight": scores}
    evaluations: dict[str, object] = {}
    for regime, windows in test_windows.items():
        aggregate, trajectory_rows = evaluate_model(model, windows, normalizer, device)
        evaluations[regime] = {"aggregate": aggregate, "trajectories": trajectory_rows}
    metadata: dict[str, object] = {
        "method": method,
        "system": spec.name,
        "seed": seed,
        "parameter_count": parameter_count,
        "hidden_dim": HIDDEN_DIMENSIONS[method],
        "train_config": asdict(config),
        "one_step_training": one_step,
        "fixed_gate_selection": fixed_gate_selection,
        "fixed_filter_selection": fixed_filter_selection,
        "device": str(device),
        "run_code_sha256": code_fingerprint,
        "data_sha256": data_fingerprint,
    }
    metadata.update(provenance)
    payload: dict[str, object] = {
        **metadata,
        "train_result": asdict(train_result),
        "evaluations": evaluations,
        "evaluation_metric_version": EVALUATION_METRIC_VERSION,
        "evaluation_code_sha256": code_fingerprint,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(result_path)
    save_checkpoint(checkpoint_path, model, train_result, metadata)
    return payload


def _parse_csv(values: str, allowed: Iterable[str]) -> list[str]:
    available = set(allowed)
    selected = [value.strip() for value in values.split(",") if value.strip()]
    unknown = set(selected) - available
    if unknown:
        raise ValueError(f"unknown values: {sorted(unknown)}")
    return selected


def _reject_confirmatory_cli_overlap(seeds: list[int], output_dir: Path) -> None:
    """Reserve frozen seeds and artifacts for the provenance-checked runner."""
    overlap = sorted(set(seeds) & set(CONFIRM_SEEDS))
    if overlap:
        raise RuntimeError(
            "confirmatory seeds are accessible only through the dedicated frozen runner "
            f"rgpwm.confirm_run; blocked seeds: {overlap}"
        )
    if output_dir.resolve() == Path("results/confirmatory").resolve():
        raise RuntimeError(
            "results/confirmatory is reserved for the dedicated frozen runner "
            "rgpwm.confirm_run"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument("--methods", default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--tuning-file", default="results/tuning/best_configs.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    systems = _parse_csv(args.systems, SYSTEMS)
    seeds = [int(value) for value in args.seeds.split(",")]
    default_methods = ABLATION_METHODS if args.ablations else PRIMARY_METHODS
    methods = _parse_csv(args.methods, (*PRIMARY_METHODS, *ABLATION_METHODS, *PIVOT_METHODS)) if args.methods else list(default_methods)
    if args.pilot:
        systems, seeds = ["forced_duffing"], [11]
        epochs = args.epochs or 3
        output_dir = Path("results/pilot")
    else:
        epochs = args.epochs or 60
        output_dir = Path(args.output_dir) if args.output_dir else Path("results/ablations" if args.ablations else "results/primary")
    _reject_confirmatory_cli_overlap(seeds, output_dir)
    base_config = TrainConfig(epochs=epochs, patience=min(10, max(2, epochs // 3)))
    tuning_path = Path(args.tuning_file)
    tuned_configs = json.loads(tuning_path.read_text(encoding="utf-8")) if args.full and tuning_path.exists() else {}
    run_start = time.perf_counter()
    deadline = run_start + args.max_hours * 3600.0 * 0.80
    completed: list[dict[str, object]] = []
    print(json.dumps({
        "event": "run_start", "pilot": args.pilot, "device": str(device),
        "torch": torch.__version__, "python": platform.python_version(),
        "systems": systems, "seeds": seeds, "methods": methods, "epochs": epochs,
    }))
    for system_name in systems:
        spec = SYSTEMS[system_name]
        for seed in seeds:
            normalizer, train_windows, validation_windows, tests = prepare_windows(spec, seed)
            for method in methods:
                if time.perf_counter() >= deadline:
                    print(json.dumps({"event": "time_guard", "completed": len(completed)}))
                    break
                system_tuning = tuned_configs.get(system_name, {})
                if args.ablations:
                    fallback = system_tuning.get("rgpwm", {})
                elif method in PIVOT_METHODS:
                    fallback = system_tuning.get("pgnn", {})
                else:
                    fallback = {}
                selected = system_tuning.get(method, fallback)
                config = replace(
                    base_config,
                    learning_rate=float(selected.get("learning_rate", base_config.learning_rate)),
                    weight_decay=float(selected.get("weight_decay", base_config.weight_decay)),
                    reliability_weight=float(selected.get("reliability_weight", base_config.reliability_weight)),
                )
                condition_start = time.perf_counter()
                payload = run_condition(
                    method, spec, seed, normalizer, train_windows, validation_windows,
                    tests, config, device, output_dir, deadline, args.overwrite,
                )
                completed.append(payload)
                print(json.dumps({
                    "event": "condition_complete", "system": system_name,
                    "seed": seed, "method": method,
                    "seconds": time.perf_counter() - condition_start,
                    "best_validation_nll": payload["train_result"]["best_validation_nll"],  # type: ignore[index]
                }))
            else:
                continue
            break
        else:
            continue
        break
    elapsed = time.perf_counter() - run_start
    if args.pilot:
        projected = elapsed * len(SYSTEMS) * 3 * (60.0 / epochs)
        print(f"TIME_ESTIMATE: {projected:.1f}s")
        summary_path = Path("results/pilot/pilot_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({
            "elapsed_seconds": elapsed,
            "projected_primary_grid_seconds": projected,
            "within_eight_hour_budget": projected <= 8 * 3600,
            "conditions_completed": len(completed),
        }, indent=2), encoding="utf-8")
    print(json.dumps({"event": "run_complete", "elapsed_seconds": elapsed, "conditions": len(completed)}))


if __name__ == "__main__":
    main()
