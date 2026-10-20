"""Equal-budget six-trial hyperparameter selection on seed 11 validation data."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .dynamics import SYSTEMS
from .experiment import PRIMARY_METHODS, _model_for, prepare_windows
from .training import TrainConfig, seed_everything, train_model


TRIALS = (
    {"learning_rate": 3e-4, "weight_decay": 1e-6, "reliability_weight": 0.1},
    {"learning_rate": 6e-4, "weight_decay": 1e-5, "reliability_weight": 0.2},
    {"learning_rate": 1e-3, "weight_decay": 1e-5, "reliability_weight": 0.2},
    {"learning_rate": 1.5e-3, "weight_decay": 1e-4, "reliability_weight": 0.3},
    {"learning_rate": 2e-3, "weight_decay": 1e-5, "reliability_weight": 0.4},
    {"learning_rate": 3e-3, "weight_decay": 1e-4, "reliability_weight": 0.2},
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", default="results/tuning")
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "trials.jsonl"
    best_path = output_dir / "best_configs.json"
    previous: dict[tuple[str, str, int], dict[str, object]] = {}
    if raw_path.exists() and not args.overwrite:
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            previous[(row["system"], row["method"], row["trial"])] = row
    if args.overwrite:
        raw_path.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best: dict[str, dict[str, object]] = {}
    start = time.perf_counter()
    for system_name, spec in SYSTEMS.items():
        normalizer, train_windows, validation_windows, _ = prepare_windows(spec, 11)
        best[system_name] = {}
        for method in PRIMARY_METHODS:
            rows: list[dict[str, object]] = []
            for trial_index, trial in enumerate(TRIALS):
                key = (system_name, method, trial_index)
                if key in previous:
                    row = previous[key]
                else:
                    initialization_seed = 11000 + trial_index
                    seed_everything(initialization_seed)
                    model = _model_for(method, spec, normalizer)
                    config = TrainConfig(
                        epochs=args.epochs,
                        patience=4,
                        learning_rate=trial["learning_rate"],
                        weight_decay=trial["weight_decay"],
                        reliability_weight=trial["reliability_weight"],
                    )
                    train_result = train_model(
                        model, train_windows, validation_windows, config, device,
                        initialization_seed,
                    )
                    row = {
                        "system": system_name,
                        "method": method,
                        "trial": trial_index,
                        **trial,
                        "best_validation_nll": train_result.best_validation_nll,
                        "best_epoch": train_result.best_epoch,
                        "epochs_completed": train_result.epochs_completed,
                        "training_seconds": train_result.training_seconds,
                    }
                    with raw_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
                print(json.dumps({"event": "tuning_trial", **row}))
            winning = min(rows, key=lambda item: item["best_validation_nll"])
            best[system_name][method] = {
                "trial": winning["trial"],
                "learning_rate": winning["learning_rate"],
                "weight_decay": winning["weight_decay"],
                "reliability_weight": winning["reliability_weight"],
                "validation_nll": winning["best_validation_nll"],
            }
            best_path.write_text(json.dumps(best, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "event": "tuning_complete", "seconds": time.perf_counter() - start,
        "trials": len(SYSTEMS) * len(PRIMARY_METHODS) * len(TRIALS),
        "best_configs": str(best_path),
    }))


if __name__ == "__main__":
    main()
