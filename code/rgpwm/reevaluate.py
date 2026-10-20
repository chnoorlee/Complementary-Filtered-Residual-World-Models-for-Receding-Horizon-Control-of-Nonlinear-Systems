"""Recompute all test metrics from saved checkpoints after metric-code audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .dynamics import SYSTEMS
from .experiment import _model_for, prepare_windows
from .metrics import EVALUATION_METRIC_VERSION, evaluation_code_sha256
from .training import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    directory = Path(args.directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cached_data = {}
    paths = sorted(directory.glob("*/seed_*/*.json"))
    for index, result_path in enumerate(paths, 1):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        key = (payload["system"], int(payload["seed"]))
        if key not in cached_data:
            normalizer, _, _, tests = prepare_windows(SYSTEMS[key[0]], key[1])
            cached_data[key] = (normalizer, tests)
        normalizer, tests = cached_data[key]
        model = _model_for(payload["method"], SYSTEMS[key[0]], normalizer)
        checkpoint = torch.load(result_path.with_suffix(".pt"), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        evaluations = {}
        for regime, windows in tests.items():
            aggregate, trajectories = evaluate_model(model, windows, normalizer, device)
            evaluations[regime] = {"aggregate": aggregate, "trajectories": trajectories}
        payload["evaluations"] = evaluations
        payload["evaluation_metric_version"] = EVALUATION_METRIC_VERSION
        payload["evaluation_code_sha256"] = evaluation_code_sha256()
        payload["evaluation_code_audit"] = (
            "recomputed_with_fixed_training_scale_nrmse_and_catastrophic_rate"
        )
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(result_path)
        print(json.dumps({"event": "reevaluated", "index": index, "total": len(paths), "path": str(result_path)}))


if __name__ == "__main__":
    main()
