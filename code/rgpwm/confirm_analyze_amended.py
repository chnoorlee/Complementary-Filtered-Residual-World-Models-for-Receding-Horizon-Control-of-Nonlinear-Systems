"""Versioned analysis-only recovery for the frozen CFR-WM run grid.

The original analyzer failed before inferential computation because the generic
loader did not copy the already-recorded top-level ``parameter_count`` field
into its aggregate table.  This entry point leaves the frozen run payloads,
estimands, statistics, random seeds, and original analysis modules unchanged.
It performs one validated many-to-one metadata join before calling the original
``analyze_confirmatory`` function.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .analyze import load_run_directory
from .confirm_analyze import (
    _validate_payload_data_pairing,
    _write_latex_summary,
    analyze_confirmatory,
)
from .confirm_protocol import (
    CONFIRM_METHODS,
    CONFIRM_SEEDS,
    analysis_code_sha256,
)
from .confirm_run import (
    CONTRACT_PATH,
    _sha256,
    _validate_contract,
    _validate_existing_output,
)
from .dynamics import SYSTEMS


AMENDMENT_PATH = Path("docs/14_cfr_analysis_amendment.json")
FAILED_STDERR_PATH = Path("logs/cfr_confirmatory_analysis.stderr.log")
FAILED_STDOUT_PATH = Path("logs/cfr_confirmatory_analysis.stdout.log")
RUN_MANIFEST_PATH = Path("results/confirmatory_run_manifest.json")
OUTPUT_PATH = Path("results/confirmatory_analysis")
TEMPORARY_OUTPUT_PATH = Path("results/confirmatory_analysis.incomplete")


def amendment_code_sha256() -> str:
    return _sha256(Path(__file__))


def attach_parameter_counts(
    aggregate: pd.DataFrame,
    payloads: list[dict[str, object]],
) -> pd.DataFrame:
    """Join descriptive parameter counts without touching any outcome column."""
    keys = ["method", "seed", "system"]
    if "parameter_count" in aggregate.columns:
        raise RuntimeError("analysis amendment expected parameter_count to be absent")
    rows: list[dict[str, object]] = []
    for payload in payloads:
        value = payload.get("parameter_count")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError("every payload must contain a positive integer parameter_count")
        rows.append({
            "method": str(payload["method"]),
            "seed": int(payload["seed"]),
            "system": str(payload["system"]),
            "parameter_count": value,
        })
    counts = pd.DataFrame(rows)
    if counts.empty or counts.duplicated(keys).any():
        raise RuntimeError("parameter-count payload keys are empty or duplicated")
    original_columns = list(aggregate.columns)
    joined = aggregate.merge(counts, on=keys, how="left", validate="many_to_one")
    if len(joined) != len(aggregate) or joined["parameter_count"].isna().any():
        raise RuntimeError("parameter-count join is incomplete or changed aggregate row count")
    if list(joined.columns[:-1]) != original_columns:
        raise RuntimeError("parameter-count join changed existing aggregate columns")
    return joined


def validate_analysis_amendment() -> dict[str, Any]:
    if not AMENDMENT_PATH.is_file():
        raise RuntimeError(f"analysis amendment is missing: {AMENDMENT_PATH}")
    amendment: dict[str, Any] = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    required = {
        "amendment_version": "cfr-confirmatory-analysis-amendment-v1",
        "status": "frozen_before_amended_analysis",
        "original_contract_path": CONTRACT_PATH.as_posix(),
        "original_contract_sha256": _sha256(CONTRACT_PATH),
        "completed_run_manifest_sha256": _sha256(RUN_MANIFEST_PATH),
        "original_analysis_code_sha256": analysis_code_sha256(),
        "amendment_code_sha256": amendment_code_sha256(),
        "failed_stdout_sha256": _sha256(FAILED_STDOUT_PATH),
        "failed_stderr_sha256": _sha256(FAILED_STDERR_PATH),
        "inference_change": "none",
        "joined_field": "parameter_count",
        "join_keys": ["method", "seed", "system"],
    }
    mismatches = {
        key: {"expected": expected, "found": amendment.get(key)}
        for key, expected in required.items()
        if amendment.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"analysis amendment mismatch: {mismatches}")
    return amendment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen CFR-WM statistics with one versioned metadata-join amendment.",
    )
    parser.parse_args()
    if OUTPUT_PATH.exists() or TEMPORARY_OUTPUT_PATH.exists():
        raise RuntimeError("refusing to overwrite an existing or incomplete confirmatory analysis")

    amendment = validate_analysis_amendment()
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    tuning = _validate_contract(contract)
    contract_hash = _sha256(CONTRACT_PATH)
    provenance: dict[str, object] = {
        "confirmatory_contract_sha256": contract_hash,
        "tuning_file_sha256": contract["training"]["tuning_file_sha256"],
        "analysis_code_sha256": contract["hashes"]["analysis_code_sha256"],
    }
    directory = Path(contract["training"]["output_directory"])
    validated_payloads = _validate_existing_output(
        directory, contract, tuning, provenance,
    )
    expected_runs = len(CONFIRM_METHODS) * len(CONFIRM_SEEDS) * len(SYSTEMS)
    if len(validated_payloads) != expected_runs:
        raise RuntimeError(
            f"expected {expected_runs} paired confirmatory payloads, "
            f"found {len(validated_payloads)}"
        )
    aggregate, trajectories, payloads = load_run_directory(directory)
    if len(payloads) != expected_runs:
        raise RuntimeError(f"expected {expected_runs} confirmatory runs, found {len(payloads)}")
    reloaded_payloads = {
        (str(payload["method"]), int(payload["seed"]), str(payload["system"])): payload
        for payload in payloads
    }
    if reloaded_payloads != validated_payloads:
        raise RuntimeError("validated and analysis-loaded payloads differ")
    unique_data_count = _validate_payload_data_pairing(payloads)
    aggregate = attach_parameter_counts(aggregate, payloads)
    decision, summary, cell = analyze_confirmatory(
        aggregate, trajectories, int(contract["statistics"]["bootstrap_replicates"]),
    )
    decision.update({
        "analysis_code_sha256": analysis_code_sha256(),
        "training_code_sha256": contract["hashes"]["training_code_sha256"],
        "evaluation_code_sha256": contract["hashes"]["training_code_sha256"],
        "confirmatory_contract_sha256": contract_hash,
        "tuning_file_sha256": contract["training"]["tuning_file_sha256"],
        "unique_data_sha256_count": unique_data_count,
        "analysis_amendment_version": amendment["amendment_version"],
        "analysis_amendment_contract_sha256": _sha256(AMENDMENT_PATH),
        "analysis_amendment_code_sha256": amendment_code_sha256(),
        "analysis_amendment_inference_change": "none",
    })

    TEMPORARY_OUTPUT_PATH.mkdir(parents=True)
    summary.to_csv(TEMPORARY_OUTPUT_PATH / "confirmatory_summary.csv", index=False)
    aggregate.to_csv(TEMPORARY_OUTPUT_PATH / "aggregate_cells.csv", index=False)
    trajectories.to_csv(TEMPORARY_OUTPUT_PATH / "trajectory_metrics.csv", index=False)
    cell.to_csv(TEMPORARY_OUTPUT_PATH / "cell_contrasts.csv", index=False)
    (TEMPORARY_OUTPUT_PATH / "confirmatory_inference.json").write_text(
        json.dumps(decision, indent=2, allow_nan=False), encoding="utf-8",
    )
    _write_latex_summary(summary, TEMPORARY_OUTPUT_PATH / "table_confirmatory.tex")
    TEMPORARY_OUTPUT_PATH.replace(OUTPUT_PATH)
    print(json.dumps(decision, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
