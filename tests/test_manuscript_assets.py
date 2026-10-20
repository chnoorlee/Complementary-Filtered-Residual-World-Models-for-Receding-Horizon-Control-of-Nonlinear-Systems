from __future__ import annotations

import numpy as np
import pandas as pd

import make_confirmatory_manuscript_assets as assets


def _summary() -> pd.DataFrame:
    methods = tuple(dict.fromkeys((*assets.MAIN_METHODS, *(name for name, _ in assets.GAINS))))
    return pd.DataFrame([
        {
            "method": method,
            "label": method.upper(),
            "nrmse_h1_mean": 0.1 + 0.005 * index,
            "nrmse_h1_sd": 0.005,
            "nrmse_h10_mean": 0.15 + 0.008 * index,
            "nrmse_h10_sd": 0.008,
            "nrmse_h25_mean": 0.2 + 0.01 * index,
            "nrmse_h25_sd": 0.01,
            "parameter_count_mean": 60_000 + index,
            "gaussian_nll_mean": 0.1 + index * 0.01,
            "calibration_error_mean": 0.05 + index * 0.001,
            "catastrophic_mean": 0.01 + index * 0.001,
        }
        for index, method in enumerate(methods)
    ])


def _trajectories() -> pd.DataFrame:
    rows = []
    for method, offset in (("pgnn", 0.0), ("cfrwm", -0.02)):
        for seed in (41, 53, 67, 79, 83, 97, 109, 113):
            for regime in assets.REGIME_LABELS:
                rows.append({
                    "method": method,
                    "seed": seed,
                    "regime": regime,
                    "nrmse_h25": 0.25 + offset + seed * 1e-5,
                })
    return pd.DataFrame(rows)


def _inference() -> dict[str, object]:
    return {
        "primary_h1": {
            "difference_proposed_minus_baseline": -0.02,
            "ci_95": [-0.03, -0.01],
            "relative_improvement_percent": 8.0,
            "relative_improvement_ci_95": [6.0, 10.0],
            "p_one_sided_relative_margin": 0.02,
            "h1_supported": True,
            "seed_level_relative_improvements_percent": [
                6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
            ],
        },
        "robustness_h2": {"supported": True},
        "cells_worse_than_pgnn_by_more_than_10_percent": 0,
        "all_metrics_finite": True,
        "unique_data_sha256_count": 40,
        "comparator_contrasts": [
            {
                "baseline": method,
                "relative_improvement_percent": float(index + 2),
                "relative_improvement_ci_95": [float(index), float(index + 4)],
                "p_two_sided": 0.01,
                "p_holm": 0.04,
            }
            for index, method in enumerate(assets.METHOD_LABELS)
        ],
        "ablation_contrasts": [
            {
                "baseline": method,
                "relative_improvement_percent": float(index + 1),
                "relative_improvement_ci_95": [float(index - 1), float(index + 3)],
                "p_two_sided": 0.01,
                "p_holm": 0.04,
            }
            for index, method in enumerate(assets.SENSITIVITY_LABELS)
        ],
    }


def test_presentation_assets_render_from_locked_schema(tmp_path, monkeypatch) -> None:
    figure_directory = tmp_path / "figures"
    macro_path = tmp_path / "results_macros.tex"
    table_directory = tmp_path / "tables"
    monkeypatch.setattr(assets, "FIGURES", figure_directory)
    monkeypatch.setattr(assets, "MACROS", macro_path)
    monkeypatch.setattr(assets, "TABLES", table_directory)

    inference = _inference()
    summary = _summary()
    assets._overview(inference, summary, _trajectories())

    cell = pd.DataFrame([
        {
                "system": system,
                "regime": regime,
                "difference_proposed_minus_baseline": -0.01 + value * 0.001,
                "ci_low": -0.02 + value * 0.001,
                "ci_high": value * 0.001,
                "relative_change_cfr_minus_pgnn_percent": value,
        }
        for value, (system, regime) in enumerate(
            (system, regime)
            for system in ("damped_oscillator", "forced_duffing", "controlled_van_der_pol", "torque_pendulum", "cartpole")
            for regime in assets.REGIME_LABELS
        )
    ])
    assets._heatmap(cell)
    inference["robustness_h2"].update({
        "corrupt_and_shifted": {
            "relative_improvement_percent": 9.0,
            "relative_improvement_ci_95": [6.0, 12.0],
            "p_one_sided_relative_margin": 0.02,
        },
        "clean": {
            "relative_improvement_percent": 4.0,
            "relative_improvement_ci_95": [-1.0, 8.0],
            "p_one_sided_relative_margin": 0.01,
        },
    })
    assets._latex_macros(inference, summary, cell)
    table_directory.mkdir(parents=True)
    assets._contrast_table(
        inference["comparator_contrasts"], assets.METHOD_LABELS,
        table_directory / "comparators.tex",
    )
    assets._contrast_table(
        inference["ablation_contrasts"], assets.SENSITIVITY_LABELS,
        table_directory / "sensitivities.tex",
    )
    assets._cell_table(cell, table_directory / "cells.tex")
    assets._secondary_table(summary, table_directory / "secondary.tex")
    assets._seed_effect_table(inference, table_directory / "seed_effects.tex")
    assets._confirmatory_table(summary, table_directory / "confirmatory.tex")

    assert (figure_directory / "confirmatory_overview.png").stat().st_size > 0
    assert (figure_directory / "cell_difference_heatmap.png").stat().st_size > 0
    assert "\\PrimaryImprovement}{8.00}" in macro_path.read_text(encoding="utf-8")
    assert "\\IDCorruptFavoredCount}{0}" in macro_path.read_text(encoding="utf-8")
    assert (table_directory / "comparators.tex").stat().st_size > 0
    assert (table_directory / "sensitivities.tex").stat().st_size > 0
    assert (table_directory / "cells.tex").stat().st_size > 0
    secondary = (table_directory / "secondary.tex").read_text(encoding="utf-8")
    assert "PGNN-style" in secondary
    assert "PG-DMM-style" in secondary
    assert "Seed & CFR-WM" in (
        table_directory / "seed_effects.tex"
    ).read_text(encoding="utf-8")
    confirmatory = (table_directory / "confirmatory.tex").read_text(encoding="utf-8")
    assert "NRMSE-25" in confirmatory
    assert r"\textbf{" in confirmatory
    assert np.isfinite(summary["nrmse_h25_mean"]).all()


def test_tex_escape_protects_table_control_characters() -> None:
    escaped = assets._tex_escape(r"A_B & 50% #1")
    assert escaped == r"A\_B \& 50\% \#1"
