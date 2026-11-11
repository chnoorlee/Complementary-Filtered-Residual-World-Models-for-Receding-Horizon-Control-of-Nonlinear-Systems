from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from reviewer_additional_analyses import optimal_scalar_gain, structured_outage
from rgpwm.data import WindowBatch


def sample_windows() -> WindowBatch:
    return WindowBatch(
        context_values=np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3),
        context_mask=np.ones((2, 8, 3), dtype=np.float32),
        context_delta=np.zeros((2, 8, 3), dtype=np.float32),
        context_actions=np.zeros((2, 8, 1), dtype=np.float32),
        future_actions=np.zeros((2, 25, 1), dtype=np.float32),
        targets=np.zeros((2, 25, 3), dtype=np.float32),
        trajectory_ids=np.array(["a", "b"]),
    )


def test_structured_outage_is_causal_and_does_not_mutate_input() -> None:
    source = sample_windows()
    original = source.context_values.copy()
    changed = structured_outage(source, start=4, length=2)
    np.testing.assert_array_equal(source.context_values, original)
    np.testing.assert_array_equal(changed.context_values[:, 4], original[:, 3])
    np.testing.assert_array_equal(changed.context_values[:, 5], original[:, 3])
    np.testing.assert_array_equal(changed.context_mask[:, 4:6], 0.0)
    np.testing.assert_array_equal(changed.context_delta[:, 4], 1.0)
    np.testing.assert_array_equal(changed.context_delta[:, 5], 2.0)
    np.testing.assert_array_equal(changed.context_values[:, :4], original[:, :4])
    np.testing.assert_array_equal(changed.context_values[:, 6:], original[:, 6:])


def test_structured_outage_rejects_invalid_ranges() -> None:
    source = sample_windows()
    with pytest.raises(ValueError):
        structured_outage(source, start=0)
    with pytest.raises(ValueError):
        structured_outage(source, start=7, length=2)


def test_optimal_scalar_gain_matches_uncorrelated_case_and_clips() -> None:
    assert optimal_scalar_gain(4.0, 1.0) == pytest.approx(0.8)
    assert optimal_scalar_gain(1.0, 4.0) == pytest.approx(0.2)
    assert optimal_scalar_gain(1.0, 2.0, covariance=1.2) == 0.0
    with pytest.raises(ValueError):
        optimal_scalar_gain(1.0, 1.0, covariance=1.0)
