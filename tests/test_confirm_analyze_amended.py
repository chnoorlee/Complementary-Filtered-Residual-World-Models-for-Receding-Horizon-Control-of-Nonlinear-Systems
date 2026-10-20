from __future__ import annotations

import pandas as pd
import pytest

from rgpwm.confirm_analyze_amended import attach_parameter_counts


def test_parameter_count_amendment_is_many_to_one_metadata_only() -> None:
    aggregate = pd.DataFrame([
        {"method": "a", "seed": 1, "system": "s", "regime": "clean", "metric": 0.1},
        {"method": "a", "seed": 1, "system": "s", "regime": "corrupt", "metric": 0.2},
        {"method": "b", "seed": 1, "system": "s", "regime": "clean", "metric": 0.3},
    ])
    payloads = [
        {"method": "a", "seed": 1, "system": "s", "parameter_count": 100},
        {"method": "b", "seed": 1, "system": "s", "parameter_count": 200},
    ]

    amended = attach_parameter_counts(aggregate, payloads)

    pd.testing.assert_frame_equal(amended[aggregate.columns], aggregate)
    assert amended["parameter_count"].tolist() == [100, 100, 200]


@pytest.mark.parametrize("value", [None, 0, -1, True, 1.5])
def test_parameter_count_amendment_rejects_invalid_values(value: object) -> None:
    aggregate = pd.DataFrame([
        {"method": "a", "seed": 1, "system": "s", "regime": "clean"},
    ])
    payloads = [
        {"method": "a", "seed": 1, "system": "s", "parameter_count": value},
    ]
    with pytest.raises(RuntimeError, match="positive integer"):
        attach_parameter_counts(aggregate, payloads)


def test_parameter_count_amendment_rejects_missing_or_duplicate_keys() -> None:
    aggregate = pd.DataFrame([
        {"method": "a", "seed": 1, "system": "s", "regime": "clean"},
    ])
    missing = [{"method": "b", "seed": 1, "system": "s", "parameter_count": 10}]
    with pytest.raises(RuntimeError, match="join is incomplete"):
        attach_parameter_counts(aggregate, missing)

    duplicated = [
        {"method": "a", "seed": 1, "system": "s", "parameter_count": 10},
        {"method": "a", "seed": 1, "system": "s", "parameter_count": 10},
    ]
    with pytest.raises(RuntimeError, match="duplicated"):
        attach_parameter_counts(aggregate, duplicated)
