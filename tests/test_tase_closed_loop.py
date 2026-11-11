from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from rgpwm.data import Normalizer
from rgpwm.dynamics import SYSTEMS
from tase_closed_loop import (
    CONTEXT_STEPS,
    ObservationHistory,
    context_tensors,
    episode_randomness,
    exact_sign_flip_p,
    prediction_costs,
    summarize,
    target_state,
)


def normalizer(state_dim: int) -> Normalizer:
    return Normalizer(
        state_mean=np.zeros(state_dim),
        state_std=np.ones(state_dim),
        action_mean=np.zeros(1),
        action_std=np.ones(1),
    )


def test_observation_history_carries_missing_values_and_tracks_elapsed_time() -> None:
    history = ObservationHistory.start(np.array([1.0, 2.0]))
    history.append(
        np.array([3.0, 4.0]),
        np.array([0.5]),
        np.array([0.1, 0.2]),
        np.array([1.0, 0.0]),
    )
    np.testing.assert_allclose(history.values[-1], [3.1, 2.0])
    np.testing.assert_allclose(history.deltas[-1], [0.0, 1.0])
    history.append(
        np.array([5.0, 6.0]),
        np.array([-0.5]),
        np.zeros(2),
        np.array([0.0, 0.0]),
    )
    np.testing.assert_allclose(history.values[-1], [3.1, 2.0])
    np.testing.assert_allclose(history.deltas[-1], [1.0, 2.0])


def test_context_action_alignment_uses_transition_into_each_state() -> None:
    history = ObservationHistory.start(np.array([0.0, 0.0]))
    for index in range(1, CONTEXT_STEPS + 2):
        history.append(
            np.array([index, -index], dtype=float),
            np.array([10.0 + index]),
            np.zeros(2),
            np.ones(2),
        )
    values, _, _, actions = context_tensors(history, normalizer(2), torch.device("cpu"))
    np.testing.assert_allclose(values[0, :, 0].numpy(), np.arange(2, 10))
    np.testing.assert_allclose(actions[0, 0].numpy(), [0.0])
    np.testing.assert_allclose(actions[0, 1:, 0].numpy(), np.arange(13.0, 20.0))


def test_prediction_cost_prefers_target_and_penalizes_terminal_error() -> None:
    spec = SYSTEMS["damped_oscillator"]
    predictions = torch.zeros(2, 8, 2)
    predictions[1, -1] = 1.0
    actions = np.zeros((2, 8, 1))
    costs = prediction_costs(predictions, actions, np.zeros(2), spec)
    assert costs[0] == pytest.approx(0.0)
    assert costs[1] == pytest.approx(3.0)


def test_targets_respect_pendulum_embedding() -> None:
    np.testing.assert_array_equal(target_state(SYSTEMS["torque_pendulum"]), [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(target_state(SYSTEMS["cartpole"]), np.zeros(4))


def test_clean_and_corrupt_id_conditions_share_the_plant_draw() -> None:
    spec = SYSTEMS["forced_duffing"]
    clean = episode_randomness(spec, 41, "id_clean", 2)
    corrupt = episode_randomness(spec, 41, "id_corrupt", 2)
    np.testing.assert_array_equal(clean[0], corrupt[0])
    assert clean[1] == corrupt[1]
    assert not np.array_equal(clean[2], corrupt[2])


def test_sign_flip_and_seed_balanced_summary() -> None:
    assert exact_sign_flip_p(np.ones(3)) == pytest.approx(1.0 / 8.0)
    rows = []
    for seed in range(8):
        for controller, loss in (
            ("nominal_last", 1.2),
            ("complementary_nominal", 1.1),
            ("pgnn", 1.0),
            ("cfrwm", 0.8),
        ):
            rows.append({
                "controller": controller,
                "condition": "id_clean",
                "seed": seed,
                "system": "damped_oscillator",
                "episode": 0,
                "mean_state_loss": loss,
                "mean_action_effort": 0.1,
                "mean_combined_cost": loss + 0.001,
            })
    overall, by_condition, inference = summarize(pd.DataFrame(rows))
    assert len(overall) == 4
    assert len(by_condition) == 4
    assert inference["mean_percent"] == pytest.approx(20.0)
    assert inference["exact_one_sided_sign_flip_p_above_zero"] == pytest.approx(1.0 / 256.0)
