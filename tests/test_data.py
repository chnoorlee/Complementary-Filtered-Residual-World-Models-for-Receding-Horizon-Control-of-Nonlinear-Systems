import numpy as np

from rgpwm.data import (
    build_normalizer,
    corrupt_observations,
    generate_trajectory_split,
    make_windows,
)
from rgpwm.dynamics import SYSTEMS


def test_trajectory_splits_are_disjoint_and_reproducible() -> None:
    spec = SYSTEMS["forced_duffing"]
    split_a = generate_trajectory_split(spec, counts={"train": 6, "val": 3, "test": 4}, steps=45, seed=19)
    split_b = generate_trajectory_split(spec, counts={"train": 6, "val": 3, "test": 4}, steps=45, seed=19)
    ids = [{trajectory.trajectory_id for trajectory in split_a[name]} for name in ("train", "val", "test")]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    np.testing.assert_allclose(split_a["train"][0].states, split_b["train"][0].states)


def test_corruption_retains_mask_and_is_reproducible() -> None:
    states = np.arange(30, dtype=np.float64).reshape(10, 3) / 10.0
    first = corrupt_observations(states, noise_std=0.1, missing_probability=0.35, seed=7)
    second = corrupt_observations(states, noise_std=0.1, missing_probability=0.35, seed=7)
    np.testing.assert_allclose(first.values, second.values)
    np.testing.assert_array_equal(first.mask, second.mask)
    np.testing.assert_allclose(first.delta, second.delta)
    assert set(np.unique(first.mask)).issubset({0.0, 1.0})
    assert np.isfinite(first.values).all()


def test_windows_have_protocol_shapes_and_no_future_observation_input() -> None:
    spec = SYSTEMS["damped_oscillator"]
    split = generate_trajectory_split(spec, counts={"train": 5, "val": 2, "test": 2}, steps=50, seed=3)
    normalizer = build_normalizer(split["train"])
    windows = make_windows(
        split["train"], normalizer, context_steps=8, forecast_steps=25, stride=7,
        noise_std=0.01, missing_probability=0.1, seed=101,
    )
    n = windows.targets.shape[0]
    assert windows.context_values.shape == (n, 8, spec.state_dim)
    assert windows.context_mask.shape == (n, 8, spec.state_dim)
    assert windows.context_delta.shape == (n, 8, spec.state_dim)
    assert windows.context_actions.shape == (n, 8, spec.action_dim)
    assert windows.future_actions.shape == (n, 25, spec.action_dim)
    assert windows.targets.shape == (n, 25, spec.state_dim)
    assert np.isfinite(windows.context_values).all()
    assert np.isfinite(windows.targets).all()

