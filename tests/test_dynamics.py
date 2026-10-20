import numpy as np
import pytest

from rgpwm.dynamics import SYSTEMS, rk4_step_numpy, sample_initial_state, sample_parameters


@pytest.mark.parametrize("system_name", sorted(SYSTEMS))
def test_rk4_step_is_finite_deterministic_and_shape_preserving(system_name: str) -> None:
    spec = SYSTEMS[system_name]
    rng = np.random.default_rng(123)
    params = sample_parameters(spec, rng, regime="train")
    state = sample_initial_state(spec, rng)
    action = np.zeros(spec.action_dim, dtype=np.float64)
    first = rk4_step_numpy(spec, state, action, params, substeps=4)
    second = rk4_step_numpy(spec, state, action, params, substeps=4)
    assert first.shape == (spec.state_dim,)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_unforced_damped_oscillator_loses_energy() -> None:
    spec = SYSTEMS["damped_oscillator"]
    params = spec.nominal_params.copy()
    state = np.array([1.0, 0.3], dtype=np.float64)
    action = np.zeros(1, dtype=np.float64)

    def energy(value: np.ndarray) -> float:
        position, velocity = value
        return 0.5 * params["mass"] * velocity**2 + 0.5 * params["stiffness"] * position**2

    initial_energy = energy(state)
    for _ in range(80):
        state = rk4_step_numpy(spec, state, action, params, substeps=4)
    assert energy(state) < initial_energy


def test_pendulum_embedding_remains_on_unit_circle() -> None:
    spec = SYSTEMS["torque_pendulum"]
    rng = np.random.default_rng(5)
    state = sample_initial_state(spec, rng)
    params = sample_parameters(spec, rng, regime="train")
    for _ in range(100):
        state = rk4_step_numpy(spec, state, np.array([0.2]), params, substeps=4)
    assert abs(state[0] ** 2 + state[1] ** 2 - 1.0) < 1e-8


def test_ood_parameters_do_not_overlap_training_ranges() -> None:
    for spec in SYSTEMS.values():
        rng = np.random.default_rng(77)
        train_draws = [sample_parameters(spec, rng, "train") for _ in range(50)]
        ood_draws = [sample_parameters(spec, rng, "ood") for _ in range(50)]
        for key in spec.shifted_param_names:
            train_low, train_high = spec.train_param_ranges[key]
            values = [draw[key] for draw in ood_draws]
            assert all(value < train_low or value > train_high for value in values)
            assert all(train_low <= draw[key] <= train_high for draw in train_draws)

