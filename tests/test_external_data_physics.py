from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from external_validity.data import Record, expected_records, fit_normalizer
from external_validity.physics import KinematicThrustPrior, rotvec_to_quaternion


def _record(split="train", coefficient=2e-6):
    dt = 0.01
    actions = np.full((10, 4), 1200.0)
    states = np.zeros((10, 12))
    acceleration = coefficient * np.square(actions[0]).sum() - 9.81
    states[:, 5] = np.arange(10) * dt * acceleration
    return Record("train_example", split, states, actions, np.arange(10) * dt, dt)


def test_external_split_has_disjoint_complete_runs():
    inventory = expected_records()
    ids = {split: {record for actual, record, _ in inventory if actual == split} for split in ("train", "val", "test")}
    assert [len(ids[split]) for split in ("train", "val", "test")] == [9, 3, 3]
    assert ids["train"].isdisjoint(ids["val"] | ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    assert all("run4" in item for item in ids["val"])
    assert all(item.startswith("melon") for item in ids["test"])


def test_training_only_normalizer_and_prior_reject_heldout_records():
    train = _record()
    normalizer = fit_normalizer([train])
    np.testing.assert_allclose(normalizer.state_mean, train.states.mean(axis=0))
    prior = KinematicThrustPrior.fit([train])
    assert float(prior.thrust_coefficient) == pytest.approx(2e-6)
    assert prior.fit_metadata["training_record_ids"] == [train.record_id]
    for split in ("val", "test"):
        with pytest.raises(ValueError, match="training records only"):
            fit_normalizer([train, replace(train, split=split, record_id=split)])
        with pytest.raises(ValueError, match="training records only"):
            KinematicThrustPrior.fit([train, replace(train, split=split, record_id=split)])


def test_hover_and_constant_velocity_are_preserved():
    coefficient = 2e-6
    prior = KinematicThrustPrior(coefficient)
    state = torch.zeros(2, 12, dtype=torch.float64)
    state[1, :3] = torch.tensor([1.0, 2.0, 3.0])
    state[1, 3:6] = torch.tensor([0.2, -0.3, 0.5])
    action = torch.full((2, 4), np.sqrt(9.81 / (4.0 * coefficient)), dtype=torch.float64)
    expected = state.clone()
    expected[:, :3] += 0.01 * expected[:, 3:6]
    torch.testing.assert_close(prior(state, action), expected, rtol=1e-12, atol=1e-12)


def test_ballistic_position_has_half_acceleration_step():
    prior = KinematicThrustPrior(0.0, dt=0.1)
    state = torch.zeros(1, 12, dtype=torch.float64)
    actual = prior(state, torch.zeros(1, 4, dtype=torch.float64))
    assert actual[0, 2].item() == pytest.approx(-0.5 * 9.81 * 0.1**2)
    assert actual[0, 5].item() == pytest.approx(-9.81 * 0.1)


@pytest.mark.parametrize("magnitude", [0.0, 1e-12, 1e-6, 0.2])
def test_rotation_update_matches_body_composition_with_finite_gradients(magnitude):
    state = torch.zeros(1, 12, dtype=torch.float64)
    state[:, 6:9] = torch.tensor([magnitude, -0.4 * magnitude, 0.3 * magnitude])
    state[:, 9:12] = torch.tensor([0.2 * magnitude, -magnitude, 0.4 * magnitude])
    state.requires_grad_()
    actions = torch.full((1, 4), 1000.0, dtype=torch.float64, requires_grad=True)
    prior = KinematicThrustPrior(2e-6)
    actual = prior(state, actions)
    expected = (Rotation.from_rotvec(state.detach().numpy()[0, 6:9]) * Rotation.from_rotvec(0.01 * state.detach().numpy()[0, 9:12])).as_rotvec()
    np.testing.assert_allclose(actual.detach().numpy()[0, 6:9], expected, rtol=1e-10, atol=1e-12)
    actual.square().sum().backward()
    assert torch.isfinite(state.grad).all()
    assert torch.isfinite(actions.grad).all()


def test_quaternion_conversion_matches_scipy():
    vectors = torch.tensor([[0.0, 0.0, 0.0], [0.5, -0.7, 1.2]], dtype=torch.float64)
    expected = Rotation.from_rotvec(vectors.numpy()).as_quat()
    np.testing.assert_allclose(rotvec_to_quaternion(vectors).numpy(), expected, atol=1e-12)
