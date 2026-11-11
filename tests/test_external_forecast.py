"""Guard observation/target separation and gradient flow in the flight extension."""
from dataclasses import replace

import numpy as np
import torch

from external_validity.data import Record
from external_validity.physics import KinematicThrustPrior
from external_validity.run import ForecastModel, make_windows
from rgpwm.data import Normalizer


def fixture():
    states = np.arange(90 * 12, dtype=float).reshape(90, 12) / 100
    actions = np.arange(90 * 4, dtype=float).reshape(90, 4)
    record = Record('example', 'train', states, actions, np.arange(90) * .01)
    norm = Normalizer(np.zeros(12), np.ones(12), np.zeros(4), np.ones(4))
    return record, norm


def test_control_alignment_and_targets_survive_context_corruption():
    record, norm = fixture()
    clean, origins = make_windows([record], norm, horizon=25, stride=50)
    damaged, damaged_origins = make_windows([record], norm, horizon=25, stride=50,
                                            noise=.1, missing=.3)
    np.testing.assert_array_equal(origins, [7, 57])
    np.testing.assert_array_equal(origins, damaged_origins)
    np.testing.assert_allclose(clean.context_actions[0, 1:], record.actions[:7])
    np.testing.assert_allclose(clean.future_actions[0], record.actions[7:32])
    np.testing.assert_allclose(clean.targets[0], record.states[8:33])
    np.testing.assert_array_equal(clean.targets, damaged.targets)
    np.testing.assert_array_equal(clean.future_actions, damaged.future_actions)
    assert not np.array_equal(clean.context_values, damaged.context_values)


def test_future_state_mutation_cannot_change_context_inputs():
    record, norm = fixture()
    changed = record.states.copy()
    changed[8:] += 999
    first, _ = make_windows([record], norm, noise=.1, missing=.3)
    second, _ = make_windows([replace(record, states=changed)], norm, noise=.1, missing=.3)
    for field in ('context_values', 'context_mask', 'context_delta', 'context_actions'):
        np.testing.assert_array_equal(getattr(first, field)[0], getattr(second, field)[0])
    assert not np.array_equal(first.targets[0], second.targets[0])


def test_learned_comparators_share_capacity_and_have_finite_rollout_gradients():
    _, norm = fixture()
    prior = KinematicThrustPrior(1e-6)
    counts = []
    values = torch.zeros(2, 8, 12)
    mask, delta = torch.ones_like(values), torch.zeros_like(values)
    context_actions, future_actions = torch.zeros(2, 8, 4), torch.zeros(2, 25, 4)
    for method in ('cfr', 'unfiltered', 'recurrent'):
        torch.manual_seed(7)
        model = ForecastModel(method, norm, prior)
        counts.append(sum(p.numel() for p in model.parameters()))
        output = model(values, mask, delta, context_actions, future_actions)
        loss = (output['mean'].square() / output['variance'] + output['variance'].log()).mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert len(set(counts)) == 1
