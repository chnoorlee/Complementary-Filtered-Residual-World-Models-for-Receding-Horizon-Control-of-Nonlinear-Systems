import numpy as np
import torch

from rgpwm.data import Normalizer, WindowBatch
from rgpwm.dynamics import SYSTEMS
from rgpwm.models import build_model
from rgpwm.training import _pooled_filter_gain_metrics, evaluate_model, rollout_loss


def test_rollout_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(4)
    spec = SYSTEMS["forced_duffing"]
    model = build_model(
        "rgpwm", spec,
        state_mean=torch.zeros(spec.state_dim), state_std=torch.ones(spec.state_dim),
        action_mean=torch.zeros(spec.action_dim), action_std=torch.ones(spec.action_dim),
        hidden_dim=16,
    )
    batch, context, horizon = 3, 8, 5
    values = torch.randn(batch, context, spec.state_dim)
    mask = torch.ones_like(values)
    delta = torch.zeros_like(values)
    context_actions = torch.randn(batch, context, spec.action_dim)
    future_actions = torch.randn(batch, horizon, spec.action_dim)
    target = torch.randn(batch, horizon, spec.state_dim)
    output = model(values, mask, delta, context_actions, future_actions)
    loss, parts = rollout_loss(output, target, reliability_weight=0.2)
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["nll"] > 0.0
    assert parts["reliability"] >= 0.0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_one_step_loss_ignores_later_targets() -> None:
    output = {
        "mean": torch.zeros(2, 4, 1),
        "variance": torch.ones(2, 4, 1),
    }
    first = torch.zeros(2, 4, 1)
    second = first.clone()
    second[:, 1:] = 100.0
    loss_a, _ = rollout_loss(output, first, one_step=True)
    loss_b, _ = rollout_loss(output, second, one_step=True)
    torch.testing.assert_close(loss_a, loss_b)


class _ConstantErrorModel(torch.nn.Module):
    def forward(self, values, mask, delta, context_actions, future_actions):
        batch, horizon = values.shape[0], future_actions.shape[1]
        mean = torch.full((batch, horizon, 1), 4.0, device=values.device)
        return {"mean": mean, "variance": torch.ones_like(mean)}

    def normalized_nominal_step(self, state, action):
        return state


def test_evaluate_model_uses_fixed_training_units_for_single_trajectory() -> None:
    windows = WindowBatch(
        context_values=np.zeros((1, 8, 1), dtype=np.float32),
        context_mask=np.ones((1, 8, 1), dtype=np.float32),
        context_delta=np.zeros((1, 8, 1), dtype=np.float32),
        context_actions=np.zeros((1, 8, 1), dtype=np.float32),
        future_actions=np.zeros((1, 25, 1), dtype=np.float32),
        targets=np.zeros((1, 25, 1), dtype=np.float32),
        trajectory_ids=np.asarray(["constant"]),
    )
    normalizer = Normalizer(
        state_mean=np.zeros(1), state_std=np.array([7.0]),
        action_mean=np.zeros(1), action_std=np.ones(1),
    )
    aggregate, rows = evaluate_model(
        _ConstantErrorModel(), windows, normalizer, torch.device("cpu"),
    )
    assert aggregate["nrmse_h1"] == 4.0
    assert rows[0]["nrmse_h1"] == 4.0
    assert aggregate["catastrophic_rollout_rate"] == 1.0
    assert rows[0]["catastrophic"] == 1.0


def test_filter_gain_pooling_uses_observation_counts_and_ignores_empty() -> None:
    # Three trajectories contribute 1, 3, and 0 observed gains respectively.
    # Pooling all observations gives 2/4 near-zero and 1/4 near-one; averaging
    # per-trajectory fractions would incorrectly give 1/3 and 1/6.
    gains = np.asarray([
        [[0.0, 0.4], [0.4, 0.4]],
        [[0.0, 1.0], [0.2, 0.4]],
        [[0.0, 1.0], [0.0, 1.0]],
    ])
    masks = np.asarray([
        [[1, 0], [0, 0]],
        [[1, 1], [1, 0]],
        [[0, 0], [0, 0]],
    ], dtype=bool)

    aggregate, rows = _pooled_filter_gain_metrics(gains, masks)

    assert aggregate["observed_filter_gain_count"] == 4.0
    assert aggregate["mean_observed_filter_gain"] == 0.3
    assert aggregate["observed_filter_gain_near_zero_fraction"] == 0.5
    assert aggregate["observed_filter_gain_near_one_fraction"] == 0.25
    assert rows[2]["observed_filter_gain_count"] == 0.0
    assert rows[2]["filter_gain_near_zero_fraction"] == 0.0
