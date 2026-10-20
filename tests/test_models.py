import pytest
import torch

from rgpwm.dynamics import SYSTEMS
from rgpwm.models import build_model


@pytest.mark.parametrize(
    "model_name",
    [
        "rssm", "simnorm", "transformer", "pgnn", "pgdmm", "rgpwm",
        "rgp_no_rel", "rgp_fixed", "rgp_no_phys", "rgp_1step",
        "pfrwm", "pfr_fixed", "cfrwm", "cfr_gain0", "cfr_gain025",
        "cfr_gain075", "cfr_gain1", "cfr_1step",
    ],
)
def test_model_rollout_contract(model_name: str) -> None:
    torch.manual_seed(0)
    spec = SYSTEMS["forced_duffing"]
    model = build_model(
        model_name,
        spec,
        state_mean=torch.zeros(spec.state_dim),
        state_std=torch.ones(spec.state_dim),
        action_mean=torch.zeros(spec.action_dim),
        action_std=torch.ones(spec.action_dim),
        hidden_dim=32,
    )
    batch, context, horizon = 4, 8, 6
    values = torch.randn(batch, context, spec.state_dim)
    mask = torch.ones_like(values)
    delta = torch.zeros_like(values)
    context_actions = torch.randn(batch, context, spec.action_dim)
    future_actions = torch.randn(batch, horizon, spec.action_dim)
    output = model(values, mask, delta, context_actions, future_actions)
    assert output["mean"].shape == (batch, horizon, spec.state_dim)
    assert output["variance"].shape == (batch, horizon, spec.state_dim)
    assert torch.isfinite(output["mean"]).all()
    assert torch.isfinite(output["variance"]).all()
    assert (output["variance"] > 0).all()
    if model_name.startswith("rgp"):
        assert output["data_weight"].shape == (batch, horizon, 1)
        assert ((output["data_weight"] >= 0) & (output["data_weight"] <= 1)).all()
        assert output["physics_mean"].shape == output["mean"].shape
        assert output["data_mean"].shape == output["mean"].shape
    if model_name.startswith(("pfr", "cfr")):
        assert output["filter_gain"].shape == (batch, context - 1, spec.state_dim)
        assert output["filtered_state"].shape == (batch, spec.state_dim)
        assert ((output["filter_gain"] >= 0) & (output["filter_gain"] <= 1)).all()


@pytest.mark.parametrize(
    ("model_name", "expected_gain"),
    [
        ("cfr_gain0", 0.0), ("cfr_gain025", 0.25), ("cfrwm", 0.5),
        ("cfr_1step", 0.5), ("cfr_gain075", 0.75), ("cfr_gain1", 1.0),
    ],
)
def test_complementary_filter_variants_use_frozen_gain(
    model_name: str,
    expected_gain: float,
) -> None:
    spec = SYSTEMS["damped_oscillator"]
    model = build_model(
        model_name, spec,
        state_mean=torch.zeros(spec.state_dim), state_std=torch.ones(spec.state_dim),
        action_mean=torch.zeros(spec.action_dim), action_std=torch.ones(spec.action_dim),
        hidden_dim=16,
    )
    values = torch.randn(2, 4, spec.state_dim)
    mask = torch.ones_like(values)
    delta = torch.zeros_like(values)
    context_actions = torch.zeros(2, 4, spec.action_dim)
    future_actions = torch.zeros(2, 2, spec.action_dim)
    gain = model(values, mask, delta, context_actions, future_actions)["filter_gain"]
    torch.testing.assert_close(gain, torch.full_like(gain, expected_gain))


def test_physics_filter_ignores_missing_measurements() -> None:
    spec = SYSTEMS["damped_oscillator"]
    model = build_model(
        "pfrwm", spec,
        state_mean=torch.zeros(spec.state_dim), state_std=torch.ones(spec.state_dim),
        action_mean=torch.zeros(spec.action_dim), action_std=torch.ones(spec.action_dim),
        hidden_dim=16,
    )
    with torch.no_grad():
        model.filter_gain_head[-1].weight.normal_(0.0, 0.2)  # type: ignore[index,union-attr]
    values = torch.zeros(1, 4, spec.state_dim)
    values[:, 1:] = 100.0
    mask = torch.zeros_like(values)
    mask[:, 0] = 1.0
    delta = torch.arange(4, dtype=torch.float32).view(1, 4, 1).expand_as(values)
    context_actions = torch.zeros(1, 4, spec.action_dim)
    future_actions = torch.zeros(1, 2, spec.action_dim)
    output = model(values, mask, delta, context_actions, future_actions)
    expected = values[:, 0]
    for step in range(1, 4):
        expected = model.normalized_nominal_step(expected, context_actions[:, step])
    torch.testing.assert_close(output["filtered_state"], expected)


def test_physics_filter_gain_at_step_is_prefix_causal() -> None:
    spec = SYSTEMS["forced_duffing"]
    model = build_model(
        "pfrwm", spec,
        state_mean=torch.zeros(spec.state_dim), state_std=torch.ones(spec.state_dim),
        action_mean=torch.zeros(spec.action_dim), action_std=torch.ones(spec.action_dim),
        hidden_dim=16,
    )
    with torch.no_grad():
        model.filter_gain_head[-1].weight.normal_(0.0, 0.2)  # type: ignore[index,union-attr]
    first = torch.randn(1, 5, spec.state_dim)
    second = first.clone()
    second[:, 2:] += 20.0
    mask = torch.ones_like(first)
    delta = torch.zeros_like(first)
    context_actions = torch.randn(1, 5, spec.action_dim)
    future_actions = torch.randn(1, 2, spec.action_dim)
    gain_first = model(first, mask, delta, context_actions, future_actions)["filter_gain"]
    gain_second = model(second, mask, delta, context_actions, future_actions)["filter_gain"]
    torch.testing.assert_close(gain_first[:, 0], gain_second[:, 0])


def test_pendulum_nominal_step_is_differentiable() -> None:
    spec = SYSTEMS["torque_pendulum"]
    model = build_model(
        "rgpwm",
        spec,
        state_mean=torch.zeros(spec.state_dim),
        state_std=torch.ones(spec.state_dim),
        action_mean=torch.zeros(spec.action_dim),
        action_std=torch.ones(spec.action_dim),
        hidden_dim=16,
    )
    state = torch.tensor([[0.0, 1.0, 0.2]], requires_grad=True)
    action = torch.tensor([[0.1]], requires_grad=True)
    next_state = model.normalized_nominal_step(state, action)
    next_state.sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert action.grad is not None and torch.isfinite(action.grad).all()
