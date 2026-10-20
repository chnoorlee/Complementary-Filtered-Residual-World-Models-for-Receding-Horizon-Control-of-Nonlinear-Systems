"""Compact, reproducible world-model baselines and the proposed RGP-WM."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .dynamics import SystemSpec, rk4_step_torch


FIXED_FILTER_GAINS = {
    "pfr_fixed": 0.5,
    "cfrwm": 0.5,
    "cfr_gain0": 0.0,
    "cfr_gain025": 0.25,
    "cfr_gain075": 0.75,
    "cfr_gain1": 1.0,
    "cfr_1step": 0.5,
}


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class WorldModelBase(nn.Module):
    """Shared causal encoder, normalized physics step, and rollout utilities."""

    def __init__(
        self,
        spec: SystemSpec,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.state_dim = spec.state_dim
        self.action_dim = spec.action_dim
        self.hidden_dim = hidden_dim
        self.register_buffer("state_mean", state_mean.float().clone())
        self.register_buffer("state_std", state_std.float().clone().clamp_min(1e-6))
        self.register_buffer("action_mean", action_mean.float().clone())
        self.register_buffer("action_std", action_std.float().clone().clamp_min(1e-6))
        encoder_dim = 3 * self.state_dim + self.action_dim
        self.encoder = nn.GRU(encoder_dim, hidden_dim, batch_first=True)
        self.rollout_cell = nn.GRUCell(self.state_dim + self.action_dim, hidden_dim)

    def encode_context(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat((values, mask, torch.log1p(delta.clamp_min(0.0)), actions), dim=-1)
        _, hidden = self.encoder(features)
        return hidden[-1]

    def normalized_nominal_step(self, normalized_state: torch.Tensor, normalized_action: torch.Tensor) -> torch.Tensor:
        state = normalized_state * self.state_std + self.state_mean
        action = normalized_action * self.action_std + self.action_mean
        next_state = rk4_step_torch(self.spec, state, action, substeps=1)
        return (next_state - self.state_mean) / self.state_std

    @staticmethod
    def _variance(raw_log_variance: torch.Tensor) -> torch.Tensor:
        return torch.exp(raw_log_variance.clamp(-8.0, 4.0)).clamp_min(1e-6)

    def _update_hidden(self, state: torch.Tensor, action: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.rollout_cell(torch.cat((state, action), dim=-1), hidden)


class LearnedWorldModel(WorldModelBase):
    def __init__(self, *args: Any, variant: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.variant = variant
        head_input = self.hidden_dim + self.state_dim + self.action_dim
        self.transition = _mlp(head_input, self.hidden_dim, 2 * self.state_dim)
        if variant == "rssm":
            self.latent_scale = nn.Linear(self.hidden_dim, self.hidden_dim)
        elif variant == "simnorm":
            self.simplex_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

    def _adapt_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.variant == "rssm":
            scale = torch.sigmoid(self.latent_scale(hidden))
            return hidden * scale
        if self.variant == "simnorm":
            projected = self.simplex_projection(hidden)
            groups = 8 if self.hidden_dim % 8 == 0 else 1
            shaped = projected.view(projected.shape[0], groups, self.hidden_dim // groups)
            return torch.softmax(shaped, dim=-1).reshape_as(projected)
        return hidden

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        context_actions: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode_context(values, mask, delta, context_actions)
        state = values[:, -1]
        means, variances = [], []
        for action in future_actions.unbind(1):
            adapted = self._adapt_hidden(hidden)
            raw = self.transition(torch.cat((state, action, adapted), dim=-1))
            change, raw_log_variance = raw.chunk(2, dim=-1)
            state = state + 0.25 * torch.tanh(change)
            variance = self._variance(raw_log_variance)
            hidden = self._update_hidden(state, action, hidden)
            means.append(state)
            variances.append(variance)
        return {"mean": torch.stack(means, 1), "variance": torch.stack(variances, 1)}


class TransformerWorldModel(WorldModelBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        encoder_dim = 3 * self.state_dim + self.action_dim
        self.input_projection = nn.Linear(encoder_dim, self.hidden_dim)
        heads = 4 if self.hidden_dim % 4 == 0 else 1
        layer = nn.TransformerEncoderLayer(
            self.hidden_dim, heads, 2 * self.hidden_dim, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        head_input = self.hidden_dim + self.state_dim + self.action_dim
        self.transition = _mlp(head_input, self.hidden_dim, 2 * self.state_dim)

    def encode_context(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat((values, mask, torch.log1p(delta.clamp_min(0.0)), actions), dim=-1)
        tokens = self.input_projection(features)
        length = tokens.shape[1]
        causal_mask = torch.triu(torch.full((length, length), float("-inf"), device=tokens.device), diagonal=1)
        return self.transformer(tokens, mask=causal_mask)[:, -1]

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        context_actions: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode_context(values, mask, delta, context_actions)
        state = values[:, -1]
        means, variances = [], []
        for action in future_actions.unbind(1):
            raw = self.transition(torch.cat((state, action, hidden), dim=-1))
            change, raw_log_variance = raw.chunk(2, dim=-1)
            state = state + 0.25 * torch.tanh(change)
            variance = self._variance(raw_log_variance)
            hidden = self._update_hidden(state, action, hidden)
            means.append(state)
            variances.append(variance)
        return {"mean": torch.stack(means, 1), "variance": torch.stack(variances, 1)}


class PhysicsGuidedWorldModel(WorldModelBase):
    def __init__(self, *args: Any, probabilistic: bool, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.probabilistic = probabilistic
        head_input = self.hidden_dim + self.state_dim + self.action_dim
        output_dim = 2 * self.state_dim if probabilistic else self.state_dim
        self.residual = _mlp(head_input, self.hidden_dim, output_dim)
        if not probabilistic:
            self.log_variance = nn.Parameter(torch.full((self.state_dim,), -2.0))

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        context_actions: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode_context(values, mask, delta, context_actions)
        state = values[:, -1]
        means, variances = [], []
        for action in future_actions.unbind(1):
            physics = self.normalized_nominal_step(state, action)
            raw = self.residual(torch.cat((state, action, hidden), dim=-1))
            if self.probabilistic:
                correction, raw_log_variance = raw.chunk(2, dim=-1)
                variance = self._variance(raw_log_variance)
            else:
                correction = raw
                variance = self._variance(self.log_variance).expand_as(state)
            state = physics + 0.25 * torch.tanh(correction)
            hidden = self._update_hidden(state, action, hidden)
            means.append(state)
            variances.append(variance)
        return {"mean": torch.stack(means, 1), "variance": torch.stack(variances, 1)}


class PhysicsFilteredResidualWorldModel(WorldModelBase):
    """Physics-predict/measurement-update context filter plus residual rollout.

    The filter uses only observations available before the forecast origin.
    Missing channels have exactly zero measurement gain, while observed
    channels receive either a learned feature-dependent gain or a gain fixed to
    the prespecified value 0.5 in the matched ablation.
    """

    def __init__(self, *args: Any, fixed_filter: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fixed_filter = fixed_filter
        filter_input = self.hidden_dim + 6 * self.state_dim + self.action_dim
        self.filter_gain_head = _mlp(filter_input, self.hidden_dim, self.state_dim)
        final = self.filter_gain_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, 1.5)
        self.register_buffer("fixed_filter_gain", torch.tensor(0.5))
        rollout_input = self.hidden_dim + self.state_dim + self.action_dim
        self.residual = _mlp(rollout_input, self.hidden_dim, 2 * self.state_dim)

    def set_fixed_filter_gain(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("fixed filter gain must be in [0, 1]")
        self.fixed_filter_gain.fill_(value)

    def _project_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        if self.spec.name != "torque_pendulum":
            return normalized_state
        physical = normalized_state * self.state_std + self.state_mean
        unit = physical[..., :2] / physical[..., :2].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        physical = torch.cat((unit, physical[..., 2:]), dim=-1)
        return (physical - self.state_mean) / self.state_std

    def _filter_context(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        actions: torch.Tensor,
        context_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = values[:, 0]
        gains: list[torch.Tensor] = []
        for index in range(1, values.shape[1]):
            action = actions[:, index]
            predicted = self.normalized_nominal_step(state, action)
            innovation = values[:, index] - predicted
            if self.fixed_filter:
                gain = torch.full_like(predicted, float(self.fixed_filter_gain))
            else:
                features = torch.cat((
                    predicted,
                    values[:, index],
                    innovation,
                    innovation.abs(),
                    mask[:, index],
                    action,
                    torch.log1p(delta[:, index].clamp_min(0.0)),
                    context_hidden[:, index],
                ), dim=-1)
                gain = torch.sigmoid(self.filter_gain_head(features))
            gain = gain * mask[:, index]
            state = self._project_state(predicted + gain * innovation)
            gains.append(gain)
        return state, torch.stack(gains, dim=1)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        context_actions: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoder_features = torch.cat((
            values,
            mask,
            torch.log1p(delta.clamp_min(0.0)),
            context_actions,
        ), dim=-1)
        context_hidden, _ = self.encoder(encoder_features)
        hidden = context_hidden[:, -1]
        state, filter_gain = self._filter_context(
            values, mask, delta, context_actions, context_hidden,
        )
        filtered_state = state
        means, variances = [], []
        for action in future_actions.unbind(1):
            physics = self.normalized_nominal_step(state, action)
            raw = self.residual(torch.cat((state, action, hidden), dim=-1))
            correction, raw_log_variance = raw.chunk(2, dim=-1)
            state = self._project_state(physics + 0.25 * torch.tanh(correction))
            variance = self._variance(raw_log_variance)
            hidden = self._update_hidden(state, action, hidden)
            means.append(state)
            variances.append(variance)
        return {
            "mean": torch.stack(means, 1),
            "variance": torch.stack(variances, 1),
            "filter_gain": filter_gain,
            "filtered_state": filtered_state,
        }


class ReliabilityGatedPhysicsResidualWorldModel(WorldModelBase):
    def __init__(self, *args: Any, ablation: str = "none", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ablation = ablation
        self.reliability_supervision = ablation != "no_rel"
        head_input = self.hidden_dim + self.state_dim + self.action_dim
        self.physics_fallback = _mlp(head_input, self.hidden_dim, self.state_dim)
        self.residual = _mlp(head_input, self.hidden_dim, self.state_dim)
        self.branch_uncertainty = _mlp(head_input, self.hidden_dim, 2 * self.state_dim)
        self.reliability = _mlp(head_input, self.hidden_dim, 2)
        self.temperature_raw = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("fixed_weight", torch.tensor(0.5))

    def set_fixed_weight(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("fixed gate weight must be in [0, 1]")
        self.fixed_weight.fill_(value)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        context_actions: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode_context(values, mask, delta, context_actions)
        state = values[:, -1]
        outputs: dict[str, list[torch.Tensor]] = {
            "mean": [], "variance": [], "data_weight": [],
            "physics_mean": [], "data_mean": [],
            "physics_reliability": [], "data_reliability": [],
        }
        for action in future_actions.unbind(1):
            features = torch.cat((state, action, hidden), dim=-1)
            nominal = self.normalized_nominal_step(state, action)
            if self.ablation == "no_phys":
                physics = state + 0.25 * torch.tanh(self.physics_fallback(features))
            else:
                physics = nominal
            data_mean = physics + 0.25 * torch.tanh(self.residual(features))
            physics_logvar, data_logvar = self.branch_uncertainty(features).chunk(2, dim=-1)
            physics_variance = self._variance(physics_logvar)
            data_variance = self._variance(data_logvar)
            reliabilities = F.softplus(self.reliability(features)) + 1e-5
            physics_error = reliabilities[:, :1]
            data_error = reliabilities[:, 1:]
            if self.ablation == "fixed":
                data_weight = torch.full_like(data_error, float(self.fixed_weight))
            else:
                temperature = F.softplus(self.temperature_raw) + 0.1
                logits = torch.cat((-physics_error / temperature, -data_error / temperature), dim=-1)
                data_weight = torch.softmax(logits, dim=-1)[:, 1:]
            mean = physics + data_weight * (data_mean - physics)
            between_branch = data_weight * (1.0 - data_weight) * (data_mean - physics).square()
            variance = (
                (1.0 - data_weight) * physics_variance
                + data_weight * data_variance
                + between_branch
            ).clamp_min(1e-6)
            state = mean
            hidden = self._update_hidden(state, action, hidden)
            outputs["mean"].append(mean)
            outputs["variance"].append(variance)
            outputs["data_weight"].append(data_weight)
            outputs["physics_mean"].append(physics)
            outputs["data_mean"].append(data_mean)
            outputs["physics_reliability"].append(physics_error)
            outputs["data_reliability"].append(data_error)
        return {name: torch.stack(items, dim=1) for name, items in outputs.items()}


def build_model(
    model_name: str,
    spec: SystemSpec,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    hidden_dim: int = 64,
) -> WorldModelBase:
    common = dict(
        spec=spec,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        hidden_dim=hidden_dim,
    )
    if model_name in {"rssm", "simnorm"}:
        return LearnedWorldModel(**common, variant=model_name)
    if model_name == "transformer":
        return TransformerWorldModel(**common)
    if model_name == "pgnn":
        return PhysicsGuidedWorldModel(**common, probabilistic=False)
    if model_name == "pgdmm":
        return PhysicsGuidedWorldModel(**common, probabilistic=True)
    if model_name == "pfrwm" or model_name in FIXED_FILTER_GAINS:
        model = PhysicsFilteredResidualWorldModel(
            **common, fixed_filter=model_name in FIXED_FILTER_GAINS,
        )
        if model_name in FIXED_FILTER_GAINS:
            model.set_fixed_filter_gain(FIXED_FILTER_GAINS[model_name])
        return model
    ablations = {
        "rgpwm": "none",
        "rgp_no_rel": "no_rel",
        "rgp_fixed": "fixed",
        "rgp_no_phys": "no_phys",
        "rgp_1step": "none",
    }
    if model_name in ablations:
        return ReliabilityGatedPhysicsResidualWorldModel(**common, ablation=ablations[model_name])
    raise ValueError(f"unknown model: {model_name}")
