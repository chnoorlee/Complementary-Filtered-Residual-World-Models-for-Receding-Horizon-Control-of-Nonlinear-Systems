"""Synthetic controlled-trajectory generation and leakage-safe windowing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .dynamics import SystemSpec, rk4_step_numpy, sample_initial_state, sample_parameters


@dataclass
class Trajectory:
    trajectory_id: str
    states: np.ndarray
    actions: np.ndarray
    parameters: dict[str, float]


@dataclass
class CorruptedObservation:
    values: np.ndarray
    mask: np.ndarray
    delta: np.ndarray


@dataclass
class Normalizer:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


@dataclass
class WindowBatch:
    context_values: np.ndarray
    context_mask: np.ndarray
    context_delta: np.ndarray
    context_actions: np.ndarray
    future_actions: np.ndarray
    targets: np.ndarray
    trajectory_ids: np.ndarray


def _sample_actions(spec: SystemSpec, steps: int, rng: np.random.Generator) -> np.ndarray:
    """Generate bounded, temporally correlated excitation without feedback."""
    white = rng.normal(0.0, 0.55 * spec.action_limit, size=(steps - 1, spec.action_dim))
    actions = np.empty_like(white)
    previous = rng.uniform(-0.2, 0.2, size=spec.action_dim) * spec.action_limit
    for index, innovation in enumerate(white):
        previous = 0.82 * previous + 0.18 * innovation
        actions[index] = np.clip(previous, -spec.action_limit, spec.action_limit)
    return actions.astype(np.float64)


def _generate_trajectory(
    spec: SystemSpec,
    steps: int,
    rng: np.random.Generator,
    trajectory_id: str,
    parameter_regime: str,
) -> Trajectory:
    params = sample_parameters(spec, rng, parameter_regime)
    states = np.empty((steps, spec.state_dim), dtype=np.float64)
    states[0] = sample_initial_state(spec, rng)
    actions = _sample_actions(spec, steps, rng)
    for time_index in range(steps - 1):
        states[time_index + 1] = rk4_step_numpy(
            spec, states[time_index], actions[time_index], params, substeps=4
        )
    if not np.isfinite(states).all():
        raise FloatingPointError(f"non-finite trajectory for {spec.name}")
    return Trajectory(trajectory_id, states, actions, params)


def generate_trajectory_split(
    spec: SystemSpec,
    counts: dict[str, int],
    steps: int,
    seed: int,
    parameter_regime: str = "train",
) -> dict[str, list[Trajectory]]:
    """Create disjoint deterministic splits from independently spawned RNGs."""
    if steps < 2:
        raise ValueError("steps must be at least two")
    root = np.random.SeedSequence(seed)
    split_names = tuple(counts)
    split_sequences = root.spawn(len(split_names))
    output: dict[str, list[Trajectory]] = {}
    for split_name, split_sequence in zip(split_names, split_sequences, strict=True):
        trajectory_sequences = split_sequence.spawn(counts[split_name])
        output[split_name] = [
            _generate_trajectory(
                spec,
                steps,
                np.random.default_rng(sequence),
                f"{spec.name}:{parameter_regime}:{split_name}:{index}:seed{seed}",
                parameter_regime,
            )
            for index, sequence in enumerate(trajectory_sequences)
        ]
    return output


def corrupt_observations(
    states: np.ndarray,
    noise_std: float,
    missing_probability: float,
    seed: int,
) -> CorruptedObservation:
    """Apply noise and MCAR missingness, returning last-value-imputed observations.

    The first sample is always observed so imputation is causal and finite.
    ``delta`` counts steps since the latest observation separately per channel.
    """
    if not 0.0 <= missing_probability < 1.0:
        raise ValueError("missing_probability must be in [0, 1)")
    rng = np.random.default_rng(seed)
    states = np.asarray(states, dtype=np.float64)
    noisy = states + rng.normal(0.0, noise_std, states.shape)
    mask = (rng.random(states.shape) >= missing_probability).astype(np.float64)
    mask[0] = 1.0
    values = noisy.copy()
    delta = np.zeros_like(states)
    for channel in range(states.shape[1]):
        last_value = values[0, channel]
        elapsed = 0.0
        for time_index in range(states.shape[0]):
            if mask[time_index, channel] == 1.0:
                last_value = values[time_index, channel]
                elapsed = 0.0
            else:
                values[time_index, channel] = last_value
                elapsed += 1.0
            delta[time_index, channel] = elapsed
    return CorruptedObservation(values, mask, delta)


def build_normalizer(trajectories: Iterable[Trajectory]) -> Normalizer:
    items = list(trajectories)
    if not items:
        raise ValueError("at least one training trajectory is required")
    states = np.concatenate([item.states for item in items], axis=0)
    actions = np.concatenate([item.actions for item in items], axis=0)
    state_std = np.maximum(states.std(axis=0), 1e-6)
    action_std = np.maximum(actions.std(axis=0), 1e-6)
    return Normalizer(states.mean(axis=0), state_std, actions.mean(axis=0), action_std)


def make_windows(
    trajectories: Iterable[Trajectory],
    normalizer: Normalizer,
    context_steps: int,
    forecast_steps: int,
    stride: int,
    noise_std: float,
    missing_probability: float,
    seed: int,
) -> WindowBatch:
    """Build normalized windows; future ground truth appears only in targets."""
    arrays: dict[str, list[np.ndarray]] = {
        "context_values": [], "context_mask": [], "context_delta": [],
        "context_actions": [], "future_actions": [], "targets": [],
    }
    ids: list[str] = []
    root = np.random.SeedSequence(seed)
    items = list(trajectories)
    corrupt_seeds = root.generate_state(len(items), dtype=np.uint32)
    for trajectory, corrupt_seed in zip(items, corrupt_seeds, strict=True):
        corrupt = corrupt_observations(
            trajectory.states,
            noise_std=noise_std,
            missing_probability=missing_probability,
            seed=int(corrupt_seed),
        )
        normalized_values = (corrupt.values - normalizer.state_mean) / normalizer.state_std
        normalized_states = (trajectory.states - normalizer.state_mean) / normalizer.state_std
        normalized_actions = (trajectory.actions - normalizer.action_mean) / normalizer.action_std
        end = trajectory.states.shape[0] - forecast_steps
        for anchor in range(context_steps, end + 1, stride):
            start = anchor - context_steps
            context_actions = np.zeros((context_steps, trajectory.actions.shape[1]), dtype=np.float64)
            if context_steps > 1:
                context_actions[1:] = normalized_actions[start:anchor - 1]
            arrays["context_values"].append(normalized_values[start:anchor])
            arrays["context_mask"].append(corrupt.mask[start:anchor])
            arrays["context_delta"].append(corrupt.delta[start:anchor])
            arrays["context_actions"].append(context_actions)
            arrays["future_actions"].append(normalized_actions[anchor - 1:anchor - 1 + forecast_steps])
            arrays["targets"].append(normalized_states[anchor:anchor + forecast_steps])
            ids.append(trajectory.trajectory_id)
    if not ids:
        raise ValueError("trajectory length is insufficient for the requested windows")
    return WindowBatch(
        **{name: np.stack(value).astype(np.float32) for name, value in arrays.items()},
        trajectory_ids=np.asarray(ids),
    )
