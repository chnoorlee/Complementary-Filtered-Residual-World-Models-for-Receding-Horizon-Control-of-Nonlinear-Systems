"""Controlled nonlinear systems used in the benchmark.

The ground-truth simulator uses sampled parameters whereas learned models only
receive the nominal parameterization.  This separation creates a controlled,
auditable parameter-shift benchmark without exposing test parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class SystemSpec:
    name: str
    state_dim: int
    action_dim: int
    dt: float
    nominal_params: dict[str, float]
    train_param_ranges: dict[str, tuple[float, float]]
    ood_param_ranges: dict[str, tuple[float, float]]
    shifted_param_names: tuple[str, ...]
    action_limit: float


SYSTEMS: dict[str, SystemSpec] = {
    "damped_oscillator": SystemSpec(
        "damped_oscillator", 2, 1, 0.05,
        {"mass": 1.0, "stiffness": 1.6, "damping": 0.18},
        {"mass": (0.9, 1.1), "stiffness": (1.4, 1.8), "damping": (0.14, 0.22)},
        {"mass": (1.25, 1.45), "stiffness": (2.05, 2.35), "damping": (0.26, 0.34)},
        ("mass", "stiffness", "damping"), 1.5,
    ),
    "forced_duffing": SystemSpec(
        "forced_duffing", 2, 1, 0.04,
        {"alpha": -1.0, "beta": 1.0, "damping": 0.25},
        {"alpha": (-1.1, -0.9), "beta": (0.9, 1.1), "damping": (0.20, 0.30)},
        {"alpha": (-0.75, -0.60), "beta": (1.25, 1.45), "damping": (0.34, 0.44)},
        ("alpha", "beta", "damping"), 1.8,
    ),
    "controlled_van_der_pol": SystemSpec(
        "controlled_van_der_pol", 2, 1, 0.035,
        {"mu": 1.2, "gain": 1.0},
        {"mu": (1.0, 1.4), "gain": (0.9, 1.1)},
        {"mu": (1.65, 1.95), "gain": (1.25, 1.45)},
        ("mu", "gain"), 1.2,
    ),
    "torque_pendulum": SystemSpec(
        "torque_pendulum", 3, 1, 0.04,
        {"gravity": 9.81, "length": 1.0, "mass": 1.0, "damping": 0.12},
        {"gravity": (9.65, 9.95), "length": (0.9, 1.1), "mass": (0.9, 1.1), "damping": (0.08, 0.16)},
        {"gravity": (10.25, 10.55), "length": (1.25, 1.45), "mass": (1.25, 1.45), "damping": (0.20, 0.28)},
        ("gravity", "length", "mass", "damping"), 2.2,
    ),
    "cartpole": SystemSpec(
        "cartpole", 4, 1, 0.025,
        {"gravity": 9.81, "cart_mass": 1.0, "pole_mass": 0.1, "half_length": 0.5, "friction": 0.04},
        {"gravity": (9.65, 9.95), "cart_mass": (0.9, 1.1), "pole_mass": (0.085, 0.115), "half_length": (0.45, 0.55), "friction": (0.02, 0.06)},
        {"gravity": (10.25, 10.55), "cart_mass": (1.25, 1.45), "pole_mass": (0.135, 0.16), "half_length": (0.65, 0.78), "friction": (0.075, 0.105)},
        ("gravity", "cart_mass", "pole_mass", "half_length", "friction"), 8.0,
    ),
}


def sample_parameters(spec: SystemSpec, rng: np.random.Generator, regime: str = "train") -> dict[str, float]:
    """Sample independently from the pre-registered train or shifted ranges."""
    if regime not in {"train", "ood"}:
        raise ValueError(f"unknown parameter regime: {regime}")
    ranges = spec.train_param_ranges if regime == "train" else spec.ood_param_ranges
    return {name: float(rng.uniform(low, high)) for name, (low, high) in ranges.items()}


def sample_initial_state(spec: SystemSpec, rng: np.random.Generator) -> np.ndarray:
    if spec.name == "torque_pendulum":
        angle = rng.uniform(-1.2, 1.2)
        return np.array([np.sin(angle), np.cos(angle), rng.uniform(-0.8, 0.8)], dtype=np.float64)
    if spec.name == "cartpole":
        return np.array([
            rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4),
            rng.uniform(-0.55, 0.55), rng.uniform(-0.5, 0.5),
        ], dtype=np.float64)
    return rng.uniform([-1.0, -0.6], [1.0, 0.6]).astype(np.float64)


def _derivative_numpy(spec: SystemSpec, state: np.ndarray, action: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    u = float(np.asarray(action).reshape(-1)[0])
    if spec.name == "damped_oscillator":
        q, v = state
        return np.array([v, (u - p["stiffness"] * q - p["damping"] * v) / p["mass"]])
    if spec.name == "forced_duffing":
        q, v = state
        return np.array([v, u - p["damping"] * v - p["alpha"] * q - p["beta"] * q**3])
    if spec.name == "controlled_van_der_pol":
        q, v = state
        return np.array([v, p["mu"] * (1.0 - q**2) * v - q + p["gain"] * u])
    if spec.name == "torque_pendulum":
        sine, cosine, omega = state
        inertia = p["mass"] * p["length"] ** 2
        return np.array([
            cosine * omega,
            -sine * omega,
            -(p["gravity"] / p["length"]) * sine - p["damping"] * omega + u / inertia,
        ])
    if spec.name == "cartpole":
        position, velocity, angle, angular_velocity = state
        del position
        sine, cosine = np.sin(angle), np.cos(angle)
        total_mass = p["cart_mass"] + p["pole_mass"]
        polemass_length = p["pole_mass"] * p["half_length"]
        force = u - p["friction"] * velocity
        temp = (force + polemass_length * angular_velocity**2 * sine) / total_mass
        angular_accel = (p["gravity"] * sine - cosine * temp) / (
            p["half_length"] * (4.0 / 3.0 - p["pole_mass"] * cosine**2 / total_mass)
        )
        linear_accel = temp - polemass_length * angular_accel * cosine / total_mass
        return np.array([velocity, linear_accel, angular_velocity, angular_accel])
    raise KeyError(spec.name)


def rk4_step_numpy(
    spec: SystemSpec,
    state: np.ndarray,
    action: np.ndarray,
    params: Mapping[str, float],
    substeps: int = 4,
) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64).copy()
    action = np.asarray(action, dtype=np.float64)
    h = spec.dt / substeps
    for _ in range(substeps):
        k1 = _derivative_numpy(spec, state, action, params)
        k2 = _derivative_numpy(spec, state + 0.5 * h * k1, action, params)
        k3 = _derivative_numpy(spec, state + 0.5 * h * k2, action, params)
        k4 = _derivative_numpy(spec, state + h * k3, action, params)
        state = state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if spec.name == "torque_pendulum":
            norm = max(float(np.linalg.norm(state[:2])), 1e-12)
            state[:2] /= norm
    return state


def _derivative_torch(spec: SystemSpec, state: torch.Tensor, action: torch.Tensor, p: Mapping[str, float]) -> torch.Tensor:
    u = action[..., 0]
    if spec.name == "damped_oscillator":
        q, v = state.unbind(-1)
        return torch.stack((v, (u - p["stiffness"] * q - p["damping"] * v) / p["mass"]), -1)
    if spec.name == "forced_duffing":
        q, v = state.unbind(-1)
        return torch.stack((v, u - p["damping"] * v - p["alpha"] * q - p["beta"] * q.pow(3)), -1)
    if spec.name == "controlled_van_der_pol":
        q, v = state.unbind(-1)
        return torch.stack((v, p["mu"] * (1.0 - q.square()) * v - q + p["gain"] * u), -1)
    if spec.name == "torque_pendulum":
        sine, cosine, omega = state.unbind(-1)
        inertia = p["mass"] * p["length"] ** 2
        return torch.stack((
            cosine * omega,
            -sine * omega,
            -(p["gravity"] / p["length"]) * sine - p["damping"] * omega + u / inertia,
        ), -1)
    if spec.name == "cartpole":
        _, velocity, angle, angular_velocity = state.unbind(-1)
        sine, cosine = torch.sin(angle), torch.cos(angle)
        total_mass = p["cart_mass"] + p["pole_mass"]
        polemass_length = p["pole_mass"] * p["half_length"]
        force = u - p["friction"] * velocity
        temp = (force + polemass_length * angular_velocity.square() * sine) / total_mass
        angular_accel = (p["gravity"] * sine - cosine * temp) / (
            p["half_length"] * (4.0 / 3.0 - p["pole_mass"] * cosine.square() / total_mass)
        )
        linear_accel = temp - polemass_length * angular_accel * cosine / total_mass
        return torch.stack((velocity, linear_accel, angular_velocity, angular_accel), -1)
    raise KeyError(spec.name)


def rk4_step_torch(
    spec: SystemSpec,
    state: torch.Tensor,
    action: torch.Tensor,
    params: Mapping[str, float] | None = None,
    substeps: int = 1,
) -> torch.Tensor:
    """Differentiable nominal integrator for batched tensors."""
    p = spec.nominal_params if params is None else params
    h = spec.dt / substeps
    value = state
    for _ in range(substeps):
        k1 = _derivative_torch(spec, value, action, p)
        k2 = _derivative_torch(spec, value + 0.5 * h * k1, action, p)
        k3 = _derivative_torch(spec, value + 0.5 * h * k2, action, p)
        k4 = _derivative_torch(spec, value + h * k3, action, p)
        value = value + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if spec.name == "torque_pendulum":
            unit = value[..., :2] / value[..., :2].norm(dim=-1, keepdim=True).clamp_min(1e-12)
            value = torch.cat((unit, value[..., 2:]), dim=-1)
    return value
