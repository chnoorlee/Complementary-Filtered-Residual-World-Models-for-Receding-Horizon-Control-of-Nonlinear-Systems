"""Independent reduced translational prior for the measured NanoDrone extension.

This prior estimates one thrust-to-mass coefficient from training transitions.
It uses constant body angular velocity and omits rotational acceleration,
motor lag, and drag. It is not the benchmark authors' full rigid-body model.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from .data import DT, Record, require_training_records


def rotvec_to_quaternion(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert a rotation vector to an xyzw unit quaternion with finite zero gradients."""
    theta = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    vector = 0.5 * torch.sinc(theta / (2.0 * math.pi)) * rotvec
    return torch.cat((vector, torch.cos(0.5 * theta)), dim=-1)


def quaternion_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lv, lw = left[..., :3], left[..., 3:]
    rv, rw = right[..., :3], right[..., 3:]
    vector = lw * rv + rw * lv + torch.linalg.cross(lv, rv, dim=-1)
    scalar = lw * rw - (lv * rv).sum(dim=-1, keepdim=True)
    return torch.cat((vector, scalar), dim=-1)


def quaternion_to_rotvec(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-12)
    quaternion = torch.where(quaternion[..., 3:] < 0.0, -quaternion, quaternion)
    vector, scalar = quaternion[..., :3], quaternion[..., 3:]
    sine = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sine, scalar)
    regular = angle / sine.clamp_min(1e-8)
    series = 2.0 + sine.square() / 3.0 + 3.0 * sine.pow(4) / 20.0
    return vector * torch.where(sine < 1e-4, series, regular)


def body_z_world(quaternion: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quaternion.unbind(-1)
    return torch.stack((2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x.square() + y.square())), dim=-1)


class KinematicThrustPrior(nn.Module):
    """One-step SI-unit prior for [position, velocity, rotvec, body omega]."""

    def __init__(self, thrust_coefficient: float, dt: float = DT, gravity: float = 9.81) -> None:
        super().__init__()
        if not math.isfinite(thrust_coefficient) or thrust_coefficient < 0.0:
            raise ValueError("Thrust coefficient must be finite and nonnegative")
        if not math.isfinite(dt) or dt <= 0.0 or not math.isfinite(gravity) or gravity <= 0.0:
            raise ValueError("dt and gravity must be finite and positive")
        self.dt = float(dt)
        self.gravity = float(gravity)
        self.register_buffer("thrust_coefficient", torch.tensor(thrust_coefficient, dtype=torch.float64))
        self.fit_metadata: dict = {"fitted": False}

    @classmethod
    def fit(cls, train_records: Iterable[Record], gravity: float = 9.81) -> "KinematicThrustPrior":
        items = require_training_records(train_records)
        dt = items[0].dt
        if any(not np.isclose(item.dt, dt, rtol=0.0, atol=1e-12) for item in items):
            raise ValueError("All training records must have the same sample interval")
        numerator = 0.0
        denominator = 0.0
        transition_count = 0
        for item in items:
            direction = Rotation.from_rotvec(item.states[:-1, 6:9]).as_matrix()[:, :, 2]
            speed_squared = np.square(item.actions[:-1]).sum(axis=1, keepdims=True)
            design = direction * speed_squared
            target = np.diff(item.states[:, 3:6], axis=0) / dt
            target[:, 2] += gravity
            numerator += float(np.sum(design * target))
            denominator += float(np.sum(design * design))
            transition_count += len(target)
        if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("Thrust fitting requires finite, nonzero training excitation")
        unconstrained = numerator / denominator
        coefficient = max(0.0, unconstrained)
        prior = cls(coefficient, dt=dt, gravity=gravity)
        prior.fit_metadata = {
            "fitted": True, "method": "nonnegative scalar least squares, no intercept",
            "training_record_ids": [item.record_id for item in items],
            "training_transitions": transition_count,
            "thrust_coefficient": coefficient, "unconstrained_coefficient": unconstrained,
            "coefficient_units": "(m/s^2)/(rad/s)^2",
            "motor_input_units": "rad/s", "gravity_m_s2": gravity, "dt_s": dt,
            "equation": "dv/dt + g*e3 = coefficient * R(rotvec)*e3 * sum(motor_speed**2)",
            "rotation": "constant body angular velocity; quaternion exponential update",
        }
        return prior

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != 12 or action.shape[-1] != 4 or state.shape[:-1] != action.shape[:-1]:
            raise ValueError("Expected matching (...,12) states and (...,4) motor speeds")
        position, velocity = state[..., :3], state[..., 3:6]
        rotation, omega = state[..., 6:9], state[..., 9:12]
        quaternion = rotvec_to_quaternion(rotation)
        coefficient = self.thrust_coefficient.to(dtype=state.dtype, device=state.device)
        acceleration = body_z_world(quaternion) * coefficient * action.square().sum(dim=-1, keepdim=True)
        gravity = state.new_tensor([0.0, 0.0, self.gravity])
        acceleration = acceleration - gravity
        next_position = position + self.dt * velocity + 0.5 * self.dt**2 * acceleration
        next_velocity = velocity + self.dt * acceleration
        increment = rotvec_to_quaternion(self.dt * omega)
        next_rotation = quaternion_to_rotvec(quaternion_product(quaternion, increment))
        return torch.cat((next_position, next_velocity, next_rotation, omega), dim=-1)
