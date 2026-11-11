"""Frozen additional closed-loop study for the T-ASE manuscript revision.

The script loads the existing confirmatory checkpoints and evaluates four
prediction interfaces under a shared receding-horizon controller. It never
writes to the confirmatory result tree.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from rgpwm.confirm_protocol import CONFIRM_SEEDS  # noqa: E402
from rgpwm.data import Normalizer  # noqa: E402
from rgpwm.dynamics import (  # noqa: E402
    SYSTEMS,
    SystemSpec,
    rk4_step_numpy,
    sample_initial_state,
    sample_parameters,
)
from rgpwm.experiment import _model_for  # noqa: E402


CONTROLLERS = ("nominal_last", "complementary_nominal", "pgnn", "cfrwm")
CONDITIONS = ("id_clean", "id_corrupt", "ood_parameter_corrupt")
EPISODES_PER_CELL = 4
CONTEXT_STEPS = 8
CONTROL_STEPS = 20
PLANNING_HORIZON = 8
CANDIDATES = 64
CEM_ITERATIONS = 2
ELITES = 8
ACTION_WEIGHT = 0.01
TERMINAL_WEIGHT = 2.0
INITIAL_STD_FRACTION = 0.6
STD_FLOOR_FRACTION = 0.05
NOISE_FRACTION = 0.08
MISSING_PROBABILITY = 0.25
BOOTSTRAP_REPLICATES = 5000
PROTOCOL_SEED = 20261108
RESULT_DIR = ROOT / "results" / "tase_closed_loop"
CHECKPOINT_DIR = ROOT / "results" / "confirmatory"


@dataclass(frozen=True)
class Protocol:
    controllers: tuple[str, ...] = CONTROLLERS
    conditions: tuple[str, ...] = CONDITIONS
    seeds: tuple[int, ...] = CONFIRM_SEEDS
    systems: tuple[str, ...] = tuple(SYSTEMS)
    episodes_per_cell: int = EPISODES_PER_CELL
    context_steps: int = CONTEXT_STEPS
    control_steps: int = CONTROL_STEPS
    planning_horizon: int = PLANNING_HORIZON
    candidates: int = CANDIDATES
    cem_iterations: int = CEM_ITERATIONS
    elites: int = ELITES
    action_weight: float = ACTION_WEIGHT
    terminal_weight: float = TERMINAL_WEIGHT
    initial_std_fraction: float = INITIAL_STD_FRACTION
    std_floor_fraction: float = STD_FLOOR_FRACTION
    noise_fraction: float = NOISE_FRACTION
    missing_probability: float = MISSING_PROBABILITY
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES
    protocol_seed: int = PROTOCOL_SEED


@dataclass
class ObservationHistory:
    values: list[np.ndarray]
    masks: list[np.ndarray]
    deltas: list[np.ndarray]
    actions: list[np.ndarray]
    last_observed: np.ndarray
    elapsed: np.ndarray

    @classmethod
    def start(cls, observation: np.ndarray) -> "ObservationHistory":
        value = np.asarray(observation, dtype=np.float64).copy()
        return cls(
            values=[value.copy()],
            masks=[np.ones_like(value)],
            deltas=[np.zeros_like(value)],
            actions=[],
            last_observed=value.copy(),
            elapsed=np.zeros_like(value),
        )

    def append(
        self,
        state: np.ndarray,
        action: np.ndarray,
        noise: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        raw = np.asarray(state, dtype=np.float64) + np.asarray(noise, dtype=np.float64)
        observed = np.asarray(mask, dtype=np.float64)
        available = observed == 1.0
        self.last_observed[available] = raw[available]
        self.elapsed[available] = 0.0
        self.elapsed[~available] += 1.0
        self.values.append(self.last_observed.copy())
        self.masks.append(observed.copy())
        self.deltas.append(self.elapsed.copy())
        self.actions.append(np.asarray(action, dtype=np.float64).copy())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint() -> dict[str, str]:
    files = (
        ROOT / "code" / "tase_closed_loop.py",
        ROOT / "code" / "rgpwm" / "dynamics.py",
        ROOT / "code" / "rgpwm" / "models.py",
        ROOT / "docs" / "25_tase_closed_loop_protocol.md",
    )
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in files}


def checkpoint_normalizer(payload: Mapping[str, object]) -> Normalizer:
    state = payload["model_state"]
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint model_state is not a mapping")

    def array(name: str) -> np.ndarray:
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"checkpoint buffer {name} is not a tensor")
        return value.detach().cpu().numpy().astype(np.float64)

    return Normalizer(
        state_mean=array("state_mean"),
        state_std=array("state_std"),
        action_mean=array("action_mean"),
        action_std=array("action_std"),
    )


def load_model(
    method: str,
    spec: SystemSpec,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, Normalizer, str]:
    path = CHECKPOINT_DIR / spec.name / f"seed_{seed}" / f"{method}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    normalizer = checkpoint_normalizer(payload)
    model = _model_for(method, spec, normalizer)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval().to(device), normalizer, sha256(path)


def target_state(spec: SystemSpec) -> np.ndarray:
    if spec.name == "torque_pendulum":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return np.zeros(spec.state_dim, dtype=np.float64)


def context_tensors(
    history: ObservationHistory,
    normalizer: Normalizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(history.values) < CONTEXT_STEPS:
        raise ValueError("insufficient context")
    start = len(history.values) - CONTEXT_STEPS
    values = np.stack(history.values[start:])
    masks = np.stack(history.masks[start:])
    deltas = np.stack(history.deltas[start:])
    context_actions = np.zeros((CONTEXT_STEPS, normalizer.action_mean.size), dtype=np.float64)
    if CONTEXT_STEPS > 1:
        context_actions[1:] = np.stack(history.actions[start:start + CONTEXT_STEPS - 1])
    values = (values - normalizer.state_mean) / normalizer.state_std
    context_actions = (context_actions - normalizer.action_mean) / normalizer.action_std
    return tuple(
        torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
        for value in (values, masks, deltas, context_actions)
    )  # type: ignore[return-value]


@torch.no_grad()
def candidate_means(
    controller: str,
    model: torch.nn.Module,
    context: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    physical_actions: np.ndarray,
    normalizer: Normalizer,
    device: torch.device,
) -> torch.Tensor:
    candidate_count = physical_actions.shape[0]
    future = (physical_actions - normalizer.action_mean) / normalizer.action_std
    future_actions = torch.as_tensor(future, dtype=torch.float32, device=device)
    values, masks, deltas, context_actions = (
        item.expand(candidate_count, -1, -1) for item in context
    )
    if controller in {"pgnn", "cfrwm"}:
        return model(values, masks, deltas, context_actions, future_actions)["mean"]

    if controller == "nominal_last":
        state = values[:, -1]
    elif controller == "complementary_nominal":
        encoder_features = torch.cat(
            (values, masks, torch.log1p(deltas.clamp_min(0.0)), context_actions), dim=-1
        )
        context_hidden, _ = model.encoder(encoder_features)
        state, _ = model._filter_context(  # type: ignore[attr-defined]
            values, masks, deltas, context_actions, context_hidden
        )
    else:
        raise ValueError(f"unknown controller: {controller}")
    rollout: list[torch.Tensor] = []
    for action in future_actions.unbind(1):
        state = model.normalized_nominal_step(state, action)
        if hasattr(model, "_project_state"):
            state = model._project_state(state)  # type: ignore[attr-defined]
        rollout.append(state)
    return torch.stack(rollout, dim=1)


def prediction_costs(
    predicted: torch.Tensor,
    actions: np.ndarray,
    target_normalized: np.ndarray,
    spec: SystemSpec,
) -> np.ndarray:
    target = torch.as_tensor(
        target_normalized, dtype=predicted.dtype, device=predicted.device
    ).view(1, 1, -1)
    state_cost = (predicted - target).square().mean(dim=-1)
    action_cost = torch.as_tensor(
        (actions / spec.action_limit) ** 2,
        dtype=predicted.dtype,
        device=predicted.device,
    ).mean(dim=-1)
    total = state_cost.sum(dim=1) + ACTION_WEIGHT * action_cost.sum(dim=1)
    total = total + TERMINAL_WEIGHT * state_cost[:, -1]
    return total.detach().cpu().numpy().astype(np.float64)


def plan_action(
    controller: str,
    model: torch.nn.Module,
    history: ObservationHistory,
    normalizer: Normalizer,
    spec: SystemSpec,
    target: np.ndarray,
    warm_start: np.ndarray,
    rng: np.random.Generator,
    device: torch.device,
    candidates: int = CANDIDATES,
    iterations: int = CEM_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(warm_start, dtype=np.float64).copy()
    std = np.full_like(mean, INITIAL_STD_FRACTION * spec.action_limit)
    context = context_tensors(history, normalizer, device)
    target_normalized = (target - normalizer.state_mean) / normalizer.state_std
    for _ in range(iterations):
        samples = mean[None] + std[None] * rng.standard_normal(
            (candidates, PLANNING_HORIZON, spec.action_dim)
        )
        samples = np.clip(samples, -spec.action_limit, spec.action_limit)
        samples[0] = np.clip(mean, -spec.action_limit, spec.action_limit)
        if candidates > 1:
            samples[1] = 0.0
        predicted = candidate_means(
            controller, model, context, samples, normalizer, device
        )
        costs = prediction_costs(predicted, samples, target_normalized, spec)
        elite = samples[np.argsort(costs, kind="stable")[:ELITES]]
        mean = elite.mean(axis=0)
        std = np.maximum(
            elite.std(axis=0, ddof=0), STD_FLOOR_FRACTION * spec.action_limit
        )
    action = np.clip(mean[0], -spec.action_limit, spec.action_limit)
    next_warm = np.concatenate((mean[1:], np.zeros_like(mean[-1:])), axis=0)
    return action, next_warm


def episode_randomness(
    spec: SystemSpec,
    seed: int,
    condition: str,
    episode: int,
) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray, int]:
    condition_index = CONDITIONS.index(condition)
    system_index = tuple(SYSTEMS).index(spec.name)
    parameter_regime_index = int(condition == "ood_parameter_corrupt")
    plant_seq = np.random.SeedSequence(
        [PROTOCOL_SEED, seed, system_index, parameter_regime_index, episode, 0]
    )
    observation_seq = np.random.SeedSequence(
        [PROTOCOL_SEED, seed, system_index, condition_index, episode, 1]
    )
    planner_seq = np.random.SeedSequence(
        [PROTOCOL_SEED, seed, system_index, condition_index, episode, 2]
    )
    plant_rng = np.random.default_rng(plant_seq)
    observation_rng = np.random.default_rng(observation_seq)
    regime = "ood" if condition == "ood_parameter_corrupt" else "train"
    initial = sample_initial_state(spec, plant_rng)
    params = sample_parameters(spec, plant_rng, regime)
    total_states = CONTEXT_STEPS + CONTROL_STEPS
    standard_noise = observation_rng.standard_normal((total_states, spec.state_dim))
    if condition == "id_clean":
        masks = np.ones_like(standard_noise)
    else:
        masks = (
            observation_rng.random(standard_noise.shape) >= MISSING_PROBABILITY
        ).astype(np.float64)
        masks[0] = 1.0
    planner_seed = int(np.random.default_rng(planner_seq).integers(0, 2**32 - 1))
    return initial, params, standard_noise, masks, planner_seed


def burn_in(
    spec: SystemSpec,
    initial: np.ndarray,
    params: Mapping[str, float],
    standard_noise: np.ndarray,
    masks: np.ndarray,
    noise_std: float,
) -> tuple[np.ndarray, ObservationHistory]:
    state = initial.copy()
    first = state + noise_std * standard_noise[0]
    history = ObservationHistory.start(first)
    zero_action = np.zeros(spec.action_dim, dtype=np.float64)
    for index in range(1, CONTEXT_STEPS):
        state = rk4_step_numpy(spec, state, zero_action, params, substeps=4)
        history.append(
            state,
            zero_action,
            noise_std * standard_noise[index],
            masks[index],
        )
    return state, history


def run_episode(
    controller: str,
    model: torch.nn.Module,
    normalizer: Normalizer,
    spec: SystemSpec,
    seed: int,
    condition: str,
    episode: int,
    device: torch.device,
) -> dict[str, object]:
    initial, params, standard_noise, masks, planner_seed = episode_randomness(
        spec, seed, condition, episode
    )
    noise_std = 0.0 if condition == "id_clean" else (
        NOISE_FRACTION * float(np.mean(normalizer.state_std))
    )
    state, history = burn_in(
        spec, initial, params, standard_noise, masks, noise_std
    )
    target = target_state(spec)
    warm_start = np.zeros((PLANNING_HORIZON, spec.action_dim), dtype=np.float64)
    planner_rng = np.random.default_rng(planner_seed)
    state_losses: list[float] = []
    action_efforts: list[float] = []
    combined_costs: list[float] = []
    for step in range(CONTROL_STEPS):
        action, warm_start = plan_action(
            controller,
            model,
            history,
            normalizer,
            spec,
            target,
            warm_start,
            planner_rng,
            device,
        )
        state = rk4_step_numpy(spec, state, action, params, substeps=4)
        index = CONTEXT_STEPS + step
        history.append(
            state,
            action,
            noise_std * standard_noise[index],
            masks[index],
        )
        state_loss = float(np.mean(((state - target) / normalizer.state_std) ** 2))
        action_effort = float(np.mean((action / spec.action_limit) ** 2))
        state_losses.append(state_loss)
        action_efforts.append(action_effort)
        combined_costs.append(state_loss + ACTION_WEIGHT * action_effort)
        if not np.isfinite(state).all():
            raise FloatingPointError(
                f"non-finite state for {controller}, {spec.name}, seed {seed}"
            )
    return {
        "controller": controller,
        "seed": seed,
        "system": spec.name,
        "condition": condition,
        "episode": episode,
        "mean_state_loss": float(np.mean(state_losses)),
        "mean_action_effort": float(np.mean(action_efforts)),
        "mean_combined_cost": float(np.mean(combined_costs)),
        "terminal_normalized_rmse": float(
            np.sqrt(np.mean(((state - target) / normalizer.state_std) ** 2))
        ),
        "initial_state": initial.tolist(),
        "plant_parameters": dict(params),
        "planner_seed": planner_seed,
    }


def exact_sign_flip_p(values: np.ndarray, margin: float = 0.0) -> float:
    centered = np.asarray(values, dtype=np.float64) - margin
    observed = float(centered.mean())
    statistics = [
        float(np.mean(centered * np.asarray(signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=centered.size)
    ]
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    seed_condition = (
        frame.groupby(["controller", "condition", "seed"], as_index=False)[
            ["mean_state_loss", "mean_action_effort", "mean_combined_cost"]
        ]
        .mean()
    )
    condition_summary = (
        seed_condition.groupby(["controller", "condition"], as_index=False)
        .agg(
            state_loss_mean=("mean_state_loss", "mean"),
            state_loss_sd=("mean_state_loss", lambda x: x.std(ddof=1)),
            action_effort_mean=("mean_action_effort", "mean"),
            combined_cost_mean=("mean_combined_cost", "mean"),
        )
    )
    seed_overall = (
        frame.groupby(["controller", "seed"], as_index=False)[
            ["mean_state_loss", "mean_action_effort", "mean_combined_cost"]
        ]
        .mean()
    )
    overall_summary = (
        seed_overall.groupby("controller", as_index=False)
        .agg(
            state_loss_mean=("mean_state_loss", "mean"),
            state_loss_sd=("mean_state_loss", lambda x: x.std(ddof=1)),
            action_effort_mean=("mean_action_effort", "mean"),
            combined_cost_mean=("mean_combined_cost", "mean"),
        )
    )
    paired = seed_overall.pivot(index="seed", columns="controller", values="mean_state_loss")
    improvement = 100.0 * (paired["pgnn"] - paired["cfrwm"]) / paired["pgnn"]
    rng = np.random.default_rng(PROTOCOL_SEED + 1)
    draws = rng.integers(0, improvement.size, size=(BOOTSTRAP_REPLICATES, improvement.size))
    bootstrap = improvement.to_numpy()[draws].mean(axis=1)
    inference = {
        "estimand": "seed-balanced relative state-loss improvement of CFR-WM over PGNN-style",
        "seed_values_percent": improvement.to_dict(),
        "mean_percent": float(improvement.mean()),
        "bootstrap_95_percent_interval": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "exact_one_sided_sign_flip_p_above_zero": exact_sign_flip_p(
            improvement.to_numpy(), margin=0.0
        ),
    }
    return overall_summary, condition_summary, inference


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(output_dir: Path, overwrite: bool = False) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output_dir}; pass --overwrite")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True)
    rows: list[dict[str, object]] = []
    checkpoint_hashes: dict[str, str] = {}
    for spec in SYSTEMS.values():
        for seed in CONFIRM_SEEDS:
            cfr, normalizer, cfr_hash = load_model("cfrwm", spec, seed, device)
            pgnn, pgnn_normalizer, pgnn_hash = load_model("pgnn", spec, seed, device)
            for name in ("state_mean", "state_std", "action_mean", "action_std"):
                if not np.array_equal(getattr(normalizer, name), getattr(pgnn_normalizer, name)):
                    raise RuntimeError(f"normalizer mismatch for {spec.name}, seed {seed}")
            checkpoint_hashes[f"{spec.name}/seed_{seed}/cfrwm.pt"] = cfr_hash
            checkpoint_hashes[f"{spec.name}/seed_{seed}/pgnn.pt"] = pgnn_hash
            model_by_controller = {
                "nominal_last": cfr,
                "complementary_nominal": cfr,
                "pgnn": pgnn,
                "cfrwm": cfr,
            }
            for condition in CONDITIONS:
                for episode in range(EPISODES_PER_CELL):
                    for controller in CONTROLLERS:
                        rows.append(
                            run_episode(
                                controller,
                                model_by_controller[controller],
                                normalizer,
                                spec,
                                seed,
                                condition,
                                episode,
                                device,
                            )
                        )
            print(
                f"completed {spec.name} seed {seed} "
                f"({len(rows)}/{len(SYSTEMS) * len(CONFIRM_SEEDS) * len(CONDITIONS) * EPISODES_PER_CELL * len(CONTROLLERS)} episodes)",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    overall, by_condition, inference = summarize(frame)
    frame.to_csv(output_dir / "episode_results.csv", index=False)
    overall.to_csv(output_dir / "overall_summary.csv", index=False)
    by_condition.to_csv(output_dir / "condition_summary.csv", index=False)
    payload: dict[str, object] = {
        "complete": True,
        "device": str(device),
        "protocol": asdict(Protocol()),
        "source_sha256": source_fingerprint(),
        "checkpoint_sha256": checkpoint_hashes,
        "row_count": int(len(frame)),
        "inference": inference,
        "outputs": {
            name: sha256(output_dir / name)
            for name in ("episode_results.csv", "overall_summary.csv", "condition_summary.csv")
        },
    }
    write_json(manifest_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    payload = run(args.output_dir, overwrite=args.overwrite)
    print(json.dumps(payload["inference"], indent=2))


if __name__ == "__main__":
    main()
