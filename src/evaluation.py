from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import imageio.v2 as imageio
import numpy as np

from .environment import PhysicsConfig, SensorNoiseConfig, make_swingup_env

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class PredictPolicy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True): ...


@dataclass
class EpisodeMetrics:
    episode_return: float
    survival_s: float
    captured: bool
    capture_time_s: float
    final_stable: bool
    upright_ratio: float
    rms_angle_deg: float
    rms_cart_position_m: float
    rms_force_n: float


@dataclass
class AggregateMetrics:
    mean_return: float
    std_return: float
    mean_survival_s: float
    capture_rate: float
    mean_capture_time_s: float
    final_stable_rate: float
    upright_ratio: float
    rms_angle_deg: float
    rms_cart_position_m: float
    rms_force_n: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _action(policy: PredictPolicy | None, env, observation: np.ndarray) -> np.ndarray:
    if policy is None:
        return np.asarray(env.action_space.sample(), dtype=np.float32)
    action, _ = policy.predict(observation, deterministic=True)
    return np.asarray(action, dtype=np.float32)


def evaluate_episode(
    policy: PredictPolicy | None,
    seed: int,
    physics: PhysicsConfig,
    sensor_noise: SensorNoiseConfig,
    video_path: Path | None = None,
) -> EpisodeMetrics:
    env = make_swingup_env(
        physics=physics,
        sensor_noise=sensor_noise,
        reset_mode="evaluation_downward",
        render_mode="rgb_array" if video_path is not None else None,
    )
    env.action_space.seed(seed + 10_000)
    observation, info = env.reset(seed=seed)

    frames: list[np.ndarray] = []
    true_angles: list[float] = []
    true_cart_positions: list[float] = []
    forces: list[float] = []
    stable_flags: list[bool] = []
    episode_return = 0.0
    capture_run = 0
    capture_steps = max(1, int(round(0.5 / physics.dt_s)))
    capture_time = math.nan

    if video_path is not None:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    terminated = False
    truncated = False
    step_index = 0
    while not (terminated or truncated):
        action = _action(policy, env, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        step_index += 1

        x, _, theta, theta_dot = [float(v) for v in info["true_state"]]
        force = float(info["actual_force_n"])
        true_angles.append(theta)
        true_cart_positions.append(x)
        forces.append(force)

        capture_condition = abs(theta) <= math.radians(12.0) and abs(theta_dot) <= 1.5
        capture_run = capture_run + 1 if capture_condition else 0
        if math.isnan(capture_time) and capture_run >= capture_steps:
            capture_time = float(info["time_s"])

        stable_flags.append(abs(theta) <= math.radians(10.0) and abs(x) <= 0.50)

        if video_path is not None and step_index % 2 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    survival_s = float(info.get("time_s", step_index * physics.dt_s))
    env.close()

    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=25, macro_block_size=1)

    angle_array = np.asarray(true_angles, dtype=np.float64)
    cart_array = np.asarray(true_cart_positions, dtype=np.float64)
    force_array = np.asarray(forces, dtype=np.float64)
    final_window_steps = max(1, int(round(2.0 / physics.dt_s)))
    final_stable = (
        len(stable_flags) >= final_window_steps
        and float(np.mean(stable_flags[-final_window_steps:])) >= 0.80
    )

    return EpisodeMetrics(
        episode_return=episode_return,
        survival_s=survival_s,
        captured=not math.isnan(capture_time),
        capture_time_s=capture_time,
        final_stable=bool(final_stable),
        upright_ratio=float(np.mean(np.abs(angle_array) <= math.radians(10.0))),
        rms_angle_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(angle_array))))),
        rms_cart_position_m=float(np.sqrt(np.mean(np.square(cart_array)))),
        rms_force_n=float(np.sqrt(np.mean(np.square(force_array)))),
    )


def evaluate_policy(
    policy: PredictPolicy | None,
    seeds: list[int],
    physics: PhysicsConfig,
    sensor_noise: SensorNoiseConfig,
) -> AggregateMetrics:
    episodes = [
        evaluate_episode(policy, seed, physics=physics, sensor_noise=sensor_noise)
        for seed in seeds
    ]
    captured_times = [item.capture_time_s for item in episodes if item.captured]
    returns = np.asarray([item.episode_return for item in episodes], dtype=np.float64)
    return AggregateMetrics(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_survival_s=float(np.mean([item.survival_s for item in episodes])),
        capture_rate=float(np.mean([item.captured for item in episodes])),
        mean_capture_time_s=float(np.mean(captured_times)) if captured_times else math.nan,
        final_stable_rate=float(np.mean([item.final_stable for item in episodes])),
        upright_ratio=float(np.mean([item.upright_ratio for item in episodes])),
        rms_angle_deg=float(np.mean([item.rms_angle_deg for item in episodes])),
        rms_cart_position_m=float(np.mean([item.rms_cart_position_m for item in episodes])),
        rms_force_n=float(np.mean([item.rms_force_n for item in episodes])),
    )
