from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class PhysicsConfig:
    gravity: float = 9.81
    cart_mass_kg: float = 1.0
    pole_mass_kg: float = 0.1
    pole_com_length_m: float = 0.5
    max_force_n: float = 10.0
    cart_viscous_friction_n_per_mps: float = 0.10
    motor_time_constant_s: float = 0.05
    dt_s: float = 0.02
    track_limit_m: float = 2.4
    max_episode_s: float = 20.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SensorNoiseConfig:
    enabled: bool = True
    position_noise_std_m: float = 0.001
    position_bias_std_m: float = 0.002
    velocity_noise_std_mps: float = 0.01
    velocity_bias_std_mps: float = 0.01
    angle_noise_std_deg: float = 0.25
    angle_bias_std_deg: float = 1.0
    gyro_noise_std_dps: float = 0.10
    gyro_bias_std_dps: float = 0.30

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


class SwingUpCartPoleEnv(gym.Env):
    """Continuous Cart-Pole swing-up plant shared with PPO and SAC experiments.

    State convention:
      x          cart position [m], right positive
      x_dot      cart velocity [m/s]
      theta      pole angle [rad], 0=upright, +/-pi=downward
      theta_dot  pole angular velocity [rad/s]

    Action is normalized to [-1, 1] and scaled to +/-max_force_n. Large pole
    angle does not terminate the episode; only leaving the finite rail does.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}
    RESET_MODES = {"near_upright", "wide", "full", "downward_mix", "evaluation_downward"}

    def __init__(
        self,
        physics: PhysicsConfig | None = None,
        reset_mode: str = "downward_mix",
        render_mode: str | None = None,
    ):
        super().__init__()
        self.physics = physics or PhysicsConfig()
        self.render_mode = render_mode
        self.set_reset_mode(reset_mode)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -1.0, -1.0, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1.0, 1.0, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        self.state = np.zeros(4, dtype=np.float64)
        self.actual_force_n = 0.0
        self.commanded_force_n = 0.0
        self.steps = 0

    @property
    def max_steps(self) -> int:
        return int(round(self.physics.max_episode_s / self.physics.dt_s))

    def set_reset_mode(self, mode: str) -> None:
        if mode not in self.RESET_MODES:
            raise ValueError(f"Unknown reset mode: {mode}")
        self.reset_mode = mode

    def _sample_angle(self) -> float:
        rng = self.np_random
        if self.reset_mode == "near_upright":
            return math.radians(rng.uniform(-25.0, 25.0))
        if self.reset_mode == "wide":
            if rng.random() < 0.25:
                return math.radians(rng.uniform(-30.0, 30.0))
            return math.radians(rng.uniform(-120.0, 120.0))
        if self.reset_mode == "full":
            if rng.random() < 0.20:
                return math.radians(rng.uniform(-35.0, 35.0))
            return rng.uniform(-math.pi, math.pi)
        if self.reset_mode == "downward_mix":
            q = rng.random()
            if q < 0.70:
                sign = -1.0 if rng.random() < 0.5 else 1.0
                return wrap_angle(sign * math.pi + math.radians(rng.uniform(-15.0, 15.0)))
            if q < 0.85:
                return rng.uniform(-math.pi, math.pi)
            return math.radians(rng.uniform(-30.0, 30.0))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        return wrap_angle(sign * math.pi + math.radians(rng.uniform(-5.0, 5.0)))

    def _observation(self) -> np.ndarray:
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)

    def _info(self) -> dict:
        return {
            "true_state": self.state.astype(float).tolist(),
            "commanded_force_n": float(self.commanded_force_n),
            "actual_force_n": float(self.actual_force_n),
            "time_s": float(self.steps * self.physics.dt_s),
            "reset_mode": self.reset_mode,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self.state = np.array(
            [
                self.np_random.uniform(-0.12, 0.12),
                self.np_random.uniform(-0.08, 0.08),
                self._sample_angle(),
                self.np_random.uniform(-0.12, 0.12),
            ],
            dtype=np.float64,
        )
        self.actual_force_n = 0.0
        self.commanded_force_n = 0.0
        self.steps = 0
        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        p = self.physics
        u = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        self.commanded_force_n = u * p.max_force_n

        alpha = 1.0 - math.exp(-p.dt_s / max(p.motor_time_constant_s, 1e-6))
        self.actual_force_n += alpha * (self.commanded_force_n - self.actual_force_n)

        x, x_dot, theta, theta_dot = self.state
        s, c = math.sin(theta), math.cos(theta)
        total_mass = p.cart_mass_kg + p.pole_mass_kg
        ml = p.pole_mass_kg * p.pole_com_length_m
        net_force = self.actual_force_n - p.cart_viscous_friction_n_per_mps * x_dot
        temp = (net_force + ml * theta_dot**2 * s) / total_mass
        theta_acc = (p.gravity * s - c * temp) / (
            p.pole_com_length_m * (4.0 / 3.0 - p.pole_mass_kg * c**2 / total_mass)
        )
        x_acc = temp - ml * theta_acc * c / total_mass

        x_dot += p.dt_s * x_acc
        theta_dot += p.dt_s * theta_acc
        x += p.dt_s * x_dot
        theta = wrap_angle(theta + p.dt_s * theta_dot)
        self.state[:] = [x, x_dot, theta, theta_dot]
        self.steps += 1

        upright = 0.5 * (1.0 + math.cos(theta))
        center = math.exp(-((x / 0.75) ** 2))
        boundary = min(abs(x) / p.track_limit_m, 1.5)
        swing_motion = (1.0 - upright) * min(abs(theta_dot) / 4.0, 1.0)
        reward = (
            0.05
            + 0.75 * upright
            + 0.15 * upright * center
            + 0.04 * swing_motion
            - 0.004 * (x_dot / 3.0) ** 2
            - 0.002 * (theta_dot / 8.0) ** 2
            - 0.003 * (self.actual_force_n / p.max_force_n) ** 2
            - 0.10 * boundary**6
        )
        if abs(theta) <= math.radians(10.0):
            reward += 0.35
        if abs(theta) <= math.radians(8.0) and abs(theta_dot) <= 1.0 and abs(x) <= 0.45:
            reward += 0.65

        terminated = bool(abs(x) > p.track_limit_m)
        truncated = bool(self.steps >= self.max_steps)
        if terminated:
            reward -= 25.0

        return self._observation(), float(reward), terminated, truncated, self._info()

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        width, height = 800, 450
        image = Image.new("RGB", (width, height), (248, 249, 251))
        draw = ImageDraw.Draw(image)
        rail_y, left_px, right_px = 315, 70, width - 70
        draw.line((left_px, rail_y + 30, right_px, rail_y + 30), fill=(90, 96, 105), width=4)
        center_px = (left_px + right_px) / 2
        draw.line((center_px, rail_y + 18, center_px, rail_y + 46), fill=(120, 125, 132), width=2)

        x, _, theta, _ = self.state
        normalized = (x + self.physics.track_limit_m) / (2.0 * self.physics.track_limit_m)
        cart_x = left_px + normalized * (right_px - left_px)
        cart_box = (int(cart_x - 45), rail_y - 42, int(cart_x + 45), rail_y)
        draw.rounded_rectangle(cart_box, radius=8, fill=(51, 65, 85), outline=(26, 34, 45), width=2)
        draw.ellipse((cart_x - 35, rail_y - 5, cart_x - 17, rail_y + 13), fill=(30, 30, 34))
        draw.ellipse((cart_x + 17, rail_y - 5, cart_x + 35, rail_y + 13), fill=(30, 30, 34))

        pivot_x, pivot_y, pole_px = cart_x, rail_y - 42, 175
        tip_x = pivot_x + pole_px * math.sin(theta)
        tip_y = pivot_y - pole_px * math.cos(theta)
        draw.line((pivot_x, pivot_y, tip_x, tip_y), fill=(205, 73, 62), width=10)
        draw.ellipse((pivot_x - 9, pivot_y - 9, pivot_x + 9, pivot_y + 9), fill=(28, 32, 38))
        draw.ellipse((tip_x - 10, tip_y - 10, tip_x + 10, tip_y + 10), fill=(205, 73, 62))

        draw.text((20, 18), f"mode = {self.reset_mode}", fill=(30, 35, 42))
        draw.text((20, 42), f"x = {x:+.3f} m", fill=(30, 35, 42))
        draw.text((20, 66), f"theta = {math.degrees(theta):+.1f} deg", fill=(30, 35, 42))
        draw.text((20, 90), f"force = {self.actual_force_n:+.2f} N", fill=(30, 35, 42))
        draw.text((20, 114), f"t = {self.steps * self.physics.dt_s:.2f} s", fill=(30, 35, 42))
        return np.asarray(image, dtype=np.uint8)


class ConsumerSensorWrapper(gym.ObservationWrapper):
    """Consumer-grade encoder/IMU error model shared across comparisons."""

    def __init__(self, env: gym.Env, noise: SensorNoiseConfig | None = None):
        super().__init__(env)
        self.noise = noise or SensorNoiseConfig()
        self._bias = np.zeros(4, dtype=np.float64)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        if self.noise.enabled:
            self._bias = np.array(
                [
                    self.np_random.normal(0.0, self.noise.position_bias_std_m),
                    self.np_random.normal(0.0, self.noise.velocity_bias_std_mps),
                    math.radians(self.np_random.normal(0.0, self.noise.angle_bias_std_deg)),
                    math.radians(self.np_random.normal(0.0, self.noise.gyro_bias_std_dps)),
                ],
                dtype=np.float64,
            )
        else:
            self._bias[:] = 0.0
        return self.observation(observation), info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if not self.noise.enabled:
            return np.asarray(observation, dtype=np.float32)
        x = float(observation[0]) + self._bias[0] + self.np_random.normal(0.0, self.noise.position_noise_std_m)
        x_dot = float(observation[1]) + self._bias[1] + self.np_random.normal(0.0, self.noise.velocity_noise_std_mps)
        theta = math.atan2(float(observation[3]), float(observation[2]))
        theta += self._bias[2] + math.radians(self.np_random.normal(0.0, self.noise.angle_noise_std_deg))
        theta_dot = float(observation[4]) + self._bias[3] + math.radians(
            self.np_random.normal(0.0, self.noise.gyro_noise_std_dps)
        )
        return np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)

    def set_reset_mode(self, mode: str) -> None:
        self.unwrapped.set_reset_mode(mode)


def make_swingup_env(
    physics: PhysicsConfig | None = None,
    sensor_noise: SensorNoiseConfig | None = None,
    reset_mode: str = "downward_mix",
    render_mode: str | None = None,
) -> gym.Env:
    return ConsumerSensorWrapper(
        SwingUpCartPoleEnv(physics=physics, reset_mode=reset_mode, render_mode=render_mode),
        noise=sensor_noise,
    )
