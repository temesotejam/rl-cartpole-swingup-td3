from __future__ import annotations

import math

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from src.environment import PhysicsConfig, SensorNoiseConfig, make_swingup_env
from src.evaluation import evaluate_policy


def test_downward_reset_and_nonnegative_survival_reward() -> None:
    env = make_swingup_env(
        physics=PhysicsConfig(),
        sensor_noise=SensorNoiseConfig(enabled=False),
        reset_mode="evaluation_downward",
    )
    obs, info = env.reset(seed=123)
    theta = float(info["true_state"][2])
    assert abs(math.degrees(theta)) > 170.0
    obs, reward, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
    assert obs.shape == (5,)
    assert np.isfinite(obs).all()
    assert reward > 0.0
    assert not terminated
    assert not truncated
    env.close()


def test_consumer_sensor_wrapper_is_finite() -> None:
    env = make_swingup_env(reset_mode="full")
    obs, _ = env.reset(seed=7)
    assert obs.shape == (5,)
    assert np.isfinite(obs).all()
    for _ in range(20):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.isfinite(obs).all()
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


def test_short_td3_training_and_evaluation() -> None:
    physics = PhysicsConfig(max_episode_s=2.0)
    noise = SensorNoiseConfig(enabled=True)
    env = make_swingup_env(physics=physics, sensor_noise=noise, reset_mode="near_upright")
    action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.2 * np.ones(1))
    model = TD3(
        "MlpPolicy", env,
        learning_rate=1e-3,
        buffer_size=2000,
        learning_starts=64,
        batch_size=64,
        train_freq=1,
        gradient_steps=1,
        action_noise=action_noise,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        policy_kwargs={"net_arch": [64, 64]},
        seed=123,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=256)
    metrics = evaluate_policy(model, [456], physics=physics, sensor_noise=noise)
    assert np.isfinite(metrics.mean_return)
    assert 0.0 <= metrics.capture_rate <= 1.0
    assert 0.0 <= metrics.final_stable_rate <= 1.0
    env.close()
