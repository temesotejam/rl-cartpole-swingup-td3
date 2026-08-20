from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from .environment import PhysicsConfig, SensorNoiseConfig, make_swingup_env
from .evaluation import evaluate_episode, evaluate_policy
from .reporting import create_plots, write_metrics_csv, write_summary

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TD3 on the continuous Cart-Pole swing-up task.")
    parser.add_argument("--preset", choices=["quick", "normal", "long"], default="normal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def load_config(preset: str) -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / f"{preset}.yaml").read_text(encoding="utf-8"))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def physics_from(config: dict) -> PhysicsConfig:
    return PhysicsConfig(**config["physics"])


def noise_from(config: dict) -> SensorNoiseConfig:
    return SensorNoiseConfig(**config["sensor_noise"])


def version_info() -> dict[str, str]:
    packages = ["gymnasium", "stable-baselines3", "torch", "numpy"]
    return {name: importlib.metadata.version(name) for name in packages}


def record_stage(
    records: list[dict], stage: str, progress: float, timesteps: int, policy,
    evaluation_seeds: list[int], video_seed: int, videos_dir: Path, video_index: int,
    physics: PhysicsConfig, noise: SensorNoiseConfig,
) -> None:
    metrics = evaluate_policy(policy, evaluation_seeds, physics=physics, sensor_noise=noise)
    evaluate_episode(policy, seed=video_seed, physics=physics, sensor_noise=noise,
                     video_path=videos_dir / f"{video_index:02d}_{stage}.mp4")
    records.append({"stage": stage, "progress": progress, "timesteps": timesteps, **metrics.to_dict()})
    capture_t = "-" if math.isnan(metrics.mean_capture_time_s) else f"{metrics.mean_capture_time_s:.2f}s"
    print(
        f"[{stage}] steps={timesteps:,} return={metrics.mean_return:.1f} "
        f"capture={metrics.capture_rate*100:.1f}% capture_t={capture_t} "
        f"final_stable={metrics.final_stable_rate*100:.1f}%"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.preset)
    set_global_seed(args.seed)

    output_dir = args.output_dir.resolve()
    models_dir, videos_dir, plots_dir = output_dir / "models", output_dir / "videos", output_dir / "plots"
    for directory in [output_dir, models_dir, videos_dir, plots_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    physics = physics_from(config)
    noise = noise_from(config)
    evaluation_episodes = int(config["evaluation_episodes"])
    evaluation_seeds = [args.seed + 100 + i for i in range(evaluation_episodes)]
    video_seed = args.seed + 999

    records: list[dict] = []
    record_stage(records, "random", 0.0, 0, None, evaluation_seeds, video_seed,
                 videos_dir, 0, physics, noise)

    train_env = make_swingup_env(physics=physics, sensor_noise=noise, reset_mode="near_upright")
    td3 = config["td3"]
    n_actions = int(np.prod(train_env.action_space.shape))
    sigma = float(td3["action_noise_sigma"])
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions, dtype=np.float32),
        sigma=sigma * np.ones(n_actions, dtype=np.float32),
    )
    model = TD3(
        "MlpPolicy", train_env,
        learning_rate=float(td3["learning_rate"]),
        buffer_size=int(td3["buffer_size"]),
        learning_starts=int(td3["learning_starts"]),
        batch_size=int(td3["batch_size"]),
        tau=float(td3["tau"]),
        gamma=float(td3["gamma"]),
        train_freq=int(td3["train_freq"]),
        gradient_steps=int(td3["gradient_steps"]),
        action_noise=action_noise,
        policy_delay=int(td3["policy_delay"]),
        target_policy_noise=float(td3["target_policy_noise"]),
        target_noise_clip=float(td3["target_noise_clip"]),
        policy_kwargs={"net_arch": [int(v) for v in td3["net_arch"]]},
        seed=args.seed, device="cpu", verbose=1,
    )

    total_timesteps = int(config["total_timesteps"])
    cumulative_target = 0
    for video_index, item in enumerate(config["curriculum"], start=1):
        stage = str(item["stage"])
        mode = str(item["reset_mode"])
        fraction = float(item["fraction"])
        target = int(round(total_timesteps * fraction))
        additional = target - cumulative_target
        if additional <= 0:
            raise ValueError("Curriculum fractions must increase strictly")

        train_env.set_reset_mode(mode)
        model.get_env().reset()
        print(f"Curriculum stage={stage} reset_mode={mode} additional_steps={additional:,}")
        model.learn(total_timesteps=additional, reset_num_timesteps=False, progress_bar=False, log_interval=20)
        cumulative_target = target
        model.save(models_dir / f"{stage}.zip")
        record_stage(records, stage, fraction, int(model.num_timesteps), model,
                     evaluation_seeds, video_seed, videos_dir, video_index, physics, noise)

    train_env.close()
    write_metrics_csv(records, output_dir / "metrics.csv")
    create_plots(records, plots_dir)
    write_summary(records, output_dir / "summary.md", args.preset, args.seed)

    metadata = {
        "environment": "custom-continuous-cartpole-swingup",
        "algorithm": "TD3",
        "comparison_targets": [
            "temesotejam/rl-cartpole-swingup-ppo",
            "temesotejam/rl-cartpole-swingup-sac",
        ],
        "preset": args.preset,
        "seed": args.seed,
        "requested_total_timesteps": total_timesteps,
        "actual_final_timesteps": int(model.num_timesteps),
        "evaluation_seeds": evaluation_seeds,
        "video_seed": video_seed,
        "physics": physics.to_dict(),
        "sensor_noise": noise.to_dict(),
        "config": config,
        "versions": version_info(),
        "replay_buffer_retained_across_curriculum": True,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
