from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from stable_baselines3 import TD3

from .environment import PhysicsConfig, SensorNoiseConfig
from .evaluation import evaluate_episode, evaluate_policy

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved Cart-Pole swing-up TD3 model.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--preset", choices=["quick", "normal", "long"], default="normal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--video", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / "configs" / f"{args.preset}.yaml").read_text(encoding="utf-8"))
    physics = PhysicsConfig(**config["physics"])
    noise = SensorNoiseConfig(**config["sensor_noise"])
    model = TD3.load(args.model, device="cpu")
    seeds = [args.seed + i for i in range(args.episodes)]
    metrics = evaluate_policy(model, seeds, physics=physics, sensor_noise=noise)
    if args.video is not None:
        evaluate_episode(model, args.seed, physics=physics, sensor_noise=noise, video_path=args.video)
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
