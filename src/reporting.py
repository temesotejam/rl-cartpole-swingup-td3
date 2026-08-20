from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


FIELDS = [
    "stage", "progress", "timesteps", "mean_return", "std_return", "mean_survival_s",
    "capture_rate", "mean_capture_time_s", "final_stable_rate", "upright_ratio",
    "rms_angle_deg", "rms_cart_position_m", "rms_force_n",
]


def write_metrics_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def _plot(records: list[dict], key: str, ylabel: str, path: Path, scale: float = 1.0) -> None:
    xs = [record["timesteps"] for record in records]
    ys = [float(record[key]) * scale for record in records]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_plots(records: list[dict], plots_dir: Path) -> None:
    _plot(records, "mean_return", "Mean evaluation return", plots_dir / "learning_curve.png")
    _plot(records, "capture_rate", "Swing-up capture rate [%]", plots_dir / "capture_rate.png", 100.0)
    _plot(records, "final_stable_rate", "Final stable rate [%]", plots_dir / "final_stable_rate.png", 100.0)
    _plot(records, "upright_ratio", "Time within ±10° [%]", plots_dir / "upright_ratio.png", 100.0)
    _plot(records, "rms_cart_position_m", "RMS cart position [m]", plots_dir / "cart_position.png")
    _plot(records, "rms_force_n", "RMS motor force [N]", plots_dir / "motor_force.png")


def _fmt_time(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.2f}s"


def write_summary(records: list[dict], path: Path, preset: str, seed: int) -> None:
    best = max(records[1:], key=lambda r: (r["final_stable_rate"], r["capture_rate"], r["mean_return"]))
    lines = [
        "# Cart-Pole Swing-up TD3 training result", "",
        f"- Preset: `{preset}`", f"- Seed: `{seed}`",
        f"- Best checkpoint: `{best['stage']}`",
        f"- Best final-stable rate: `{best['final_stable_rate'] * 100:.1f}%`", "",
        "## Downward-start evaluation", "",
        "| Stage | Timesteps | Return | Capture | Time to capture | Final stable | Upright ±10° | RMS cart x | RMS force |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {r['stage']} | {int(r['timesteps']):,} | {r['mean_return']:.1f} | "
            f"{r['capture_rate']*100:.1f}% | {_fmt_time(float(r['mean_capture_time_s']))} | "
            f"{r['final_stable_rate']*100:.1f}% | {r['upright_ratio']*100:.1f}% | "
            f"{r['rms_cart_position_m']:.3f}m | {r['rms_force_n']:.2f}N |"
        )
    lines.extend([
        "", "Capture means the pole stayed within ±12° with low angular velocity for at least 0.5 s.",
        "Final stable means at least 80% of the final 2 s was within ±10° and ±0.50 m.", "",
        "Training uses TD3 with one environment and a replay buffer retained across curriculum stages.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
