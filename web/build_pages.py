from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from pathlib import Path


def pct(value: str) -> str:
    return f"{float(value) * 100:.1f}%"


def build(input_dir: Path, output_dir: Path, run_url: str) -> None:
    rows = list(csv.DictReader((input_dir / "metrics.csv").open(encoding="utf-8")))
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for directory in ["videos", "plots"]:
        source = input_dir / directory
        if source.exists():
            shutil.copytree(source, output_dir / directory)

    table_rows = []
    for row in rows:
        capture_time = row["mean_capture_time_s"]
        try:
            capture_time_text = "-" if capture_time.lower() == "nan" else f"{float(capture_time):.2f}s"
        except Exception:
            capture_time_text = capture_time
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['stage'])}</td>"
            f"<td>{int(float(row['timesteps'])):,}</td>"
            f"<td>{float(row['mean_return']):.1f}</td>"
            f"<td>{pct(row['capture_rate'])}</td>"
            f"<td>{capture_time_text}</td>"
            f"<td>{pct(row['final_stable_rate'])}</td>"
            f"<td>{pct(row['upright_ratio'])}</td>"
            f"<td>{float(row['rms_cart_position_m']):.3f} m</td>"
            f"<td>{float(row['rms_force_n']):.2f} N</td>"
            "</tr>"
        )

    videos = [
        ("random", "videos/00_random.mp4"), ("25%", "videos/01_25_percent.mp4"),
        ("50%", "videos/02_50_percent.mp4"), ("75%", "videos/03_75_percent.mp4"),
        ("100%", "videos/04_100_percent.mp4"),
    ]
    video_buttons = "".join(
        f'<button data-src="{src}" onclick="switchVideo(this)">{label}</button>' for label, src in videos
    )
    td3 = metadata.get("config", {}).get("td3", {})
    sensor = metadata.get("sensor_noise", {})
    html_text = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cart-Pole Swing-up TD3</title>
<style>
:root{{font-family:system-ui,sans-serif}}body{{margin:0;background:#0d1117;color:#e6edf3}}main{{max-width:1120px;margin:auto;padding:32px 20px 64px}}
a{{color:#58a6ff}}.hero,.card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:22px;margin:18px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}video,img{{width:100%;border-radius:10px;background:#000}}.controls{{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}}button{{padding:8px 14px;border-radius:8px;border:1px solid #30363d;background:#21262d;color:#e6edf3;cursor:pointer}}button.active{{border-color:#58a6ff;background:#1f3b5b}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{border-bottom:1px solid #30363d;padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.badge{{display:inline-block;padding:4px 9px;margin:3px;border-radius:999px;background:#21262d}}.small{{color:#8b949e;font-size:14px}}
</style></head><body><main>
<section class="hero"><h1>Cart-Pole Swing-up — TD3</h1>
<p>PPO / SACと同じ連続入力Cart-Pole環境を <strong>Twin Delayed DDPG (TD3)</strong> で学習した結果です。</p>
<span class="badge">seed {metadata.get('seed')}</span><span class="badge">{int(metadata.get('actual_final_timesteps',0)):,} steps</span>
<span class="badge">Replay buffer: {int(td3.get('buffer_size',0)):,}</span><span class="badge">Action noise σ={td3.get('action_noise_sigma')}</span>
<p><a href="{html.escape(run_url)}">元のGitHub Actions run</a> · <a href="https://temesotejam.github.io/rl-cartpole-swingup-ppo/">PPO版</a> · <a href="https://temesotejam.github.io/rl-cartpole-swingup-sac/">SAC版</a></p></section>
<section class="card"><h2>学習の進行を動画で比較</h2><div class="controls">{video_buttons}</div><video id="trainingVideo" controls muted loop playsinline src="videos/00_random.mp4"></video></section>
<section class="card"><h2>評価指標</h2><div style="overflow:auto"><table><thead><tr><th>Stage</th><th>Steps</th><th>Return</th><th>Capture</th><th>Capture time</th><th>Final stable</th><th>±10°</th><th>RMS cart x</th><th>RMS force</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div></section>
<section class="grid"><div class="card"><h2>Return</h2><img src="plots/learning_curve.png"></div><div class="card"><h2>Capture rate</h2><img src="plots/capture_rate.png"></div><div class="card"><h2>Final stable</h2><img src="plots/final_stable_rate.png"></div><div class="card"><h2>Upright ratio</h2><img src="plots/upright_ratio.png"></div><div class="card"><h2>Cart position</h2><img src="plots/cart_position.png"></div><div class="card"><h2>Motor force</h2><img src="plots/motor_force.png"></div></section>
<section class="card"><h2>TD3設定</h2><p>learning rate <code>{td3.get('learning_rate')}</code> / gamma <code>{td3.get('gamma')}</code> / tau <code>{td3.get('tau')}</code> / policy delay <code>{td3.get('policy_delay')}</code> / target noise <code>{td3.get('target_policy_noise')}</code></p><p>探索にはGaussian action noiseを使い、criticを2本持ち、actor更新を遅らせて過大評価を抑えます。</p></section>
<section class="card"><h2>センサノイズ</h2><p>角度 σ={sensor.get('angle_noise_std_deg')}°、角度bias σ={sensor.get('angle_bias_std_deg')}°、gyro σ={sensor.get('gyro_noise_std_dps')}°/s、位置 σ={sensor.get('position_noise_std_m')} m。</p></section>
<script>function switchVideo(button){{const video=document.getElementById('trainingVideo');document.querySelectorAll('.controls button').forEach(b=>b.classList.remove('active'));button.classList.add('active');video.src=button.dataset.src;video.load();video.play().catch(()=>{{}})}}document.querySelector('.controls button').classList.add('active');</script>
</main></body></html>"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-url", default="")
    args = parser.parse_args()
    build(args.input, args.output, args.run_url)


if __name__ == "__main__":
    main()
