# rl-cartpole-swingup-td3

同じ連続入力 Cart-Pole swing-up 問題を **Twin Delayed DDPG (TD3)** で学習し、既存の PPO / SAC 実験と比較するリポジトリです。

- PPO: `temesotejam/rl-cartpole-swingup-ppo`
- SAC: `temesotejam/rl-cartpole-swingup-sac`
- TD3: このリポジトリ

## 目的

比較で変えるのは基本的に学習アルゴリズムだけです。物理、reward、センサノイズ、初期状態カリキュラム、評価seed、評価指標、学習step数を揃えます。

主に比較するもの:

- Swing-up capture を覚えるまでの環境step数
- Capture成功率
- Captureまでの時間
- 最終2秒の安定化率
- ±10°以内の滞在率
- RMS台車位置
- RMSモータ力
- GitHub Actions上の学習時間

## 物理モデル

| 項目 | 値 |
|---|---:|
| 台車質量 | 1.0 kg |
| 振り子質量 | 0.1 kg |
| 振り子重心長 | 0.5 m |
| 最大駆動力 | ±10 N |
| 制御周期 | 20 ms (50 Hz) |
| レール制限 | ±2.4 m |
| モータ時定数 | 50 ms |
| 台車粘性摩擦 | 0.10 N/(m/s) |
| 1 episode | 20 s |

角度は `0 rad = 直立`, `±π rad = 真下` です。Swing-up中は角度が大きくても終了しません。台車がレール端を越えた場合だけ早期終了します。

## 観測

TD3に渡す観測は

```text
[x, x_dot, cos(theta), sin(theta), theta_dot]
```

です。ただし真値ではなく、民生用エンコーダ / IMU相当のノイズを加えます。

初期値:

- 台車位置 white noise σ = 1 mm
- 台車位置 episode bias σ = 2 mm
- 台車速度 white noise σ = 0.01 m/s
- 台車速度 episode bias σ = 0.01 m/s
- 角度 white noise σ = 0.25°
- 角度 episode bias σ = 1.0°
- gyro white noise σ = 0.10°/s
- gyro episode bias σ = 0.30°/s

シミュレータ真値は評価にだけ使い、学習器には見せません。

## reward

PPO / SAC版の修正版rewardをそのまま使います。真下で静止しているだけでも1step rewardが極端に負にならないようにし、意図的にレール端へ行ってepisodeを早く終える抜け道を防いでいます。

rewardは概ね、

- 直立しているほど高い
- 直立時に台車が中央ほど高い
- 真下付近では振り子運動を少し奨励
- 大きな速度・角速度・モータ力を小さく抑える
- レール端へ近づくほど強く罰する
- ±10°以内、さらに低角速度 + 中央付近なら追加報酬

という構成です。

## TD3

TD3はDDPGを安定化したoff-policyアルゴリズムです。

この実験では次の3点が中心です。

1. **Twin critics** — 2本のQ関数の小さい方を使い、Q値の過大評価を抑える
2. **Delayed policy update** — criticよりactorを低頻度で更新する
3. **Target policy smoothing** — target actionに小さなノイズを加え、狭いQピークへの過適合を抑える

探索時にはGaussian action noiseを加えます。SACのentropy自動調整とは違い、探索ノイズ量は明示的に設定します。

normal初期設定:

```yaml
learning_rate: 0.001
buffer_size: 500000
learning_starts: 5000
batch_size: 256
tau: 0.005
gamma: 0.99
train_freq: 1
gradient_steps: 1
action_noise_sigma: 0.20
policy_delay: 2
target_policy_noise: 0.20
target_noise_clip: 0.50
net_arch: [256, 256]
```

Replay Bufferはカリキュラム段階をまたいで保持します。

## カリキュラム

400k normalでは4段階です。

| checkpoint | 累積step | 学習初期角度 |
|---|---:|---|
| 25% | 100k | 直立近傍 |
| 50% | 200k | ±120°を含む |
| 75% | 300k | 全角度 |
| 100% | 400k | 真下70% + 全角度 + 直立近傍 |

ただし各checkpointの評価はカリキュラム状態ではなく、毎回同じ **ほぼ真下スタート** です。

## 評価

### Capture

振り子が

- ±12°以内
- |角速度| ≤ 1.5 rad/s

を0.5秒以上連続で満たしたときSwing-up capture成功とします。

### Final stable

最後の2秒の80%以上で

- ±10°以内
- 台車位置 ±0.50 m以内

を満たすと最終安定化成功です。

## 出力

学習ごとに生成します。

```text
results/
├─ models/
│  ├─ 25_percent.zip
│  ├─ 50_percent.zip
│  ├─ 75_percent.zip
│  └─ 100_percent.zip
├─ videos/
│  ├─ 00_random.mp4
│  ├─ 01_25_percent.mp4
│  ├─ 02_50_percent.mp4
│  ├─ 03_75_percent.mp4
│  └─ 04_100_percent.mp4
├─ plots/
├─ metrics.csv
├─ metadata.json
└─ summary.md
```

## 学習量

```text
quick   20,000 step
normal 400,000 step
long   800,000 step
```

PPO / SACとの比較ではnormal 400kを基本にします。

## GitHub Actions

PR CIではテストだけでなく、quick 20kのTD3学習を実際に実行します。モデル、5段階動画、評価CSV、グラフ、Pages用HTMLまで生成できることを確認してからmainへ入れます。

`main`へ初回実装をマージすると、`.github/training-trigger/`によって `normal / seed 42 / 400k` が自動実行されます。

## GitHub Pages

成功した最新学習結果はGitHub Pagesへ自動公開する構成です。

予定URL:

`https://temesotejam.github.io/rl-cartpole-swingup-td3/`

ページでは random / 25% / 50% / 75% / 100% の動画、Capture率、Final stable率、±10°滞在率、台車位置、モータ力を確認できます。

## 比較で重要な点

PPO / SAC / TD3は性質がかなり違います。

- PPO: on-policy。サンプル再利用は少ないが、今回のPPOでは最終安定化が強かった
- SAC: off-policy + stochastic actor。今回のSACではSwing-up習得が非常に速かった
- TD3: off-policy + deterministic actor。SACのentropy探索とは別の方法で連続制御を学ぶ

このためTD3で特に見たいのは、**SAC並みのサンプル効率を保ちながら、PPOに近い最終整定性を出せるか**です。
