# Cart-Pole Swing-up TD3 training result

- Preset: `normal`
- Seed: `42`
- Best checkpoint: `100_percent`
- Best final-stable rate: `100.0%`

## Downward-start evaluation

| Stage | Timesteps | Return | Capture | Time to capture | Final stable | Upright ±10° | RMS cart x | RMS force |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0 | 9.4 | 0.0% | - | 0.0% | 0.0% | 1.062m | 2.58N |
| 25_percent | 100,000 | 336.7 | 33.3% | 10.34s | 8.3% | 16.9% | 0.434m | 1.91N |
| 50_percent | 200,000 | 1617.0 | 100.0% | 3.63s | 91.7% | 84.1% | 0.325m | 2.68N |
| 75_percent | 300,000 | 1592.9 | 100.0% | 2.33s | 75.0% | 90.3% | 0.418m | 2.76N |
| 100_percent | 400,000 | 1705.7 | 100.0% | 2.14s | 100.0% | 91.5% | 0.340m | 2.50N |

Capture means the pole stayed within ±12° with low angular velocity for at least 0.5 s.
Final stable means at least 80% of the final 2 s was within ±10° and ±0.50 m.

Training uses TD3 with one environment and a replay buffer retained across curriculum stages.
