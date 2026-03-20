# 2x2 SUMO-RL Thesis Pipeline

Minimal, reproducible experiment pipeline for traffic-light control on a 2x2 grid using SUMO, SUMO-RL, and CleanRL-style PPO implementations.

## What is implemented

- Reproducible route generation with low / medium / high demand regimes
- SUMO default baseline evaluation
- Centralized single-agent PPO with factorized joint actions
- Multi-agent PPO with independent agents
- Multi-agent PPO with parameter sharing
- MAPPO with shared actors and a centralized critic
- Checkpoint evaluation and result aggregation

The code forces `libsumo` at runtime because the local environment blocks socket binding, which breaks standard TraCI startup.

## Layout

```text
configs/
  env.yaml
  ppo_common.yaml
  ppo_centralized.yaml
  ppo_shared.yaml
  ppo_independent.yaml
  ppo_mappo.yaml
  experiment_matrix.yaml
nets/2x2/
  2x2.net.xml
scripts/
  generate_routes.py
  run_baseline.py
  train_centralized_ppo.py
  train_shared_ppo.py
  train_independent_ppo.py
  train_mappo.py
  eval_policy.py
  aggregate_results.py
  compare_results.py
src/traffic_rl/
  runtime.py
  routes.py
  envs.py
  models.py
  algo_*.py
```

## Recommended workflow

Generate reproducible route files:

```bash
python scripts/generate_routes.py
```

Run the SUMO baseline on the held-out test routes:

```bash
python scripts/run_baseline.py --split test
```

Render the SUMO UI for the baseline:

```bash
python scripts/run_baseline.py --split test --intensity medium --gui
```

Train one controller:

```bash
python scripts/train_shared_ppo.py --intensity medium --seed 0
python scripts/train_centralized_ppo.py --intensity medium --seed 0
python scripts/train_independent_ppo.py --intensity medium --seed 0
python scripts/train_mappo.py --intensity medium --seed 0
```

Evaluate a checkpoint:

```bash
python scripts/eval_policy.py \
  --checkpoint results/checkpoints/shared_ppo/medium/seed_0/final.pt \
  --split test \
  --intensity medium
```

Render the SUMO UI during evaluation:

```bash
python scripts/eval_policy.py \
  --checkpoint results/checkpoints/shared_ppo/medium/seed_0/final.pt \
  --split test \
  --intensity medium \
  --gui
```

Aggregate evaluation tables:

```bash
python scripts/aggregate_results.py
```

Build a comparison table and an HTML dashboard with charts:

```bash
python scripts/compare_results.py --split test --intensity medium
```

This writes:

```text
results/compare/comparison_test_medium.csv
results/compare/comparison_test_medium.md
results/compare/comparison_test_medium.html
```

Omit `--intensity` to build a single overall dashboard spanning all available intensities:

```bash
python scripts/compare_results.py --split test
```

This writes:

```text
results/compare/comparison_test_all.csv
results/compare/comparison_test_all.md
results/compare/comparison_test_all.html
```

Open TensorBoard for training curves:

```bash
tensorboard --logdir results/tensorboard
```

Each training run writes a separate TensorBoard directory under:

```text
results/tensorboard/<algorithm>/<intensity>/seed_<seed>/<timestamp>/
```

Logged scalars include:

- `train/learning_rate`
- `train/sps`
- `train/recent_return_mean`
- `losses/policy_loss`
- `losses/value_loss`
- `losses/entropy`
- `losses/approx_kl`
- `losses/clipfrac`
- `losses/explained_variance`
- episode traffic metrics such as waiting time, speed, throughput, queue length, time loss, and teleports

## Smoke-test overrides

For quick checks on a laptop, reduce training budget from the command line:

```bash
python scripts/train_shared_ppo.py \
  --intensity low \
  --seed 0 \
  --total-timesteps 64 \
  --num-steps 16
```

## Notes

- `total_timesteps` is counted in environment decision steps, not per-agent samples.
- Multi-agent methods therefore consume more PPO samples per environment step than centralized PPO.
- Generated test routes are deterministic by seed and can be reused across all algorithms for fair comparison.
- Training logs are written as JSONL under `results/train_logs/`.
- Evaluation outputs are stored under `results/eval/<algorithm>/<split>/<intensity>/`.
- Training prints one progress line per PPO update with percent complete, steps, recent return, SPS, and ETA.
