# Prism Handoff

This folder contains copied files for writing the thesis chapters:

- System Model and Proposed Method
- Experimental Setup and Results

The original project files were not moved.

## Main Result Sources

Use these as the authoritative result tables:

- `results/compare/comparison_test_all.csv`
- `results/compare/comparison_test_all.html`
- `results/tables/evaluation_across_seeds.csv`
- `results/tables/evaluation_per_seed.csv`

The comparison CSV is built from `results/eval_across_seeds`.
RL rows have `seed_count = 5`, `train_seeds = 0,1,2,3,4`, and `episodes = 50`.
Baseline rows have `seed_count = 1` and `episodes = 10` because the baseline has no training seed.

## Experiment Context

- Network: 2x2 SUMO grid
- Controlled traffic lights: `1`, `2`, `5`, `6`
- Episode length: 1800 simulated seconds
- Decision interval: 5 seconds
- Demand levels: low, medium, high
- Variants: vehicle-only and pedestrian-enabled
- Test route seeds: 100-109
- PPO training budget: 300000 environment decision steps for every learned approach

Learned methods:

- Centralized PPO
- Independent PPO
- Shared-parameter PPO
- MAPPO

Baseline:

- SUMO default/fixed traffic-light control

## Suggested Claims

- PPO controllers improve vehicle-only performance over the SUMO baseline.
- Multi-agent PPO variants outperform centralized PPO.
- MAPPO is strongest overall in vehicle-only settings, with shared PPO close behind.
- Shared PPO slightly outperforms MAPPO in high-demand vehicle-only traffic.
- Pedestrian-enabled scenarios are harder; centralized PPO is fragile under high pedestrian demand.
- MAPPO and shared PPO provide the best overall vehicle-pedestrian trade-off.

Avoid claiming statistical significance unless formal tests are added.
