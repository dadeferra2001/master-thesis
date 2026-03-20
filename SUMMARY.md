# Codebase Summary

This repository is a reproducible experiment pipeline for traffic-light control on a 2x2 SUMO grid.
It covers the full loop:

- build or reuse the network
- generate deterministic train/test route sets
- train multiple PPO variants
- evaluate baseline and learned controllers
- aggregate metrics
- render comparison tables and HTML dashboards

Most of the logic lives in `src/traffic_rl/`.
Most of the disk usage comes from generated artifacts under `routes/` and `results/`.

## At A Glance

Current repo contents that matter operationally:

- 18 Python modules under `src/traffic_rl/`
- 10 executable scripts under `scripts/`
- 60 generated training route files under `routes/train/`
- 30 generated test route files under `routes/test/`
- 15 evaluation aggregate files under `results/eval/`
- 12 comparison outputs under `results/compare/`

Implemented controllers:

- `baseline`: SUMO fixed/default traffic light logic
- `centralized_ppo`: one policy controls all 4 lights jointly
- `independent_ppo`: 4 separate PPO agents
- `shared_ppo`: one shared PPO policy for all 4 agents
- `mappo`: shared actor plus centralized critic

Important environment constraint:

- the code defaults to `libsumo` because the local environment blocks the normal socket-based TraCI startup
- GUI runs switch back to pure `traci`

## Big Picture

```text
                 +-------------------+
                 |  configs/*.yaml   |
                 +---------+---------+
                           |
                           v
+-------------+    +-------+--------+    +----------------------+
| nets/2x2/*  | -> | scripts/*.py   | -> | src/traffic_rl/*.py  |
+-------------+    +-------+--------+    +----------+-----------+
                           |                        |
                           |                        |
                           v                        v
                +-------------------+    +-----------------------+
                | routes/*.rou.xml  |    | SUMO / SUMO-RL envs   |
                | manifests/*.json  |    | PPO models/controllers|
                +---------+---------+    +-----------+-----------+
                          \                         /
                           \                       /
                            v                     v
                         +---------------------------+
                         | results/                  |
                         | checkpoints/              |
                         | train_logs/               |
                         | tensorboard/              |
                         | eval/                     |
                         | compare/                  |
                         | tables/                   |
                         +---------------------------+
```

## Scenario Model

The experiment uses a 2x2 road grid with four signalized intersections.
The traffic light ids are fixed as `1`, `2`, `5`, and `6`.

Approximate layout:

```text
                 north
                   |
                   v
      west -->   [1] ---- [2]   --> east
                  |        |
                  |        |
      west -->   [5] ---- [6]   --> east
                   ^
                   |
                 south
```

Mental model:

- `1`: top-left
- `2`: top-right
- `5`: bottom-left
- `6`: bottom-right

Traffic demand is generated from 8 boundary origins:

- west-north, west-south
- east-north, east-south
- north-west, north-east
- south-west, south-east

Each origin emits vehicles that go:

- straight
- left
- right

With the default route config:

- `num_windows = 4`
- `8 origins * 3 turns * 4 windows = 96 flows` per standard route file

The smoke-test manifest uses:

- `episode_seconds = 60`
- `num_windows = 2`
- `8 * 3 * 2 = 48 flows`

## End-To-End Workflow

```text
1. scripts/build_2x2_network.py
   plain XML parts -> nets/2x2/2x2.net.xml

2. scripts/generate_routes.py
   configs/env.yaml -> routes/train/* + routes/test/* + manifest JSON

3. scripts/train_*.py
   train manifest + env config + PPO config -> checkpoints + JSONL logs + TensorBoard

4. scripts/run_baseline.py / scripts/eval_policy.py
   route files + controller -> step_metrics.csv + summary.json + aggregate.json

5. scripts/aggregate_results.py
   aggregate.json files -> one CSV table

6. scripts/compare_results.py
   aggregate.json files -> comparison CSV + Markdown + HTML dashboard
```

A more detailed data-flow view:

```text
generate_routes.py
    |
    +--> routes/train/<intensity>/seed_XXX.rou.xml
    +--> routes/test/<intensity>/seed_XXX.rou.xml
    +--> routes/manifests/routes_manifest.json

train_shared_ppo.py / train_centralized_ppo.py / train_independent_ppo.py / train_mappo.py
    |
    +--> results/checkpoints/<algo>/<intensity>/seed_<seed>/*.pt
    +--> results/train_logs/<algo>/<intensity>/seed_<seed>.jsonl
    +--> results/tensorboard/<algo>/<intensity>/seed_<seed>/<timestamp>/

run_baseline.py / eval_policy.py
    |
    +--> results/eval/<algo>/<split>/<intensity>/route_<seed>/step_metrics.csv
    +--> results/eval/<algo>/<split>/<intensity>/route_<seed>/summary.json
    +--> results/eval/<algo>/<split>/<intensity>/route_<seed>/tripinfo.xml
    +--> results/eval/<algo>/<split>/<intensity>/episodes.csv
    +--> results/eval/<algo>/<split>/<intensity>/aggregate.json

aggregate_results.py
    |
    +--> results/tables/evaluation_summary.csv

compare_results.py
    |
    +--> results/compare/comparison_<split>_<intensity-or-all>.csv
    +--> results/compare/comparison_<split>_<intensity-or-all>.md
    +--> results/compare/comparison_<split>_<intensity-or-all>.html
```

## Top-Level Directory Map

| Path | Purpose | Notes |
| --- | --- | --- |
| `configs/` | YAML configuration files | Scenario config, common PPO config, per-algorithm overrides |
| `nets/2x2/` | SUMO network files | Contains final `2x2.net.xml`, source XML parts, and `tls_ids.json` |
| `routes/` | Generated route inputs | Train/test route files plus manifests |
| `scripts/` | User-facing CLI entry points | Thin wrappers around `src/traffic_rl` |
| `src/traffic_rl/` | Core implementation | Envs, models, algorithms, metrics, reporting |
| `tests/` | Smoke tests | Validates route generation, env stepping, one training path, report generation |
| `results/` | Generated experiment outputs | Checkpoints, logs, eval artifacts, comparison dashboards |
| `README.md` | Usage-oriented overview | Good for command examples |
| `PLAN.md` | Research/experiment plan | Higher-level thesis design notes |
| `METRICS.md` | Metric definitions | Best reference for what each output field means |

## Config System

Config loading is intentionally simple.
`traffic_rl.config.load_experiment_config(...)` loads YAML files and recursively merges dictionaries from left to right.

That is why the training scripts pass both:

- `configs/ppo_common.yaml`
- one algorithm-specific override like `configs/ppo_shared.yaml`

### `configs/env.yaml`

This file has three major sections:

- `scenario`
- `route_generation`
- `evaluation`

Key defaults:

| Key | Default |
| --- | --- |
| `scenario.net_file` | `nets/2x2/2x2.net.xml` |
| `scenario.tls_ids` | `["1", "2", "5", "6"]` |
| `scenario.episode_seconds` | `1800` |
| `scenario.delta_time` | `5` |
| `scenario.yellow_time` | `2` |
| `scenario.min_green` | `5` |
| `scenario.max_green` | `50` |
| `scenario.reward_fn` | `diff-waiting-time` |
| `scenario.time_to_teleport` | `-1` |
| `route_generation.intensity_scale.low` | `0.55` |
| `route_generation.intensity_scale.medium` | `1.0` |
| `route_generation.intensity_scale.high` | `1.30` |
| `route_generation.base_origin_vehs_per_hour` | `420` |
| `evaluation.train_route_seeds` | `0..19` |
| `evaluation.test_route_seeds` | `100..109` |

Meaningful implications:

- one decision step usually represents 5 simulated seconds
- teleports are disabled by default
- the generator makes train and test route sets from separate seed ranges

### PPO configs

`configs/ppo_common.yaml` defines shared PPO defaults:

- `total_timesteps = 300000`
- `learning_rate = 3e-4`
- `gamma = 0.99`
- `gae_lambda = 0.95`
- `num_steps = 256`
- `update_epochs = 4`
- `num_minibatches = 4`
- `clip_coef = 0.2`
- `vf_coef = 0.5`
- `ent_coef = 0.01`
- `hidden_sizes = [128, 128]`

Per-algorithm overrides:

- `ppo_centralized.yaml`: larger network and `total_timesteps = 400000`
- `ppo_shared.yaml`: enables `agent_id: true`
- `ppo_independent.yaml`: keeps local non-shared setup
- `ppo_mappo.yaml`: enables `agent_id: true`, lower `total_timesteps`

### `configs/experiment_matrix.yaml`

This file looks like a batch-run plan, but the current codebase does not consume it directly.
It is documentation/config inventory, not an active orchestrator input.

Important nuance:

- `mappo` is marked `enabled: false` here
- but MAPPO is fully implemented and there are MAPPO checkpoints/evaluations already present under `results/`

## Scripts

The scripts are thin wrappers.
They mainly:

- parse CLI args
- load configs and manifests
- call into `src/traffic_rl`
- print output paths

| Script | What it does |
| --- | --- |
| `scripts/build_2x2_network.py` | Runs `netconvert` on the plain XML pieces to rebuild `nets/2x2/2x2.net.xml` |
| `scripts/generate_routes.py` | Generates train/test route files and a manifest JSON |
| `scripts/run_baseline.py` | Evaluates SUMO fixed/default traffic light control on selected routes |
| `scripts/train_centralized_ppo.py` | Trains centralized PPO |
| `scripts/train_independent_ppo.py` | Trains one PPO per traffic light |
| `scripts/train_shared_ppo.py` | Trains parameter-sharing PPO |
| `scripts/train_mappo.py` | Trains MAPPO |
| `scripts/eval_policy.py` | Loads a checkpoint, infers its algorithm, evaluates it, writes aggregates |
| `scripts/aggregate_results.py` | Collects all `aggregate.json` files into one CSV |
| `scripts/compare_results.py` | Builds comparison CSV/Markdown/HTML reports from evaluation aggregates |

Operational note:

- the four training scripts are nearly identical at the CLI layer
- the real differences live in the `algo_*.py` modules

## `src/traffic_rl/` Module Map

### Configuration and runtime

| Module | Responsibility |
| --- | --- |
| `config.py` | Path resolution, YAML loading, recursive config merge, JSON helpers |
| `runtime.py` | Prepares `SUMO_HOME`, injects SUMO tools into `sys.path`, and preloads `libstdc++` if needed |
| `scenario.py` | Defines `TLS_IDS`, the 8 route origins, turn destination logic, and agent index mapping |
| `routes.py` | Route generation and manifest generation |

### Environment and inference

| Module | Responsibility |
| --- | --- |
| `envs.py` | Builds raw SUMO-RL environments and exposes centralized and parallel wrappers |
| `controllers.py` | Inference-time wrappers around trained models |
| `models.py` | Neural network architectures used by PPO variants |

### Training

| Module | Responsibility |
| --- | --- |
| `train_common.py` | GAE, minibatch shuffling, LR annealing, checkpoints, TensorBoard progress reporting |
| `algo_centralized.py` | Centralized PPO training and checkpoint loading |
| `algo_independent.py` | Independent PPO training and checkpoint loading |
| `algo_shared.py` | Parameter-sharing PPO training and checkpoint loading |
| `algo_mappo.py` | MAPPO training and checkpoint loading |

### Evaluation and reporting

| Module | Responsibility |
| --- | --- |
| `evaluation.py` | Runs one baseline or policy episode and writes episode outputs |
| `metrics.py` | Converts step history and `tripinfo.xml` into episode summaries |
| `reporting.py` | Aggregates many episode summaries into mean/std rows |
| `comparison.py` | Builds comparison rows and renders Markdown/HTML dashboards |
| `utils.py` | Seeds RNGs, picks device, resolves standard output locations, writes CSV/JSON/JSONL |

## Route Generation

Route generation happens in `traffic_rl.routes`.

High-level logic:

```text
for split in [train, test]:
  for intensity in [low, medium, high]:
    for seed in chosen seeds:
      create one .rou.xml file
      for each time window:
        for each boundary origin:
          sample noisy origin demand
          sample turn shares from a Dirichlet
          create straight/left/right flow entries
```

Key design choices:

- deterministic per seed because NumPy RNG is explicitly seeded
- separate train and test seeds
- stochastic demand inside each episode via intensity scaling
- stochastic demand inside each episode via per-window profile multipliers
- stochastic demand inside each episode via lognormal multiplicative noise
- stochastic demand inside each episode via Dirichlet turn share sampling
- a manifest JSON is written so every route file can be rediscovered later

Default manifest contents:

- generation timestamp
- episode length
- active intensities
- one record per route file with path, intensity, seed, split, flow count, and base rate

## Environment Layer

`traffic_rl.envs` is the bridge between the repo and SUMO-RL.

### Backend switching

`configure_sumo_backend(use_gui=False)` does this:

- if `use_gui=False`, patch SUMO-RL to use `libsumo`
- if `use_gui=True`, patch SUMO-RL back to socket-based `traci`

This is one of the most important implementation details in the repo.
Without it, the sandboxed environment would break normal TraCI startup.

### Raw environment builder

`build_env_kwargs(...)` maps the YAML config to SUMO-RL arguments, including:

- network path
- route file
- episode length
- action interval
- min/max green
- reward function
- traffic light ids
- tripinfo output flags

### Two wrappers

`CentralizedGridEnv`

- wraps the raw multi-agent environment as one `gymnasium.Env`
- observation = all 4 local observations concatenated into one vector
- action = `MultiDiscrete([A, A, A, A])`
- reward = sum of the 4 local rewards

`ParallelGridEnv`

- exposes the SUMO-RL parallel multi-agent API directly
- observation and action stay keyed by traffic light id
- used by independent PPO, shared PPO, MAPPO, and decentralized evaluation

## Model Layer

There are three neural-network families.

### `LocalActorCritic`

Used by:

- `independent_ppo`
- `shared_ppo`

Shape:

- MLP encoder
- single categorical actor head
- single scalar critic head

### `FactorizedJointActorCritic`

Used by:

- `centralized_ppo`

Shape:

- one shared encoder over the global observation
- one linear actor head that emits `num_agents * action_dim` logits
- logits are reshaped into 4 categorical distributions
- one scalar critic for the full system state

### `MAPPOActorCritic`

Used by:

- `mappo`

Shape:

- actor encoder on local observation
- critic encoder on concatenated global observation
- actor outputs one local categorical action
- critic outputs one value per agent index

## Controller Layer

The controller classes in `controllers.py` are inference adapters.
They hide model-specific tensor preparation from evaluation code.

Helpers:

- `stack_local_observations(...)`
- `concatenate_global_observation(...)`
- `append_agent_id_features(...)`

Controllers:

- `CentralizedController`: takes one flat global observation, returns 4 actions
- `SharedController`: takes observation dict, optionally appends one-hot agent identity, returns per-agent actions
- `IndependentController`: runs 4 separate models, one per intersection
- `MAPPOController`: builds both local and global tensors before acting

## Training Architecture

All PPO trainers share the same skeleton:

```text
sample one training route
reset env

for each PPO update:
  collect num_steps of rollout data
  if an episode ends:
    summarize env.metrics
    append JSONL training row
    log to TensorBoard
    sample another route and reset

  bootstrap next value
  compute GAE returns
  run PPO minibatch updates
  log losses and SPS
  save checkpoint every N updates

save final.pt
```

Common PPO machinery comes from `train_common.py`:

- `compute_gae(...)`
- `minibatch_indices(...)`
- `maybe_anneal_lr(...)`
- `save_checkpoint(...)`
- `ProgressReporter`

`ProgressReporter` writes:

- console progress lines
- TensorBoard scalars
- recent episode return trend
- loss metrics

### Algorithm differences

| Algorithm | Env wrapper | Observation | Action | Model | Critic style |
| --- | --- | --- | --- | --- | --- |
| `centralized_ppo` | `CentralizedGridEnv` | one concatenated global vector | one 4-part joint action | `FactorizedJointActorCritic` | centralized scalar |
| `independent_ppo` | `ParallelGridEnv` | one local vector per agent | one local discrete action per agent | 4 separate `LocalActorCritic` models | local, one critic per model |
| `shared_ppo` | `ParallelGridEnv` | local obs, optionally with one-hot agent id | local discrete action | one shared `LocalActorCritic` | local shared critic |
| `mappo` | `ParallelGridEnv` | actor sees local obs, critic sees concatenated global obs | local discrete action | `MAPPOActorCritic` | centralized critic with per-agent values |

Another useful view:

```text
Centralized PPO
  [obs1|obs2|obs5|obs6] -> one network -> [a1 a2 a5 a6]

Independent PPO
  obs1 -> net1 -> a1
  obs2 -> net2 -> a2
  obs5 -> net5 -> a5
  obs6 -> net6 -> a6

Shared PPO
  obs1 + id1 --\
  obs2 + id2 --- one shared network -> per-agent actions
  obs5 + id5 --/
  obs6 + id6 -/

MAPPO
  actor:  local obs (+ id) -> per-agent actions
  critic: [obs1|obs2|obs5|obs6] -> value for each agent index
```

### Training semantics that matter

- `total_timesteps` means environment decision steps, not total per-agent samples
- because of that, decentralized methods process more PPO samples per environment step than centralized PPO
- centralized PPO batch size is `num_steps`
- shared PPO and MAPPO batch size is `num_steps * 4`
- independent PPO updates 4 separate `num_steps` batches

## Evaluation Pipeline

Evaluation logic lives in `traffic_rl.evaluation`.

### Baseline evaluation

`run_baseline_episode(...)`:

- builds a raw env with `fixed_ts=True`
- steps it using empty action dicts
- writes `step_metrics.csv`
- writes `summary.json`

### Learned policy evaluation

`run_policy_episode(...)`:

- loads either centralized or decentralized env path depending on the algorithm
- acts deterministically by default
- accumulates total episode return
- writes the same output files as the baseline

Per-route output directory:

```text
results/eval/<algorithm>/<split>/<intensity>/route_<seed>/
```

Then each script also writes:

- `episodes.csv`: one row per evaluated route
- `aggregate.json`: mean/std over all evaluated routes in that group

## Metrics Pipeline

The cleanest explanation is:

```text
SUMO / SUMO-RL step history
    |
    +--> step_metrics.csv        (raw decision-step rows)
    |
    +--> summarize_episode(...)
           |
           +--> summary.json     (one evaluated route / episode)
           |
           +--> aggregate.json   (mean/std over many episodes)
```

Main metric code:

- `metrics.py`
- `reporting.py`
- `evaluation.py`
- `METRICS.md`

### What `metrics.py` computes

From per-step history:

- `num_decision_steps`
- `mean_average_speed`
- `mean_average_waiting_time`
- `mean_queue_length`
- `max_queue_length`
- `throughput`
- `teleports`
- `mean_backlog`
- `final_backlog`

From `tripinfo.xml`:

- `completed_trips`
- `mean_travel_time`
- `mean_time_loss`
- `mean_trip_waiting_time`
- `mean_depart_delay`

Derived flag:

- `congestion_failure = teleports > 0 or final_backlog > 0`

### Reporting behavior

`reporting.aggregate_episode_summaries(...)` aggregates every numeric field it sees.
That includes some fields that are bookkeeping rather than scientific metrics, such as:

- `route_seed`
- `update`
- `global_step`
- booleans coerced to `0/1`

That is why `aggregate.json` and the global CSV can contain extra columns beyond the main traffic metrics.

### Training-log caveat

Training JSONL rows are built from `env.history` only.
They do not pass a `tripinfo.xml` path into `summarize_episode(...)`.

So during training logs:

- `mean_travel_time`
- `mean_time_loss`
- `mean_trip_waiting_time`
- `completed_trips`

can be zero or incomplete even when the policy is behaving sensibly.

Those fields are only meaningful in the evaluation outputs that actually write `tripinfo.xml`.

## Comparison Reporting

`traffic_rl.comparison` turns evaluation aggregates into:

- comparison CSV
- Markdown report
- HTML dashboard

Features:

- metric-direction aware improvement percentages
- algorithm labels and colors
- per-intensity tables
- per-metric comparison cards
- multi-intensity overview heatmap
- simple SVG trend charts across low/medium/high

The default metrics used in the comparison layer are:

- mean speed
- mean waiting time
- mean travel time
- throughput
- mean queue length
- mean time loss
- teleports

## Testing

There is one main smoke test file: `tests/test_pipeline.py`.

What it checks:

- route manifest generation works
- centralized env can reset and step
- parallel env can reset and step
- backend switching between `libsumo` and `traci` works
- centralized env truly controls all four lights
- shared PPO can run a tiny smoke training session and write a checkpoint plus TensorBoard events
- comparison report generation works for one intensity and across all intensities

What it does not do:

- exhaustive training validation for all algorithms
- regression tests for metric values
- deep correctness checks on HTML layout

So this is a useful smoke suite, not a full verification harness.

## Current Artifact State

The repo already contains generated experiment outputs.

Visible structure on disk:

- checkpoints for all four RL algorithms under `results/checkpoints/`
- TensorBoard runs for all four RL algorithms
- training JSONL logs for all four RL algorithms
- evaluation aggregates for all five algorithms, including baseline
- comparison dashboards for `low`
- comparison dashboards for `medium`
- comparison dashboards for `high`
- comparison dashboards for `all`

This means the repo is not just source code.
It is also acting as an experiment snapshot.

## Important Quirks And Gotchas

### 1. `libsumo` is not optional in practice

The code is written to prefer `libsumo` specifically because the environment blocks the normal socket binding used by TraCI.
If you remove that behavior without changing the environment, training and evaluation will likely fail.

### 2. GUI mode changes the backend

Passing `--gui` is not just a rendering toggle.
It changes the backend selection from `libsumo` to `traci`.

### 3. `experiment_matrix.yaml` is not wired into execution

It documents the intended experiment matrix, but no script currently reads it.

### 4. `aggregate_results.py` is just a collector

Despite the name, it does not recompute evaluation aggregates from route-level summaries.
It only scans existing `aggregate.json` files and combines them into one CSV.

### 5. Training summaries are not evaluation summaries

Training JSONL logs come from step-history-only summaries.
Evaluation summaries include `tripinfo.xml`.
Do not compare those two outputs as if they were identical measurements.

### 6. The repo contains generated noise

There are many generated files that are not core source:

- `results/**`
- `routes/**`
- `__pycache__/`

If you are trying to understand the implementation, start with:

- `README.md`
- `configs/`
- `scripts/`
- `src/traffic_rl/`
- `tests/test_pipeline.py`

## Where To Edit What

If you want to change a specific behavior, this is the shortest route:

- change SUMO timing, reward, GUI defaults, or route seed sets: `configs/env.yaml`
- change demand generation: `src/traffic_rl/routes.py`, `src/traffic_rl/scenario.py`
- change environment wiring or backend behavior: `src/traffic_rl/envs.py`, `src/traffic_rl/runtime.py`
- change model architectures: `src/traffic_rl/models.py`
- change PPO logic shared by all algorithms: `src/traffic_rl/train_common.py`
- change centralized PPO: `src/traffic_rl/algo_centralized.py`
- change independent PPO: `src/traffic_rl/algo_independent.py`
- change shared PPO: `src/traffic_rl/algo_shared.py`
- change MAPPO: `src/traffic_rl/algo_mappo.py`
- change evaluation outputs or metric definitions: `src/traffic_rl/metrics.py`, `src/traffic_rl/evaluation.py`, `src/traffic_rl/reporting.py`
- change comparison dashboard layout or metrics: `src/traffic_rl/comparison.py`
- add or change CLI flags: the relevant file under `scripts/`

## Suggested Reading Order

If you are new to the repo, this order gives the fastest mental model:

1. `README.md`
2. `configs/env.yaml`
3. `scripts/generate_routes.py`
4. `src/traffic_rl/routes.py`
5. `src/traffic_rl/envs.py`
6. `src/traffic_rl/models.py`
7. one algorithm module, usually `src/traffic_rl/algo_shared.py`
8. `src/traffic_rl/evaluation.py`
9. `src/traffic_rl/metrics.py`
10. `src/traffic_rl/comparison.py`

## Bottom Line

This codebase is organized around one clear idea:

```text
same network
+ same route-generation scheme
+ same metrics
+ multiple controller types
= a reproducible benchmark for traffic-light control
```

The design is practical rather than abstract.
Most modules are small, explicit, and purpose-built for this exact experiment pipeline.
