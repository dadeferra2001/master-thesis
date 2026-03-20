# Metrics in This Project

This document is the source of truth for how metrics are computed in the current pipeline.

It covers:

- raw step-level metrics saved in `step_metrics.csv`
- episode-level metrics saved in `summary.json`
- aggregate metrics saved in `aggregate.json`
- comparison metrics produced by `compare_results.py`
- training diagnostics written to TensorBoard

The main code paths are:

- [src/traffic_rl/metrics.py](/home/dado/lol/src/traffic_rl/metrics.py)
- [src/traffic_rl/reporting.py](/home/dado/lol/src/traffic_rl/reporting.py)
- [src/traffic_rl/evaluation.py](/home/dado/lol/src/traffic_rl/evaluation.py)
- [src/traffic_rl/train_common.py](/home/dado/lol/src/traffic_rl/train_common.py)

## Metric Pipeline

For evaluation, metrics are produced in three layers:

1. SUMO-RL emits raw per-step information during the episode.
   This is written to `step_metrics.csv`.
2. We collapse one episode into one summary row.
   This is written to `summary.json`.
3. We aggregate many episodes for one `(algorithm, split, intensity)` group.
   This is written to `aggregate.json`.

So when you see a field like `mean_average_speed_mean`, that is:

- first a per-step metric: `system_mean_speed`
- then averaged within one episode: `mean_average_speed`
- then averaged again across episodes: `mean_average_speed_mean`

## Time Scale and Notation

Let:

- `e` be one episode
- `t = 1, ..., T_e` be decision steps inside episode `e`
- `delta_time` be the SUMO time between decisions
- `V_t` be the set of vehicles currently running in the network at decision step `t`
- `P_t` be the set of pending vehicles that have not yet entered the network at step `t`
- `K_e` be the set of `<tripinfo>` records written for episode `e`

Important:

- A row in `step_metrics.csv` is one decision step, not one simulated second.
- In the current config, `delta_time = 5`, so one decision step usually corresponds to 5 simulated seconds.
- Some metrics are computed over currently running vehicles.
- Some metrics are computed over trip records in `tripinfo.xml`.
- Some metrics are computed over episodes.

## Files and What They Mean

### `step_metrics.csv`

One row per decision step.

Source:

- `env.metrics` in SUMO-RL
- written by [evaluation.py](/home/dado/lol/src/traffic_rl/evaluation.py)

### `summary.json`

One row per evaluated episode / route seed.

Source:

- `summarize_episode(...)` in [metrics.py](/home/dado/lol/src/traffic_rl/metrics.py)

### `aggregate.json`

One row per `(algorithm, split, intensity)` group.

Source:

- `aggregate_episode_summaries(...)` in [reporting.py](/home/dado/lol/src/traffic_rl/reporting.py)

## Raw Step Metrics

These are the main raw system fields produced by SUMO-RL in each step row.

They come from `SumoEnvironment._get_system_info()` and `SumoEnvironment._get_per_agent_info()` in `sumo-rl`.

### System metrics

#### `system_total_running`

Definition:

- Number of vehicles currently in the network at the decision step.

Formula:

`system_total_running_t = |V_t|`

Interpretation:

- Snapshot count, not cumulative.

#### `system_total_backlogged`

Definition:

- Number of pending vehicles waiting to be inserted into the network.

Formula:

`system_total_backlogged_t = |P_t|`

Interpretation:

- This is a network loading / demand overflow indicator.
- It is not the same as queue length at intersections.

#### `system_total_stopped`

Definition:

- Number of running vehicles considered halting at the decision step.

Formula:

`system_total_stopped_t = sum(1[speed(v, t) < 0.1] for v in V_t)`

Interpretation:

- In SUMO-RL this is based on speed `< 0.1 m/s`.
- In this project we use it as a queue proxy.
- So `mean_queue_length` is really "mean number of halting vehicles in the network", not a detector-based physical queue length.

#### `system_total_arrived`

Definition:

- Cumulative number of vehicles that have arrived by this step.

Interpretation:

- This is cumulative from the beginning of the episode.
- At the final step we use it as throughput.

#### `system_total_departed`

Definition:

- Cumulative number of vehicles that have departed by this step.

Interpretation:

- Useful for debugging loading and insertion behavior.
- Not currently promoted to the episode summary.

#### `system_total_teleported`

Definition:

- Cumulative number of teleported vehicles by this step.

Interpretation:

- In the current config `time_to_teleport = -1`, so teleports are disabled unless the setup changes.
- This means `teleports` is usually zero in the current experiments.

#### `system_total_waiting_time`

Definition:

- Sum of the current waiting times of all running vehicles.

Formula:

`system_total_waiting_time_t = sum(waiting_time(v, t) for v in V_t)`

Interpretation:

- This uses SUMO's current waiting time signal for each running vehicle.
- It is not the same as accumulated waiting time over the whole trip.

#### `system_mean_waiting_time`

Definition:

- Mean current waiting time over running vehicles at the step.

Formula:

- `0.0` if `|V_t| = 0`
- otherwise
  `system_mean_waiting_time_t = (1 / |V_t|) * sum(waiting_time(v, t) for v in V_t)`

Interpretation:

- This is a step snapshot over running vehicles only.
- It is not averaged over completed trips.
- It is not accumulated waiting time over a vehicle's whole life.

#### `system_mean_speed`

Definition:

- Mean raw speed over running vehicles at the step.

Formula:

- `0.0` if `|V_t| = 0`
- otherwise
  `system_mean_speed_t = (1 / |V_t|) * sum(speed(v, t) for v in V_t)`

Unit:

- meters per second

Interpretation:

- This is a step snapshot over running vehicles only.
- It is not normalized.

### Per-intersection raw metrics

These are also stored in `step_metrics.csv` but are not currently promoted into `summary.json`.

For each traffic light `ts` in `{1, 2, 5, 6}`:

#### `<ts>_stopped`

Definition:

- Total number of halting vehicles on lanes controlled by that traffic signal.

#### `<ts>_accumulated_waiting_time`

Definition:

- Sum of accumulated waiting time over all vehicles currently on the lanes controlled by that signal.

Interpretation:

- This is closer to the reward signal than `system_mean_waiting_time`.

#### `<ts>_average_speed`

Definition:

- Mean speed normalized by allowed speed for vehicles in that intersection area.

Formula:

- if no vehicles are present, SUMO-RL returns `1.0`
- otherwise it averages `speed / allowed_speed`

Interpretation:

- This is not in m/s.
- It is a normalized intersection-local metric.

### Aggregated per-agent raw fields

#### `agents_total_stopped`

- Sum of `<ts>_stopped` over all controlled intersections.

#### `agents_total_accumulated_waiting_time`

- Sum of `<ts>_accumulated_waiting_time` over all controlled intersections.

## Episode Metrics in `summary.json`

These are computed in [summarize_step_metrics(...) in metrics.py](/home/dado/lol/src/traffic_rl/metrics.py#L55) and [parse_tripinfo(...) in metrics.py](/home/dado/lol/src/traffic_rl/metrics.py#L16).

### `num_decision_steps`

Definition:

- Number of rows in `step_history`

Formula:

`num_decision_steps_e = T_e`

Interpretation:

- This is literally `len(step_history)`.
- It is not derived from `episode_seconds / delta_time`.
- If SUMO-RL emits or omits an initial metrics row, this count will reflect that.

### `mean_average_speed`

Definition:

- Mean of the step-level system mean speed within one episode.

Formula:

`mean_average_speed_e = (1 / T_e) * sum(system_mean_speed_t for t = 1..T_e)`

Unit:

- meters per second

Interpretation:

- "Average of per-step average speed", not direct average over all vehicle-seconds.
- Since each decision step represents the same `delta_time`, this is effectively a uniform time average over decision points.

### `mean_average_waiting_time`

Definition:

- Mean of the step-level system mean waiting time within one episode.

Formula:

`mean_average_waiting_time_e = (1 / T_e) * sum(system_mean_waiting_time_t for t = 1..T_e)`

Unit:

- seconds

Interpretation:

- "Average of per-step average waiting time", not average accumulated waiting time per trip.

### `mean_queue_length`

Definition:

- Mean of the step-level halting-vehicle count.

Formula:

`mean_queue_length_e = (1 / T_e) * sum(system_total_stopped_t for t = 1..T_e)`

Interpretation:

- This is a time-averaged halting count.
- In this project we use it as queue length.
- Strictly speaking it is a network-level halting proxy, not a lane-by-lane geometric queue measurement.

### `max_queue_length`

Definition:

- Maximum halting-vehicle count observed in the episode.

Formula:

`max_queue_length_e = max(system_total_stopped_t for t = 1..T_e)`

Interpretation:

- Stress indicator.
- Useful for spotting transient overload or spillback-like behavior.

### `throughput`

Definition:

- Number of vehicles that have arrived by the final decision step.

Formula:

`throughput_e = system_total_arrived_(T_e)`

Interpretation:

- This is the best "completed vehicles" metric in the current pipeline.
- It is cumulative and episode-level.

### `teleports`

Definition:

- Number of teleported vehicles by the final decision step.

Formula:

`teleports_e = system_total_teleported_(T_e)`

Interpretation:

- Episode-level failure signal.
- Usually zero in the current config because teleports are disabled.

### `mean_backlog`

Definition:

- Mean pending-vehicle backlog over decision steps.

Formula:

`mean_backlog_e = (1 / T_e) * sum(system_total_backlogged_t for t = 1..T_e)`

Interpretation:

- Measures how much demand was waiting to enter the network on average.

### `final_backlog`

Definition:

- Pending-vehicle backlog at the final decision step.

Formula:

`final_backlog_e = system_total_backlogged_(T_e)`

Interpretation:

- Strong episode-end overload indicator.
- This is part of `congestion_failure`.

## Trip-Based Episode Metrics from `tripinfo.xml`

The `tripinfo.xml` file is written with:

- `--tripinfo-output`
- `--tripinfo-output.write-unfinished true`

That second setting matters a lot.

It means unfinished trips at the episode cutoff are still written to the file.

So these trip-based metrics are currently computed over all `<tripinfo>` elements in the file, not strictly only over vehicles that physically arrived before the end of the episode.

This is a known interpretation caveat.

### `completed_trips`

Current implementation:

- `completed_trips_e = number of <tripinfo> entries in tripinfo.xml`

Important caveat:

- Because unfinished trips are written too, this field is currently closer to "number of tripinfo records" than "strictly completed trips".
- If you need strict completions, the parser should exclude records with `arrival = -1` or `vaporized="end"`.

### `mean_travel_time`

Current implementation:

- Mean of the `duration` attribute over all `<tripinfo>` records.

Formula:

`mean_travel_time_e = (1 / |K_e|) * sum(duration(k) for k in K_e)`

Important caveat:

- Since unfinished trips are included, some durations are truncated-at-end values rather than true completed trip durations.

### `mean_time_loss`

Current implementation:

- Mean of the `timeLoss` attribute over all `<tripinfo>` records.

Formula:

`mean_time_loss_e = (1 / |K_e|) * sum(timeLoss(k) for k in K_e)`

Interpretation:

- Standard traffic efficiency measure relative to free-flow.
- Same unfinished-trip caveat applies.

### `mean_trip_waiting_time`

Current implementation:

- Mean of the `waitingTime` attribute over all `<tripinfo>` records.

Formula:

`mean_trip_waiting_time_e = (1 / |K_e|) * sum(waitingTime(k) for k in K_e)`

Interpretation:

- This is trip-level waiting time.
- It is different from `mean_average_waiting_time`, which comes from per-step system snapshots.

### `mean_depart_delay`

Current implementation:

- Mean of the `departDelay` attribute over all `<tripinfo>` records.

Formula:

`mean_depart_delay_e = (1 / |K_e|) * sum(departDelay(k) for k in K_e)`

Interpretation:

- Useful for diagnosing insertion pressure and demand overflow.

## Derived Episode Flag

### `congestion_failure`

Definition:

- Episode is marked as a failure if either:
  - at least one teleport occurred
  - or the final backlog is positive

Formula:

`congestion_failure_e = 1 if teleports_e > 0 or final_backlog_e > 0 else 0`

Interpretation:

- In the current config this is mostly driven by `final_backlog > 0`, because teleports are disabled.

## RL Reward Metric

### `episode_return`

This is an RL training / evaluation metric, not a direct traffic engineering metric.

For RL methods only:

- `episode_return_e = sum of rewards over the episode`

In the centralized setup:

- the environment wrapper already sums the 4 local traffic-signal rewards each step

In the multi-agent setups:

- we explicitly sum the 4 local rewards each step before accumulating `episode_return`

So in all RL algorithms:

`episode_return_e = sum_t sum_i r_i(t)`

where `i` indexes the 4 traffic lights.

### Current reward function

The current config uses:

- `reward_fn: diff-waiting-time`

In SUMO-RL, for each traffic signal:

`ts_wait_i(t) = sum(accumulated_waiting_time_per_lane_i(t)) / 100`

`r_i(t) = ts_wait_i(t-1) - ts_wait_i(t)`

Interpretation:

- Positive reward means accumulated waiting decreased since the previous decision.
- Negative reward means accumulated waiting increased.
- The division by `100` is only a scale factor.

Important:

- `episode_return` is not directly comparable to speed or travel time in physical units.
- It is mainly useful for monitoring RL optimization.
- The baseline does not have this metric.

## Aggregated Metrics in `aggregate.json`

These are computed by [aggregate_episode_summaries(...) in reporting.py](/home/dado/lol/src/traffic_rl/reporting.py).

For every numeric field in the episode summaries:

- `field_mean = mean(field_e over episodes)`
- `field_std = std(field_e over episodes)`

If there are `N` evaluated route seeds in one group:

`field_mean = (1 / N) * sum(field_e for e = 1..N)`

`field_std = standard deviation over the same episode-level values`

### Examples

#### `mean_average_speed_mean`

- Mean of per-episode `mean_average_speed`

#### `mean_average_speed_std`

- Standard deviation of per-episode `mean_average_speed`

#### `throughput_mean`

- Mean throughput across evaluated route seeds

#### `congestion_failure_mean`

- Mean of a boolean field after conversion to `0/1`
- This is therefore the fraction of episodes flagged as congestion failures

#### `congestion_failure_std`

- Standard deviation of that same `0/1` field

### Generic aggregation caveat

Aggregation is intentionally generic: every numeric field is aggregated.

This means some bookkeeping fields appear in `aggregate.json`, for example:

- `route_seed_mean`
- `route_seed_std`

These are not performance metrics.
They are artifacts of the generic aggregation logic and should not be interpreted as results.

## Comparison Metrics from `compare_results.py`

The comparison script creates baseline-relative fields such as:

- `<metric>_improvement_abs`
- `<metric>_improvement_pct`

These are computed from `aggregate.json`.

### Higher-is-better metrics

For metrics such as:

- `mean_average_speed`
- `throughput`

the comparison uses:

`improvement_abs = method_value - baseline_value`

`improvement_pct = 100 * (method_value - baseline_value) / abs(baseline_value)`

### Lower-is-better metrics

For metrics such as:

- `mean_average_waiting_time`
- `mean_travel_time`
- `mean_queue_length`
- `mean_time_loss`
- `teleports`

the comparison uses:

`improvement_abs = baseline_value - method_value`

`improvement_pct = 100 * (baseline_value - method_value) / abs(baseline_value)`

Interpretation:

- Positive percentage means "better than baseline"
- Negative percentage means "worse than baseline"

## Training Diagnostics in TensorBoard

These metrics are not traffic KPIs. They are training diagnostics.

They are logged in [train_common.py](/home/dado/lol/src/traffic_rl/train_common.py).

### Episode-level TensorBoard metrics

#### `episode/return`

- The episode return described above

#### `episode/count`

- Cumulative number of completed episodes seen by the trainer

#### `episode/<traffic_metric>`

When an episode ends, the trainer also logs the numeric fields from `episode_summary`.

Examples:

- `episode/mean_average_speed`
- `episode/mean_average_waiting_time`
- `episode/throughput`
- `episode/teleports`

These are the same episode metrics described earlier, but emitted during training.

### Training-progress metrics

#### `train/update`

- Current PPO update number

#### `train/progress_pct`

- `100 * current_update / total_updates`

#### `train/learning_rate`

- Current optimizer learning rate after optional linear annealing

#### `train/sps`

- Environment decision steps per second of wall-clock time

Formula:

`sps = global_step / elapsed_wall_time`

Important:

- `global_step` counts environment decision steps, not per-agent samples.
- So multi-agent methods process more total agent samples than `sps` alone suggests.

#### `train/episodes_completed`

- Number of completed episodes so far

#### `train/recent_return_mean`

- Mean of the most recent 10 completed episode returns

Implementation detail:

- This comes from a deque with `maxlen = 10`

### PPO loss diagnostics

#### `losses/policy_loss`

- PPO clipped policy surrogate loss

#### `losses/value_loss`

- PPO value-function loss

#### `losses/entropy`

- Mean policy entropy used for entropy regularization

#### `losses/total_loss`

- Combined optimization objective:
  `policy_loss - ent_coef * entropy + vf_coef * value_loss`

#### `losses/approx_kl`

- Approximate KL divergence proxy used in many PPO implementations:
  `mean((ratio - 1) - logratio)`

#### `losses/clipfrac`

- Fraction of samples whose importance ratio exceeded the PPO clip threshold in magnitude

#### `losses/explained_variance`

- Explained variance of the value targets:
  `1 - Var(y_true - y_pred) / Var(y_true)`

Interpretation:

- Near `1`: value function explains targets well
- Near `0`: weak fit
- Negative: very poor fit

### Loss logging caveat

The loss scalars are not averaged in exactly the same way across all trainers:

- `independent_ppo` logs means over collected loss values across agents and minibatches
- `centralized_ppo`, `shared_ppo`, and `mappo` log the last seen minibatch value for several losses, while `clipfrac` is averaged

So loss curves are useful for debugging, but they should not be over-interpreted as perfectly standardized across algorithms.

## Recommended Interpretation Rules

If you want the most robust story for the thesis:

- Use `throughput` for completed-vehicle productivity
- Use `mean_average_waiting_time` for time-averaged online congestion
- Use `mean_queue_length` and `max_queue_length` for network pressure
- Use `mean_travel_time` and `mean_time_loss` as trip-level efficiency indicators
- Treat `episode_return` only as an RL optimization signal

## Known Caveats in the Current Implementation

### Trip metrics include unfinished trips

Because `tripinfo-output.write-unfinished=true`, the current parser includes unfinished trips.

This affects:

- `completed_trips`
- `mean_travel_time`
- `mean_time_loss`
- `mean_trip_waiting_time`
- `mean_depart_delay`

If you want strict completed-trip metrics later, update `parse_tripinfo(...)` to filter out records with:

- `arrival = -1`
- or `vaporized = "end"`

### Queue length is a halting proxy

`mean_queue_length` and `max_queue_length` are based on `system_total_stopped`, which is:

- number of vehicles with speed `< 0.1 m/s`

That is practical and consistent, but it is not a lane detector queue measurement.

### Aggregate files contain bookkeeping statistics

Fields like `route_seed_mean` are generated automatically.
They should not be used in result interpretation.
