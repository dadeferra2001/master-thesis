# Thesis Plan: Reproducible PPO Traffic-Light Control on a 2x2 SUMO Grid

  ## Summary

  Study whether PPO-based RL can outperform SUMO’s default fixed/automatic traffic-light control on a 4-intersection
  grid under stochastic low/medium/high demand, while staying computationally feasible on a laptop GPU.

  Recommended main thesis scope:

  - Baseline: SUMO default traffic-light control
  - RL 1: Centralized single-agent PPO controlling all 4 lights jointly
  - RL 2: Multi-agent independent PPO with 4 separate policies
  - RL 3: Multi-agent PPO with parameter sharing
  - MAPPO: planned as a stretch goal, added only after the first 4 setups are stable and runtime is acceptable

  Primary recommendation:

  - Reuse the existing sumo_rl/nets/2x2grid topology as the network backbone
  - Generate your own thesis-specific route files and evaluation pipeline around it
  - Keep one common observation/reward definition across all RL methods unless a method structurally requires
    otherwise
  - Treat parameter-sharing PPO as the strongest practical multi-agent method for the main thesis on laptop compute

  ## Experiment Definition

  ### Research question

  Can PPO-based RL traffic-light control improve traffic performance over SUMO default control on a 2x2 urban grid
  under stochastic demand, and how do centralized vs multi-agent PPO formulations trade off performance, stability,
  and compute cost?

  ### Hypotheses

  - H1: RL controllers reduce mean waiting time and queue length relative to the SUMO baseline in medium and high
    demand.
  - H2: Centralized single-agent PPO reaches stronger coordination than independent PPO but is less sample-efficient
    due to a larger joint action space.
  - H3: Parameter-sharing PPO provides the best performance/compute tradeoff on a laptop because it exploits symmetry
    while keeping decentralized execution.
  - H4: MAPPO can improve stability over independent PPO, but on this problem size its added implementation/training
    cost may not justify inclusion in the main thesis unless results are clearly better.

  ### Variables

  - Independent variables:
      - Control method: baseline, centralized PPO, independent PPO, parameter-sharing PPO, optional MAPPO
      - Traffic intensity: low, medium, high
  - Controlled variables:
      - Same 2x2 network
      - Same simulation horizon
      - Same action interval, yellow time, min green, teleport setting, vehicle dynamics
      - Same train/test split policy for route seeds
      - Same reward definition across RL methods where possible
      - Same evaluation metrics and reporting pipeline
  - Dependent variables:
      - Mean average speed
      - Mean average waiting time
      - Mean travel time
      - Throughput
      - Mean queue / halting vehicles
      - Time loss
      - Teleports / congestion failure rate
      - Training stability indicators: episodic return, variance across seeds

  ### Fair comparison protocol

  - Train each RL method only on training route seeds.
  - Evaluate every method, including baseline, on the exact same held-out test route seeds.
  - Use identical route files per (intensity, seed) across all methods.
  - Use fixed episode duration and common SUMO parameters.
  - Report mean and standard deviation across random seeds; include per-intensity results and an overall summary.
  - Do not tune each method independently beyond a small shared hyperparameter budget; keep PPO settings aligned
    unless a method structurally requires a difference.

  ## SUMO Scenario Design

  ### Network

  Use the existing 4-light network at /home/dado/sumo-rl/sumo_rl/nets/2x2grid/2x2.net.xml.
  Observed traffic-light IDs are 1, 2, 5, 6, which matches the intended 2x2 grid.

  ### Demand model

  Replace the bundled static 2x2.rou.xml with thesis-owned route generation:

  - Generate route files from seeded stochastic flows
  - Keep the OD structure symmetric by default
  - Add mild per-episode randomness so policies do not overfit one deterministic schedule

  Recommended default:

  - Use SUMO flow-based route generation with seeded probabilities or periods, not hand-authored vehicle lists
  - Create one route file per (split, intensity, seed) so experiments are exactly replayable
  - Start with stationary demand within an episode; add time-varying rush-hour profiles only if the core pipeline is
    already stable

  ### Intensity regimes

  Define intensity by scaling the same base OD pattern:

  - low: uncongested, queues rarely spill back
  - medium: moderate recurring queues, still recoverable
  - high: near-saturation but not gridlock under reasonable control

  Recommended concrete starting point:

  - Use one base flow template and multiply all OD rates by 0.5 / 1.0 / 1.5
  - Calibrate once by running the SUMO baseline and checking that:
      - low has near-zero teleports
      - medium has sustained queues without collapse
      - high is difficult but usually still operable
  - If high causes chronic collapse, reduce to 1.3 instead of 1.5

  ### Reproducibility rules

  - Separate route-generation seeds from training seeds
  - Commit generated config templates and the route-generation script
  - Either commit generated test route files or regenerate them deterministically from a manifest
  - Store a manifest such as routes_manifest.json listing intensity, split, seed, generator params, and file path

  ### Recommended folder structure

  project/
    configs/
      env.yaml
      ppo_shared.yaml
      experiment_matrix.yaml
    nets/
      2x2/
        2x2.net.xml
        tls_ids.json
    routes/
      train/
        low/
        medium/
        high/
      test/
        low/
        medium/
        high/
      manifests/
    scripts/
      generate_routes.py
      train_centralized_ppo.py
      train_independent_ppo.py
      train_shared_ppo.py
      train_mappo.py
      eval_policy.py
      run_baseline.py
      aggregate_results.py
    src/
      envs/
      agents/
      utils/
      evaluation/
    results/
      raw/
      tables/
      figures/
    thesis_notes/

  ## RL Setup Design

  ### Common defaults

  - Simulator interface: SUMO-RL
  - Training library: CleanRL style
  - Simulator mode: libsumo during training for speed
  - Action interval: delta_time = 5
  - Yellow time: 2
  - Min green: 5
  - Episode duration: start with 1800 simulated seconds
  - Reward default: diff-waiting-time
  - Observation default per intersection: SUMO-RL default vector
  - Action per intersection: discrete next green phase

  ### 1. Centralized single-agent PPO

  Definition:

  - One policy controls all 4 lights jointly
  - One environment step outputs a 4-action joint decision

  Implementation shape:

  - Observation: concatenate the 4 local observations in a fixed order, optionally append normalized global stats such
    as total queued vehicles
  - Action: MultiDiscrete([A1, A2, A3, A4]); if CleanRL code path expects categorical heads, use one shared encoder
    with 4 actor heads
  - Reward: sum of the 4 local rewards, or a single global system reward computed from total delay change
  - Critic: centralized
  - Policy structure: one actor-critic network over the global state

  Recommendation:

  - Prefer this over flattening to a single huge categorical action. MultiDiscrete with factorized heads is much
    lighter and avoids A^4 blow-up.

  ### 2. Independent PPO

  Definition:

  - 4 separate PPO agents
  - Each agent trains only from its own observation, action, reward stream
  - No shared parameters and no centralized critic

  Implementation shape:

  - Observation: local observation only
  - Action: local discrete phase selection
  - Reward: local reward from the corresponding traffic signal
  - Critic: local
  - Policy structure: 4 separate actor-critic models and optimizers

  Important distinction:

  - This is not parameter sharing. Data, gradients, and weights remain separate.
  - It is the cleanest conceptual multi-agent baseline, but usually the least sample-efficient.

  Implementation note:

  - Do not force full CleanRL purity here if it becomes awkward. A lightweight custom PPO loop in CleanRL style is
    acceptable and easier than launching 4 isolated training processes.

  ### 3. Parameter-sharing PPO

  Definition:

  - One shared actor-critic is used by all 4 agents
  - Each agent acts from its own local observation
  - Trajectories from all agents are pooled into one PPO update

  Implementation shape:

  - Observation: local observation only; optionally append agent ID one-hot or intersection coordinates
  - Action: local discrete phase selection
  - Reward: local reward
  - Critic: local by default for the main thesis version
  - Policy structure: one shared network, one optimizer, batched data from all agents

  Important distinction:

  - Same decentralized observation/action interface as independent PPO
  - Unlike independent PPO, all agents share parameters and training data
  - This is the recommended primary multi-agent method for the main thesis

  Recommended default:

  - Add an agent identity embedding or one-hot. The network is shared, but the 4 intersections are not perfectly
    identical in context.

  ### 4. MAPPO

  Definition:

  - Multi-agent PPO with decentralized actors and a centralized critic

  Implementation shape:

  - Actor observation: local observation plus optional agent ID
  - Critic input: concatenated observations from all 4 intersections plus optional global stats
  - Action: local discrete phase selection for each agent
  - Reward: use local rewards for policy loss and a centralized value baseline; alternatively use a shared global
    reward if you want a cooperative setting, but then keep it consistent across methods and state it explicitly
  - Critic: centralized
  - Policy structure: shared actor across agents plus centralized value network

  Important distinction:

  - Compared with parameter-sharing PPO, the crucial difference is the value function input, not just shared weights
  - Compared with centralized single-agent PPO, actors remain per-agent and execution is decentralized

  Recommendation:

  - Keep MAPPO out of the MVP
  - Implement only after parameter-sharing PPO is working and evaluation infrastructure is stable

  ### Baseline: SUMO default control

  Use the same route files and simulation horizon with fixed_ts=True or the network’s default signal program.
  This is the mandatory non-RL reference.

  ## Public Interfaces and Minimal Code Architecture

  ### Environment wrappers

  Add thin wrappers over SUMO-RL:

  - CentralizedGridEnv: converts the 4-light environment into a Gym-style single-agent env with concatenated
    observation, factorized multi-discrete action, global reward
  - MultiAgentGridEnv: thin helper around parallel_env that returns observations/actions in a fixed agent order and
    exposes shared utilities for batching, masks, and logging

  ### Core training interfaces

  Define a small common API:

  - make_env(config, route_file, mode)
  - collect_rollout(...)
  - compute_ppo_loss(...)
  - evaluate_policy(policy, route_files, deterministic=True)
  - extract_episode_metrics(episode_output)

  ### Config surface

  Keep configs small and explicit:

  - environment params
  - PPO params
  - route split manifest
  - experiment matrix
  - random seeds

  ## Training Budget and Feasibility

  ### Recommended main budget

  For laptop feasibility:

  - Episode length: 1800 s
  - PPO rollout length: target 2048-8192 agent-steps per update depending on setup
  - Training seeds: 3
  - Test route seeds per intensity: 10
  - Main methods in thesis core: 4
  - Total demand levels: 3

  Practical starting training horizon:

  - Centralized PPO: 300k-500k env steps
  - Independent PPO: 300k per agent-equivalent, but monitor runtime carefully
  - Parameter-sharing PPO: 300k-500k shared-agent steps
  - MAPPO stretch: 200k-300k first, only if runtime permits

  ### What to reduce first if runtime is too high

  1. Drop MAPPO from the main matrix
  2. Reduce training seeds from 3 to 2
  3. Reduce train route seed count
  4. Reduce episode horizon from 1800 to 1200
  5. Reduce total training timesteps
  6. Keep test repetitions intact as long as possible

  ### Staged development order

  1. Baseline evaluation on low/medium/high
  2. Route generator and reproducible evaluation pipeline
  3. Centralized single-agent PPO
  4. Parameter-sharing PPO
  5. Independent PPO
  6. MAPPO only if the above are stable and runtime remains acceptable

  ## Evaluation Metrics

  ### Mandatory metrics

  - Mean average speed
      - Why: direct mobility efficiency indicator
      - Source: SUMO-RL info already exposes system_mean_speed; can also derive from vehicle states
      - Aggregation: average over simulation steps, then over episodes
  - Mean average waiting time
      - Why: core congestion/service-quality metric
      - Source: system_mean_waiting_time from SUMO-RL info, plus trip-level waiting from tripinfo if needed
      - Aggregation: average over steps and episodes; also report final per-vehicle average on completed trips if
        available

  ### Additional recommended metrics

  - Mean travel time
      - Why: end-to-end user-facing performance
      - Source: SUMO tripinfo.xml
      - Aggregation: average over completed vehicles, then episodes
  - Throughput
      - Why: measures network productivity
      - Source: system_total_arrived
      - Aggregation: final count per episode
  - Queue length / halting vehicles
      - Why: captures spillback pressure better than speed alone
      - Source: system_total_stopped and per-agent stopped counts
      - Aggregation: average over steps; also report max over episode for stress
  - Time loss
      - Why: standard traffic-efficiency metric relative to free-flow
      - Source: tripinfo.xml
      - Aggregation: average over completed vehicles
  - Teleports / congestion failures
      - Why: signals broken or gridlocked control
      - Source: system_total_teleported
      - Aggregation: final count per episode; also report fraction of episodes with any teleport
  - Backlogged/pending vehicles
      - Why: shows unmet demand when insertion fails
      - Source: system_total_backlogged
      - Aggregation: average and final value per episode

  Recommended reporting rule:

  - Use step-averaged system metrics for online congestion behavior
  - Use trip-based metrics from tripinfo.xml for end-to-end traffic quality
  - Report both, because they capture different failure modes

  ## Experimental Protocol

  ### Train/test split

  - Generate separate route seeds for train and test
  - Example default:
      - train route seeds per intensity: 20
      - test route seeds per intensity: 10
  - During training, sample only from train routes
  - During final evaluation, use all held-out test routes

  ### Repetitions and reporting

  - Training seeds: 3
  - Evaluation: every trained model tested on all held-out seeds for each intensity
  - Report:
      - mean ± std across training seeds
      - per-intensity tables
      - normalized improvement over baseline
      - learning curves for the RL methods

  ### Statistical reporting

  Recommended thesis-safe minimum:

  - Report mean, standard deviation, and 95% bootstrap confidence intervals
  - Use paired comparisons against the baseline on the same test route seeds
  - Avoid overclaiming significance with tiny n; if sample size stays small, frame results as empirical trends

  ### Comparison across intensities

  - Keep train/eval structure identical for low/medium/high
  - Produce:
      - one table per intensity
      - one summary figure showing relative improvement vs baseline across intensities
  - Do not average all intensities into one headline number without also showing per-intensity breakdown

  ### Fair baseline comparison

  - Evaluate baseline on exactly the same route files, horizon, and SUMO settings as RL
  - If RL uses action interval and yellow constraints, baseline remains the network default program; do not hand-tune
    it after seeing RL results
  - If you later add an actuated SUMO baseline, report it as a second baseline, not a replacement

  ## Recommended Experiment Matrix

  ### Minimal viable thesis matrix

  | Method | Low | Medium | High | Seeds |
  |---|---:|---:|---:|---:|
  | SUMO baseline | yes | yes | yes | 10 eval seeds |
  | Centralized PPO | yes | yes | yes | 3 train seeds |
  | Shared PPO | yes | yes | yes | 3 train seeds |

  ### Strong but still feasible thesis matrix

  | Method | Low | Medium | High | Seeds |
  |---|---:|---:|---:|---:|
  | SUMO baseline | yes | yes | yes | 10 eval seeds |
  | Centralized PPO | yes | yes | yes | 3 train seeds |
  | Independent PPO | yes | yes | yes | 3 train seeds |
  | Shared PPO | yes | yes | yes | 3 train seeds |

  ### Stretch matrix

  | Method | Low | Medium | High | Seeds |
  |---|---:|---:|---:|---:|
  | MAPPO | yes | yes | yes | 2-3 train seeds |

  ## MVP vs Strong Thesis

  ### Minimal viable version

  - One stable 2x2 network
  - Three traffic intensities
  - Baseline + centralized PPO + shared PPO
  - Three core metrics: speed, waiting time, throughput
  - Three training seeds only for RL
  - Held-out test route seeds
  - Reproducible scripts for generation, training, evaluation, aggregation

  ### Strong thesis version

  - Add independent PPO
  - Add trip-based metrics from tripinfo.xml
  - Add per-agent analysis and coordination discussion
  - Add MAPPO only if it finishes within budget and shows meaningful differences
  - Include ablation on reward choice or agent-ID feature only if time remains

  ### Too ambitious for a laptop

  - Large hyperparameter sweeps for every method
  - Full MAPPO tuning plus full 5-method x 3-intensity x many-seed matrix
  - Time-varying nonstationary demand plus all methods plus statistical depth
  - Repeated GUI-based debugging during training instead of libsumo

  ## Main Risks and Pitfalls

  - Joint centralized action space can become unstable if implemented as one flat categorical; use factorized
    MultiDiscrete heads instead.
  - Independent PPO may look weaker simply because it gets less data efficiency; document that this is part of the
    method tradeoff.
  - High demand can create pathological gridlock; calibrate intensity before large training runs.
  - SUMO metrics measured over currently running vehicles can differ from trip-completion metrics; report both.
  - Route leakage between train and test seeds will invalidate the comparison.
  - Reward mismatch can distort conclusions; keep one default reward across RL methods unless the thesis explicitly
    studies reward design.
  - MAPPO implementation complexity can consume thesis time without guaranteeing insight.

  ## Step-by-Step Roadmap

  1. Freeze network and SUMO parameters.
  2. Implement seeded route generation and manifest logging.
  3. Run the baseline on low/medium/high and calibrate intensity multipliers.
  4. Implement evaluation outputs: per-step CSV plus tripinfo.xml.
  5. Build centralized PPO wrapper and verify one full training/eval run.
  6. Build parameter-sharing PPO and compare against centralized PPO.
  7. Add independent PPO.
  8. Aggregate results into tables/plots and identify whether MAPPO is still worth adding.
  9. If runtime permits, implement a minimal MAPPO variant with shared actors and centralized critic.
  10. Write the thesis around the stable experiment matrix, not around unfinished stretch goals.

  ## Test Plan

  - Route generation determinism: same seed produces byte-identical route files or equivalent manifest-verified
    outputs.
  - Environment wrapper correctness: fixed agent order, observation shapes, action shapes, reward aggregation.
  - Baseline reproducibility: repeated runs on the same route seed produce identical metrics up to simulator
    determinism settings.
  - PPO smoke tests: 1-2 short episodes complete without NaNs, deadlocks, or shape errors.
  - Evaluation pipeline: metrics extracted consistently for baseline and RL runs.
  - Split integrity: no overlap between train and test route seeds.
  - Congestion calibration: low/medium/high remain meaningfully distinct under the baseline.

  - Week 2: implement route generator, manifests, and baseline runner
  - Week 3: calibrate low/medium/high and finalize evaluation metrics
  - Week 4: implement centralized PPO and short smoke runs
  - Week 5: stabilize centralized PPO and produce first baseline comparison
  - Week 6: implement parameter-sharing PPO
  - Week 7: train centralized and shared PPO across seeds
  - Week 8: implement independent PPO
  - Week 9: run full main experiment matrix
  - Week 10: aggregate tables/plots, inspect failures, rerun missing cases
  - Week 11: MAPPO feasibility decision; implement only if budget remains
  - Week 12: thesis writing focused on methods, protocol, and results

  ## Assumptions and Defaults

  - MAPPO is a stretch goal, not part of the mandatory core matrix.
  - The local SUMO-RL 2x2 network is acceptable as the thesis network backbone.
  - CleanRL compatibility means CleanRL-style PPO implementations and logging, with lightweight custom code where
    CleanRL has no direct multi-agent script.
  - Default reward is diff-waiting-time.
  - Default episode horizon is 1800 s.
  - Default core result set is baseline + centralized PPO + independent PPO + shared PPO across low/medium/high with 3
    training seeds.