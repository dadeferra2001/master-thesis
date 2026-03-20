"""Shared evaluation runners for baseline and trained policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .envs import CentralizedGridEnv, ParallelGridEnv, make_raw_env
from .metrics import summarize_episode
from .utils import write_csv, write_json


def run_baseline_episode(
    config: dict[str, Any],
    route_file: str | Path,
    route_seed: int,
    output_dir: str | Path,
    use_gui: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    tripinfo_output = output_dir / "tripinfo.xml"
    env = make_raw_env(
        config,
        route_file,
        fixed_ts=True,
        seed=route_seed,
        tripinfo_output=tripinfo_output,
        use_gui=use_gui,
    )
    env.reset(seed=route_seed)
    done = False
    while not done:
        _, _, dones, _ = env.step({})
        done = bool(dones["__all__"])
    step_history = list(env.metrics)
    env.close()

    summary = summarize_episode(
        step_history,
        tripinfo_output,
        extra={"algorithm": "baseline", "route_seed": int(route_seed)},
    )
    write_csv(output_dir / "step_metrics.csv", step_history)
    write_json(output_dir / "summary.json", summary)
    return summary


def run_policy_episode(
    algo: str,
    controller: Any,
    config: dict[str, Any],
    route_file: str | Path,
    route_seed: int,
    output_dir: str | Path,
    deterministic: bool = True,
    use_gui: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    tripinfo_output = output_dir / "tripinfo.xml"
    episode_return = 0.0

    if algo == "centralized_ppo":
        env = CentralizedGridEnv(
            config,
            route_file,
            seed=route_seed,
            tripinfo_output=tripinfo_output,
            use_gui=use_gui,
        )
        obs, _ = env.reset(seed=route_seed)
        done = False
        while not done:
            action = controller.act(obs, deterministic=deterministic)
            obs, reward, _, truncated, _ = env.step(action)
            episode_return += float(reward)
            done = bool(truncated)
        step_history = env.history
        env.close()
    else:
        env = ParallelGridEnv(
            config,
            route_file,
            seed=route_seed,
            tripinfo_output=tripinfo_output,
            use_gui=use_gui,
        )
        obs, _ = env.reset(seed=route_seed)
        done = False
        while not done:
            actions = controller.act(obs, deterministic=deterministic)
            obs, rewards, terminations, truncations, _ = env.step(actions)
            episode_return += float(sum(float(value) for value in rewards.values()))
            done = bool(all(terminations.values()) or all(truncations.values()))
        step_history = env.history
        env.close()

    summary = summarize_episode(
        step_history,
        tripinfo_output,
        extra={
            "algorithm": algo,
            "route_seed": int(route_seed),
            "episode_return": episode_return,
        },
    )
    write_csv(output_dir / "step_metrics.csv", step_history)
    write_json(output_dir / "summary.json", summary)
    return summary
