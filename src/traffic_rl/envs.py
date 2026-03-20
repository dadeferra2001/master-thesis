"""SUMO environment builders and wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .runtime import bootstrap_sumo

bootstrap_sumo(prefer_libsumo=True)

import gymnasium as gym
import libsumo
import numpy as np
import sumo_rl.environment.env as sumo_env_module
import traci

from .config import resolve_path
from .controllers import concatenate_global_observation
from .scenario import TLS_IDS

parallel_env = sumo_env_module.parallel_env  # noqa: E402
SumoEnvironment = sumo_env_module.SumoEnvironment  # noqa: E402


def configure_sumo_backend(use_gui: bool = False) -> str:
    """Select the SUMO-RL backend for the next environment instance.

    `libsumo` is faster and avoids sandbox socket restrictions, but it cannot
    render the SUMO GUI. GUI runs therefore fall back to pure-python TraCI.
    """

    if use_gui:
        sumo_env_module.traci = traci
        sumo_env_module.LIBSUMO = False
        return "traci"

    sumo_env_module.traci = libsumo
    sumo_env_module.LIBSUMO = True
    return "libsumo"


def build_env_kwargs(
    config: dict[str, Any],
    route_file: str | Path,
    fixed_ts: bool = False,
    seed: int | None = None,
    tripinfo_output: str | Path | None = None,
    use_gui: bool | None = None,
) -> dict[str, Any]:
    scenario_cfg = config["scenario"]
    effective_use_gui = bool(scenario_cfg["use_gui"] if use_gui is None else use_gui)
    additional_cmd_parts: list[str] = []
    if tripinfo_output is not None:
        additional_cmd_parts.extend(
            [
                "--tripinfo-output",
                str(resolve_path(tripinfo_output)),
                "--tripinfo-output.write-unfinished",
                "true",
            ]
        )

    return {
        "net_file": str(resolve_path(scenario_cfg["net_file"])),
        "route_file": str(resolve_path(route_file)),
        "use_gui": effective_use_gui,
        "num_seconds": int(scenario_cfg["episode_seconds"]),
        "delta_time": int(scenario_cfg["delta_time"]),
        "yellow_time": int(scenario_cfg["yellow_time"]),
        "min_green": int(scenario_cfg["min_green"]),
        "max_green": int(scenario_cfg["max_green"]),
        "max_depart_delay": int(scenario_cfg["max_depart_delay"]),
        "waiting_time_memory": int(scenario_cfg["waiting_time_memory"]),
        "time_to_teleport": int(scenario_cfg["time_to_teleport"]),
        "reward_fn": scenario_cfg["reward_fn"],
        "ts_ids": list(scenario_cfg["tls_ids"]),
        "fixed_ts": fixed_ts,
        "add_system_info": bool(scenario_cfg["add_system_info"]),
        "add_per_agent_info": bool(scenario_cfg["add_per_agent_info"]),
        "sumo_warnings": bool(scenario_cfg["sumo_warnings"]),
        "sumo_seed": int(seed) if seed is not None else 0,
        "additional_sumo_cmd": " ".join(additional_cmd_parts) if additional_cmd_parts else None,
    }


def make_raw_env(
    config: dict[str, Any],
    route_file: str | Path,
    fixed_ts: bool = False,
    seed: int | None = None,
    tripinfo_output: str | Path | None = None,
    use_gui: bool | None = None,
) -> SumoEnvironment:
    effective_use_gui = bool(config["scenario"]["use_gui"] if use_gui is None else use_gui)
    configure_sumo_backend(use_gui=effective_use_gui)
    return SumoEnvironment(
        **build_env_kwargs(
            config,
            route_file,
            fixed_ts=fixed_ts,
            seed=seed,
            tripinfo_output=tripinfo_output,
            use_gui=effective_use_gui,
        )
    )


class CentralizedGridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any],
        route_file: str | Path,
        seed: int | None = None,
        tripinfo_output: str | Path | None = None,
        use_gui: bool | None = None,
    ) -> None:
        super().__init__()
        self.agent_order = tuple(config["scenario"]["tls_ids"])
        effective_use_gui = bool(config["scenario"]["use_gui"] if use_gui is None else use_gui)
        self.env = make_raw_env(
            config,
            route_file,
            fixed_ts=False,
            seed=seed,
            tripinfo_output=tripinfo_output,
            use_gui=effective_use_gui,
        )
        single_obs_dim = int(np.prod(self.env.observation_spaces(self.agent_order[0]).shape))
        num_agents = len(self.agent_order)
        action_dim = int(self.env.action_spaces(self.agent_order[0]).n)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(single_obs_dim * num_agents,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete([action_dim] * num_agents)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self.env.metrics)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        del options
        obs_dict = self.env.reset(seed=seed)
        obs = concatenate_global_observation(obs_dict, self.agent_order)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_array = np.asarray(action, dtype=np.int64).reshape(-1)
        action_dict = {agent: int(action_array[idx]) for idx, agent in enumerate(self.agent_order)}
        obs_dict, rewards, dones, info = self.env.step(action_dict)
        obs = concatenate_global_observation(obs_dict, self.agent_order)
        reward = float(sum(float(rewards[agent]) for agent in self.agent_order))
        truncated = bool(dones["__all__"])
        return obs, reward, False, truncated, info

    def close(self) -> None:
        self.env.close()


class ParallelGridEnv:
    def __init__(
        self,
        config: dict[str, Any],
        route_file: str | Path,
        seed: int | None = None,
        tripinfo_output: str | Path | None = None,
        fixed_ts: bool = False,
        use_gui: bool | None = None,
    ) -> None:
        self.agent_order = tuple(config["scenario"]["tls_ids"])
        effective_use_gui = bool(config["scenario"]["use_gui"] if use_gui is None else use_gui)
        configure_sumo_backend(use_gui=effective_use_gui)
        self.env = parallel_env(
            **build_env_kwargs(
                config,
                route_file,
                fixed_ts=fixed_ts,
                seed=seed,
                tripinfo_output=tripinfo_output,
                use_gui=effective_use_gui,
            )
        )
        self.base_env = self.env.unwrapped.env

    @property
    def observation_dim(self) -> int:
        return int(np.prod(self.env.observation_space(self.agent_order[0]).shape))

    @property
    def action_dim(self) -> int:
        return int(self.env.action_space(self.agent_order[0]).n)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self.base_env.metrics)

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        output = self.env.reset(seed=seed)
        if isinstance(output, tuple):
            return output
        return output, {agent: {} for agent in self.agent_order}

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        return self.env.step(actions)

    def close(self) -> None:
        self.env.close()
