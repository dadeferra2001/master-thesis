"""Pedestrian-aware SUMO-RL helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from gymnasium import spaces

from .runtime import bootstrap_sumo

bootstrap_sumo(prefer_libsumo=True)

from sumo_rl.environment.observations import DefaultObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal

from .config import pedestrian_scenario_config


@dataclass(frozen=True)
class PedestrianCrossingState:
    walkingarea_edge: str
    crossing_edge: str
    link_index: int
    queue_length: int
    waiting_time: float
    max_waiting_time: float
    served: bool


def _lane_edge_id(lane_id: str) -> str:
    return lane_id.rsplit("_", 1)[0]


def pedestrian_controlled_links(ts: TrafficSignal) -> list[tuple[str, str, int]]:
    cached = getattr(ts, "_pedestrian_links_cache", None)
    if cached is not None:
        return cached

    pairs: list[tuple[str, str, int]] = []
    for link_index, controlled_link in enumerate(ts.sumo.trafficlight.getControlledLinks(ts.id)):
        if not controlled_link:
            continue
        from_lane, to_lane, _ = controlled_link[0]
        if not from_lane.startswith(f":{ts.id}_w") or not to_lane.startswith(f":{ts.id}_c"):
            continue
        pairs.append((_lane_edge_id(from_lane), _lane_edge_id(to_lane), int(link_index)))

    pairs = sorted(set(pairs), key=lambda pair: pair[2])
    ts._pedestrian_links_cache = pairs
    return pairs


def pedestrian_links_for_signal(ts: TrafficSignal) -> list[tuple[str, str]]:
    return [(walkingarea_edge, crossing_edge) for walkingarea_edge, crossing_edge, _ in pedestrian_controlled_links(ts)]


def pedestrian_crossing_states(ts: TrafficSignal) -> list[PedestrianCrossingState]:
    current_state = ts.sumo.trafficlight.getRedYellowGreenState(ts.id)
    crossing_states: list[PedestrianCrossingState] = []

    for walkingarea_edge, crossing_edge, link_index in pedestrian_controlled_links(ts):
        person_ids = list(ts.sumo.edge.getLastStepPersonIDs(walkingarea_edge))
        waiting_times = [float(ts.sumo.person.getWaitingTime(person_id)) for person_id in person_ids]
        crossing_states.append(
            PedestrianCrossingState(
                walkingarea_edge=walkingarea_edge,
                crossing_edge=crossing_edge,
                link_index=link_index,
                queue_length=len(person_ids),
                waiting_time=float(sum(waiting_times)),
                max_waiting_time=float(max(waiting_times, default=0.0)),
                served=current_state[link_index] in {"g", "G"},
            )
        )

    return crossing_states


def total_pedestrian_waiting_time(ts: TrafficSignal) -> float:
    return float(sum(crossing.waiting_time for crossing in pedestrian_crossing_states(ts)))


def pedestrian_queue_lengths(ts: TrafficSignal) -> list[int]:
    return [crossing.queue_length for crossing in pedestrian_crossing_states(ts)]


def max_pedestrian_waiting_time(ts: TrafficSignal) -> float:
    return float(max((crossing.max_waiting_time for crossing in pedestrian_crossing_states(ts)), default=0.0))


def pedestrian_step_metrics(env: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    total_waiting_time = 0.0
    total_queue_length = 0
    total_max_waiting_time = 0.0

    for ts_id in env.ts_ids:
        ts = env.traffic_signals[ts_id]
        crossing_states = pedestrian_crossing_states(ts)
        waiting_time = sum(crossing.waiting_time for crossing in crossing_states)
        queue_length = sum(crossing.queue_length for crossing in crossing_states)
        max_waiting_time = max((crossing.max_waiting_time for crossing in crossing_states), default=0.0)
        mean_waiting_time = waiting_time / float(queue_length) if queue_length > 0 else 0.0

        metrics[f"{ts_id}_pedestrian_waiting_time"] = float(waiting_time)
        metrics[f"{ts_id}_pedestrian_queue_length"] = float(queue_length)
        metrics[f"{ts_id}_pedestrian_mean_waiting_time"] = float(mean_waiting_time)
        metrics[f"{ts_id}_pedestrian_max_waiting_time"] = float(max_waiting_time)

        total_waiting_time += float(waiting_time)
        total_queue_length += int(queue_length)
        total_max_waiting_time = max(total_max_waiting_time, float(max_waiting_time))

    metrics["pedestrian_total_waiting_time"] = float(total_waiting_time)
    metrics["pedestrian_total_queue_length"] = float(total_queue_length)
    metrics["pedestrian_mean_waiting_time"] = (
        float(total_waiting_time / float(total_queue_length)) if total_queue_length > 0 else 0.0
    )
    metrics["pedestrian_max_waiting_time"] = float(total_max_waiting_time)
    return metrics


def attach_pedestrian_metrics(env: Any) -> Any:
    if getattr(env, "_pedestrian_metrics_attached", False):
        return env

    original_compute_info = env._compute_info

    def _compute_info_with_pedestrians():
        info = original_compute_info()
        ped_info = pedestrian_step_metrics(env)
        info.update(ped_info)
        if env.metrics:
            env.metrics[-1].update(ped_info)
        return info

    env._compute_info = _compute_info_with_pedestrians
    env._pedestrian_metrics_attached = True
    return env


def make_pedestrian_observation_class(config: dict[str, Any]) -> type[DefaultObservationFunction]:
    ped_cfg = pedestrian_scenario_config(config)
    include_ped_observation = bool(ped_cfg["observation"])
    waiting_time_scale = float(max(ped_cfg["waiting_time_scale"], 1.0))
    queue_scale = float(max(ped_cfg["queue_scale"], 1.0))

    class PedestrianObservationFunction(DefaultObservationFunction):
        def __call__(self) -> np.ndarray:
            observation = super().__call__()
            if not include_ped_observation:
                return observation

            features: list[float] = []
            for walkingarea_edge, _ in pedestrian_links_for_signal(self.ts):
                person_ids = self.ts.sumo.edge.getLastStepPersonIDs(walkingarea_edge)
                queue = len(person_ids)
                waiting = sum(float(self.ts.sumo.person.getWaitingTime(person_id)) for person_id in person_ids)
                features.extend(
                    [
                        min(queue / queue_scale, 1.0),
                        min(waiting / waiting_time_scale, 1.0),
                    ]
                )
            return np.concatenate((observation, np.asarray(features, dtype=np.float32))).astype(np.float32, copy=False)

        def observation_space(self) -> spaces.Box:
            base_space = super().observation_space()
            if not include_ped_observation:
                return base_space
            feature_count = 2 * len(pedestrian_links_for_signal(self.ts))
            low = np.concatenate((base_space.low, np.zeros(feature_count, dtype=np.float32))).astype(
                np.float32,
                copy=False,
            )
            high = np.concatenate((base_space.high, np.ones(feature_count, dtype=np.float32))).astype(
                np.float32,
                copy=False,
            )
            return spaces.Box(low=low, high=high, dtype=np.float32)

    return PedestrianObservationFunction


def make_pedestrian_reward_fn(config: dict[str, Any]) -> Callable[[TrafficSignal], float]:
    ped_cfg = pedestrian_scenario_config(config)
    waiting_time_scale = float(max(ped_cfg["waiting_time_scale"], 1.0))
    reward_weight = float(ped_cfg["reward_weight"])
    fairness_wait_threshold = float(max(ped_cfg["fairness_wait_threshold"], 1.0))
    max_wait_penalty = float(ped_cfg["max_wait_penalty"])
    starvation_penalty = float(ped_cfg["starvation_penalty"])
    vehicle_reward_fn = TrafficSignal.reward_fns["diff-waiting-time"]

    def pedestrian_reward(ts: TrafficSignal) -> float:
        vehicle_reward = float(vehicle_reward_fn(ts))
        crossing_states = pedestrian_crossing_states(ts)
        ped_wait = float(sum(crossing.waiting_time for crossing in crossing_states)) / waiting_time_scale
        last_wait = float(getattr(ts, "_last_pedestrian_waiting_time", 0.0))
        setattr(ts, "_last_pedestrian_waiting_time", ped_wait)
        max_wait = float(max((crossing.max_waiting_time for crossing in crossing_states), default=0.0))
        max_wait_cost = max(0.0, max_wait - fairness_wait_threshold) / fairness_wait_threshold

        starvation_cost = 0.0
        decision_step = float(max(getattr(ts, "delta_time", 1), 1))
        for crossing in crossing_states:
            timer_attr = f"_pedestrian_starvation_seconds_{crossing.link_index}"
            blocked_seconds = float(getattr(ts, timer_attr, 0.0))
            if crossing.queue_length > 0 and not crossing.served:
                blocked_seconds += decision_step
            else:
                blocked_seconds = 0.0
            setattr(ts, timer_attr, blocked_seconds)
            starvation_cost += max(0.0, blocked_seconds - fairness_wait_threshold) / fairness_wait_threshold

        return (
            vehicle_reward
            + reward_weight * (last_wait - ped_wait)
            - max_wait_penalty * max_wait_cost
            - starvation_penalty * starvation_cost
        )

    pedestrian_reward.__name__ = "diff-waiting-time-pedestrians"
    return pedestrian_reward


def pedestrian_mode_enabled(config: dict[str, Any]) -> bool:
    return bool(pedestrian_scenario_config(config)["enabled"])
