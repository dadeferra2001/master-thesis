#!/usr/bin/env python
from __future__ import annotations

import argparse
import heapq
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.config import effective_net_file, load_experiment_config, resolve_path
from traffic_rl.utils import load_manifest, route_specs_for_split


@dataclass(frozen=True)
class SignalConnection:
    tl_id: str
    link_index: int
    from_edge: str
    to_edge: str
    direction: str


@dataclass(frozen=True)
class Stage:
    phase_index: int
    duration: int
    green_links: tuple[int, ...]
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Webster-style baseline timings for the existing network.")
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--manifest", default="routes/manifests/routes_manifest.json")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--intensity", action="append", choices=["low", "medium", "high"])
    parser.add_argument("--saturation-flow-per-lane", type=float, default=1900.0)
    parser.add_argument(
        "--lost-time-per-stage",
        type=float,
        default=3.0,
        help="Lost time assumed in Webster's formula for each green stage. Defaults to the current yellow length.",
    )
    parser.add_argument("--output", default="results/tables/webster_baseline.json")
    return parser.parse_args()


def _edge_lengths(root: ET.Element) -> dict[str, float]:
    lengths: dict[str, float] = {}
    for edge in root.findall("edge"):
        edge_id = edge.attrib["id"]
        if edge_id.startswith(":"):
            continue
        lane_lengths = [float(lane.attrib["length"]) for lane in edge.findall("lane")]
        lengths[edge_id] = sum(lane_lengths) / len(lane_lengths)
    return lengths


def _graph_and_connections(
    root: ET.Element,
) -> tuple[dict[str, list[tuple[float, str]]], dict[tuple[str, str], SignalConnection], dict[str, list[SignalConnection]]]:
    edge_lengths = _edge_lengths(root)
    graph: dict[str, list[tuple[float, str]]] = defaultdict(list)
    signal_connection_by_pair: dict[tuple[str, str], SignalConnection] = {}
    signal_connections_by_tl: dict[str, list[SignalConnection]] = defaultdict(list)

    for connection in root.findall("connection"):
        from_edge = connection.attrib.get("from")
        to_edge = connection.attrib.get("to")
        if not from_edge or not to_edge:
            continue
        if from_edge.startswith(":") or to_edge.startswith(":"):
            continue

        graph[from_edge].append((edge_lengths.get(to_edge, 1.0), to_edge))
        tl_id = connection.attrib.get("tl")
        if tl_id is None:
            continue

        signal_connection = SignalConnection(
            tl_id=tl_id,
            link_index=int(connection.attrib["linkIndex"]),
            from_edge=from_edge,
            to_edge=to_edge,
            direction=connection.attrib.get("dir", ""),
        )
        signal_connection_by_pair[(from_edge, to_edge)] = signal_connection
        signal_connections_by_tl[tl_id].append(signal_connection)

    return graph, signal_connection_by_pair, signal_connections_by_tl


def _shortest_edge_path(
    graph: dict[str, list[tuple[float, str]]],
    start_edge: str,
    end_edge: str,
) -> list[str]:
    if start_edge == end_edge:
        return [start_edge]

    queue: list[tuple[float, str]] = [(0.0, start_edge)]
    distance = {start_edge: 0.0}
    previous: dict[str, str] = {}

    while queue:
        current_cost, current_edge = heapq.heappop(queue)
        if current_edge == end_edge:
            break
        if current_cost > distance.get(current_edge, float("inf")):
            continue

        for weight, next_edge in graph.get(current_edge, []):
            next_cost = current_cost + weight
            if next_cost >= distance.get(next_edge, float("inf")):
                continue
            distance[next_edge] = next_cost
            previous[next_edge] = current_edge
            heapq.heappush(queue, (next_cost, next_edge))

    if end_edge not in distance:
        raise ValueError(f"No path found between edges {start_edge} and {end_edge}")

    path = [end_edge]
    while path[-1] != start_edge:
        path.append(previous[path[-1]])
    path.reverse()
    return path


def _phase_name(movements: list[SignalConnection]) -> str:
    orientation = "mixed"
    if movements and all("v" in movement.from_edge for movement in movements):
        orientation = "north_south"
    elif movements and all("h" in movement.from_edge for movement in movements):
        orientation = "east_west"

    directions = {movement.direction for movement in movements}
    if directions == {"l"}:
        turn_group = "left"
    elif directions <= {"r", "s"}:
        turn_group = "through_right"
    else:
        turn_group = "mixed"
    return f"{orientation}_{turn_group}"


def _parse_stages(root: ET.Element, signal_connections_by_tl: dict[str, list[SignalConnection]]) -> dict[str, list[Stage]]:
    stages_by_tl: dict[str, list[Stage]] = {}
    connection_lookup = {
        tl_id: {connection.link_index: connection for connection in connections}
        for tl_id, connections in signal_connections_by_tl.items()
    }

    for tl_logic in root.findall("tlLogic"):
        tl_id = tl_logic.attrib["id"]
        stages: list[Stage] = []
        for phase_index, phase in enumerate(tl_logic.findall("phase")):
            state = phase.attrib["state"]
            green_links = tuple(index for index, signal in enumerate(state) if signal in {"G", "g"})
            if not green_links:
                continue
            movements = [connection_lookup[tl_id][link_index] for link_index in green_links]
            stages.append(
                Stage(
                    phase_index=phase_index,
                    duration=int(phase.attrib["duration"]),
                    green_links=green_links,
                    name=_phase_name(movements),
                )
            )
        stages_by_tl[tl_id] = stages
    return stages_by_tl


def _yellow_durations(root: ET.Element) -> dict[str, list[int]]:
    yellow_by_tl: dict[str, list[int]] = {}
    for tl_logic in root.findall("tlLogic"):
        tl_id = tl_logic.attrib["id"]
        yellow_by_tl[tl_id] = [
            int(phase.attrib["duration"])
            for phase in tl_logic.findall("phase")
            if any(signal in {"y", "Y"} for signal in phase.attrib["state"])
        ]
    return yellow_by_tl


def _route_window_key(flow: ET.Element) -> str:
    begin = int(float(flow.attrib.get("begin", "0")))
    end = int(float(flow.attrib.get("end", "0")))
    return f"{begin:04d}-{end:04d}"


def _round_preserving_sum(values: list[float], total: int) -> list[int]:
    floors = [int(value) for value in values]
    remainder = total - sum(floors)
    fractions = sorted(
        ((value - int(value), index) for index, value in enumerate(values)),
        reverse=True,
    )
    rounded = list(floors)
    for _, index in fractions[: max(remainder, 0)]:
        rounded[index] += 1
    return rounded


def _compute_webster_plan(
    stage_ratios: list[float],
    stage_names: list[str],
    *,
    lost_time_per_stage: float,
    yellow_durations: list[int],
) -> dict[str, Any]:
    total_ratio = sum(stage_ratios)
    total_lost = lost_time_per_stage * len(stage_ratios)
    result: dict[str, Any] = {
        "stage_names": stage_names,
        "critical_flow_ratios": stage_ratios,
        "sum_critical_flow_ratio": total_ratio,
        "total_lost_time": total_lost,
        "yellow_durations": yellow_durations,
    }

    if total_ratio >= 1.0:
        result["status"] = "oversaturated"
        return result

    cycle = (1.5 * total_lost + 5.0) / (1.0 - total_ratio)
    total_effective_green = cycle - total_lost
    raw_greens = [
        (ratio / total_ratio) * total_effective_green if total_ratio > 0.0 else total_effective_green / len(stage_ratios)
        for ratio in stage_ratios
    ]
    rounded_greens = _round_preserving_sum(raw_greens, max(int(round(total_effective_green)), 0))

    phase_program: list[int] = []
    for green_duration, yellow_duration in zip(rounded_greens, yellow_durations):
        phase_program.extend([green_duration, yellow_duration])

    result.update(
        {
            "status": "ok",
            "webster_cycle": cycle,
            "effective_green_total": total_effective_green,
            "raw_green_durations": raw_greens,
            "rounded_green_durations": rounded_greens,
            "rounded_phase_program": phase_program,
            "rounded_cycle": sum(phase_program),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.env_config)
    net_path = resolve_path(effective_net_file(config))
    manifest = load_manifest(args.manifest)
    intensities = args.intensity or manifest["intensities"]

    root = ET.parse(net_path).getroot()
    graph, signal_connection_by_pair, signal_connections_by_tl = _graph_and_connections(root)
    stages_by_tl = _parse_stages(root, signal_connections_by_tl)
    yellow_by_tl = _yellow_durations(root)

    output: dict[str, Any] = {
        "net_file": str(net_path.relative_to(ROOT)),
        "manifest": args.manifest,
        "split": args.split,
        "saturation_flow_per_lane": args.saturation_flow_per_lane,
        "lost_time_per_stage": args.lost_time_per_stage,
        "intensities": {},
    }

    for intensity in intensities:
        route_specs = route_specs_for_split(manifest, split=args.split, intensity=intensity)
        if not route_specs:
            output["intensities"][intensity] = {"status": "no_routes"}
            continue

        episode_seconds = float(config["route_generation"]["episode_seconds"])
        overall_connection_volume: dict[str, dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
        window_connection_volume: dict[str, dict[str, dict[tuple[str, str], float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        for route_spec in route_specs:
            route_root = ET.parse(resolve_path(route_spec["path"])).getroot()
            path_cache: dict[tuple[str, str], list[str]] = {}
            for flow in route_root.findall("flow"):
                start_edge = flow.attrib["from"]
                end_edge = flow.attrib["to"]
                route_key = (start_edge, end_edge)
                if route_key not in path_cache:
                    path_cache[route_key] = _shortest_edge_path(graph, start_edge, end_edge)

                vehs_per_hour = float(flow.attrib["vehsPerHour"])
                window_weight = (float(flow.attrib["end"]) - float(flow.attrib["begin"])) / episode_seconds
                window_key = _route_window_key(flow)
                path = path_cache[route_key]
                for from_edge, to_edge in zip(path, path[1:]):
                    connection = signal_connection_by_pair.get((from_edge, to_edge))
                    if connection is None:
                        continue
                    connection_key = (connection.from_edge, connection.to_edge)
                    overall_connection_volume[connection.tl_id][connection_key] += (
                        vehs_per_hour * window_weight / len(route_specs)
                    )
                    window_connection_volume[window_key][connection.tl_id][connection_key] += vehs_per_hour / len(route_specs)

        per_tl: dict[str, Any] = {}
        average_stage_ratios = [0.0] * len(next(iter(stages_by_tl.values())))
        max_stage_ratios = [0.0] * len(next(iter(stages_by_tl.values())))
        stage_names = [stage.name for stage in next(iter(stages_by_tl.values()))]

        for tl_id, stages in stages_by_tl.items():
            stage_ratios: list[float] = []
            stage_volumes: list[float] = []
            for stage in stages:
                movements = [
                    connection
                    for connection in signal_connections_by_tl[tl_id]
                    if connection.link_index in stage.green_links
                ]
                grouped: dict[str, list[float]] = defaultdict(list)
                stage_volume = 0.0
                for movement in movements:
                    volume = overall_connection_volume[tl_id].get((movement.from_edge, movement.to_edge), 0.0)
                    grouped[movement.from_edge].append(volume / args.saturation_flow_per_lane)
                    stage_volume += volume
                stage_ratio = sum(max(values) for values in grouped.values())
                stage_ratios.append(stage_ratio)
                stage_volumes.append(stage_volume)

            for index, stage_ratio in enumerate(stage_ratios):
                average_stage_ratios[index] += stage_ratio / len(stages_by_tl)
                max_stage_ratios[index] = max(max_stage_ratios[index], stage_ratio)

            per_tl[tl_id] = {
                "stage_names": stage_names,
                "stage_flow_veh_per_hour": stage_volumes,
                "plan": _compute_webster_plan(
                    stage_ratios,
                    stage_names,
                    lost_time_per_stage=args.lost_time_per_stage,
                    yellow_durations=yellow_by_tl[tl_id],
                ),
            }

        per_window: dict[str, Any] = {}
        for window_key, tls_volume in sorted(window_connection_volume.items()):
            network_average_window_ratios = [0.0] * len(stage_names)
            for tl_id, stages in stages_by_tl.items():
                stage_ratios: list[float] = []
                for stage in stages:
                    movements = [
                        connection
                        for connection in signal_connections_by_tl[tl_id]
                        if connection.link_index in stage.green_links
                    ]
                    grouped: dict[str, list[float]] = defaultdict(list)
                    for movement in movements:
                        volume = tls_volume[tl_id].get((movement.from_edge, movement.to_edge), 0.0)
                        grouped[movement.from_edge].append(volume / args.saturation_flow_per_lane)
                    stage_ratios.append(sum(max(values) for values in grouped.values()))
                for index, stage_ratio in enumerate(stage_ratios):
                    network_average_window_ratios[index] += stage_ratio / len(stages_by_tl)

            per_window[window_key] = _compute_webster_plan(
                network_average_window_ratios,
                stage_names,
                lost_time_per_stage=args.lost_time_per_stage,
                yellow_durations=yellow_by_tl[next(iter(yellow_by_tl))],
            )

        output["intensities"][intensity] = {
            "status": "ok",
            "stage_names": stage_names,
            "per_tl": per_tl,
            "network_average": _compute_webster_plan(
                average_stage_ratios,
                stage_names,
                lost_time_per_stage=args.lost_time_per_stage,
                yellow_durations=yellow_by_tl[next(iter(yellow_by_tl))],
            ),
            "network_max": _compute_webster_plan(
                max_stage_ratios,
                stage_names,
                lost_time_per_stage=args.lost_time_per_stage,
                yellow_durations=yellow_by_tl[next(iter(yellow_by_tl))],
            ),
            "per_window_network_average": per_window,
        }

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
