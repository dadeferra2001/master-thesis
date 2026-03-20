"""Episode metric extraction from SUMO and SUMO-RL outputs."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def parse_tripinfo(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "completed_trips": 0,
            "mean_travel_time": 0.0,
            "mean_time_loss": 0.0,
            "mean_trip_waiting_time": 0.0,
        }

    target = Path(path)
    if not target.exists():
        return {
            "completed_trips": 0,
            "mean_travel_time": 0.0,
            "mean_time_loss": 0.0,
            "mean_trip_waiting_time": 0.0,
        }

    root = ET.parse(target).getroot()
    durations: list[float] = []
    waiting_times: list[float] = []
    time_losses: list[float] = []
    depart_delays: list[float] = []

    for tripinfo in root.findall("tripinfo"):
        durations.append(float(tripinfo.attrib.get("duration", 0.0)))
        waiting_times.append(float(tripinfo.attrib.get("waitingTime", 0.0)))
        time_losses.append(float(tripinfo.attrib.get("timeLoss", 0.0)))
        depart_delays.append(float(tripinfo.attrib.get("departDelay", 0.0)))

    return {
        "completed_trips": len(durations),
        "mean_travel_time": _safe_mean(durations),
        "mean_time_loss": _safe_mean(time_losses),
        "mean_trip_waiting_time": _safe_mean(waiting_times),
        "mean_depart_delay": _safe_mean(depart_delays),
    }


def summarize_step_metrics(step_history: list[dict[str, Any]]) -> dict[str, Any]:
    if not step_history:
        return {
            "num_decision_steps": 0,
            "mean_average_speed": 0.0,
            "mean_average_waiting_time": 0.0,
            "mean_queue_length": 0.0,
            "max_queue_length": 0.0,
            "throughput": 0,
            "teleports": 0,
            "mean_backlog": 0.0,
            "final_backlog": 0.0,
        }

    speeds = [float(record.get("system_mean_speed", 0.0)) for record in step_history]
    waits = [float(record.get("system_mean_waiting_time", 0.0)) for record in step_history]
    queues = [float(record.get("system_total_stopped", 0.0)) for record in step_history]
    backlogs = [float(record.get("system_total_backlogged", 0.0)) for record in step_history]
    final_record = step_history[-1]

    return {
        "num_decision_steps": len(step_history),
        "mean_average_speed": _safe_mean(speeds),
        "mean_average_waiting_time": _safe_mean(waits),
        "mean_queue_length": _safe_mean(queues),
        "max_queue_length": float(max(queues)),
        "throughput": int(final_record.get("system_total_arrived", 0)),
        "teleports": int(final_record.get("system_total_teleported", 0)),
        "mean_backlog": _safe_mean(backlogs),
        "final_backlog": float(final_record.get("system_total_backlogged", 0.0)),
    }


def summarize_episode(
    step_history: list[dict[str, Any]],
    tripinfo_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summarize_step_metrics(step_history)
    summary.update(parse_tripinfo(tripinfo_path))
    summary["congestion_failure"] = bool(summary["teleports"] > 0 or summary["final_backlog"] > 0)
    if extra:
        summary.update(extra)
    return summary
