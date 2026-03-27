"""Project configuration helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PEDESTRIAN_SCENARIO = {
    "enabled": False,
    "observation": True,
    "reward_weight": 5.0,
    "waiting_time_scale": 100.0,
    "queue_scale": 10.0,
    "fairness_wait_threshold": 30.0,
    "max_wait_penalty": 2.5,
    "starvation_penalty": 3.0,
    "net_file": "nets/2x2/2x2_peds.net.xml",
}
DEFAULT_PEDESTRIAN_ROUTE_GENERATION = {
    "base_origin_peds_per_hour": 90,
    "min_flow_peds_per_hour": 5,
    "intensity_scale": {"low": 0.55, "medium": 1.0, "high": 1.30},
    "window_profile": [0.95, 1.05, 1.10, 0.90],
    "lognormal_sigma": 0.15,
}


def project_root() -> Path:
    return ROOT


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_experiment_config(*paths: str | Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for path in paths:
        config = merge_dicts(config, load_yaml(path))
    return normalize_config(config)


def ensure_dir(path: str | Path) -> Path:
    target = resolve_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: Any) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: str | Path) -> Any:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    scenario = normalized.setdefault("scenario", {})
    route_generation = normalized.setdefault("route_generation", {})
    scenario["pedestrians"] = merge_dicts(DEFAULT_PEDESTRIAN_SCENARIO, scenario.get("pedestrians", {}))

    ped_route_cfg = merge_dicts(DEFAULT_PEDESTRIAN_ROUTE_GENERATION, route_generation.get("pedestrians", {}))
    ped_route_cfg.setdefault("intensity_scale", deepcopy(route_generation.get("intensity_scale", {})))
    ped_route_cfg.setdefault("window_profile", deepcopy(route_generation.get("window_profile", [])))
    ped_route_cfg.setdefault("lognormal_sigma", route_generation.get("lognormal_sigma", 0.15))
    if not route_generation.get("pedestrians", {}).get("intensity_scale"):
        ped_route_cfg["intensity_scale"] = deepcopy(route_generation.get("intensity_scale", ped_route_cfg["intensity_scale"]))
    if not route_generation.get("pedestrians", {}).get("window_profile"):
        ped_route_cfg["window_profile"] = deepcopy(route_generation.get("window_profile", ped_route_cfg["window_profile"]))
    if "lognormal_sigma" not in route_generation.get("pedestrians", {}):
        ped_route_cfg["lognormal_sigma"] = route_generation.get("lognormal_sigma", ped_route_cfg["lognormal_sigma"])
    route_generation["pedestrians"] = ped_route_cfg
    return normalized


def pedestrian_scenario_config(config: dict[str, Any]) -> dict[str, Any]:
    scenario = config.get("scenario", {})
    return merge_dicts(DEFAULT_PEDESTRIAN_SCENARIO, scenario.get("pedestrians", {}))


def pedestrian_route_generation_config(config: dict[str, Any]) -> dict[str, Any]:
    route_generation = config.get("route_generation", {})
    ped_cfg = merge_dicts(DEFAULT_PEDESTRIAN_ROUTE_GENERATION, route_generation.get("pedestrians", {}))
    if "intensity_scale" not in route_generation.get("pedestrians", {}):
        ped_cfg["intensity_scale"] = deepcopy(route_generation.get("intensity_scale", ped_cfg["intensity_scale"]))
    if "window_profile" not in route_generation.get("pedestrians", {}):
        ped_cfg["window_profile"] = deepcopy(route_generation.get("window_profile", ped_cfg["window_profile"]))
    if "lognormal_sigma" not in route_generation.get("pedestrians", {}):
        ped_cfg["lognormal_sigma"] = route_generation.get("lognormal_sigma", ped_cfg["lognormal_sigma"])
    return ped_cfg


def pedestrians_enabled(config: dict[str, Any]) -> bool:
    return bool(pedestrian_scenario_config(config)["enabled"])


def set_pedestrians_enabled(config: dict[str, Any], enabled: bool) -> dict[str, Any]:
    config.setdefault("scenario", {}).setdefault("pedestrians", {})
    config["scenario"]["pedestrians"]["enabled"] = bool(enabled)
    return config


def experiment_variant(config: dict[str, Any]) -> str | None:
    return "peds" if pedestrians_enabled(config) else None


def effective_net_file(config: dict[str, Any]) -> str:
    scenario = config.get("scenario", {})
    if pedestrians_enabled(config):
        return str(pedestrian_scenario_config(config)["net_file"])
    return str(scenario["net_file"])


def default_manifest_path(config: dict[str, Any]) -> str:
    return "routes/manifests/routes_manifest_peds.json" if pedestrians_enabled(config) else "routes/manifests/routes_manifest.json"


def default_routes_root(config: dict[str, Any]) -> Path:
    return Path("routes_peds") if pedestrians_enabled(config) else Path("routes")
