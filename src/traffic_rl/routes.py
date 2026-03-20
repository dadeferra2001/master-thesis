"""Route generation utilities for reproducible traffic demand."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import ensure_dir, resolve_path, utc_timestamp, write_json
from .scenario import ORIGINS, turn_destinations


def _window_bounds(episode_seconds: int, num_windows: int) -> list[tuple[int, int]]:
    width = episode_seconds // num_windows
    bounds = []
    for idx in range(num_windows):
        begin = idx * width
        end = episode_seconds if idx == num_windows - 1 else (idx + 1) * width
        bounds.append((begin, end))
    return bounds


def generate_route_file(
    output_path: str | Path,
    intensity: str,
    seed: int,
    config: dict,
) -> dict:
    route_cfg = config["route_generation"]
    rng = np.random.default_rng(seed)
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    turn_probs = route_cfg["turn_probabilities"]
    alpha = route_cfg["dirichlet_concentration"] * np.array(
        [turn_probs["straight"], turn_probs["left"], turn_probs["right"]],
        dtype=np.float64,
    )
    base_origin_rate = route_cfg["base_origin_vehs_per_hour"] * route_cfg["intensity_scale"][intensity]
    min_flow_rate = route_cfg["min_flow_vehs_per_hour"]
    profile = route_cfg["window_profile"]
    lognormal_sigma = route_cfg["lognormal_sigma"]

    root = ET.Element("routes")
    flow_count = 0
    for window_idx, (begin, end) in enumerate(
        _window_bounds(route_cfg["episode_seconds"], route_cfg["num_windows"])
    ):
        for origin in ORIGINS:
            origin_rate = base_origin_rate * profile[window_idx]
            origin_rate *= float(rng.lognormal(mean=0.0, sigma=lognormal_sigma))
            turn_shares = rng.dirichlet(alpha)
            destinations = turn_destinations(origin)
            for turn_idx, turn_name in enumerate(("straight", "left", "right")):
                flow_rate = max(min_flow_rate, origin_rate * float(turn_shares[turn_idx]))
                flow = ET.SubElement(root, "flow")
                flow.set("id", f"{origin.name}_{turn_name}_w{window_idx}")
                flow.set("from", origin.edge)
                flow.set("to", destinations[turn_name])
                flow.set("begin", str(begin))
                flow.set("end", str(end))
                flow.set("vehsPerHour", str(int(round(flow_rate))))
                flow.set("departSpeed", "max")
                flow.set("departPos", "base")
                flow.set("departLane", "best")
                flow_count += 1

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="utf-8", xml_declaration=False)

    return {
        "path": str(output_path.relative_to(resolve_path("."))),
        "intensity": intensity,
        "seed": seed,
        "flow_count": flow_count,
        "base_origin_vehs_per_hour": base_origin_rate,
    }


def generate_route_manifest(
    config: dict,
    train_seeds: Iterable[int],
    test_seeds: Iterable[int],
    intensities: Iterable[str],
    output_path: str | Path = "routes/manifests/routes_manifest.json",
) -> dict:
    manifest = {
        "generated_at": utc_timestamp(),
        "episode_seconds": config["route_generation"]["episode_seconds"],
        "intensities": list(intensities),
        "routes": [],
    }

    for split, seeds in (("train", train_seeds), ("test", test_seeds)):
        for intensity in intensities:
            output_dir = ensure_dir(Path("routes") / split / intensity)
            for seed in seeds:
                route_path = output_dir / f"seed_{seed:03d}.rou.xml"
                route_record = generate_route_file(route_path, intensity=intensity, seed=int(seed), config=config)
                route_record["split"] = split
                manifest["routes"].append(route_record)

    write_json(output_path, manifest)
    return manifest
