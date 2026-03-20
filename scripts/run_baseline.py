#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.config import load_experiment_config
from traffic_rl.evaluation import run_baseline_episode
from traffic_rl.reporting import aggregate_episode_summaries
from traffic_rl.utils import eval_dir, load_manifest, route_specs_for_split, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the SUMO default baseline.")
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--manifest", default="routes/manifests/routes_manifest.json")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--intensity", choices=["low", "medium", "high"])
    parser.add_argument("--gui", action="store_true", help="Render the SUMO GUI during baseline evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.env_config)
    manifest = load_manifest(args.manifest)
    intensities = [args.intensity] if args.intensity else manifest["intensities"]

    for intensity in intensities:
        rows = []
        for route_spec in route_specs_for_split(manifest, split=args.split, intensity=intensity):
            run_dir = eval_dir("baseline", args.split, intensity, int(route_spec["seed"]))
            summary = run_baseline_episode(
                config,
                route_file=route_spec["path"],
                route_seed=int(route_spec["seed"]),
                output_dir=run_dir,
                use_gui=args.gui,
            )
            summary["intensity"] = intensity
            rows.append(summary)
        aggregate = aggregate_episode_summaries(rows)
        aggregate.update({"algorithm": "baseline", "split": args.split, "intensity": intensity})
        output_root = ROOT / "results" / "eval" / "baseline" / args.split / intensity
        write_csv(output_root / "episodes.csv", rows)
        write_json(output_root / "aggregate.json", aggregate)
        print(output_root / "aggregate.json")


if __name__ == "__main__":
    main()
