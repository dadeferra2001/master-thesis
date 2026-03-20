#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.runtime import bootstrap_sumo

bootstrap_sumo(prefer_libsumo=True)

import torch

from traffic_rl import algo_centralized, algo_independent, algo_mappo, algo_shared
from traffic_rl.config import load_experiment_config
from traffic_rl.evaluation import run_policy_episode
from traffic_rl.reporting import aggregate_episode_summaries
from traffic_rl.utils import eval_dir, get_device, load_manifest, route_specs_for_split, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--manifest", default="routes/manifests/routes_manifest.json")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--intensity", choices=["low", "medium", "high"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Render the SUMO GUI during evaluation.")
    return parser.parse_args()


def load_controller(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    algo = checkpoint["algo"]
    if algo == "centralized_ppo":
        controller = algo_centralized.load_controller(checkpoint_path, device=device)
    elif algo == "shared_ppo":
        controller = algo_shared.load_controller(checkpoint_path, device=device)
    elif algo == "independent_ppo":
        controller = algo_independent.load_controller(checkpoint_path, device=device)
    elif algo == "mappo":
        controller = algo_mappo.load_controller(checkpoint_path, device=device)
    else:
        raise ValueError(f"Unsupported checkpoint algo: {algo}")
    return algo, controller


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    algo, controller = load_controller(args.checkpoint, device)
    config = load_experiment_config(args.env_config)
    manifest = load_manifest(args.manifest)
    intensities = [args.intensity] if args.intensity else manifest["intensities"]

    for intensity in intensities:
        rows = []
        for route_spec in route_specs_for_split(manifest, split=args.split, intensity=intensity):
            run_dir = eval_dir(algo, args.split, intensity, int(route_spec["seed"]))
            summary = run_policy_episode(
                algo=algo,
                controller=controller,
                config=config,
                route_file=route_spec["path"],
                route_seed=int(route_spec["seed"]),
                output_dir=run_dir,
                deterministic=not args.stochastic,
                use_gui=args.gui,
            )
            summary["intensity"] = intensity
            rows.append(summary)
        aggregate = aggregate_episode_summaries(rows)
        aggregate.update({"algorithm": algo, "split": args.split, "intensity": intensity})
        output_root = ROOT / "results" / "eval" / algo / args.split / intensity
        write_csv(output_root / "episodes.csv", rows)
        write_json(output_root / "aggregate.json", aggregate)
        print(output_root / "aggregate.json")


if __name__ == "__main__":
    main()
