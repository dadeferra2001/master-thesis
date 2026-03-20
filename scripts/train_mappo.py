#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.algo_mappo import train
from traffic_rl.config import load_experiment_config
from traffic_rl.utils import load_manifest, route_specs_for_split, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MAPPO.")
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--ppo-common", default="configs/ppo_common.yaml")
    parser.add_argument("--ppo-config", default="configs/ppo_mappo.yaml")
    parser.add_argument("--manifest", default="routes/manifests/routes_manifest.json")
    parser.add_argument("--intensity", choices=["low", "medium", "high"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.env_config)
    ppo_config = load_experiment_config(args.ppo_common, args.ppo_config)
    if args.total_timesteps is not None:
        ppo_config["train"]["total_timesteps"] = args.total_timesteps
    if args.num_steps is not None:
        ppo_config["train"]["num_steps"] = args.num_steps
    if args.device:
        ppo_config["train"]["device"] = args.device

    seed_everything(args.seed)
    manifest = load_manifest(args.manifest)
    route_specs = route_specs_for_split(manifest, split="train", intensity=args.intensity)
    checkpoint_path = train(config, ppo_config, route_specs, intensity=args.intensity, seed=args.seed)
    print(checkpoint_path)


if __name__ == "__main__":
    main()
