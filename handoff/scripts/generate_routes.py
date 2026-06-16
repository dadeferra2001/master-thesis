#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.config import default_manifest_path, load_experiment_config, set_pedestrians_enabled
from traffic_rl.routes import generate_route_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible train/test route files.")
    parser.add_argument("--config", default="configs/env.yaml")
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=20)
    parser.add_argument("--test-start", type=int, default=100)
    parser.add_argument("--test-count", type=int, default=10)
    parser.add_argument("--intensities", nargs="+", default=["low", "medium", "high"])
    parser.add_argument("--output")
    parser.add_argument("--with-pedestrians", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.with_pedestrians:
        set_pedestrians_enabled(config, True)
    train_seeds = range(args.train_start, args.train_start + args.train_count)
    test_seeds = range(args.test_start, args.test_start + args.test_count)
    output_path = args.output or default_manifest_path(config)
    manifest = generate_route_manifest(
        config=config,
        train_seeds=train_seeds,
        test_seeds=test_seeds,
        intensities=args.intensities,
        output_path=output_path,
    )
    print(f"Generated {len(manifest['routes'])} route files at {output_path}")


if __name__ == "__main__":
    main()
