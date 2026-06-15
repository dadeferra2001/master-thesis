#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.comparison import DEFAULT_METRICS, write_comparison_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build comparison tables and charts from evaluation results.")
    parser.add_argument(
        "--results-root",
        help="Evaluation aggregate root. Defaults to results/eval_across_seeds if it exists, otherwise results/eval.",
    )
    parser.add_argument("--output-dir", default="results/compare")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--intensity", action="append", choices=["low", "medium", "high"])
    parser.add_argument("--variant", action="append", choices=["vehicle", "peds"])
    parser.add_argument(
        "--algorithm",
        action="append",
        help="Filter to one or more algorithms, for example --algorithm baseline --algorithm shared_ppo",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help=(
            "Metric to include. Can be repeated. Defaults to: "
            + ", ".join(spec.key for spec in DEFAULT_METRICS)
        ),
    )
    parser.add_argument("--baseline", default="baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root
    if results_root is None:
        across_seed_root = ROOT / "results" / "eval_across_seeds"
        results_root = "results/eval_across_seeds" if across_seed_root.exists() else "results/eval"
    outputs = write_comparison_outputs(
        results_root=results_root,
        output_dir=args.output_dir,
        split=args.split,
        intensities=args.intensity,
        variants=args.variant,
        algorithms=args.algorithm,
        metric_keys=args.metric,
        baseline_algo=args.baseline,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
