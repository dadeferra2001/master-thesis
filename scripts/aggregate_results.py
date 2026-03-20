#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.utils import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-run evaluation summaries into a single CSV.")
    parser.add_argument("--results-root", default="results/eval")
    parser.add_argument("--output", default="results/tables/evaluation_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = ROOT / args.results_root
    rows = []
    for path in sorted(results_root.rglob("aggregate.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["source_file"] = str(path.relative_to(ROOT))
        rows.append(payload)
    write_csv(args.output, rows)
    print(args.output)


if __name__ == "__main__":
    main()
