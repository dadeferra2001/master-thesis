#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.utils import eval_group_dir, write_csv, write_json


GROUP_KEYS = ("algorithm", "variant", "split", "intensity")
INTENSITY_ORDER = {"low": 0, "medium": 1, "high": 2}
VARIANT_ORDER = {"vehicle": 0, "peds": 1}
TRAIN_SEED_PATTERN = re.compile(r"(?:^|/)train_seed_(-?\d+)(?:/|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-training-seed and across-training-seed evaluation summaries."
    )
    parser.add_argument("--results-root", default="results/eval")
    parser.add_argument(
        "--output",
        default="results/tables/evaluation_summary.csv",
        help="Backward-compatible copy of the per-seed table.",
    )
    parser.add_argument("--per-seed-output", default="results/tables/evaluation_per_seed.csv")
    parser.add_argument("--across-seeds-output", default="results/tables/evaluation_across_seeds.csv")
    parser.add_argument("--across-seeds-root", default="results/eval_across_seeds")
    parser.add_argument("--variant", choices=["vehicle", "peds"])
    parser.add_argument(
        "--legacy-train-seed",
        type=int,
        default=0,
        help=(
            "Training seed assigned to old RL aggregate.json files that predate train_seed_<n> "
            "evaluation directories. Baseline rows are never assigned a train seed."
        ),
    )
    return parser.parse_args()


def resolve_input_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def normalize_variant(payload: dict[str, Any], source_file: str) -> str:
    variant = str(payload.get("variant") or "").strip().lower()
    if variant in VARIANT_ORDER:
        return variant
    return "peds" if "/peds/" in source_file.replace("\\", "/") else "vehicle"


def infer_train_seed(payload: dict[str, Any], source_file: str, legacy_train_seed: int) -> tuple[int | None, str]:
    if payload.get("algorithm") == "baseline":
        return None, "baseline"

    payload_seed = payload.get("train_seed")
    if payload_seed not in (None, ""):
        return int(payload_seed), "payload"

    match = TRAIN_SEED_PATTERN.search(source_file.replace("\\", "/"))
    if match:
        return int(match.group(1)), "path"

    return legacy_train_seed, "legacy"


def load_seed_rows(results_root: Path, *, variant: str | None, legacy_train_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("aggregate.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        try:
            payload["source_file"] = str(path.relative_to(ROOT))
        except ValueError:
            payload["source_file"] = str(path)
        payload["variant"] = normalize_variant(payload, payload["source_file"])
        train_seed, train_seed_source = infer_train_seed(payload, payload["source_file"], legacy_train_seed)
        payload["train_seed"] = train_seed
        payload["train_seed_source"] = train_seed_source
        payload["aggregation_level"] = "per_train_seed"
        if variant and payload["variant"] != variant:
            continue
        rows.append(payload)
    return deduplicate_seed_rows(rows)


def row_preference(row: dict[str, Any]) -> tuple[int, str]:
    source = str(row.get("train_seed_source", ""))
    source_rank = {"payload": 3, "path": 2, "legacy": 1, "baseline": 0}.get(source, 0)
    return source_rank, str(row.get("source_file", ""))


def deduplicate_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(key_name) for key_name in GROUP_KEYS) + (row.get("train_seed"),)
        existing = selected.get(key)
        if existing is None or row_preference(row) > row_preference(existing):
            selected[key] = row
    return sorted(selected.values(), key=sort_key)


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    train_seed = row.get("train_seed")
    return (
        VARIANT_ORDER.get(str(row.get("variant")), 99),
        INTENSITY_ORDER.get(str(row.get("intensity")), 99),
        str(row.get("algorithm", "")),
        10**9 if train_seed in (None, "") else int(train_seed),
        str(row.get("source_file", "")),
    )


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)) and not math.isnan(float(value)):
            values.append(float(value))
    return values


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return float(math.sqrt(sum((value - avg) ** 2 for value in values) / len(values)))


def aggregate_across_seeds(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        key = tuple(row.get(key_name) for key_name in GROUP_KEYS)
        groups[key].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for key, rows in groups.items():
        algorithm, variant, split, intensity = key
        train_seeds = sorted(
            {int(row["train_seed"]) for row in rows if row.get("train_seed") not in (None, "")}
        )
        record: dict[str, Any] = {
            "algorithm": algorithm,
            "variant": variant,
            "split": split,
            "intensity": intensity,
            "aggregation_level": "across_train_seeds",
            "seed_count": len(rows),
            "train_seeds": ",".join(str(seed) for seed in train_seeds),
            "episodes": int(sum(numeric_values(rows, "episodes"))),
            "source_files": ";".join(str(row.get("source_file", "")) for row in rows),
        }

        mean_keys = sorted(
            {
                key_name
                for row in rows
                for key_name, value in row.items()
                if key_name.endswith("_mean") and isinstance(value, (int, float, bool))
            }
        )
        for mean_key in mean_keys:
            values = numeric_values(rows, mean_key)
            if not values:
                continue
            record[mean_key] = mean(values)
            record[f"{mean_key[:-5]}_std"] = population_std(values)

        aggregate_rows.append(record)

    return sorted(aggregate_rows, key=sort_key)


def write_across_seed_json(rows: list[dict[str, Any]], output_root: str | Path) -> None:
    namespace = str(output_root)
    if namespace.startswith("results/"):
        namespace = namespace[len("results/") :]

    for row in rows:
        variant = None if row.get("variant") == "vehicle" else str(row.get("variant"))
        target = eval_group_dir(
            str(row["algorithm"]),
            str(row["split"]),
            str(row["intensity"]),
            variant=variant,
            results_namespace=namespace,
        )
        write_json(target / "aggregate.json", row)


def main() -> None:
    args = parse_args()
    seed_rows = load_seed_rows(
        resolve_input_path(args.results_root),
        variant=args.variant,
        legacy_train_seed=args.legacy_train_seed,
    )
    across_seed_rows = aggregate_across_seeds(seed_rows)

    write_csv(args.per_seed_output, seed_rows)
    if args.output and args.output != args.per_seed_output:
        write_csv(args.output, seed_rows)
    write_csv(args.across_seeds_output, across_seed_rows)
    write_across_seed_json(across_seed_rows, args.across_seeds_root)

    print(f"per_seed: {args.per_seed_output}")
    if args.output and args.output != args.per_seed_output:
        print(f"legacy_summary: {args.output}")
    print(f"across_seeds: {args.across_seeds_output}")
    print(f"across_seed_json: {args.across_seeds_root}")


if __name__ == "__main__":
    main()
