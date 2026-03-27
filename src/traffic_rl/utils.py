"""General utility functions shared by scripts and trainers."""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ensure_dir, project_root, resolve_path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        has_cuda = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                has_cuda = torch.cuda.is_available()
            except Exception:
                has_cuda = False
        return torch.device("cuda" if has_cuda else "cpu")
    return torch.device(requested)


def flatten_dict_keys(records: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for record in records:
        keys.update(record.keys())
    return sorted(keys)


def write_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = flatten_dict_keys(records)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_json(path: str | Path, payload: Any) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_manifest(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def route_specs_for_split(
    manifest: dict[str, Any],
    split: str,
    intensity: str | None = None,
) -> list[dict[str, Any]]:
    matches = [
        route
        for route in manifest["routes"]
        if route["split"] == split and (intensity is None or route["intensity"] == intensity)
    ]
    return sorted(matches, key=lambda item: (item["intensity"], item["seed"]))


def _results_root(*parts: str | Path, variant: str | None = None) -> Path:
    root = project_root() / "results"
    for part in parts:
        root /= Path(part)
    if variant:
        root /= variant
    return root


def checkpoint_dir(algo: str, intensity: str, seed: int, variant: str | None = None) -> Path:
    return ensure_dir(_results_root("checkpoints", algo, variant=variant) / intensity / f"seed_{seed}")


def train_log_path(algo: str, intensity: str, seed: int, variant: str | None = None) -> Path:
    root = ensure_dir(_results_root("train_logs", algo, variant=variant) / intensity)
    return root / f"seed_{seed}.jsonl"


def eval_dir(algo: str, split: str, intensity: str, route_seed: int, variant: str | None = None) -> Path:
    return ensure_dir(_results_root("eval", algo, variant=variant) / split / intensity / f"route_{route_seed:03d}")


def tensorboard_run_dir(algo: str, intensity: str, seed: int, variant: str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(_results_root("tensorboard", algo, variant=variant) / intensity / f"seed_{seed}" / timestamp)
