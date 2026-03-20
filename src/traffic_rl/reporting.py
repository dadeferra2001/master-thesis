"""Result aggregation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def aggregate_episode_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"episodes": 0}

    aggregate: dict[str, Any] = {"episodes": len(rows)}
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, bool))
        }
    )
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row]
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values))
    return aggregate
