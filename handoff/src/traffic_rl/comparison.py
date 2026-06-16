"""Comparison report generation for evaluation aggregates."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import ensure_dir, project_root, resolve_path
from .utils import write_csv


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    higher_is_better: bool
    unit: str = ""
    decimals: int = 2

    @property
    def mean_key(self) -> str:
        return f"{self.key}_mean"

    @property
    def std_key(self) -> str:
        return f"{self.key}_std"

    @property
    def direction_label(self) -> str:
        return "higher is better" if self.higher_is_better else "lower is better"


DEFAULT_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("mean_average_speed", "Mean speed", True, "m/s", 2),
    MetricSpec("mean_average_waiting_time", "Mean waiting time", False, "s", 2),
    MetricSpec("mean_average_pedestrian_waiting_time", "Pedestrian waiting time", False, "s", 2),
    MetricSpec("mean_travel_time", "Mean travel time", False, "s", 2),
    MetricSpec("throughput", "Throughput", True, "veh", 1),
    MetricSpec("mean_queue_length", "Mean queue length", False, "veh", 2),
    MetricSpec("mean_time_loss", "Mean time loss", False, "s", 2),
    MetricSpec("mean_backlog", "Mean backlog", False, "veh", 2),
    MetricSpec("congestion_failure", "Congestion-failure rate", False, "", 2),
    MetricSpec("teleports", "Teleports", False, "veh", 1),  # tienila o toglila, è scelta a parte
)

ALGORITHM_ORDER = {
    "baseline": 0,
    "centralized_ppo": 1,
    "independent_ppo": 2,
    "shared_ppo": 3,
    "mappo": 4,
}

ALGORITHM_LABELS = {
    "baseline": "SUMO baseline",
    "centralized_ppo": "Centralized PPO",
    "independent_ppo": "Independent PPO",
    "shared_ppo": "Shared PPO",
    "mappo": "MAPPO",
}

ALGORITHM_COLORS = {
    "baseline": "#6b7280",
    "centralized_ppo": "#0f766e",
    "independent_ppo": "#2563eb",
    "shared_ppo": "#c2410c",
    "mappo": "#b45309",
}

INTENSITY_ORDER = {"low": 0, "medium": 1, "high": 2}
VARIANT_ORDER = {"vehicle": 0, "peds": 1}
VARIANT_LABELS = {"vehicle": "Vehicle Only", "peds": "Pedestrians"}


def normalize_variant(value: Any, source_file: str = "") -> str:
    variant = str(value or "").strip().lower()
    if variant in VARIANT_ORDER:
        return variant
    normalized_source = source_file.replace("\\", "/")
    if "/peds/" in normalized_source:
        return "peds"
    return "vehicle"


def load_aggregate_rows(results_root: str | Path = "results/eval") -> list[dict[str, Any]]:
    root = resolve_path(results_root)
    repo_root = project_root()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("aggregate.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        try:
            payload["source_file"] = str(path.relative_to(repo_root))
        except ValueError:
            payload["source_file"] = str(path)
        payload["variant"] = normalize_variant(payload.get("variant"), payload["source_file"])
        rows.append(payload)
    return rows


def metric_specs_from_keys(metric_keys: Sequence[str] | None) -> list[MetricSpec]:
    if not metric_keys:
        return list(DEFAULT_METRICS)
    specs_by_key = {spec.key: spec for spec in DEFAULT_METRICS}
    missing = [key for key in metric_keys if key not in specs_by_key]
    if missing:
        raise ValueError(f"Unsupported metric keys: {', '.join(sorted(missing))}")
    return [specs_by_key[key] for key in metric_keys]


def filter_rows(
    rows: Iterable[dict[str, Any]],
    *,
    split: str | None = None,
    intensities: Sequence[str] | None = None,
    variants: Sequence[str] | None = None,
    algorithms: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    intensity_filter = set(intensities or [])
    variant_filter = set(variants or [])
    algorithm_filter = set(algorithms or [])
    filtered = [
        row
        for row in rows
        if (split is None or row.get("split") == split)
        and (not intensity_filter or row.get("intensity") in intensity_filter)
        and (not variant_filter or row.get("variant") in variant_filter)
        and (not algorithm_filter or row.get("algorithm") in algorithm_filter)
    ]
    return sorted(
        filtered,
        key=lambda row: (
            VARIANT_ORDER.get(str(row.get("variant")), 99),
            INTENSITY_ORDER.get(str(row.get("intensity")), 99),
            ALGORITHM_ORDER.get(str(row.get("algorithm")), 99),
            str(row.get("algorithm")),
        ),
    )


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def build_comparison_rows(
    rows: Sequence[dict[str, Any]],
    metric_specs: Sequence[MetricSpec],
    *,
    baseline_algo: str = "baseline",
) -> list[dict[str, Any]]:
    baseline_by_variant_and_intensity = {
        (str(row.get("variant")), str(row.get("intensity"))): row
        for row in rows
        if row.get("algorithm") == baseline_algo
    }
    comparison_rows: list[dict[str, Any]] = []

    for row in rows:
        variant = normalize_variant(row.get("variant"), str(row.get("source_file", "")))
        record: dict[str, Any] = {
            "algorithm": row.get("algorithm"),
            "algorithm_label": ALGORITHM_LABELS.get(str(row.get("algorithm")), str(row.get("algorithm"))),
            "split": row.get("split"),
            "variant": variant,
            "variant_label": VARIANT_LABELS.get(variant, variant),
            "intensity": row.get("intensity"),
            "episodes": row.get("episodes"),
            "aggregation_level": row.get("aggregation_level", ""),
            "seed_count": row.get("seed_count", ""),
            "train_seeds": row.get("train_seeds", ""),
            "source_file": row.get("source_file", ""),
        }
        baseline = baseline_by_variant_and_intensity.get((variant, str(row.get("intensity"))))

        for spec in metric_specs:
            value = _safe_float(row.get(spec.mean_key))
            std = _safe_float(row.get(spec.std_key))
            record[spec.mean_key] = value
            record[spec.std_key] = std

            improvement_abs = None
            improvement_pct = None
            baseline_value = _safe_float(baseline.get(spec.mean_key)) if baseline else None
            if value is not None and baseline_value is not None:
                if spec.higher_is_better:
                    improvement_abs = value - baseline_value
                else:
                    improvement_abs = baseline_value - value
                if abs(baseline_value) < 1e-12:
                    if abs(improvement_abs) < 1e-12:
                        improvement_pct = 0.0
                else:
                    improvement_pct = (improvement_abs / abs(baseline_value)) * 100.0

            record[f"{spec.key}_improvement_abs"] = improvement_abs
            record[f"{spec.key}_improvement_pct"] = improvement_pct
        comparison_rows.append(record)

    return comparison_rows


def _format_number(value: float | None, decimals: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def _format_mean_std(row: dict[str, Any], spec: MetricSpec) -> str:
    mean = _safe_float(row.get(spec.mean_key))
    std = _safe_float(row.get(spec.std_key))
    mean_text = _format_number(mean, spec.decimals)
    std_text = _format_number(std, spec.decimals)
    unit = f" {spec.unit}" if spec.unit else ""
    return f"{mean_text} +/- {std_text}{unit}"


def _format_improvement_pct(row: dict[str, Any], spec: MetricSpec) -> str:
    improvement_pct = _safe_float(row.get(f"{spec.key}_improvement_pct"))
    if improvement_pct is None:
        return "n/a"
    return f"{improvement_pct:+.{1}f}%"


def _ordered_intensities(rows: Sequence[dict[str, Any]]) -> list[str]:
    intensities = {
        str(row.get("intensity"))
        for row in rows
        if row.get("intensity") not in (None, "")
    }
    return sorted(intensities, key=lambda intensity: INTENSITY_ORDER.get(intensity, 99))


def _ordered_variants(rows: Sequence[dict[str, Any]]) -> list[str]:
    variants = {
        normalize_variant(row.get("variant"), str(row.get("source_file", "")))
        for row in rows
        if row.get("variant") not in (None, "")
    } or {"vehicle"}
    return sorted(variants, key=lambda variant: VARIANT_ORDER.get(variant, 99))


def _ordered_algorithms(rows: Sequence[dict[str, Any]]) -> list[str]:
    algorithms = {
        str(row.get("algorithm"))
        for row in rows
        if row.get("algorithm") not in (None, "")
    }
    return sorted(algorithms, key=lambda algorithm: (ALGORITHM_ORDER.get(algorithm, 99), algorithm))


def _algorithm_color(algorithm: str) -> str:
    return ALGORITHM_COLORS.get(algorithm, "#1d4ed8")


def _available_metric_specs(rows: Sequence[dict[str, Any]], metric_specs: Sequence[MetricSpec]) -> list[MetricSpec]:
    return [
        spec
        for spec in metric_specs
        if any(row.get(spec.mean_key) not in (None, "") for row in rows)
    ]


def render_markdown_report(
    rows: Sequence[dict[str, Any]],
    metric_specs: Sequence[MetricSpec],
    *,
    baseline_algo: str,
) -> str:
    if not rows:
        return "# Result Comparison\n\nNo matching evaluation aggregates were found.\n"

    lines = ["# Result Comparison", ""]
    for variant in _ordered_variants(rows):
        lines.append(f"## {VARIANT_LABELS.get(variant, variant)}")
        lines.append("")
        for intensity in _ordered_intensities([row for row in rows if row["variant"] == variant]):
            group = [row for row in rows if row["variant"] == variant and row["intensity"] == intensity]
            group_metric_specs = _available_metric_specs(group, metric_specs)
            lines.append(f"### {intensity.title()}")
            lines.append("")
            headers = ["Algorithm"] + [
                f"{spec.label} ({spec.unit})" if spec.unit else spec.label for spec in group_metric_specs
            ]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in group:
                cells = [str(row["algorithm_label"])]
                for spec in group_metric_specs:
                    mean = _safe_float(row.get(spec.mean_key))
                    if mean is None:
                        cells.append("n/a")
                        continue
                    improvement = _format_improvement_pct(row, spec)
                    if row["algorithm"] == baseline_algo:
                        cells.append(f"{mean:.{spec.decimals}f} (base)")
                    else:
                        cells.append(f"{mean:.{spec.decimals}f} ({improvement})")
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    lines.append("The CSV output contains the full mean/std values and baseline-relative improvements.")
    lines.append("")
    return "\n".join(lines)


def _bar_color(row: dict[str, Any], spec: MetricSpec) -> str:
    algorithm = str(row.get("algorithm"))
    if algorithm == "baseline":
        return "#6b7280"
    improvement = _safe_float(row.get(f"{spec.key}_improvement_pct"))
    if improvement is None:
        return "#2563eb"
    return "#059669" if improvement >= 0.0 else "#dc2626"


def _render_metric_card(rows: Sequence[dict[str, Any]], spec: MetricSpec) -> str:
    values = [_safe_float(row.get(spec.mean_key)) for row in rows]
    finite_values = [value for value in values if value is not None]
    max_value = max(finite_values) if finite_values else 0.0
    max_value = max(max_value, 1e-9)

    parts = [
        '<section class="metric-card">',
        f"<h3>{escape(spec.label)}</h3>",
        f'<p class="metric-subtitle">{escape(spec.direction_label)}</p>',
    ]
    for row in rows:
        value = _safe_float(row.get(spec.mean_key))
        width_pct = 0.0 if value is None else (value / max_value) * 100.0
        parts.append('<div class="bar-row">')
        parts.append(f'<div class="bar-label">{escape(str(row["algorithm_label"]))}</div>')
        parts.append('<div class="bar-track">')
        parts.append(
            f'<div class="bar-fill" style="width: {width_pct:.1f}%; background: {_bar_color(row, spec)};"></div>'
        )
        parts.append("</div>")
        value_text = _format_mean_std(row, spec)
        improvement_text = "base" if row["algorithm"] == "baseline" else _format_improvement_pct(row, spec)
        parts.append(f'<div class="bar-value">{escape(value_text)} | {escape(improvement_text)}</div>')
        parts.append("</div>")
    parts.append("</section>")
    return "\n".join(parts)


def _render_summary_heatmap(
    rows: Sequence[dict[str, Any]],
    metric_specs: Sequence[MetricSpec],
    *,
    baseline_algo: str,
) -> str:
    summary_rows: list[dict[str, Any]] = []
    for algorithm in _ordered_algorithms(rows):
        algorithm_rows = [row for row in rows if row.get("algorithm") == algorithm]
        summary: dict[str, Any] = {
            "algorithm": algorithm,
            "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm),
        }
        for spec in metric_specs:
            improvements = [
                _safe_float(row.get(f"{spec.key}_improvement_pct"))
                for row in algorithm_rows
                if _safe_float(row.get(f"{spec.key}_improvement_pct")) is not None
            ]
            summary[spec.key] = sum(improvements) / len(improvements) if improvements else None
        summary_rows.append(summary)

    max_abs = max(
        (
            abs(value)
            for summary in summary_rows
            if summary["algorithm"] != baseline_algo
            for spec in metric_specs
            for value in [_safe_float(summary.get(spec.key))]
            if value is not None
        ),
        default=0.0,
    )
    max_abs = max(max_abs, 1e-9)

    def cell_style(algorithm: str, value: float | None) -> str:
        if algorithm == baseline_algo:
            return "background: #ece8df; color: #51483c;"
        if value is None:
            return "background: #f3efe7; color: #9ca3af;"
        alpha = 0.16 + (min(abs(value) / max_abs, 1.0) * 0.48)
        if value >= 0.0:
            return f"background: rgba(5, 150, 105, {alpha:.3f}); color: {'#ffffff' if alpha >= 0.45 else '#065f46'};"
        return f"background: rgba(220, 38, 38, {alpha:.3f}); color: {'#ffffff' if alpha >= 0.45 else '#7f1d1d'};"

    header_cells = "".join(
        f"<th>{escape(spec.label)}<br><span>avg improvement</span></th>"
        for spec in metric_specs
    )
    body_rows: list[str] = []
    for summary in summary_rows:
        cells: list[str] = []
        for spec in metric_specs:
            value = _safe_float(summary.get(spec.key))
            if summary["algorithm"] == baseline_algo:
                text = "base"
            elif value is None:
                text = "n/a"
            else:
                text = f"{value:+.1f}%"
            cells.append(
                "<td>"
                f'<div class="heatmap-cell" style="{cell_style(str(summary["algorithm"]), value)}">{escape(text)}</div>'
                "</td>"
            )
        body_rows.append(
            "<tr>"
            f'<th scope="row">{escape(str(summary["algorithm_label"]))}</th>'
            + "".join(cells)
            + "</tr>"
        )

    return f"""
<section class="heatmap-panel">
  <div class="subsection-header">
    <h3>Average Improvement vs Baseline</h3>
    <p>Positive percentages mean better performance after accounting for metric direction.</p>
  </div>
  <div class="heatmap-wrap">
    <table class="heatmap-table">
      <thead>
        <tr>
          <th>Algorithm</th>
          {header_cells}
        </tr>
      </thead>
      <tbody>
        {"".join(body_rows)}
      </tbody>
    </table>
  </div>
</section>
"""


def _render_trend_chart(
    rows: Sequence[dict[str, Any]],
    spec: MetricSpec,
    intensities: Sequence[str],
) -> str:
    algorithms = _ordered_algorithms(rows)
    lookup = {
        (str(row.get("algorithm")), str(row.get("intensity"))): _safe_float(row.get(spec.mean_key))
        for row in rows
    }
    finite_values = [value for value in lookup.values() if value is not None]
    if not finite_values:
        return ""

    y_min = min(finite_values)
    y_max = max(finite_values)
    if abs(y_max - y_min) < 1e-12:
        pad = max(abs(y_max) * 0.1, 1.0)
    else:
        pad = (y_max - y_min) * 0.15
    y_min -= pad
    y_max += pad

    width = 420
    height = 220
    left = 52
    right = 16
    top = 16
    bottom = 36
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_pos(index: int) -> float:
        if len(intensities) == 1:
            return left + (plot_width / 2.0)
        return left + (index * plot_width / (len(intensities) - 1))

    def y_pos(value: float) -> float:
        return top + ((y_max - value) / (y_max - y_min)) * plot_height

    grid_lines: list[str] = []
    tick_count = 5
    for index in range(tick_count):
        y = top + (index * plot_height / (tick_count - 1))
        tick_value = y_max - (index * (y_max - y_min) / (tick_count - 1))
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#ddd4c6" stroke-width="1"></line>'
        )
        grid_lines.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#6b7280">{escape(_format_number(tick_value, spec.decimals))}</text>'
        )

    x_ticks = [
        f'<text x="{x_pos(index):.1f}" y="{height - 10}" text-anchor="middle" font-size="12" fill="#6b7280">{escape(intensity.title())}</text>'
        for index, intensity in enumerate(intensities)
    ]

    series_markup: list[str] = []
    for algorithm in algorithms:
        points: list[tuple[float, float]] = []
        circles: list[str] = []
        for index, intensity in enumerate(intensities):
            value = lookup.get((algorithm, intensity))
            if value is None:
                continue
            point = (x_pos(index), y_pos(value))
            points.append(point)
            circles.append(
                f'<circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="4.5" fill="{_algorithm_color(algorithm)}" stroke="#fffdf8" stroke-width="2"></circle>'
            )
        if len(points) >= 2:
            series_markup.append(
                f'<polyline fill="none" stroke="{_algorithm_color(algorithm)}" stroke-width="3" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}"></polyline>'
            )
        series_markup.extend(circles)

    legend = "".join(
        f'<span class="legend-item"><span class="legend-swatch" style="background: {_algorithm_color(algorithm)};"></span>{escape(ALGORITHM_LABELS.get(algorithm, algorithm))}</span>'
        for algorithm in algorithms
    )
    unit_text = spec.unit or "value"

    return f"""
<section class="trend-card">
  <div class="subsection-header">
    <h3>{escape(spec.label)}</h3>
    <p>{escape(spec.direction_label)} | unit: {escape(unit_text)}</p>
  </div>
  <svg class="trend-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(spec.label)} across intensities">
    {"".join(grid_lines)}
    <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#9f9482" stroke-width="1.2"></line>
    <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#9f9482" stroke-width="1.2"></line>
    {"".join(series_markup)}
    {"".join(x_ticks)}
  </svg>
  <div class="trend-legend">
    {legend}
  </div>
</section>
"""


def _render_overview_section(
    rows: Sequence[dict[str, Any]],
    metric_specs: Sequence[MetricSpec],
    *,
    variant: str,
    split: str | None,
    intensities: Sequence[str],
    baseline_algo: str,
) -> str:
    intensity_text = ", ".join(intensity.title() for intensity in intensities)
    active_metric_specs = _available_metric_specs(rows, metric_specs)
    trend_cards = "\n".join(
        _render_trend_chart(rows, spec, intensities)
        for spec in active_metric_specs
    )
    return f"""
<section class="overview-section">
  <div class="section-header">
    <h2>Overall Overview</h2>
    <p>Variant: {escape(VARIANT_LABELS.get(variant, variant))} | Split: {escape(split or 'all')} | Intensities: {escape(intensity_text)}</p>
  </div>
  <p class="section-note">
    The heatmap summarizes average baseline-relative improvement across the selected intensities.
    The trend charts below show how each algorithm's absolute metric values move from low to high demand.
  </p>
  {_render_summary_heatmap(rows, active_metric_specs, baseline_algo=baseline_algo)}
  <div class="trend-grid">
    {trend_cards}
  </div>
</section>
"""


def render_html_report(
    rows: Sequence[dict[str, Any]],
    metric_specs: Sequence[MetricSpec],
    *,
    split: str | None,
    baseline_algo: str,
) -> str:
    title = "Traffic RL Result Comparison"
    if not rows:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
</head>
<body>
  <h1>{title}</h1>
  <p>No matching evaluation aggregates were found.</p>
</body>
</html>
"""

    intensities = _ordered_intensities(rows)
    sections: list[str] = []
    overview_sections: list[str] = []
    for variant in _ordered_variants(rows):
        variant_rows = [row for row in rows if row["variant"] == variant]
        variant_intensities = _ordered_intensities(variant_rows)
        if len(variant_intensities) > 1:
            overview_sections.append(
                _render_overview_section(
                    variant_rows,
                    metric_specs,
                    variant=variant,
                    split=split,
                    intensities=variant_intensities,
                    baseline_algo=baseline_algo,
                )
            )

        for intensity in variant_intensities:
            group = [row for row in variant_rows if row["intensity"] == intensity]
            group_metric_specs = _available_metric_specs(group, metric_specs)
            table_headers = "".join(
                f"<th>{escape(spec.label)}<br><span>{escape(spec.unit or '-')}</span></th>" for spec in group_metric_specs
            )
            table_rows: list[str] = []
            for row in group:
                metric_cells = []
                for spec in group_metric_specs:
                    metric_cells.append(
                        "<td>"
                        f'<div class="cell-primary">{escape(_format_mean_std(row, spec))}</div>'
                        f'<div class="cell-secondary">{"base" if row["algorithm"] == baseline_algo else escape(_format_improvement_pct(row, spec))}</div>'
                        "</td>"
                    )
                table_rows.append(
                    "<tr>"
                    f'<td><strong>{escape(str(row["algorithm_label"]))}</strong><br><span class="cell-secondary">{escape(str(row.get("episodes", "n/a")))} episodes</span></td>'
                    + "".join(metric_cells)
                    + "</tr>"
                )

            cards = "\n".join(_render_metric_card(group, spec) for spec in group_metric_specs)
            sections.append(
                f"""
<section class="intensity-section">
  <div class="section-header">
    <h2>{escape(VARIANT_LABELS.get(variant, variant))} | {escape(intensity.title())}</h2>
    <p>Split: {escape(split or 'all')} | Baseline: {escape(ALGORITHM_LABELS.get(baseline_algo, baseline_algo))}</p>
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Algorithm</th>
          {table_headers}
        </tr>
      </thead>
      <tbody>
        {"".join(table_rows)}
      </tbody>
    </table>
  </div>
  <div class="metric-grid">
    {cards}
  </div>
</section>
"""
            )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f3ee;
      --panel: #fffdf8;
      --line: #d7d1c4;
      --ink: #1f2937;
      --muted: #6b7280;
      --accent: #0f766e;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      padding: 32px;
      background: linear-gradient(180deg, #f7f4ec 0%, #efe9dd 100%);
      color: var(--ink);
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    h1, h2, h3 {{
      margin: 0;
      font-weight: 700;
    }}
    p {{
      margin: 0;
    }}
    .page-header {{
      margin-bottom: 24px;
    }}
    .page-header h1 {{
      font-size: 32px;
      margin-bottom: 8px;
    }}
    .page-header p {{
      color: var(--muted);
      max-width: 900px;
    }}
    .overview-section,
    .intensity-section {{
      margin-bottom: 28px;
      padding: 24px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(31, 41, 55, 0.08);
    }}
    .section-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 16px;
    }}
    .subsection-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 12px;
    }}
    .section-header p {{
      color: var(--muted);
      text-align: right;
    }}
    .subsection-header p,
    .section-note {{
      color: var(--muted);
    }}
    .section-note {{
      margin-bottom: 18px;
      max-width: 920px;
    }}
    .table-wrap {{
      overflow-x: auto;
      margin-bottom: 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 920px;
    }}
    th, td {{
      padding: 12px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    th span {{
      text-transform: none;
      letter-spacing: normal;
      font-size: 12px;
    }}
    .cell-primary {{
      font-weight: 600;
    }}
    .cell-secondary {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 2px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .heatmap-panel {{
      margin-bottom: 22px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fffefb;
    }}
    .heatmap-table {{
      width: 100%;
      min-width: 760px;
      border-collapse: separate;
      border-spacing: 8px;
    }}
    .heatmap-table th,
    .heatmap-table td {{
      border: 0;
      padding: 0;
      vertical-align: middle;
    }}
    .heatmap-table thead th:first-child,
    .heatmap-table tbody th {{
      padding-right: 8px;
      text-align: left;
      color: var(--ink);
      font-size: 14px;
    }}
    .heatmap-table thead th {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .heatmap-table thead th span {{
      text-transform: none;
      letter-spacing: normal;
      font-size: 11px;
    }}
    .heatmap-cell {{
      padding: 14px 10px;
      border-radius: 12px;
      text-align: center;
      font-weight: 700;
      font-size: 14px;
    }}
    .metric-card {{
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fffefb;
    }}
    .metric-card h3 {{
      font-size: 18px;
      margin-bottom: 4px;
    }}
    .metric-subtitle {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 14px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 110px 1fr 150px;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }}
    .bar-label {{
      font-size: 13px;
      font-weight: 600;
    }}
    .bar-track {{
      height: 14px;
      background: #ebe4d7;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
    }}
    .bar-value {{
      font-size: 12px;
      color: var(--muted);
      text-align: right;
    }}
    .trend-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .trend-card {{
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fffefb;
    }}
    .trend-svg {{
      width: 100%;
      height: auto;
      display: block;
      margin-bottom: 10px;
    }}
    .trend-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
    }}
    .legend-swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
      flex: 0 0 auto;
    }}
    @media (max-width: 900px) {{
      body {{
        padding: 20px;
      }}
      .section-header {{
        display: block;
      }}
      .subsection-header {{
        display: block;
      }}
      .section-header p {{
        margin-top: 8px;
        text-align: left;
      }}
      .subsection-header p {{
        margin-top: 6px;
      }}
      .bar-row {{
        grid-template-columns: 1fr;
      }}
      .bar-value {{
        text-align: left;
      }}
    }}
  </style>
</head>
<body>
  <header class="page-header">
    <h1>{escape(title)}</h1>
    <p>
      Comparison built directly from evaluation aggregates under <code>results/eval</code>.
      Positive percentages indicate improvement relative to the baseline for the metric direction.
      Vehicle-only and pedestrian-enabled evaluations are shown in separate sections.
      Higher-is-better and lower-is-better metrics are handled automatically.
    </p>
  </header>
  {''.join(overview_sections)}
  {''.join(sections)}
</body>
</html>
"""


def write_comparison_outputs(
    *,
    results_root: str | Path = "results/eval",
    output_dir: str | Path = "results/compare",
    split: str | None = "test",
    intensities: Sequence[str] | None = None,
    variants: Sequence[str] | None = None,
    algorithms: Sequence[str] | None = None,
    metric_keys: Sequence[str] | None = None,
    baseline_algo: str = "baseline",
) -> dict[str, Path]:
    metric_specs = metric_specs_from_keys(metric_keys)
    rows = filter_rows(
        load_aggregate_rows(results_root),
        split=split,
        intensities=intensities,
        variants=variants,
        algorithms=algorithms,
    )
    if not rows:
        raise ValueError("No matching evaluation aggregates were found under results/eval.")

    comparison_rows = build_comparison_rows(rows, metric_specs, baseline_algo=baseline_algo)
    output_root = ensure_dir(output_dir)

    intensity_slug = "all" if not intensities else "-".join(intensities)
    variant_slug = "" if not variants else "_" + "-".join(variants)
    split_slug = split or "all"
    stem = f"comparison_{split_slug}_{intensity_slug}{variant_slug}"

    csv_path = output_root / f"{stem}.csv"
    md_path = output_root / f"{stem}.md"
    html_path = output_root / f"{stem}.html"

    write_csv(csv_path, comparison_rows)
    md_path.write_text(
        render_markdown_report(comparison_rows, metric_specs, baseline_algo=baseline_algo),
        encoding="utf-8",
    )
    html_path.write_text(
        render_html_report(comparison_rows, metric_specs, split=split, baseline_algo=baseline_algo),
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": md_path, "html": html_path}
