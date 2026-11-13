#!/usr/bin/env python3
"""EDA for PubChem activity metrics (IC50, Inhibition, Ki) by activity outcome.

The script reads the PubChem CSV and builds separate summaries and plots for
Active vs Inactive rows. It focuses on three standard types:
- IC50
- Inhibition
- Ki

For Inhibition, only rows with Standard Relation in {"=", "~"} are used for
range-based plots because those are the most comparable numeric measurements.
For IC50 and Ki, values are kept in nM and the script reports the share of
rows with inequality relations (<, >) that were excluded from the current
threshold-oriented EDA.
"""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "pubchem_pfalciparum" / "pfalciparum_compounds_raw.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "eda_results" / "activity_metric_eda"


def parse_extra_fields(raw_value: Optional[str]) -> Dict[str, object]:
    if not raw_value:
        return {}
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        try:
            fixed = raw_value.replace('""', '"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            return {}


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_metric_value(extra: Dict[str, object], standard_type: str) -> Optional[float]:
    if standard_type in {"IC50", "Ki"}:
        standard_value = to_float(extra.get("Standard Value"))
        if standard_value is not None:
            return standard_value
        pubchem_value = to_float(extra.get("PubChem Standard Value"))
        if pubchem_value is not None:
            return pubchem_value * 1000.0
        return None

    if standard_type == "Inhibition":
        return to_float(extra.get("Standard Value"))

    return None


def classify_outcome(value: Optional[str]) -> Optional[str]:
    outcome = (value or "").strip()
    if outcome == "Active":
        return "Active"
    if outcome == "Inactive":
        return "Inactive"
    return None


def load_records(input_path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with input_path.open(newline="", encoding="utf-8") as fh:
        import csv
        reader = csv.DictReader(fh)
        for row in reader:
            outcome = classify_outcome(row.get("activity_outcome"))
            if outcome is None:
                continue

            extra = parse_extra_fields(row.get("extra_fields_json"))
            standard_type = (extra.get("Standard Type") or "").strip()
            if not standard_type:
                continue

            standard_type = standard_type.strip()
            if standard_type.lower() not in {"ic50", "inhibition", "ki"}:
                continue

            relation = (extra.get("Standard Relation") or "").strip()
            value = normalize_metric_value(extra, standard_type)
            if value is None:
                continue

            records.append({
                "outcome": outcome,
                "metric": standard_type,
                "value": value,
                "relation": relation,
                "units": (extra.get("Standard Units") or "").strip(),
            })
    return records


def describe_values(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build_summary(records: List[Dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_lines: List[str] = []
    summary_lines.append("Activity metric EDA summary")
    summary_lines.append("==========================")
    summary_lines.append(f"Rows analyzed: {len(records)}")
    summary_lines.append("")

    for metric in ["IC50", "Inhibition", "Ki"]:
        subset = [r for r in records if r["metric"] == metric]
        if not subset:
            summary_lines.append(f"{metric}: no rows found")
            summary_lines.append("")
            continue

        active_values = [r["value"] for r in subset if r["outcome"] == "Active"]
        inactive_values = [r["value"] for r in subset if r["outcome"] == "Inactive"]

        summary_lines.append(f"{metric}")
        summary_lines.append(f"  Global tests: {len(subset)}")
        summary_lines.append(f"  Active rows: {len(active_values)}")
        summary_lines.append(f"  Inactive rows: {len(inactive_values)}")

        if metric in {"IC50", "Ki"}:
            inequality_relations = [r for r in subset if r["relation"] in {"<", ">"}]
            summary_lines.append(
                f"  Rows with '<' or '>' relation: {len(inequality_relations)} / {len(subset)} ({100.0 * len(inequality_relations) / len(subset):.1f}%)"
            )
            summary_lines.append("  These relation symbols were not used for the current range-based EDA.")

        if metric == "Inhibition":
            comparable = [r for r in subset if r["relation"] in {"=", "~"}]
            comparable_active = [r["value"] for r in comparable if r["outcome"] == "Active"]
            comparable_inactive = [r["value"] for r in comparable if r["outcome"] == "Inactive"]
            summary_lines.append(
                f"  Comparable numeric rows for range EDA (relation '=' or '~'): {len(comparable)} / {len(subset)}"
            )
            summary_lines.append(
                f"  Active comparable mean/std: {describe_values(comparable_active)['mean']:.3f} / {describe_values(comparable_active)['std']:.3f}"
            )
            summary_lines.append(
                f"  Inactive comparable mean/std: {describe_values(comparable_inactive)['mean']:.3f} / {describe_values(comparable_inactive)['std']:.3f}"
            )
        else:
            summary_lines.append(
                f"  Active mean/std (nM): {describe_values(active_values)['mean']:.3f} / {describe_values(active_values)['std']:.3f}"
            )
            summary_lines.append(
                f"  Inactive mean/std (nM): {describe_values(inactive_values)['mean']:.3f} / {describe_values(inactive_values)['std']:.3f}"
            )

        summary_lines.append("")

    summary_path = output_dir / "activity_metric_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Summary written to {summary_path}")


def make_box_plot(values: List[float], metric: str, outcome: str, output_dir: Path) -> None:
    if not values:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(values, patch_artist=True, boxprops=dict(facecolor="#4C78A8", alpha=0.8))
    ax.set_title(f"{metric} - {outcome} rows")
    ax.set_ylabel("Value (nM)" if metric in {"IC50", "Ki"} else "Percent inhibition")
    ax.set_xticks([1])
    ax.set_xticklabels([outcome])
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    output_path = output_dir / f"{metric.lower()}_{outcome.lower()}_boxplot.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot written to {output_path}")


def make_range_bar_chart(values: List[float], metric: str, outcome: str, output_dir: Path) -> None:
    if not values:
        return

    if metric in {"IC50", "Ki"}:
        positive = [v for v in values if v > 0]
        if not positive:
            return
        log_vals = np.log10(positive)
        if np.ptp(log_vals) < 1e-9:
            bins = np.linspace(log_vals[0] - 0.5, log_vals[0] + 0.5, 6)
        else:
            bins = np.linspace(log_vals.min(), log_vals.max(), 10)
        counts, edges = np.histogram(log_vals, bins=bins)
        labels = []
        for start, end in zip(edges[:-1], edges[1:]):
            labels.append(f"{10 ** start:.3g}–{10 ** end:.3g}")
    else:
        if max(values) == min(values):
            bins = np.linspace(min(values) - 1, max(values) + 1, 6)
        else:
            bins = np.linspace(min(values), max(values), 10)
        counts, edges = np.histogram(values, bins=bins)
        labels = [f"{start:.1f}–{end:.1f}" for start, end in zip(edges[:-1], edges[1:])]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(counts))
    bars = ax.bar(x, counts, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Count of rows")
    ax.set_xlabel("Value range")
    ax.set_title(f"{metric} - {outcome} rows by value range")
    ax.grid(alpha=0.25, axis="y")
    ax.bar_label(bars, labels=[str(c) for c in counts], padding=3)
    fig.tight_layout()
    output_path = output_dir / f"{metric.lower()}_{outcome.lower()}_range_counts.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot written to {output_path}")


def make_plots(records: List[Dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["IC50", "Inhibition", "Ki"]:
        metric_records = [r for r in records if r["metric"] == metric]
        if not metric_records:
            continue

        if metric == "Inhibition":
            plot_records = [r for r in metric_records if r["relation"] in {"=", "~"}]
            if not plot_records:
                continue
        else:
            plot_records = metric_records

        active_values = [r["value"] for r in plot_records if r["outcome"] == "Active"]
        inactive_values = [r["value"] for r in plot_records if r["outcome"] == "Inactive"]
        if not active_values or not inactive_values:
            continue

        make_box_plot(active_values, metric, "Active", output_dir)
        make_box_plot(inactive_values, metric, "Inactive", output_dir)
        make_range_bar_chart(active_values, metric, "Active", output_dir)
        make_range_bar_chart(inactive_values, metric, "Inactive", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for active/inactive activity metrics in the PubChem CSV")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    records = load_records(args.input)
    build_summary(records, args.output_dir)
    make_plots(records, args.output_dir)

    print(f"All outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
