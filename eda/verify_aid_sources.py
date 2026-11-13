#!/usr/bin/env python3
"""Verify PubChem AID source metadata and extract assay names.

This script reads a list of AIDs from a CSV/TSV or command line, queries the
PubChem assay summary endpoint, and saves a CSV with the extracted assay names
and source metadata. It also prints a source repartition summary and writes
per-source statistics to a second CSV.

Example:
    python eda/verify_aid_sources.py --input data/chembl_drugs_targets.csv \
        --output-dir eda/eda_results

The input file can be:
- a CSV/TSV file with a column named `aid` or `AID`
- a plain file with one AID per line
- a comma-separated list of AIDs passed via `--aids`
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PUBCHEM_AID_SUMMARY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/summary/JSON"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "assaysInfo"


def parse_aid_values_from_file(path: Path) -> list[int]:
    values: list[int] = []
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if "aid" in reader.fieldnames or "AID" in reader.fieldnames:
                field = "aid" if "aid" in reader.fieldnames else "AID"
                for row in reader:
                    raw = row.get(field, "")
                    if raw is None:
                        continue
                    raw = raw.strip()
                    if raw:
                        try:
                            values.append(int(raw))
                        except ValueError:
                            continue
                return values

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            for part in raw.replace(",", " ").split():
                try:
                    values.append(int(part))
                except ValueError:
                    continue
    return values


def fetch_aid_summary(aid: int, timeout: int = 15) -> dict | None:
    url = PUBCHEM_AID_SUMMARY_URL.format(aid=aid)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
            return data
    except HTTPError as exc:
        print(f"AID {aid}: HTTP error {exc.code} - {exc.reason}", file=sys.stderr)
    except URLError as exc:
        print(f"AID {aid}: URL error {exc.reason}", file=sys.stderr)
    except json.JSONDecodeError as exc:
        print(f"AID {aid}: failed to decode JSON: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"AID {aid}: unexpected error: {exc}", file=sys.stderr)
    return None


def extract_assay_summaries(data: dict) -> list[dict]:
    summaries = []
    if not data:
        return summaries

    assay_summaries = data.get("AssaySummaries")
    if not assay_summaries:
        return summaries

    assay_summary = assay_summaries.get("AssaySummary")
    if assay_summary is None:
        return summaries

    if isinstance(assay_summary, dict):
        assay_summary = [assay_summary]

    for entry in assay_summary:
        if not isinstance(entry, dict):
            continue
        summaries.append(
            {
                "aid": entry.get("AID"),
                "source_name": entry.get("SourceName"),
                "source_id": entry.get("SourceID"),
                "name": entry.get("Name"),
                "assay_type": entry.get("AssayType"),
                "description": entry.get("Description"),
            }
        )
    return summaries


def write_csv(path: Path, column_names: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=column_names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in column_names})


def analyze_source_distribution(rows: list[dict]) -> list[dict]:
    source_counter = Counter()
    source_ids_by_name: dict[str, set[str]] = {}

    for row in rows:
        source_name = row.get("source_name") or "UNKNOWN"
        source_counter[source_name] += 1
        source_ids_by_name.setdefault(source_name, set()).add(str(row.get("source_id", "")))

    result = []
    for source_name, count in source_counter.most_common():
        result.append(
            {
                "source_name": source_name,
                "aid_count": count,
                "distinct_source_ids": len({sid for sid in source_ids_by_name.get(source_name, set()) if sid}),
                "source_ids": ";".join(sorted(sid for sid in source_ids_by_name.get(source_name, set()) if sid)),
            }
        )
    return result


def write_text_summary(path: Path, rows: list[dict], source_stats: list[dict], total_aids: int, failed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("PubChem AID source verification summary")
    lines.append("=" * 60)
    lines.append(f"Total AIDs requested: {total_aids}")
    lines.append(f"Total summaries extracted: {len(rows)}")
    lines.append(f"Failed requests or missing data: {failed}")
    lines.append("")
    lines.append("Source repartition:")
    for stat in source_stats:
        lines.append(
            f"  {stat['source_name']}: aids={stat['aid_count']}, "
            f"distinct_source_ids={stat['distinct_source_ids']}"
        )
    lines.append("")
    assay_types = Counter(row.get("assay_type") or "UNKNOWN" for row in rows)
    if assay_types:
        lines.append("Assay type repartition:")
        for assay_type, count in assay_types.most_common():
            lines.append(f"  {assay_type}: {count}")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def make_plots(output_dir: Path, rows: list[dict], source_stats: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot generation.")
        return

    sources = [stat["source_name"] for stat in source_stats]
    counts = [stat["aid_count"] for stat in source_stats]
    if sources and counts:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(sources, counts, color="#4C78A8")
        ax.set_title("PubChem AID Source Repartition")
        ax.set_ylabel("Number of AIDs")
        ax.set_xlabel("SourceName")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plot_path = output_dir / "aid_source_repartition.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"Saved source repartition plot: {plot_path}")

    assay_types = Counter(row.get("assay_type") or "UNKNOWN" for row in rows)
    if assay_types:
        labels, counts = zip(*assay_types.most_common())
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(labels, counts, color="#54A24B")
        ax.set_title("PubChem AID Assay Type Repartition")
        ax.set_ylabel("Number of records")
        ax.set_xlabel("AssayType")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plot_path = output_dir / "aid_assay_type_repartition.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"Saved assay type repartition plot: {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify PubChem AID source metadata and extract assay names")
    parser.add_argument("--input", type=Path, help="Path to a file containing AIDs or an aid column")
    parser.add_argument("--aids", type=str, help="Comma-separated list of AIDs to verify")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-csv", type=Path, default="eda/assaysInfo/aid_assay_names.csv", help="CSV file path for extracted assay names")
    parser.add_argument("--source-stats-csv", type=Path, default="eda/assaysInfo/aid_source_stats.csv", help="CSV file path for source repartition stats")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to wait between PubChem requests")
    parser.add_argument("--timeout", type=int, default=15, help="Network timeout for each PubChem request")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input is None and args.aids is None:
        raise SystemExit("Provide either --input or --aids")

    if args.input is not None and not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    if args.aids:
        raw_aids = [raw.strip() for raw in args.aids.replace(",", " ").split() if raw.strip()]
        aids = []
        for raw in raw_aids:
            try:
                aids.append(int(raw))
            except ValueError:
                print(f"Skipping invalid AID value: {raw}", file=sys.stderr)
    else:
        aids = parse_aid_values_from_file(args.input)

    if not aids:
        raise SystemExit("No valid AIDs were found in the input")

    rows: list[dict] = []
    failed = 0
    seen_aids = set()
    for aid in aids:
        if aid in seen_aids:
            continue
        seen_aids.add(aid)
        data = fetch_aid_summary(aid, timeout=args.timeout)
        if not data:
            failed += 1
            continue

        summaries = extract_assay_summaries(data)
        if not summaries:
            print(f"AID {aid}: no assay summaries found", file=sys.stderr)
            failed += 1
            continue

        for summary in summaries:
            if summary["aid"] is None:
                summary["aid"] = aid
            rows.append(summary)
        time.sleep(args.delay)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv_path = args.output_csv or output_dir / "aid_assay_names.csv"
    source_stats_csv_path = args.source_stats_csv or output_dir / "aid_source_repartition.csv"
    summary_txt_path = output_dir / "aid_source_summary.txt"

    write_csv(
        output_csv_path,
        ["aid", "source_name", "source_id", "name", "assay_type", "description"],
        rows,
    )
    source_stats = analyze_source_distribution(rows)
    write_csv(
        source_stats_csv_path,
        ["source_name", "aid_count", "distinct_source_ids", "source_ids"],
        source_stats,
    )
    write_text_summary(summary_txt_path, rows, source_stats, total_aids=len(aids), failed=failed)
    make_plots(output_dir, rows, source_stats)

    print(f"Extracted assay names CSV: {output_csv_path}")
    print(f"Source repartition CSV: {source_stats_csv_path}")
    print(f"Text summary: {summary_txt_path}")


if __name__ == "__main__":
    main()
