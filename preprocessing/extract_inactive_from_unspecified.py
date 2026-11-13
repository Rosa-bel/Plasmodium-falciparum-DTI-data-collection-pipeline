#!/usr/bin/env python3
"""Extract inactive rows from unspecified activity outcomes using metric thresholds.

Rules:
- IC50: inactive if value >= 10000 nM
- Ki: inactive if value >= 10000 nM
- Inhibition: inactive if value <= 35%

The output is written to a CSV file named inactive_extract.csv in the same
output directory as the script, containing the rows classified as inactive.
"""

import argparse
import csv
import json
from ntpath import join
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_INPUT = "data/pubchem_pfalciparum/pfalciparum_compounds_raw.csv"
DEFAULT_OUTPUT = "preprocessing" 


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


def should_be_inactive(extra: Dict[str, object]) -> bool:
    standard_type = (extra.get("Standard Type") or "").strip()
    if standard_type == "IC50":
        value = normalize_metric_value(extra, standard_type)
        return value is not None and value >= 10000.0
    if standard_type == "Ki":
        value = normalize_metric_value(extra, standard_type)
        return value is not None and value >= 10000.0
    if standard_type == "Inhibition":
        value = normalize_metric_value(extra, standard_type)
        return value is not None and value <= 50.0
    return False

def should_be_active(extra: Dict[str, object]) -> bool:
    standard_type = (extra.get("Standard Type") or "").strip()
    if standard_type == "IC50":
        value = normalize_metric_value(extra, standard_type)
        return value is not None and value < 10000.0
    if standard_type == "Ki":
        value = normalize_metric_value(extra, standard_type)
        return value is not None and value < 10000.0
    if standard_type == "Inhibition":
        value = normalize_metric_value(extra, standard_type)
        return value is not None and value > 35.0
    return False


def extract_inactive_rows(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with input_path.open(newline="", encoding="utf-8") as fh_in:
        reader = csv.DictReader(fh_in)
        fieldnames = reader.fieldnames or []
        with output_path.open("w", newline="", encoding="utf-8") as fh_out:
            writer = csv.DictWriter(fh_out, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                outcome = (row.get("activity_outcome") or "").strip()
                if outcome != "Unspecified":
                    continue

                extra = parse_extra_fields(row.get("extra_fields_json"))
                if should_be_inactive(extra):
                    writer.writerow(row)
                    rows_written += 1

    print(f"Wrote {rows_written} inactive rows to {output_path}")
    return rows_written


def extract_rows(input_path: Path, output_path: Path , active=False) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with input_path.open(newline="", encoding="utf-8") as fh_in:
        reader = csv.DictReader(fh_in)
        fieldnames = reader.fieldnames or []
        with output_path.open("w", newline="", encoding="utf-8") as fh_out:
            writer = csv.DictWriter(fh_out, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                outcome = (row.get("activity_outcome") or "").strip()
                if outcome != "Unspecified":
                    continue

                extra = parse_extra_fields(row.get("extra_fields_json"))
                if active and should_be_active(extra):
                    writer.writerow(row)
                    rows_written += 1
                elif not active and should_be_inactive(extra):
                    writer.writerow(row)
                    rows_written += 1

    print(f"Wrote {rows_written} rows to {output_path}")
    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract inactive/active rows from unspecified activity outcomes")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--active", action="store_true", help="Extract active rows instead of inactive rows")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    output_path = Path(args.output) / ("active_extract.csv" if args.active else "inactive_extract.csv")
    extract_rows(args.input, output_path, active=args.active)



if __name__ == "__main__":
    main()
