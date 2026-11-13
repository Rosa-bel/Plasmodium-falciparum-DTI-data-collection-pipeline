#!/usr/bin/env python3
"""Combine Active/Inactive rows from pfalciparum_compounds_raw.csv with inactive_extract.csv.

Rows with any null/empty values in the key columns are ignored.
"""

import argparse
import csv
from pathlib import Path


DEFAULT_RAW = Path(__file__).resolve().parents[1] / "data" / "pubchem_pfalciparum" / "pfalciparum_compounds_raw.csv"
DEFAULT_INACTIVE = Path(__file__).resolve().parents[1] / "preprocessing" / "inactive_extract.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "preprocessing" / "final_data.csv"

KEY_COLUMNS = ["protein_accession", "aid", "sid", "cid", "smiles", "activity_outcome", "extra_fields_json"]


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def is_complete(row: dict) -> bool:
    return all((row.get(col) or "").strip() not in {"", "nan", "None", "null"} for col in KEY_COLUMNS)


def merge_rows(raw_rows, inactive_rows):
    kept = []
    seen = set()

    for row in raw_rows:
        outcome = (row.get("activity_outcome") or "").strip()
        if outcome not in {"Active", "Inactive"}:
            continue
        if not is_complete(row):
            continue

        key = (
            row.get("protein_accession", ""),
            row.get("aid", ""),
            row.get("sid", ""),
            row.get("cid", ""),
            row.get("smiles", ""),
            row.get("activity_outcome", ""),
            row.get("extra_fields_json", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)

    for row in inactive_rows:
        outcome = (row.get("activity_outcome") or "").strip()
        if outcome not in {"Active", "Inactive"}:
            continue
        if not is_complete(row):
            continue

        key = (
            row.get("protein_accession", ""),
            row.get("aid", ""),
            row.get("sid", ""),
            row.get("cid", ""),
            row.get("smiles", ""),
            row.get("activity_outcome", ""),
            row.get("extra_fields_json", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)

    return kept


def write_rows(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=KEY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in KEY_COLUMNS})


def main():
    parser = argparse.ArgumentParser(description="Merge Active/Inactive rows from the raw PubChem CSV with inactive_extract.csv")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--inactive", type=Path, default=DEFAULT_INACTIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    raw_rows = load_rows(args.raw)
    inactive_rows = load_rows(args.inactive)
    combined = merge_rows(raw_rows, inactive_rows)
    write_rows(combined, args.output)

    print(f"Wrote {len(combined)} rows to {args.output}")


if __name__ == "__main__":
    main()
