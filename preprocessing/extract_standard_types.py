import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_extra_fields_json(raw_value):
    if not raw_value:
        return {}

    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        # Try to fix common quoting issues by replacing doubled quotes
        try:
            fixed = raw_value.replace("\"\"", "\"")
            return json.loads(fixed)
        except json.JSONDecodeError:
            return {}


def is_unspecified_activity(row):
    outcome = (row.get("activity_outcome") or "").strip()
    return outcome.casefold().startswith("unspecified")


def summarize_standard_fields(csv_path):
    standard_summary = defaultdict(lambda: {
        "relations": set(),
        "units": set(),
        "missing_relation": 0,
        "missing_units": 0,
        "count": 0,
    })

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not is_unspecified_activity(row):
                continue

            raw_json = row.get("extra_fields_json", "")
            extra = parse_extra_fields_json(raw_json)
            standard_type = extra.get("Standard Type")
            if not standard_type:
                continue

            standard_summary[standard_type]["count"] += 1

            relation = extra.get("Standard Relation")
            if relation is None or relation == "":
                standard_summary[standard_type]["missing_relation"] += 1
            else:
                standard_summary[standard_type]["relations"].add(relation)

            units = extra.get("Standard Units")
            if units is None or units == "":
                standard_summary[standard_type]["missing_units"] += 1
            else:
                standard_summary[standard_type]["units"].add(units)

    return standard_summary


def format_summary(summary):
    lines = []
    if not summary:
        lines.append("No Standard Type values found in the CSV file.")
        return "\n".join(lines)

    lines.append("Summary of Standard Types and observed Standard Relation / Standard Units values")
    lines.append("-")
    for standard_type, data in sorted(summary.items()):
        relations = sorted(data["relations"])
        units = sorted(data["units"])

        lines.append(f"Standard Type: {standard_type}")
        lines.append(f"  Occurrences: {data['count']}")
        lines.append(f"  Standard Relation values: {relations if relations else ['<none found>']}")
        lines.append(f"  Standard Units values: {units if units else ['<none found>']}")
        if data["missing_relation"]:
            lines.append(f"  Rows missing Standard Relation: {data['missing_relation']}")
        if data["missing_units"]:
            lines.append(f"  Rows missing Standard Units: {data['missing_units']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def print_summary(summary):
    print(format_summary(summary), end="")


def save_summary(summary, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_summary(summary), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Extract Standard Type, Standard Relation, and Standard Units values from a PubChem CSV extra_fields_json column."
    )
    parser.add_argument(
        "--output",
        default="standard_types_summary.txt",
        help="File path where the summary should be written.",
    )

    args = parser.parse_args()
    csvpath = "data/pubchem_pfalciparum/pfalciparum_compounds_raw.csv"

    summary = summarize_standard_fields(csvpath)
    print_summary(summary)
    output_path = save_summary(summary, args.output)
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
