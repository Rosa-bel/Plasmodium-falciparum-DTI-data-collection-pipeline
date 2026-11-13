#!/usr/bin/env python3
"""Exploratory analysis for PubChem protein-compound activity CSV.

This script summarizes:
- Repartition of activity outcomes: Active, Inactive, Unspecified, Inconclusive
- Missing values in key columns
- Number of compounds per protein
- Number of proteins per compound
- Per-protein active/inactive compound ratios
- Optional plots for the main summaries
"""

import argparse
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "final" /"combined_active_inactive_rows.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "final" / "eda"


def load_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, dtype=str, keep_default_na=True)
    expected_columns = [
        "protein_accession",
        "protein_name",
        "gene_id",
        "organism",
        "aid",
        "cid",
        "sid",
        "smiles",
        "activity_outcome",
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan

    for col in expected_columns:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    df["activity_outcome"] = df["activity_outcome"].fillna("").astype(str).str.strip()
    return df


def summarize_outcomes(df: pd.DataFrame) -> pd.Series:
    outcomes = df["activity_outcome"].fillna("")
    counts = outcomes.value_counts(dropna=False)
    return counts


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    total = len(df)
    summary = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": (missing / total * 100).round(2),
    })
    return summary.sort_values("missing_count", ascending=False)


def summarize_protein_compound_counts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # compounds per protein, based on unique compounds for each protein accession
    protein_level = (
        df.groupby("protein_accession", dropna=False)
        .agg(
            protein_name=("protein_name", lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan),
            n_compounds=("cid", lambda s: s.dropna().nunique()),
            n_active_compounds=("activity_outcome", lambda s: int(((s == "Active").sum()))),
            n_inactive_compounds=("activity_outcome", lambda s: int(((s == "Inactive").sum()))),
            n_unspecified_compounds=("activity_outcome", lambda s: int(((s == "Unspecified").sum()))),
            n_inconclusive_compounds=("activity_outcome", lambda s: int(((s == "Inconclusive").sum()))),
        )
        .reset_index()
    )

    protein_level["active_ratio"] = protein_level["n_active_compounds"] / np.where(
        protein_level["n_compounds"] > 0, protein_level["n_compounds"], np.nan
    )
    protein_level["inactive_ratio"] = protein_level["n_inactive_compounds"] / np.where(
        protein_level["n_compounds"] > 0, protein_level["n_compounds"], np.nan
    )
    protein_level["active_inactive_ratio"] = protein_level["n_active_compounds"] / np.where(
        protein_level["n_inactive_compounds"] > 0, protein_level["n_inactive_compounds"], np.nan
    )

    # proteins per compound, based on unique proteins for each compound (cid)
    compound_level = (
        df.groupby("cid", dropna=False)
        .agg(
            n_proteins=("protein_accession", lambda s: s.dropna().nunique()),
            n_active_proteins=("activity_outcome", lambda s: int(((s == "Active").sum()))),
            n_inactive_proteins=("activity_outcome", lambda s: int(((s == "Inactive").sum()))),
            n_unspecified_proteins=("activity_outcome", lambda s: int(((s == "Unspecified").sum()))),
            n_inconclusive_proteins=("activity_outcome", lambda s: int(((s == "Inconclusive").sum()))),
        )
        .reset_index()
    )

    return protein_level, compound_level


def write_text_summary(output_dir: Path, df: pd.DataFrame, outcome_counts: pd.Series, missing_summary: pd.DataFrame, protein_level: pd.DataFrame, compound_level: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("PubChem activity EDA summary")
    lines.append("=" * 80)
    lines.append(f"Rows: {len(df)}")
    lines.append("")
    lines.append("Activity outcome repartition:")
    for label, count in outcome_counts.items():
        lines.append(f"  {label}: {count}")
    lines.append("")
    lines.append("Missing values:")
    for _, row in missing_summary.iterrows():
        lines.append(f"  {row.name}: {int(row['missing_count'])} ({row['missing_percent']:.2f}%)")
    lines.append("")
    lines.append("Top proteins by number of compounds:")
    top_proteins = protein_level.sort_values("n_compounds", ascending=False).head(15)
    for _, row in top_proteins.iterrows():
        lines.append(
            f"  {row['protein_accession']}: compounds={int(row['n_compounds'])}, active={int(row['n_active_compounds'])}, inactive={int(row['n_inactive_compounds'])}, unspecified={int(row['n_unspecified_compounds'])}, inconclusive={int(row['n_inconclusive_compounds'])}"
        )
    lines.append("")
    lines.append("Top compounds by number of proteins:")
    top_compounds = compound_level.sort_values("n_proteins", ascending=False).head(15)
    for _, row in top_compounds.iterrows():
        lines.append(
            f"  {row['cid']}: proteins={int(row['n_proteins'])}, active={int(row['n_active_proteins'])}, inactive={int(row['n_inactive_proteins'])}, unspecified={int(row['n_unspecified_proteins'])}, inconclusive={int(row['n_inconclusive_proteins'])}"
        )
    lines.append("")
    lines.append("Per-protein active/inactive ratio overview:")
    for _, row in protein_level.sort_values("active_inactive_ratio", ascending=False).head(20).iterrows():
        ratio = row["active_inactive_ratio"]
        lines.append(
            f"  {row['protein_accession']}: active/inactive={ratio:.3f}, active_ratio={row['active_ratio']:.3f}, inactive_ratio={row['inactive_ratio']:.3f}"
        )

    summary_path = output_dir / "pubchem_activity_eda_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary written to {summary_path}")


def make_plots(df: pd.DataFrame, protein_level: pd.DataFrame, compound_level: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    outcome_counts = df["activity_outcome"].fillna("").value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    outcome_counts.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_title("Outcome distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Outcome")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)
    fig.tight_layout()
    fig.savefig(output_dir / "outcome_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    top_proteins = protein_level.sort_values("n_compounds", ascending=False).head(20)
    if not top_proteins.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(top_proteins["protein_accession"].astype(str), top_proteins["n_compounds"], color="#4C78A8")
        ax.set_title("Number of compounds per protein")
        ax.set_ylabel("Compounds")
        ax.set_xlabel("Protein accession")
        plt.xticks(rotation=90)
        fig.tight_layout()
        fig.savefig(output_dir / "compounds_per_protein.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    top_compounds = compound_level.sort_values("n_proteins", ascending=False).head(20)
    if not top_compounds.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(top_compounds["cid"].astype(str), top_compounds["n_proteins"], color="#54A24B")
        ax.set_title("Number of proteins per compound")
        ax.set_ylabel("Proteins")
        ax.set_xlabel("Compound CID")
        plt.xticks(rotation=90)
        fig.tight_layout()
        fig.savefig(output_dir / "proteins_per_compound.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    protein_level = protein_level.sort_values("n_compounds", ascending=False).head(20)
    x = np.arange(len(protein_level))
    ax.bar(x - 0.2, protein_level["n_active_compounds"], width=0.35, label="Active")
    ax.bar(x + 0.2, protein_level["n_inactive_compounds"], width=0.35, label="Inactive")
    ax.set_xticks(x)
    ax.set_xticklabels(protein_level["protein_accession"].astype(str), rotation=90)
    ax.set_title("Active vs inactive compounds by protein")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "active_inactive_by_protein.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for PubChem protein-compound activity CSV")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = load_data(args.input)
    outcome_counts = summarize_outcomes(df)
    missing_summary = summarize_missing_values(df)
    protein_level, compound_level = summarize_protein_compound_counts(df)

    write_text_summary(args.output_dir, df, outcome_counts, missing_summary, protein_level, compound_level)

    if not args.no_plots:
        make_plots(df, protein_level, compound_level, args.output_dir)

    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
