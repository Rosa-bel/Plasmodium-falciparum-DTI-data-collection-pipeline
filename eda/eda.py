#!/usr/bin/env python3
"""
eda_pf_bioassays.py

Exploratory data analysis for the main scraper output CSV with columns:
    assay_id, protein_target, gene_symbol, gi, gene_id, chembl_target_id,
    chembl_organism, active_compounds, inactive_compounds

Key design point: the SAME protein is often tested across MULTIPLE assays.
So all "per protein" statistics here are computed by first GROUPING rows by
protein (using `target_uniprot`-like identity -- here we use `protein_target`
name combined with `chembl_target_id` as the grouping key, see NOTE below),
UNIONING the SIDs from every assay that tested that protein, and de-duplicating
SIDs before counting -- otherwise a compound re-tested in two assays against
the same protein would be double-counted.

NOTE on grouping key: `protein_target` is a free-text name and could in theory
collide across truly different proteins, or the same protein could appear
with slightly different ChEMBL target IDs (e.g. due to a homologous-target
relationship). This script groups by `chembl_target_id` when present (most
reliable single-protein key tied to a verified organism), and falls back to
`protein_target` text when chembl_target_id is null, then reports both so you
can sanity-check pages where the two disagree.

Outputs:
    - Printed summary to stdout
    - <out_prefix>_protein_summary.csv      : per-protein stats
    - <out_prefix>_global_summary.txt        : global numbers as text
    - <out_prefix>_no_compounds_assays.csv   : assays with NO active or
                                                inactive compounds at all
    - A few PNG charts (distribution histograms, top proteins bar chart)
      saved alongside the CSVs, IF matplotlib is available.

Usage:
    python eda_pf_bioassays.py --input pf_bioassays.csv --out-prefix eda_out/pf
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "..", "data", "pf_bioassays.csv")
DEFAULT_OUT_PREFIX = os.path.join(SCRIPT_DIR, "eda_results", "eda_pf")


def split_sids(cell):
    """Parse a ';'-joined SID string into a clean list of non-empty SID strings."""
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [s.strip() for s in str(cell).split(";") if s.strip()]


def load_data(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=True)

    # normalize expected columns; tolerate missing ones gracefully
    expected_cols = [
        "assay_id", "protein_target", "gene_symbol", "gi", "gene_id",
        "chembl_target_id", "chembl_organism", "active_compounds", "inactive_compounds"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    # treat empty strings as NaN uniformly (e.g. "" from CSV)
    for col in expected_cols:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    df["active_sids"] = df["active_compounds"].apply(split_sids)
    df["inactive_sids"] = df["inactive_compounds"].apply(split_sids)
    df["n_active"] = df["active_sids"].apply(len)
    df["n_inactive"] = df["inactive_sids"].apply(len)
    df["n_total_compounds"] = df["n_active"] + df["n_inactive"]

    # grouping key for "per protein" stats: prefer chembl_target_id, else
    # fall back to the protein_target free-text name.
    df["protein_group_key"] = df["chembl_target_id"].fillna(
        "NAME::" + df["protein_target"].fillna("UNKNOWN_PROTEIN")
    )

    return df


def global_stats(df):
    lines = []
    n_assays = len(df)
    lines.append(f"Total assays (rows) in file:                 {n_assays}")

    # null / missing value report
    lines.append("")
    lines.append("Null / missing value counts per column:")
    cols_to_check = [
        "protein_target", "gene_symbol", "gi", "gene_id",
        "chembl_target_id", "chembl_organism", "active_compounds", "inactive_compounds"
    ]
    for col in cols_to_check:
        n_null = df[col].isna().sum()
        pct_null = 100.0 * n_null / n_assays if n_assays else 0.0
        lines.append(f"  {col:<20s}: {n_null:>6d} null  ({pct_null:5.1f}%)")

    # chembl_target_id is explicitly called out since it's the organism-verification key
    n_null_chembl = df["chembl_target_id"].isna().sum()
    pct_null_chembl = 100.0 * n_null_chembl / n_assays if n_assays else 0.0
    lines.append("")
    lines.append(f"Assays missing chembl_target_id: {n_null_chembl} / {n_assays} "
                 f"({pct_null_chembl:.1f}%) -- these could not be organism-verified "
                 f"upstream and should already have been filtered out by the scraper; "
                 f"any non-zero count here is worth double-checking.")

    # unique proteins
    n_unique_proteins_by_chembl = df["chembl_target_id"].dropna().nunique()
    n_unique_proteins_by_name = df["protein_target"].dropna().nunique()
    n_unique_protein_groups = df["protein_group_key"].nunique()
    lines.append("")
    lines.append(f"Unique proteins by chembl_target_id:   {n_unique_proteins_by_chembl}")
    lines.append(f"Unique proteins by protein_target name: {n_unique_proteins_by_name}")
    lines.append(f"Unique protein GROUPS used for stats below "
                 f"(chembl_target_id, falling back to name): {n_unique_protein_groups}")

    # global active/inactive compound percentages (counting all SID mentions,
    # i.e. NOT deduped across assays -- see protein-level table for deduped numbers)
    total_active_mentions = df["n_active"].sum()
    total_inactive_mentions = df["n_inactive"].sum()
    total_mentions = total_active_mentions + total_inactive_mentions
    pct_active = 100.0 * total_active_mentions / total_mentions if total_mentions else 0.0
    pct_inactive = 100.0 * total_inactive_mentions / total_mentions if total_mentions else 0.0

    lines.append("")
    lines.append("Global active/inactive compound mentions (raw SID mentions across all "
                 "assay rows, NOT deduplicated across assays testing the same protein):")
    lines.append(f"  total active mentions:   {total_active_mentions}")
    lines.append(f"  total inactive mentions: {total_inactive_mentions}")
    lines.append(f"  total mentions:          {total_mentions}")
    lines.append(f"  % active:   {pct_active:.2f}%")
    lines.append(f"  % inactive: {pct_inactive:.2f}%")

    # assays with no compounds at all
    n_no_compounds = (df["n_total_compounds"] == 0).sum()
    pct_no_compounds = 100.0 * n_no_compounds / n_assays if n_assays else 0.0
    lines.append("")
    lines.append(f"Assays with NO active AND NO inactive compounds listed: "
                 f"{n_no_compounds} / {n_assays} ({pct_no_compounds:.1f}%)")

    return "\n".join(lines), n_no_compounds


def protein_level_stats(df):
    """
    Group by protein_group_key, UNION+DEDUPE SIDs across all assays testing
    that protein, and compute per-protein active/inactive counts.
    """
    records = []
    for key, sub in df.groupby("protein_group_key"):
        active_union = set()
        inactive_union = set()
        for sids in sub["active_sids"]:
            active_union.update(sids)
        for sids in sub["inactive_sids"]:
            inactive_union.update(sids)

        # a SID could in principle show up as active in one assay and
        # inactive in another for the same protein -- flag the overlap
        # rather than silently double counting it in both buckets.
        overlap = active_union & inactive_union

        protein_name = sub["protein_target"].dropna().iloc[0] if sub["protein_target"].notna().any() else "UNKNOWN"
        chembl_id = sub["chembl_target_id"].dropna().iloc[0] if sub["chembl_target_id"].notna().any() else None
        organism = sub["chembl_organism"].dropna().iloc[0] if sub["chembl_organism"].notna().any() else None
        gene_symbol = sub["gene_symbol"].dropna().iloc[0] if sub["gene_symbol"].notna().any() else None

        n_active = len(active_union)
        n_inactive = len(inactive_union)
        n_total = len(active_union | inactive_union)

        records.append({
            "protein_group_key": key,
            "protein_target": protein_name,
            "gene_symbol": gene_symbol,
            "chembl_target_id": chembl_id,
            "chembl_organism": organism,
            "n_assays": len(sub),
            "n_assays_with_no_compounds": (sub["n_total_compounds"] == 0).sum(),
            "n_unique_active_compounds": n_active,
            "n_unique_inactive_compounds": n_inactive,
            "n_unique_total_compounds": n_total,
            "n_overlap_active_and_inactive_sids": len(overlap),
            "pct_active_of_protein": (100.0 * n_active / n_total) if n_total else 0.0,
            "pct_inactive_of_protein": (100.0 * n_inactive / n_total) if n_total else 0.0,
        })

    protein_df = pd.DataFrame.from_records(records)
    protein_df = protein_df.sort_values("n_unique_total_compounds", ascending=False).reset_index(drop=True)
    return protein_df


def no_compounds_count(df):
    """Just the count/percentage -- no CSV export needed."""
    n_no_compounds = (df["n_total_compounds"] == 0).sum()
    n_assays = len(df)
    pct = 100.0 * n_no_compounds / n_assays if n_assays else 0.0
    return n_no_compounds, pct


def print_protein_distribution_summary(protein_df):
    lines = []
    lines.append("Per-protein UNIQUE compound count distribution "
                 "(deduplicated across all assays testing each protein):")
    for col, label in [
        ("n_unique_active_compounds", "active"),
        ("n_unique_inactive_compounds", "inactive"),
        ("n_unique_total_compounds", "total"),
    ]:
        desc = protein_df[col].describe()
        lines.append(f"\n  {label.upper()} compounds per protein:")
        lines.append(f"    count(proteins): {int(desc['count'])}")
        lines.append(f"    mean:   {desc['mean']:.2f}")
        lines.append(f"    std:    {desc['std']:.2f}")
        lines.append(f"    min:    {desc['min']:.0f}")
        lines.append(f"    25%:    {desc['25%']:.0f}")
        lines.append(f"    50%:    {desc['50%']:.0f}")
        lines.append(f"    75%:    {desc['75%']:.0f}")
        lines.append(f"    max:    {desc['max']:.0f}")

    n_proteins_zero_compounds = (protein_df["n_unique_total_compounds"] == 0).sum()
    lines.append(f"\n  Proteins with ZERO unique compounds across ALL their assays: "
                 f"{n_proteins_zero_compounds} / {len(protein_df)}")

    lines.append("\nNumber of assays per protein (how many assays tested each protein):")
    desc2 = protein_df["n_assays"].describe()
    lines.append(f"    mean:   {desc2['mean']:.2f}")
    lines.append(f"    max:    {desc2['max']:.0f}")
    lines.append(f"    proteins tested in >1 assay: "
                 f"{(protein_df['n_assays'] > 1).sum()} / {len(protein_df)}")

    return "\n".join(lines)


def _protein_label(row, max_len=45):
    """Prefer the protein name; fall back to the chembl_target_id / group key."""
    name = row.get("protein_target")
    if pd.isna(name) or str(name).strip() == "":
        name = row.get("chembl_target_id") or row.get("protein_group_key")
    name = str(name)
    return (name[:max_len] + "...") if len(name) > max_len else name


def make_plots(df, protein_df, out_prefix, n_no_compounds, pct_no_compounds):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not available, skipping chart generation.", file=sys.stderr)
        return

    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plot_df = protein_df.copy()
    plot_df["label"] = plot_df.apply(_protein_label, axis=1)
    n_proteins = len(plot_df)

    # ------------------------------------------------------------------
    # 1) Compounds tested per protein -- active vs inactive, one bar per
    #    protein, each bar labeled with its protein name / chembl ID.
    # ------------------------------------------------------------------
    by_compounds = plot_df.sort_values("n_unique_total_compounds", ascending=True)
    fig_height = max(6, 0.32 * n_proteins)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.barh(by_compounds["label"], by_compounds["n_unique_active_compounds"],
            color="seagreen", label="Active")
    ax.barh(by_compounds["label"], by_compounds["n_unique_inactive_compounds"],
            left=by_compounds["n_unique_active_compounds"], color="indianred", label="Inactive")
    ax.set_xlabel("Unique tested compounds (deduplicated across assays)")
    ax.set_title(f"Unique compounds tested per protein (n={n_proteins} proteins)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_compounds_per_protein.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2) Number of assays per protein, one bar per protein, labeled.
    # ------------------------------------------------------------------
    by_assays = plot_df.sort_values("n_assays", ascending=True)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.barh(by_assays["label"], by_assays["n_assays"], color="darkorange")
    ax.set_xlabel("Number of assays testing this protein")
    ax.set_title(f"Assays per protein (n={n_proteins} proteins)")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_assays_per_protein.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3) Global active vs inactive compound mentions, with key-note
    #    annotations (counts, percentages, and the no-compounds-assay note).
    # ------------------------------------------------------------------
    total_active = int(df["n_active"].sum())
    total_inactive = int(df["n_inactive"].sum())
    total = total_active + total_inactive
    pct_active = 100.0 * total_active / total if total else 0.0
    pct_inactive = 100.0 * total_inactive / total if total else 0.0

    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.bar(["Active", "Inactive"], [total_active, total_inactive],
                   color=["seagreen", "indianred"], edgecolor="black")
    for bar, count, pct in zip(bars, [total_active, total_inactive], [pct_active, pct_inactive]):
        ax.annotate(f"{count}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total compound mentions (raw, across all assay rows)")
    ax.set_title("Global active vs. inactive compound mentions")
    ax.set_ylim(0, max(total_active, total_inactive, 1) * 1.18)

    note = (f"Total mentions: {total}\n"
            f"Assays with NO compounds listed: {n_no_compounds} ({pct_no_compounds:.1f}%)\n"
            f"Note: counts are raw SID mentions across assay rows,\n"
            f"not deduplicated across assays sharing the same protein.")
    ax.text(0.5, -0.22, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="whitesmoke", edgecolor="gray"))

    fig.tight_layout()
    fig.savefig(f"{out_prefix}_global_active_inactive.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plots] saved 3 PNGs with prefix: {out_prefix}_*", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                         help=f"Path to pf_bioassays.csv (default: {DEFAULT_INPUT})")
    parser.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX,
                         help="Prefix (can include directory) for output files")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG chart generation")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = load_data(args.input)

    global_summary_text, n_no_compounds = global_stats(df)
    protein_df = protein_level_stats(df)
    dist_summary_text = print_protein_distribution_summary(protein_df)
    n_no_compounds, pct_no_compounds = no_compounds_count(df)

    full_summary = global_summary_text + "\n\n" + "=" * 70 + "\n\n" + dist_summary_text

    print(full_summary)

    # write outputs
    protein_csv_path = f"{args.out_prefix}_protein_summary.csv"
    protein_df.to_csv(protein_csv_path, index=False)

    summary_txt_path = f"{args.out_prefix}_global_summary.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(full_summary + "\n")

    print(f"\n\nWrote:\n  {protein_csv_path}\n  {summary_txt_path}")

    if not args.no_plots:
        make_plots(df, protein_df, args.out_prefix, n_no_compounds, pct_no_compounds)


if __name__ == "__main__":
    main()