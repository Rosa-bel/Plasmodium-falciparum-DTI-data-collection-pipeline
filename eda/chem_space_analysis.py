import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def build_descriptor_frame(smiles_series: pd.Series) -> pd.DataFrame:
    """Create a descriptor table from a SMILES series using RDKit descriptors."""
    rows = []
    for smiles in smiles_series:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        rows.append({
            "smiles": smiles,
            "mol_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "rot_bonds": Descriptors.NumRotatableBonds(mol),
            "heavy_atoms": Descriptors.HeavyAtomCount(mol),
            "ring_count": Descriptors.RingCount(mol),
            "fraction_csp3": Descriptors.FractionCSP3(mol),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "smiles", "mol_weight", "logp", "tpsa", "hbd", "hba",
            "rot_bonds", "heavy_atoms", "ring_count", "fraction_csp3"
        ])
    return pd.DataFrame(rows)


def prepare_dataset(input_path: Path, activity: str) -> pd.DataFrame:
    """Load the PubChem CSV, keep the requested activity, ignore unspecified rows."""
    df = pd.read_csv(input_path)
    df = df[df["activity_outcome"].eq(activity)].copy()
    if df.empty:
        return df

    df = df[["protein_accession", "aid", "cid", "sid", "smiles", "activity_outcome"]].copy()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[df["smiles"].ne("")]

    # Keep one row per compound per activity class; aggregate proteins for the same SMILES.
    grouped = []
    for _, sub in df.groupby("smiles", sort=False):
        grouped.append({
            "smiles": sub["smiles"].iloc[0],
            "activity_outcome": activity,
            "protein_accessions": ";".join(sorted(sub["protein_accession"].dropna().astype(str).unique())),
            "n_proteins": sub["protein_accession"].nunique(),
            "aid": sub["aid"].iloc[0],
            "cid": sub["cid"].iloc[0],
            "sid": sub["sid"].iloc[0],
        })
    return pd.DataFrame(grouped)


def build_compound_protein_bar_plots(input_path: Path, output_dir: Path) -> None:
    """Generate bar plots for compound counts per protein and proteins per compound."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, dtype=str, keep_default_na=True)
    if "activity_outcome" not in df.columns or "protein_accession" not in df.columns or "cid" not in df.columns:
        print("Required columns not found for compound/protein EDA plots; skipping.")
        return

    df = df[df["activity_outcome"].isin(["Active", "Inactive"])].copy()
    df = df[["protein_accession", "cid", "activity_outcome"]].copy()
    df["protein_accession"] = df["protein_accession"].astype(str).str.strip()
    df["cid"] = df["cid"].astype(str).str.strip()
    df = df[df["protein_accession"].ne("") & df["cid"].ne("")]
    df = df.drop_duplicates(subset=["protein_accession", "cid", "activity_outcome"])

    def save_split_bar_plot(records: pd.DataFrame, label_col: str, value_col: str, title: str, ylabel: str, output_path: Path, max_bars_per_fig: int = 50) -> None:
        if records.empty:
            return
        records = records.sort_values(value_col, ascending=False).reset_index(drop=True)
        labels = records[label_col].astype(str).tolist()
        values = records[value_col].astype(int).tolist()

        if len(labels) <= max_bars_per_fig:
            fig, ax = plt.subplots(figsize=(14, max(4.5, 0.28 * len(labels))))
            bars = ax.bar(labels, values, color="#4C78A8")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xlabel(label_col.replace("_", " ").title())
            ax.grid(alpha=0.25, axis="y")
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom", fontsize=8)
            plt.xticks(rotation=90)
            fig.tight_layout()
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return

        n_panels = math.ceil(len(labels) / max_bars_per_fig)
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.2 * n_panels), squeeze=False)
        for idx in range(n_panels):
            start = idx * max_bars_per_fig
            end = min(start + max_bars_per_fig, len(labels))
            ax = axes[idx][0]
            chunk_labels = labels[start:end]
            chunk_values = values[start:end]
            bars = ax.bar(chunk_labels, chunk_values, color="#4C78A8")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} (part {idx + 1}/{n_panels})")
            ax.set_xlabel(label_col.replace("_", " ").title())
            ax.grid(alpha=0.25, axis="y")
            for bar, value in zip(bars, chunk_values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom", fontsize=8)
            plt.setp(ax.get_xticklabels(), rotation=90)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    for outcome in ["Active", "Inactive"]:
        outcome_df = df[df["activity_outcome"] == outcome]
        if outcome_df.empty:
            continue

        compounds_per_protein = (
            outcome_df.groupby("protein_accession", as_index=False)["cid"]
            .nunique()
            .rename(columns={"cid": "n_compounds"})
        )
        save_split_bar_plot(
            compounds_per_protein,
            label_col="protein_accession",
            value_col="n_compounds",
            title=f"Compounds per protein ({outcome})",
            ylabel="Number of compounds",
            output_path=output_dir / f"compounds_per_protein_{outcome.lower()}.png",
        )

        proteins_per_compound = (
            outcome_df.groupby("cid", as_index=False)["protein_accession"]
            .nunique()
            .rename(columns={"protein_accession": "n_proteins"})
        )
        save_split_bar_plot(
            proteins_per_compound,
            label_col="cid",
            value_col="n_proteins",
            title=f"Proteins tested per compound ({outcome})",
            ylabel="Number of proteins",
            output_path=output_dir / f"proteins_per_compound_{outcome.lower()}.png",
        )


def build_pca_and_distance_plots(df: pd.DataFrame, output_dir: Path, activity: str) -> None:
    """Generate a PCA scatter plot and a distance-based heatmap/dendrogram."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print(f"No {activity.lower()} compounds available after filtering. Skipping plots.")
        return

    descriptor_df = build_descriptor_frame(df["smiles"])
    descriptor_df = descriptor_df.merge(df[["smiles", "activity_outcome", "protein_accessions", "n_proteins"]], on="smiles", how="left")
    descriptor_df = descriptor_df.dropna(subset=["smiles"]).reset_index(drop=True)

    if descriptor_df.shape[0] < 3:
        print(f"Only {descriptor_df.shape[0]} compounds available for {activity.lower()} analysis; skipping PCA and clustering plots.")
        return

    feature_cols = [
        "mol_weight", "logp", "tpsa", "hbd", "hba", "rot_bonds", "heavy_atoms", "ring_count", "fraction_csp3"
    ]
    X = descriptor_df[feature_cols].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)

    descriptor_df["pc1"] = pcs[:, 0]
    descriptor_df["pc2"] = pcs[:, 1]
    descriptor_df.to_csv(output_dir / f"chem_space_{activity.lower()}_compounds.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        descriptor_df["pc1"],
        descriptor_df["pc2"],
        c=descriptor_df["n_proteins"],
        cmap="viridis",
        s=70,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.4,
    )
    for _, row in descriptor_df.iterrows():
        ax.text(row["pc1"], row["pc2"], "•", fontsize=12, ha="center", va="center")
    ax.set_title(f"PCA of {activity} compounds (RDKit descriptors)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Number of proteins tested")
    fig.tight_layout()
    fig.savefig(output_dir / f"chem_space_pca_{activity.lower()}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Distance-based clustering on the standardized descriptor matrix.
    dist_matrix = squareform(pdist(X_scaled, metric="euclidean"))
    linkage_matrix = linkage(dist_matrix, method="average")

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    dendrogram(linkage_matrix, labels=None, orientation="top", ax=ax, leaf_font_size=7)
    ax.set_title(f"Dendrogram for {activity} compounds")
    ax.set_xlabel("Compound")
    ax.set_ylabel("Euclidean distance")
    fig.tight_layout()
    fig.savefig(output_dir / f"chem_space_dendrogram_{activity.lower()}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Heatmap from the distance matrix.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    heatmap_data = pd.DataFrame(dist_matrix, index=[f"Compound {i+1}" for i in range(len(descriptor_df))], columns=[f"Compound {i+1}" for i in range(len(descriptor_df))])
    sns.heatmap(heatmap_data, cmap="viridis", square=True, ax=ax)
    ax.set_title(f"Distance heatmap for {activity} compounds")
    fig.tight_layout()
    fig.savefig(output_dir / f"chem_space_heatmap_{activity.lower()}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chemical space analysis for PubChem protein-compound activity pairs")
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "final" / "combined_active_inactive_rows.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "final" / "eda" / "chem_space")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    build_compound_protein_bar_plots(args.input, args.output_dir)

    activities = ["Active", "Inactive"]
    for activity in activities:
        subset = prepare_dataset(args.input, activity)
        build_pca_and_distance_plots(subset, args.output_dir, activity)

    print(f"Chemical-space outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()