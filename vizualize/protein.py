"""Visualize a protein structure from a PDB file using Graphein.

Builds a protein graph (Graphein) from a local .pdb file and exports an
interactive 3D Plotly visualization as a standalone HTML file, written into
its own output folder.

Usage:
    python visualize_protein.py --pdb-path structures/PF3D7_0102200.pdb \
        --output-dir viz_output

    # Customize edge construction / node coloring:
    python visualize_protein.py --pdb-path my_protein.pdb \
        --output-dir viz_output \
        --distance-threshold 8.0 \
        --colour-nodes-by residue_name \
        --colour-edges-by kind
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import add_distance_threshold, add_peptide_bonds
from graphein.protein.graphs import construct_graph
from graphein.protein.visualisation import plotly_protein_structure_graph


def build_graph(pdb_path: Path, distance_threshold: float):
    """Construct a Graphein protein graph: peptide-bond backbone edges plus
    distance-threshold edges for spatial contacts."""
    edge_funcs = [
        add_peptide_bonds,
        partial(add_distance_threshold, threshold=distance_threshold, long_interaction_threshold=2),
    ]
    config = ProteinGraphConfig(edge_construction_functions=edge_funcs)
    return construct_graph(config=config, path=str(pdb_path))


def visualize(graph, title: str, colour_nodes_by: str, colour_edges_by: str):
    return plotly_protein_structure_graph(
        graph,
        plot_title=title,
        colour_nodes_by=colour_nodes_by,
        colour_edges_by=colour_edges_by,
        label_node_ids=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Visualize a PDB structure as an interactive HTML graph via Graphein")
    parser.add_argument("--pdb-path", type=Path, required=True, help="Path to a local .pdb file")
    parser.add_argument("--output-dir", type=Path, default=Path("protein_viz"),
                        help="Folder the HTML file is written into (created if missing)")
    parser.add_argument("--distance-threshold", type=float, default=6.0,
                        help="Distance (Angstrom) for spatial contact edges")
    parser.add_argument("--colour-nodes-by", type=str, default="residue_name",
                        help="Node attribute to colour by, e.g. residue_name, degree, chain_id")
    parser.add_argument("--colour-edges-by", type=str, default="kind",
                        help="Edge attribute to colour by, e.g. kind, distance")
    args = parser.parse_args(argv)

    if not args.pdb_path.is_file():
        raise FileNotFoundError(f"PDB file not found: {args.pdb_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building protein graph from {args.pdb_path} ...")
    graph = build_graph(args.pdb_path, args.distance_threshold)
    print(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    protein_name = args.pdb_path.stem
    fig = visualize(graph, title=protein_name, colour_nodes_by=args.colour_nodes_by,
                    colour_edges_by=args.colour_edges_by)

    output_path = args.output_dir / f"{protein_name}.html"
    fig.write_html(str(output_path))
    print(f"Saved interactive visualization to {output_path}")


if __name__ == "__main__":
    main()