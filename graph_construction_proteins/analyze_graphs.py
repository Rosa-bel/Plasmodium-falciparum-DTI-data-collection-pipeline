"""
Utility script to analyze and inspect constructed protein graphs.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd

AMINO_ACIDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']


def compute_residue_stats(node_features: np.ndarray) -> dict:
    """
    Compute per-graph residue-type counts, available chains, and the
    residue-number range from the raw node feature matrix.

    Assumes column layout: [0:3]=coords, [3:23]=residue one-hot (20),
    [23]=chain_id, [24]=residue_number -- matches analyze_single_graph().
    """
    residue_onehot = node_features[:, 3:23]
    chain_id = node_features[:, 23:24]
    residue_number = node_features[:, 24:25]

    residue_types = np.argmax(residue_onehot, axis=1)
    residue_names = [AMINO_ACIDS[t] for t in residue_types]

    # Count occurrences of each amino acid, including zero counts for
    # residues absent from this particular protein.
    counts = {aa: 0 for aa in AMINO_ACIDS}
    for name in residue_names:
        counts[name] += 1

    chains = sorted(np.unique(chain_id).astype(int).tolist())
    chain_letters = [chr(c + ord('A')) for c in chains]

    return {
        "residue_counts": counts,
        "chains": chain_letters,
        "residue_number_min": int(residue_number.min()),
        "residue_number_max": int(residue_number.max()),
    }


def analyze_single_graph(graph_path: str) -> dict:
    """
    Analyze a single protein graph.

    Args:
        graph_path: Path to the .pt graph file

    Returns:
        Dictionary with graph statistics
    """
    try:
        graph = torch.load(graph_path, weights_only=False)
    except Exception as e:
        print(f"Error loading {graph_path}: {e}")
        return None

    graph = graph["protein_graph"] if "protein_graph" in graph else graph  # Handle dict case
    node_features = graph.x.numpy()

    # Parse features
    coordinates = node_features[:, 0:3]
    residue_stats = compute_residue_stats(node_features)

    num_nodes = graph.x.shape[0]
    num_edges = graph.edge_index.shape[1] if getattr(graph, 'edge_index', None) is not None else 0

    stats = {
        'filename': Path(graph_path).name,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_node_degree': 2 * num_edges / max(1, num_nodes),
        'coord_x_min': coordinates[:, 0].min(),
        'coord_x_max': coordinates[:, 0].max(),
        'coord_y_min': coordinates[:, 1].min(),
        'coord_y_max': coordinates[:, 1].max(),
        'coord_z_min': coordinates[:, 2].min(),
        'coord_z_max': coordinates[:, 2].max(),
        'residue_range_min': residue_stats['residue_number_min'],
        'residue_range_max': residue_stats['residue_number_max'],
        'num_chains': len(residue_stats['chains']),
        'chains': ','.join(residue_stats['chains']),
        'unique_residues': sum(1 for c in residue_stats['residue_counts'].values() if c > 0),
        'node_feature_dim': graph.x.shape[1],
        'edge_attr_dim': (getattr(graph, 'edge_attr', None).shape[1]
                          if getattr(graph, 'edge_attr', None) is not None and hasattr(getattr(graph, 'edge_attr', None), 'shape')
                          else 0)
    }

    return stats


def extract_detailed_node_edge_info(graph_path: str, output_file, num_nodes: int = 50, num_edges: int = 50) -> str:
    """
    Extract and format detailed information: residue-type counts, chains,
    residue-number range, then the first N nodes and N edges.

    Args:
        graph_path: Path to the .pt graph file
        output_file: File object to write to
        num_nodes: Number of nodes to extract (default 50)
        num_edges: Number of edges to extract (default 50)

    Returns:
        Formatted string with all extracted details
    """
    try:
        graph = torch.load(graph_path, weights_only=False)
    except Exception as e:
        return f"Error loading {graph_path}: {e}\n"

    graph = graph["protein_graph"] if "protein_graph" in graph else graph

    node_features = graph.x.numpy()
    residue_stats = compute_residue_stats(node_features)

    output_text = ""

    # Header
    output_text += f"\n{'='*90}\n"
    output_text += f"PROTEIN GRAPH: {Path(graph_path).name}\n"
    output_text += f"{'='*90}\n"
    output_text += f"Total Nodes: {graph.x.shape[0]}, Total Edges: {graph.edge_index.shape[1] if hasattr(graph, 'edge_index') else 0}\n\n"

    # --- Residue type counts / chains / residue number range ---
    output_text += f"{'-'*90}\n"
    output_text += "RESIDUE TYPE COUNTS\n"
    output_text += f"{'-'*90}\n"
    for aa, count in sorted(residue_stats['residue_counts'].items(), key=lambda kv: -kv[1]):
        if count > 0:
            output_text += f"  {aa}: {count}\n"
    output_text += "\n"
    output_text += f"Available chains: {', '.join(residue_stats['chains'])}\n"
    output_text += f"Residue number range: {residue_stats['residue_number_min']} to {residue_stats['residue_number_max']}\n\n"

    # Node information
    output_text += f"{'='*90}\n"
    output_text += f"FIRST {min(num_nodes, graph.x.shape[0])} NODES INFORMATION\n"
    output_text += f"{'='*90}\n"
    output_text += f"{'Node_ID':<8} {'Coord_X':<12} {'Coord_Y':<12} {'Coord_Z':<12} {'Residue':<8} {'Chain':<6} {'Res_Num':<8}\n"
    output_text += "-" * 90 + "\n"

    for node_idx in range(min(num_nodes, graph.x.shape[0])):
        features = node_features[node_idx]
        coords = features[0:3]
        residue_onehot = features[3:23]
        chain_id = int(features[23])
        residue_num = int(features[24])

        residue_idx = np.argmax(residue_onehot)
        residue_name = AMINO_ACIDS[residue_idx] if residue_idx < len(AMINO_ACIDS) else 'UNK'
        chain_letter = chr(chain_id + ord('A'))

        output_text += f"{node_idx:<8} {coords[0]:<12.4f} {coords[1]:<12.4f} {coords[2]:<12.4f} {residue_name:<8} {chain_letter:<6} {residue_num:<8}\n"

    # Edge information
    output_text += f"\n{'='*90}\n"
    output_text += f"FIRST {min(num_edges, graph.edge_index.shape[1])} EDGES INFORMATION\n"
    output_text += f"{'='*90}\n"

    if hasattr(graph, 'edge_index'):
        edge_index = graph.edge_index.numpy()
        output_text += f"{'Edge_ID':<8} {'Source_Node':<12} {'Target_Node':<12} {'Edge_Attr':<30}\n"
        output_text += "-" * 90 + "\n"

        for edge_idx in range(min(num_edges, edge_index.shape[1])):
            source = edge_index[0, edge_idx]
            target = edge_index[1, edge_idx]

            edge_attr_str = ""
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                edge_attr = graph.edge_attr[edge_idx].numpy()
                edge_attr_str = str(np.round(edge_attr, 4))
            else:
                edge_attr_str = "N/A"

            output_text += f"{edge_idx:<8} {source:<12} {target:<12} {edge_attr_str:<30}\n"
    else:
        output_text += "No edge information available\n"

    output_text += "\n"
    output_file.write(output_text)
    return output_text


def analyze_directory(directory: str, max_files: int = None, recursive: bool = False) -> pd.DataFrame:
    """
    Analyze all graphs in a directory.

    Args:
        directory: Path to directory containing .pt files
        max_files: Maximum number of files to analyze (None for all)

    Returns:
        DataFrame with statistics for all graphs
    """
    dir_path = Path(directory)
    graph_files = sorted(dir_path.rglob('*.pt')) if recursive else sorted(dir_path.glob('*.pt'))

    if max_files is not None:
        graph_files = graph_files[:max_files]

    print(f"Found {len(graph_files)} graph files in {directory}")

    results = []
    for graph_file in tqdm(graph_files, desc="Analyzing graphs"):
        stats = analyze_single_graph(str(graph_file))
        if stats is not None:
            results.append(stats)

    return pd.DataFrame(results)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze protein graphs")
    parser.add_argument(
        '--directory',
        type=str,
        default='graphproteins',
        help='Directory containing .pt graph files'
    )
    parser.add_argument(
        '--single',
        type=str,
        help='Analyze a single graph file'
    )
    parser.add_argument(
        '--output',
        default='graph_construction_proteins/output.csv',
        type=str,
        help='Save analysis results to CSV'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively search for .pt files in subdirectories'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of files to analyze'
    )
    parser.add_argument(
        '--detail-txt',
        type=str,
        default="graph_construction_proteins/output.txt",
        help='Save detailed residue stats, node and edge information to a .txt file'
    )
    parser.add_argument(
        '--num-nodes',
        type=int,
        default=50,
        help='Number of nodes to extract details for (default 50)'
    )
    parser.add_argument(
        '--num-edges',
        type=int,
        default=50,
        help='Number of edges to extract details for (default 50)'
    )

    args = parser.parse_args()

    if args.single:
        # Analyze single file
        print(f"\nAnalyzing: {args.single}")
        print("="*70)
        stats = analyze_single_graph(args.single)
        if stats:
            for key, value in stats.items():
                print(f"{key:.<40} {value}")

        # Extract detailed residue stats + node/edge info if requested
        if args.detail_txt:
            print(f"\nExtracting detailed residue stats, node and edge information...")
            with open(args.detail_txt, 'w') as f:
                extract_detailed_node_edge_info(args.single, f, args.num_nodes, args.num_edges)
            print(f"Detailed information saved to: {args.detail_txt}")
    else:
        # Analyze directory
        if args.recursive:
            all_files = list(Path(args.directory).rglob('*.pt'))
            print(f"Found {len(all_files)} graph files (recursive)")

        df = analyze_directory(args.directory, max_files=args.limit, recursive=args.recursive)

        print("\n" + "="*70)
        print("GRAPH ANALYSIS SUMMARY")
        print("="*70)
        print(f"\nTotal graphs analyzed: {len(df)}")
        if df.empty:
            print("No graph statistics were collected. Check the directory or file permissions.")
            return

        # Ensure expected columns exist
        required_cols = ['num_nodes', 'num_edges', 'avg_node_degree', 'num_chains', 'unique_residues']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Missing expected columns in results: {missing}. Columns present: {list(df.columns)}")
            return
        print(f"\nNode statistics:")
        print(f"  Mean nodes per graph: {df['num_nodes'].mean():.1f}")
        print(f"  Min nodes: {df['num_nodes'].min()}")
        print(f"  Max nodes: {df['num_nodes'].max()}")

        print(f"\nEdge statistics:")
        print(f"  Mean edges per graph: {df['num_edges'].mean():.1f}")
        print(f"  Mean node degree: {df['avg_node_degree'].mean():.2f}")

        print(f"\nStructure statistics:")
        print(f"  Mean chains per graph: {df['num_chains'].mean():.2f}")
        print(f"  Mean unique residues: {df['unique_residues'].mean():.1f}")

        print("\nDetailed results:")
        print(df.to_string())

        # Save if requested
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")

        # Extract and save residue stats + detailed node/edge info to txt file
        if args.detail_txt:
            print(f"\nExtracting detailed residue stats, node and edge information for all graphs...")
            dir_path = Path(args.directory)
            graph_files = sorted(dir_path.rglob('*.pt')) if args.recursive else sorted(dir_path.glob('*.pt'))

            if args.limit is not None:
                graph_files = graph_files[:args.limit]

            with open(args.detail_txt, 'w') as f:
                f.write("="*90 + "\n")
                f.write("DETAILED RESIDUE, NODE AND EDGE INFORMATION FOR ALL PROTEIN GRAPHS\n")
                f.write(f"Total graphs: {len(graph_files)}\n")
                f.write("="*90 + "\n")

                for graph_file in tqdm(graph_files, desc="Extracting details"):
                    extract_detailed_node_edge_info(str(graph_file), f, args.num_nodes, args.num_edges)

            print(f"Detailed information saved to: {args.detail_txt}")

        print("="*70)


if __name__ == "__main__":
    main()