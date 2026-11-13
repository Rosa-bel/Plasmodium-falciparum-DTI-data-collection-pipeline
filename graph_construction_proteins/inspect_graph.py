#!/usr/bin/env python3
"""Inspect a PyTorch Geometric graph saved in a .pt file."""

import argparse
from pathlib import Path
from collections import Counter

import torch
import numpy as np

AMINO_ACIDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'UNK']


def decode_residue_types(onehot: np.ndarray) -> np.ndarray:
    if onehot.ndim != 2 or onehot.shape[1] not in {20, 21}:
        return np.array([])
    indices = np.argmax(onehot, axis=1)
    return np.array([AMINO_ACIDS[i] if i < len(AMINO_ACIDS) else 'UNK' for i in indices])


def summarize_graph(graph, name: str) -> None:
    print(f"Graph: {name}")
    print("=" * 80)

    if hasattr(graph, 'keys') and callable(graph.keys):
        keys = list(graph.keys())
    else:
        keys = [k for k in dir(graph) if not k.startswith('_') and not callable(getattr(graph, k))]

    print("Available graph attributes:")
    print("  " + ", ".join(keys))
    print()

    def safe_get(attr, default=None):
        return getattr(graph, attr, default)

    x = safe_get('x')
    edge_index = safe_get('edge_index')
    edge_attr = safe_get('edge_attr')
    num_nodes = safe_get('num_nodes', None)
    num_edges = safe_get('num_edges', None)

    if num_nodes is None and x is not None:
        num_nodes = x.shape[0]
    if num_edges is None and edge_index is not None:
        num_edges = edge_index.shape[1]

    print(f"Total nodes: {num_nodes}")
    print(f"Total edges: {num_edges}")
    print(f"Node feature tensor: {getattr(x, 'shape', None)}")
    print(f"Edge index tensor: {getattr(edge_index, 'shape', None)}")
    print(f"Edge attribute tensor: {getattr(edge_attr, 'shape', None)}")
    print()

    if x is not None:
        x_np = x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
        coord_dims = x_np.shape[1] if x_np.ndim == 2 else 0
        if coord_dims >= 3:
            coords = x_np[:, 0:3]
            print("Coordinate ranges:")
            print(f"  X: {coords[:, 0].min():.3f} to {coords[:, 0].max():.3f}")
            print(f"  Y: {coords[:, 1].min():.3f} to {coords[:, 1].max():.3f}")
            print(f"  Z: {coords[:, 2].min():.3f} to {coords[:, 2].max():.3f}")
            print()

        if coord_dims >= 24:
            residue_onehot = x_np[:, 3:23]
            residue_names = decode_residue_types(residue_onehot)
            residue_counts = Counter(residue_names)
            print("Residue type counts:")
            for residue, count in residue_counts.most_common():
                print(f"  {residue}: {count}")
            print()

        if coord_dims >= 25:
            chain_values = x_np[:, 23:24].astype(int).flatten()
            unique_chains = sorted({chr(int(val) + ord('A')) for val in chain_values})
            chain_counts = Counter(chain_values)
            print(f"Chains: {', '.join(unique_chains)}")
            print(f"Chain counts: {dict(chain_counts)}")
            print()

            residue_numbers = x_np[:, 24:25].flatten().astype(int)
            print(f"Residue number range: {residue_numbers.min()} to {residue_numbers.max()}")
            print()

    if edge_attr is not None:
        edge_attr_np = edge_attr.detach().cpu().numpy() if hasattr(edge_attr, 'detach') else np.asarray(edge_attr)
        if edge_attr_np.size > 0:
            print(f"Edge attribute min: {edge_attr_np.min():.4f}")
            print(f"Edge attribute max: {edge_attr_np.max():.4f}")
            print(f"Edge attribute mean: {edge_attr_np.mean():.4f}")
            print()

    print("Summary complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect a saved PyTorch Geometric .pt graph file.')
    parser.add_argument('graph_path', type=str, help='Path to the .pt graph file')
    args = parser.parse_args()

    path = Path(args.graph_path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    graph = torch.load(path , weights_only=False)
    summary_name = path.name
    summarize_graph(graph, summary_name)


if __name__ == '__main__':
    main()
