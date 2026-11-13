# Protein Graph Construction Pipeline

This module constructs PyTorch Geometric graphs from protein structures using UniProt IDs and the Graphein library.

## Overview

The pipeline:
1. Loads protein data from a merged UniProt/InterPro CSV file
2. Downloads PDB structures for each protein via UniProt ID mapping
3. Constructs protein graphs using Graphein
4. Converts graphs to PyTorch Geometric format with standardized node features
5. **Filters edges using a distance cutoff of 10 Ångströms** to represent spatial proximity
6. Saves graphs as `.pt` files organized by source (common, uniprot_only, interpro_only)

## Node Features Format

Each node (residue) has 25-dimensional features:

```
[0:3]   -> 3D Cartesian coordinates (X, Y, Z)
[3:23]  -> Residue type one-hot encoding (20 amino acids)
[23:24] -> Chain ID (numeric: A=0, B=1, etc.)
[24:25] -> Residue number (sequential position in chain)
```

## Files

### `functions.py`
Core functions for graph construction:
- `encode_residue_onehot()` - Convert amino acid codes to one-hot vectors
- `extract_node_features()` - Parse nodes from Graphein graph
- `construct_edge_index()` - Extract edge connectivity and attributes
- `build_pytorch_graph()` - Create PyTorch Geometric Data object
- `construct_protein_graph_from_uniprot()` - Full pipeline for one protein
- `print_graph_info()` - Display detailed graph statistics

### `main.py`
Pipeline orchestrator:
- `ProteinGraphPipeline` - Main class managing the full workflow
- Loads merged CSV dataset
- Processes proteins in batches
- Saves graphs with organized directory structure
- Generates execution statistics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python main.py \
    --input ../uniprot_interpro_plasmodium_falciparum_merged.csv \
    --output protein_graphs
```

### With Custom Distance Cutoff
```bash
python main.py \
    --input ../uniprot_interpro_plasmodium_falciparum_merged.csv \
    --output protein_graphs \
    --distance-cutoff 10.0  # Default is 10.0 Ångströms
```

### With Options
```bash
python main.py \
    --input ../uniprot_interpro_plasmodium_falciparum_merged.csv \
    --output protein_graphs \
    --distance-cutoff 8.0 \
    --verbose \
    --limit 10  # Process only 10 proteins for testing
```

### Arguments
- `--input`: Path to merged UniProt/InterPro CSV (default: `../uniprot_interpro_plasmodium_falciparum_merged.csv`)
- `--output`: Output directory for graphs (default: `protein_graphs`)
- `--distance-cutoff`: Maximum distance in Ångströms for residue-residue contacts (default: `10.0`)
- `--verbose`: Print detailed information for each graph
- `--limit`: Maximum number of proteins to process (optional)

## Output Structure

```
protein_graphs/
├── common/              # Proteins in both datasets
│   ├── P12345.pt
│   ├── Q67890.pt
│   └── ...
├── uniprot_only/        # Proteins only in UniProt
│   ├── P99999.pt
│   └── ...
└── interpro_only/       # Proteins only in InterPro
    ├── A0A001.pt
    └── ...
```

## Loading and Using Graphs

```python
import torch
from torch_geometric.data import Data

# Load a graph
graph = torch.load('protein_graphs/common/P12345.pt')

# Access graph information
print(f"Nodes: {graph.num_nodes}")
print(f"Edges: {graph.num_edges}")
print(f"Node features shape: {graph.x.shape}")  # (num_nodes, 25)
print(f"Edge attributes shape: {graph.edge_attr.shape}")

# Parse node features
coordinates = graph.x[:, 0:3]          # (num_nodes, 3)
residue_onehot = graph.x[:, 3:23]     # (num_nodes, 20)
chain_id = graph.x[:, 23:24]          # (num_nodes, 1)
residue_number = graph.x[:, 24:25]    # (num_nodes, 1)

# Get edge information
edge_index = graph.edge_index           # (2, num_edges)
edge_distances = graph.edge_attr        # (num_edges, 1)
```

## Graph Properties

For each constructed graph:

```
Total Nodes: Number of residues in the protein
Total Edges: Contacts/interactions between residues
Node features shape: (num_residues, 25)
Edge attributes shape: (num_contacts, 1) - typically contains edge distances
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- Graphein 1.0+
- Pandas, NumPy, BioPython, tqdm

## Notes

- PDB files are downloaded automatically via Graphein
- First run may take time due to PDB downloads
- Graphs are cached in the output directory
- If a graph file already exists, it's skipped
- Failed proteins are logged but don't stop the pipeline

## Troubleshooting

### PDB Download Fails
- Ensure internet connection is stable
- Some proteins may not have PDB structures
- Check the log file for specific protein IDs

### Memory Issues
Use the `--limit` flag to process proteins in smaller batches

### Graph Construction Fails
- Verify the input CSV format matches the merged dataset
- Check that UniProt IDs are valid
- Review logs for specific error messages
