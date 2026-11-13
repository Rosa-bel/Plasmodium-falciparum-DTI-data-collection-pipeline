"""
Functions to construct protein graphs from UniProt IDs using Graphein and PyTorch Geometric.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import requests
import torch
from torch_geometric.data import Data
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from functools import partial
from graphein.protein.edges.distance import add_distance_threshold
from torch_geometric.utils import remove_self_loops
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Amino acid types (20 standard amino acids plus unknown)
AMINO_ACIDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'UNK']

AMINO_ACID_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def get_pdp_url(uniprot_id: str) -> Optional[str]:
    """Fetch Alphafold prediction metadata and return the PDB download URL."""
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}?include_complexes=false"
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return data[0].get('pdbUrl')
    except Exception as e:
        logger.warning(f"Failed to fetch Alphafold metadata for {uniprot_id}: {e}")
    return None


def download_alphafold_pdb(uniprot_id: str, output_dir: Optional[str] = None) -> Optional[Path]:
    """Download an Alphafold PDB file for the given UniProt accession."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / 'pdb_files'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = output_dir / f"{uniprot_id}_model.pdb"

    if pdb_path.exists():
        logger.info(f"Using cached Alphafold PDB: {pdb_path}")
        return pdb_path

    pdb_url = get_pdp_url(uniprot_id)
    if not pdb_url:
        logger.warning(f"No PDB URL available for {uniprot_id}")
        return None

    try:
        response = requests.get(pdb_url, stream=True, timeout=60)
        response.raise_for_status()
        with open(pdb_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"Downloaded Alphafold PDB for {uniprot_id} to {pdb_path}")
        return pdb_path
    except Exception as e:
        logger.error(f"Failed to download PDB for {uniprot_id} from {pdb_url}: {e}")
        return None

url = "https://search.rcsb.org"
endpoint = "/rcsbsearch/v2/query"

def get_pdb_ids_from_uniprot(uniprot_id):
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "operator": "exact_match",
                "value": uniprot_id,
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession"
            }
        },
        "return_type": "entry",
        "request_options": {
            "return_all_hits": True
        }
    }
    try:
        response = requests.post(url + endpoint, json=query)
        response.raise_for_status()
        data = response.json()
        pdb_id = data.get("result_set")[0]["identifier"]
        return(pdb_id)
    except Exception as e :
        logger.error(f"Failed to get pdb id for uniprot: {uniprot_id} from {url}: {e}")
        return None


    



def download_pdb_file_rcsb(pdb_id, save_dir="graph_construction_proteins/pdb_files"):
    """
    Download a structure file from RCSB.
    """
    os.makedirs(save_dir, exist_ok=True)

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    response = requests.get(url)
    response.raise_for_status()

    try : 
        filepath = os.path.join(save_dir, f"{pdb_id}.pdb")
        with open(filepath, "wb") as f:
          f.write(response.content)
        return filepath

    except Exception as e : 
        logger.error(f"Failed to download PDB file of {pdb_id} from {url}: {e}")
        return None


def download_pdb_from_uniprot(uniprotid):
    try : 
        id = get_pdb_ids_from_uniprot(uniprot_id=uniprotid)
        pdb_path = download_pdb_file_rcsb(pdb_id=id)
        return pdb_path
    except Exception as e :
        return None

 


def encode_residue_onehot(residue_type: str) -> np.ndarray:
    """
    Encode amino acid residue type as one-hot vector (21 dims).
    
    Args:
        residue_type: Three-letter amino acid code (e.g., 'ALA')
        
    Returns:
        One-hot encoded residue (21-dimensional vector)
    """
    onehot = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    residue = residue_type if residue_type in AMINO_ACID_TO_INDEX else 'UNK'
    onehot[AMINO_ACID_TO_INDEX[residue]] = 1.0
    if residue == 'UNK' and residue_type != 'UNK':
        logger.warning(f"Unknown residue type: {residue_type}")
    return onehot


def extract_node_features(graph) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract node features from Graphein protein graph.
    
    Returns:
        Tuple of:
        - coordinates: (N, 3) 3D coordinates
        - residue_onehot: (N, 21) one-hot encoded residue types
        - chain_ids: (N, 1) chain IDs
        - residue_numbers: (N, 1) residue numbers
    """
    coordinates = []
    residue_onehot = []
    chain_ids = []
    residue_numbers = []
    
    nodes = sorted(graph.nodes())
    
    for node in nodes:
        node_data = graph.nodes[node]
        
        # Extract 3D coordinates
        if 'coords' in node_data:
            coords = node_data['coords']
            coordinates.append(coords)
        else:
            coordinates.append([0.0, 0.0, 0.0])
        
        # Extract residue type
        residue = node_data.get('residue_name', 'UNK')
        residue_onehot.append(encode_residue_onehot(residue))
        
        # Extract chain ID
        chain = node_data.get('chain_id', 'A')
        chain_id_numeric = ord(chain) - ord('A')  # Convert A->0, B->1, etc.
        chain_ids.append([chain_id_numeric])
        
        # Extract residue number
        res_num = node_data.get('residue_number', 0)
        residue_numbers.append([res_num])
    
    coordinates = np.array(coordinates, dtype=np.float32)
    residue_onehot = np.array(residue_onehot, dtype=np.float32)
    chain_ids = np.array(chain_ids, dtype=np.float32)
    residue_numbers = np.array(residue_numbers, dtype=np.float32)
    
    return coordinates, residue_onehot, chain_ids, residue_numbers


def construct_edge_index(graph, distance_cutoff: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract edge connectivity and attributes from Graphein graph.
    Only includes edges where distance <= distance_cutoff (in Angstroms).
    
    Args:
        graph: Graphein protein graph
        distance_cutoff: Maximum distance in Angstroms to consider as edge (default: 10.0)
    
    Returns:
        Tuple of:
        - edge_index: (2, E) edge connectivity
        - edge_attr: (E, F) edge attributes (distance)
    """
    edges = list(graph.edges(data=True))
    
    if len(edges) == 0:
        return np.array([[], []], dtype=np.int64), np.array([], dtype=np.float32).reshape(0, 1)
    
    nodes = sorted(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    edge_list = []
    edge_attrs = []
    
    for src, dst, data in edges:
        # Extract edge distance
        distance = data.get('distance', float('inf'))
        
        # Only include edges within distance cutoff
        if distance <= distance_cutoff:
            src_idx = node_to_idx[src]
            dst_idx = node_to_idx[dst]
            
            edge_list.append([src_idx, dst_idx])
            edge_attrs.append([distance])
    
    if len(edge_list) == 0:
        return np.array([[], []], dtype=np.int64), np.array([], dtype=np.float32).reshape(0, 1)
    
    edge_index = np.array(edge_list, dtype=np.int64).T
    edge_attr = np.array(edge_attrs, dtype=np.float32)
    
    return edge_index, edge_attr


def build_pytorch_graph(graph, distance_cutoff: float = 10.0) -> Optional[Data]:
    """
    Convert Graphein protein graph to PyTorch Geometric Data object.
    
    Node features structure (26 dims):
    - [0:3] 3D coordinates
    - [3:24] Residue type one-hot (21 dims)
    - [24:25] Chain ID
    - [25:26] Residue number
    
    Args:
        graph: Graphein protein graph
        distance_cutoff: Maximum distance in Angstroms for edges (default: 10.0 A)
        
    Returns:
        PyTorch Geometric Data object or None if graph is empty
    """
    try:
        # Extract node features
        coordinates, residue_onehot, chain_ids, residue_numbers = extract_node_features(graph)
        
        if len(coordinates) == 0:
            logger.warning("No nodes found in graph")
            return None
        
        # Concatenate all node features
        # Structure: coords (3) | residue_onehot (20) | chain_id (1) | residue_num (1)
        node_features = np.concatenate([
            coordinates,
            residue_onehot,
            chain_ids,
            residue_numbers
        ], axis=1)
        
        # Extract edge information with distance cutoff
        edge_index, edge_attr = construct_edge_index(graph, distance_cutoff=distance_cutoff)
        
        # Convert to torch tensors
        x = torch.from_numpy(node_features).float()
        
        if edge_index.shape[1] > 0:
            edge_index = torch.from_numpy(edge_index).long()
            edge_attr = torch.from_numpy(edge_attr).float()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float32)
        
        # Create PyTorch Geometric Data object
        edge_index, edge_attr = remove_self_loops(edge_index,edge_attr)
        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return pyg_graph
        
    except Exception as e:
        logger.error(f"Error building PyTorch graph: {e}")
        return None


def construct_protein_graph_from_uniprot(uniprot_id: str, distance_cutoff: float = 10.0) -> Optional[Data]:
    """
    Download PDB structure for UniProt ID and construct protein graph.
    
    Args:
        uniprot_id: UniProt accession ID (e.g., 'P12345')
        distance_cutoff: Maximum distance in Angstroms for edges (default: 10.0 A)
        
    Returns:
        PyTorch Geometric Data object or None if construction fails
    """
    try:
        logger.info(f"Constructing graph for {uniprot_id} (cutoff: {distance_cutoff}A)...")
        pdb_path = download_pdb_from_uniprot(uniprotid=uniprot_id)
        if pdb_path is None:
            logger.warning(f"Could not download RCSB PDB for {uniprot_id}")
            pdb_path = download_alphafold_pdb(uniprot_id)
            if pdb_path is None : 
                logger.warning(f"Could not download Alphafold PDB for {uniprot_id}")
                return None
    

        config = ProteinGraphConfig( granularity="CA",
          edge_construction_functions=[
        partial(add_distance_threshold, threshold=10.0, long_interaction_threshold=0)
    ],)
        graph = construct_graph(config=config, path=str(pdb_path))

        pyg_graph = build_pytorch_graph(graph, distance_cutoff=distance_cutoff)
        return pyg_graph

    except Exception as e:
        logger.error(f"Error constructing graph for {uniprot_id}: {e}")
        return None


def print_graph_info(graph: Data, uniprot_id: str) -> None:
    """
    Print detailed information about constructed protein graph.
    
    Args:
        graph: PyTorch Geometric Data object
        uniprot_id: UniProt ID for reference
    """
    if graph is None:
        print(f"[{uniprot_id}] Graph is None")
        return
    
    print(f"\n{'='*60}")
    print(f"Protein: {uniprot_id}")
    print(f"{'='*60}")
    print(f"  Total Nodes: {graph.num_nodes}")
    print(f"  Total Edges: {graph.num_edges}")
    print(f"  Node features shape: {graph.x.shape}")
    print(f"  Edge attributes shape: {graph.edge_attr.shape}\n")
    
    # Parse node features
    # Structure: 3D coords (0:3) | Residue type one-hot (3:24) | Chain ID (24:25) | Residue num (25:26)
    node_features = graph.x.numpy()
    
    coordinates = node_features[:, 0:3]
    residue_onehot = node_features[:, 3:24]
    chain_id = node_features[:, 24:25]
    residue_number = node_features[:, 25:26]
    
    # Decode residue type from one-hot
    residue_types = np.argmax(residue_onehot, axis=1)
    
    print(f"  Coordinates range: X[{coordinates[:, 0].min():.2f}, {coordinates[:, 0].max():.2f}], "
          f"Y[{coordinates[:, 1].min():.2f}, {coordinates[:, 1].max():.2f}], "
          f"Z[{coordinates[:, 2].min():.2f}, {coordinates[:, 2].max():.2f}]")
    print(f"  Chain IDs: {np.unique(chain_id).flatten()}")
    print(f"  Residue range: {int(residue_number.min())} - {int(residue_number.max())}")
    print(f"  Unique residues: {len(np.unique(residue_types))}")
    print(f"{'='*60}\n")
