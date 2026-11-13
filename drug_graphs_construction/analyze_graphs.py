import os
import argparse
import glob
import torch
import statistics
from rdkit import Chem


ATOM_TYPES = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
]
HYBRIDIZATIONS = [
    'UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'SP2D', 'OTHER'
]

BOND_TYPES = [
    Chem.rdchem.BondType.ZERO,                      # 0
    Chem.rdchem.BondType.SINGLE,                    # 1
    Chem.rdchem.BondType.DOUBLE,                    # 2
    Chem.rdchem.BondType.TRIPLE,                    # 3
    Chem.rdchem.BondType.QUADRUPLE,                 # 4
    Chem.rdchem.BondType.QUINTUPLE,                 # 5
    Chem.rdchem.BondType.HEXTUPLE,                  # 6
    Chem.rdchem.BondType.ONEANDAHALF,               # 7
    Chem.rdchem.BondType.TWOANDAHALF,               # 8
    Chem.rdchem.BondType.THREEANDAHALF,             # 9
    Chem.rdchem.BondType.FOURANDAHALF,              # 10
    Chem.rdchem.BondType.FIVEANDAHALF,              # 11
    Chem.rdchem.BondType.IONIC,                     # 12
    Chem.rdchem.BondType.HYDROGEN,                  # 13
    Chem.rdchem.BondType.THREECENTER,               # 14
    Chem.rdchem.BondType.DATIVEONE,                 # 15
    Chem.rdchem.BondType.DATIVE,                    # 16
    Chem.rdchem.BondType.DATIVEL,                   # 17
    Chem.rdchem.BondType.DATIVER,                   # 18
    Chem.rdchem.BondType.AROMATIC,                  # 19 ← matches reference!
    Chem.rdchem.BondType.OTHER,                     # 20
    Chem.rdchem.BondType.UNSPECIFIED,               # 21
]

FEATURE_SEGMENTS = {
    'atom_type': (0, 46),
    'degree': (46, 57),
    'implicit_valence': (57, 64),
    'total_num_hs': (64, 74),
    'hybridization': (74, 83),
    'formal_charge': (83, 84),
    'num_radical_electrons': (84, 85),
    'aromatic': (85, 86),
}


def _argmax_index(vec):
    if vec is None or len(vec) == 0:
        return None
    arr = torch.as_tensor(vec, dtype=torch.float)
    if arr.numel() == 0:
        return None
    idx = int(torch.argmax(arr).item())
    val = float(arr[idx].item())
    return idx, val


def decode_atom_feature_vector(atom_vector):
    if atom_vector is None:
        return None

    vec = torch.as_tensor(atom_vector, dtype=torch.float)
    if vec.dim() == 0:
        return None
    if vec.numel() < 86:
        return {
            'atom_type': 'UNKNOWN',
            'degree': None,
            'implicit_valence': None,
            'total_num_hs': None,
            'hybridization': None,
            'formal_charge': None,
            'num_radical_electrons': None,
            'aromatic': None,
            'raw_length': int(vec.numel()),
        }

    atom_type_segment = vec[:46]
    degree_segment = vec[46:57]
    implicit_valence_segment = vec[57:64]
    total_hs_segment = vec[64:74]
    hybridization_segment = vec[74:83]

    atom_type = 'UNKNOWN'
    atom_type_idx = _argmax_index(atom_type_segment)
    if atom_type_idx is not None:
        idx, val = atom_type_idx
        if val > 0.5 and idx < len(ATOM_TYPES):
            atom_type = ATOM_TYPES[idx]

    degree_idx = _argmax_index(degree_segment)
    degree = None if degree_idx is None else int(degree_idx[0])

    valence_idx = _argmax_index(implicit_valence_segment)
    implicit_valence = None if valence_idx is None else int(valence_idx[0])

    hs_idx = _argmax_index(total_hs_segment)
    total_num_hs = None if hs_idx is None else int(hs_idx[0])

    hybrid_idx = _argmax_index(hybridization_segment)
    hybridization = None if hybrid_idx is None else HYBRIDIZATIONS[hybrid_idx[0]] if hybrid_idx[0] < len(HYBRIDIZATIONS) else 'UNKNOWN'

    formal_charge = float(vec[83].item())
    num_radical_electrons = float(vec[84].item())
    aromatic = bool(float(vec[85].item()) > 0.5)

    return {
        'atom_type': atom_type,
        'degree': degree,
        'implicit_valence': implicit_valence,
        'total_num_hs': total_num_hs,
        'hybridization': hybridization,
        'formal_charge': formal_charge,
        'num_radical_electrons': num_radical_electrons,
        'aromatic': aromatic,
    }


def format_atom_summary(atom_info, atom_index):
    if atom_info is None:
        return f'    atom {atom_index}: <no features>'
    return (
        f"    atom {atom_index}: type={atom_info['atom_type']}, "
        f"degree={atom_info['degree']}, implicit_valence={atom_info['implicit_valence']}, "
        f"total_num_hs={atom_info['total_num_hs']}, hybridization={atom_info['hybridization']}, "
        f"formal_charge={atom_info['formal_charge']}, radical_electrons={atom_info['num_radical_electrons']}, "
        f"aromatic={atom_info['aromatic']}"
    )


def decode_bond_feature_vector(bond_vector):
    if bond_vector is None:
        return None

    vec = torch.as_tensor(bond_vector, dtype=torch.float)
    if vec.dim() == 0:
        return None
    if vec.numel() < 24:
        return {
            'bond_type': 'UNKNOWN',
            'is_in_ring': None,
            'same_ring': None,
            'raw_length': int(vec.numel()),
        }

    bond_type_segment = vec[:22]
    ring_segment = vec[22]
    same_ring_segment = vec[23]

    bond_type = 'UNKNOWN'
    bond_idx = _argmax_index(bond_type_segment)
    if bond_idx is not None:
        idx, val = bond_idx
        if val > 0.5 and idx < len(BOND_TYPES):
            bond_type = str(BOND_TYPES[idx]).split('.')[-1]

    return {
        'bond_type': bond_type,
        'is_in_ring': bool(float(ring_segment.item()) > 0.5),
        'same_ring': bool(float(same_ring_segment.item()) > 0.5),
    }


def format_bond_summary(edge_index, edge_attr, edge_position):
    if edge_attr is None:
        return f'    edge {edge_position}: <no features>'

    src, dst = int(edge_index[0, edge_position]), int(edge_index[1, edge_position])
    bond_info = decode_bond_feature_vector(edge_attr[edge_position])
    if bond_info is None:
        return f'    edge {edge_position}: {src}->{dst}: <no features>'

    return (
        f"    edge {edge_position}: {src}->{dst}, "
        f"bond_type={bond_info['bond_type']}, "
        f"is_in_ring={bond_info['is_in_ring']}, same_ring={bond_info['same_ring']}"
    )


def analyze_graph(path):
    data = torch.load(path, weights_only=False)

    num_nodes = int(data.x.shape[0]) if getattr(data, 'x', None) is not None else 0
    feat_dim = int(data.x.shape[1]) if getattr(data, 'x', None) is not None else 0
    directed_edges = int(data.edge_index.shape[1]) if getattr(data, 'edge_index', None) is not None else 0
    undirected_edges = directed_edges // 2

    avg_degree = 0.0
    if directed_edges > 0 and num_nodes > 0:
        src = data.edge_index[0]
        deg = torch.bincount(src, minlength=num_nodes).float()
        avg_degree = float(deg.mean().item())

    aromatic_count = None
    if getattr(data, 'x', None) is not None and data.x.shape[1] >= 1:
        try:
            aromatic_count = int((data.x[:, -1] > 0.5).sum().item())
        except Exception:
            aromatic_count = None

    first_atoms = []
    if getattr(data, 'x', None) is not None and data.x.dim() >= 2:
        for i in range(min(10, data.x.shape[0])):
            atom_info = decode_atom_feature_vector(data.x[i])
            first_atoms.append({
                'atom_index': i,
                'atom_type': atom_info['atom_type'] if atom_info else 'UNKNOWN',
                'degree': atom_info['degree'] if atom_info else None,
                'implicit_valence': atom_info['implicit_valence'] if atom_info else None,
                'total_num_hs': atom_info['total_num_hs'] if atom_info else None,
                'hybridization': atom_info['hybridization'] if atom_info else None,
                'formal_charge': atom_info['formal_charge'] if atom_info else None,
                'num_radical_electrons': atom_info['num_radical_electrons'] if atom_info else None,
                'aromatic': atom_info['aromatic'] if atom_info else None,
            })

    first_edges = []
    if getattr(data, 'edge_index', None) is not None and getattr(data, 'edge_attr', None) is not None:
        edge_count = min(100, int(data.edge_index.shape[1]))
        for i in range(edge_count):
            src = int(data.edge_index[0, i].item())
            dst = int(data.edge_index[1, i].item())
            bond_info = decode_bond_feature_vector(data.edge_attr[i])
            first_edges.append({
                'edge_index': i,
                'src': src,
                'dst': dst,
                'bond_type': bond_info['bond_type'] if bond_info else 'UNKNOWN',
                'is_in_ring': bond_info['is_in_ring'] if bond_info else None,
                'same_ring': bond_info['same_ring'] if bond_info else None,
            })

    return {
        'num_nodes': num_nodes,
        'feat_dim': feat_dim,
        'directed_edges': directed_edges,
        'undirected_edges': undirected_edges,
        'avg_degree': avg_degree,
        'aromatic_count': aromatic_count,
        'first_atoms': first_atoms,
        'first_edges': first_edges,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze saved PyG .pt graphs and write a summary')
    parser.add_argument('graphs_dir', help='Directory containing .pt graph files')
    parser.add_argument('--out', default='drug_graphs_construction/graphs_summary.txt', help='Output summary text file')
    parser.add_argument('--pattern', default='*.pt', help='Filename pattern to match (default: *.pt)')
    args = parser.parse_args()

    pattern = os.path.join(args.graphs_dir, args.pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'No files matching {pattern}')
        return

    per_file = []
    for p in files:
        try:
            stats = analyze_graph(p)
            stats['path'] = p
            per_file.append(stats)
        except Exception as e:
            per_file.append({'path': p, 'error': str(e)})

    # Build aggregate stats
    node_counts = [f['num_nodes'] for f in per_file if 'num_nodes' in f]
    undirected_edges = [f['undirected_edges'] for f in per_file if 'undirected_edges' in f]
    feat_dims = list(sorted({f['feat_dim'] for f in per_file if 'feat_dim' in f}))

    with open(args.out, 'w') as fh:
        fh.write('Per-file graph summary:\n')
        for f in per_file:
            if 'error' in f:
                fh.write(f"{os.path.basename(f['path'])}: ERROR: {f['error']}\n")
                continue
            fh.write(f"{os.path.basename(f['path'])}: nodes={f['num_nodes']}, feat_dim={f['feat_dim']}, undirected_edges={f['undirected_edges']}, avg_degree={f['avg_degree']:.3f}, aromatic_count={f['aromatic_count']}\n")
            if f.get('first_atoms'):
                fh.write('  Feature layout: 46 atom type + 11 degree + 7 implicit valence + 10 total H + 9 hybridization + 1 formal charge + 1 radical electrons + 1 aromatic\n')
                fh.write('  First 10 atoms (decoded):\n')
                for atom in f['first_atoms']:
                    atom_info = {
                        'atom_type': atom['atom_type'],
                        'degree': atom['degree'],
                        'implicit_valence': atom['implicit_valence'],
                        'total_num_hs': atom['total_num_hs'],
                        'hybridization': atom['hybridization'],
                        'formal_charge': atom['formal_charge'],
                        'num_radical_electrons': atom['num_radical_electrons'],
                        'aromatic': atom['aromatic'],
                    }
                    fh.write(format_atom_summary(atom_info, atom['atom_index']) + '\n')

            if f.get('first_edges'):
                fh.write('  Bond layout: 22 bond type + 1 ring + 1 same_ring\n')
                fh.write('  First 20 edges (decoded):\n')
                for edge in f['first_edges']:
                    fh.write(
                        f"    edge {edge['edge_index']}: {edge['src']}->{edge['dst']}, "
                        f"bond_type={edge['bond_type']}, is_in_ring={edge['is_in_ring']}, same_ring={edge['same_ring']}\n"
                    )

        fh.write('\nAggregate summary:\n')
        fh.write(f"Total graphs: {len(per_file)}\n")
        if node_counts:
            fh.write(f"Nodes -- min: {min(node_counts)}, max: {max(node_counts)}, mean: {statistics.mean(node_counts):.2f}, median: {statistics.median(node_counts)}\n")
        if undirected_edges:
            fh.write(f"Undirected edges -- min: {min(undirected_edges)}, max: {max(undirected_edges)}, mean: {statistics.mean(undirected_edges):.2f}\n")
        fh.write(f"Feature dimensions present: {feat_dims}\n")

        # Top 10 largest graphs by nodes
        top10 = sorted([f for f in per_file if 'num_nodes' in f], key=lambda x: x['num_nodes'], reverse=True)[:10]
        fh.write('\nTop graphs by node count:\n')
        for f in top10:
            fh.write(f"{os.path.basename(f['path'])}: nodes={f['num_nodes']}, undirected_edges={f['undirected_edges']}\n")

    print(f"Wrote summary to {args.out}")


if __name__ == '__main__':
    main()
