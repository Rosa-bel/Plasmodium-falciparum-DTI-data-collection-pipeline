import torch
from rdkit import Chem
from torch_geometric.data import Data

# --------------------------------------------------------------------------
# Vocabularies (fixed order, so one-hot indices are stable across all molecules)
# --------------------------------------------------------------------------

# Match the reference graph encoding used in the test graphs: atom types follow
# the periodic table ordering, and hybridization follows the one-hot order used
# by the original reference generator.
ATOM_TYPES = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
]  # 46 elements; matches the reference graph index layout

HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.UNSPECIFIED,
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.SP2D,
    Chem.rdchem.HybridizationType.OTHER,
]  # 9 states, ordered to match the reference graphs

# Match the original graph encoding used by the saved reference graphs.
# In those files, bond-type index 1 is SINGLE, 2 is DOUBLE, and 19 is AROMATIC.
# The vocabulary must begin with ZERO at index 0, and one-hot indices must match.
BOND_TYPES = [
    Chem.rdchem.BondType.ZERO,                      # 0: ZERO
    Chem.rdchem.BondType.SINGLE,                    # 1: SINGLE
    Chem.rdchem.BondType.DOUBLE,                    # 2: DOUBLE
    Chem.rdchem.BondType.TRIPLE,                    # 3: TRIPLE
    Chem.rdchem.BondType.QUADRUPLE,                 # 4: QUADRUPLE
    Chem.rdchem.BondType.QUINTUPLE,                 # 5: QUINTUPLE
    Chem.rdchem.BondType.HEXTUPLE,                  # 6: HEXTUPLE
    Chem.rdchem.BondType.ONEANDAHALF,               # 7: 1.5
    Chem.rdchem.BondType.TWOANDAHALF,               # 8: 2.5
    Chem.rdchem.BondType.THREEANDAHALF,             # 9: 3.5
    Chem.rdchem.BondType.FOURANDAHALF,              # 10: 4.5
    Chem.rdchem.BondType.FIVEANDAHALF,              # 11: 5.5
    Chem.rdchem.BondType.IONIC,                     # 12: IONIC
    Chem.rdchem.BondType.HYDROGEN,                  # 13: HYDROGEN
    Chem.rdchem.BondType.THREECENTER,               # 14: THREECENTER
    Chem.rdchem.BondType.DATIVEONE,                 # 15: DATIVEONE
    Chem.rdchem.BondType.DATIVE,                    # 16: DATIVE
    Chem.rdchem.BondType.DATIVEL,                   # 17: DATIVEL
    Chem.rdchem.BondType.DATIVER,                   # 18: DATIVER
    Chem.rdchem.BondType.AROMATIC,                  # 19: AROMATIC ← matches reference graphs!
    Chem.rdchem.BondType.OTHER,                     # 20: OTHER
    Chem.rdchem.BondType.UNSPECIFIED,               # 21: UNSPECIFIED
]  # 22 bond types, ordered to match the reference graph files


# --------------------------------------------------------------------------
# One-hot helper
# --------------------------------------------------------------------------
def one_hot(value, allowed_set):
    """Returns a one-hot vector; unknown values map to the last bucket if
    you want an explicit 'other' slot — here, unknown -> all zeros unless
    you append an explicit 'UNKNOWN' entry to allowed_set."""
    return [1.0 if value == s else 0.0 for s in allowed_set]

 
# --------------------------------------------------------------------------
# Node (atom) features -- 46 + 11 + 7 + 10 + 9 + 1 + 1 + 1 = 86 dims
# --------------------------------------------------------------------------
def atom_features(atom):
    features = []
    features += one_hot(atom.GetSymbol(), ATOM_TYPES)                     # 46
    features += one_hot(atom.GetDegree(), list(range(11)))                # 11 (0-10)
    features += one_hot(atom.GetImplicitValence(), list(range(7)))        # 7  (0-6)
    features += one_hot(atom.GetTotalNumHs(), list(range(10)))            # 10 (0-9)
    features += one_hot(atom.GetHybridization(), HYBRIDIZATIONS)          # 9
    features.append(float(atom.GetFormalCharge()))                       # 1
    features.append(float(atom.GetNumRadicalElectrons()))                # 1
    features.append(1.0 if atom.GetIsAromatic() else 0.0)                # 1
    return features  # length 86


# --------------------------------------------------------------------------
# Edge (bond) features -- 22 + 1 + 1 = 24 dims
# --------------------------------------------------------------------------
def bond_features(bond, ring_info):
    features = []
    features += one_hot(bond.GetBondType(), BOND_TYPES)                  # 22
    features.append(1.0 if bond.IsInRing() else 0.0)                     # 1
    a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    same_ring = ring_info.AreAtomsInSameRing(a1, a2) if hasattr(
        ring_info, "AreAtomsInSameRing"
    ) else _atoms_share_ring(ring_info, a1, a2)
    features.append(1.0 if same_ring else 0.0)                           # 1
    return features  # length 24


def _atoms_share_ring(ring_info, a1, a2):
    """Fallback for RDKit versions without AreAtomsInSameRing()."""
    for ring in ring_info.AtomRings():
        if a1 in ring and a2 in ring:
            return True
    return False


# --------------------------------------------------------------------------
# SMILES -> PyG Data object
# --------------------------------------------------------------------------
def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # invalid SMILES -- log and skip upstream

    # Node features
    x = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(x, dtype=torch.float)  # [num_atoms, 86]

    # Edge features -- add BOTH directions (i->j and j->i) since molecular
    # graphs are undirected but PyG expects directed edge_index pairs
    ring_info = mol.GetRingInfo()
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond, ring_info)
        edge_index.append([i, j])
        edge_attr.append(feats)
        edge_index.append([j, i])
        edge_attr.append(feats)

    if len(edge_index) == 0:
        # single-atom molecule edge case: no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 24), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)                    # [num_edges, 24]

    # Do not store the raw SMILES string in the saved Data object to keep
    # serialized graph files compact and free of extra string fields.
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# --------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------
if __name__ == "__main__":
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    data = smiles_to_graph(smiles)
    print(data)
    print("Node feature shape:", data.x.shape)       # [num_atoms, 86]
    print("Edge index shape:", data.edge_index.shape)
    print("Edge attr shape:", data.edge_attr.shape)   # [num_edges*2, 24]