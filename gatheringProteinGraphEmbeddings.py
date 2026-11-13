import torch
import os
from pathlib import Path
from tqdm import tqdm


def convert_distance_dict_to_tensor(distance_dict, edge_index):
    """
    distance_dict: {(i,j): tensor(distance_dim,)}
    edge_index   : (2, num_edges)
    Returns      : (num_edges, distance_dim)
    """
    src_list  = edge_index[0].tolist()
    dst_list  = edge_index[1].tolist()
    num_edges = len(src_list)
    distance_dim = next(iter(distance_dict.values())).shape[0]

    distance_tensor = torch.zeros(num_edges, distance_dim)
    for idx, (i, j) in enumerate(zip(src_list, dst_list)):
        if (i, j) in distance_dict:
            distance_tensor[idx] = distance_dict[(i, j)]
        elif (j, i) in distance_dict:
            distance_tensor[idx] = distance_dict[(j, i)]
        # else: stays zero (missing edge)

    return distance_tensor  # (num_edges, distance_dim)


def convert_angular_dict_to_tensor(angular_dict, edge_index):
    """
    angular_dict: {(i,j): [(neighbor_idx, tensor(angle_dim,)), ...]}
    edge_index  : (2, num_edges)
    Returns     : (num_pairs, angle_dim)
    """
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()

    angle_rows = []
    for (i, j) in zip(src_list, dst_list):
        neighbors = angular_dict.get((i, j), angular_dict.get((j, i), []))
        for (_, angle_feat) in neighbors:
            if isinstance(angle_feat, torch.Tensor):
                angle_rows.append(angle_feat)
            else:
                angle_rows.append(torch.tensor(angle_feat, dtype=torch.float32))

    if angle_rows:
        return torch.stack(angle_rows)  # (num_pairs, angle_dim)
    else:
        # infer angle_dim from first available entry
        for v in angular_dict.values():
            if v:
                angle_dim = v[0][1].shape[0] if isinstance(v[0][1], torch.Tensor) else len(v[0][1])
                return torch.zeros(0, angle_dim)
        return torch.zeros(0, 32)  # fallback


def build_preloaded_tensor(
    drug_graphs_dir: str,
    protein_graphs_dir: str,
    embeddings_dir: str,
    output_path: str,
):
    preloaded_data = {
        "drug_graphs": {},
        "protein_graphs": {},
        "distance_embeddings": {},
        "angular_embeddings": {},
    }

    # ── Drug graphs ───────────────────────────────────────────────────────────
    drug_files = list(Path(drug_graphs_dir).glob("*.pt"))
    print(f"Loading {len(drug_files)} drug graphs...")
    for fpath in tqdm(drug_files, desc="Drugs"):
        drug_id = fpath.stem.replace("_drug_graph", "")
        preloaded_data["drug_graphs"][drug_id] = torch.load(fpath, map_location="cpu", weights_only=False)

    # ── Protein graphs ────────────────────────────────────────────────────────
    protein_files = list(Path(protein_graphs_dir).glob("*.pt"))
    print(f"Loading {len(protein_files)} protein graphs...")
    for fpath in tqdm(protein_files, desc="Proteins"):
        hsa_id = fpath.stem
        preloaded_data["protein_graphs"][hsa_id] = torch.load(fpath, map_location="cpu", weights_only=False)

    # ── Embeddings (distance + angular) ──────────────────────────────────────
    emb_files = list(Path(embeddings_dir).glob("*.pt"))
    print(f"Loading {len(emb_files)} embedding files...")
    missing_keys, no_graph, failed = [], [], []

    for fpath in tqdm(emb_files, desc="Embeddings"):
        hsa_id = fpath.stem.replace("_embeddings", "")
        emb = torch.load(fpath, map_location="cpu", weights_only=False)

        if "distance_embeddings" not in emb or "angular_embeddings" not in emb:
            missing_keys.append(str(fpath))
            continue

        protein_graph = preloaded_data["protein_graphs"].get(hsa_id)
        if protein_graph is None:
            no_graph.append(hsa_id)
            continue

        try:
            preloaded_data["distance_embeddings"][hsa_id] = convert_distance_dict_to_tensor(
                emb["distance_embeddings"], protein_graph.edge_index
            )
            preloaded_data["angular_embeddings"][hsa_id] = convert_angular_dict_to_tensor(
                emb["angular_embeddings"], protein_graph.edge_index
            )
        except Exception as e:
            failed.append((hsa_id, str(e)))

    # ── Warnings ──────────────────────────────────────────────────────────────
    if missing_keys:
        print(f"\n⚠️  {len(missing_keys)} files missing distance/angular keys:")
        for p in missing_keys: print(f"   {p}")
    if no_graph:
        print(f"\n⚠️  {len(no_graph)} embeddings skipped (no matching protein graph):")
        for h in no_graph: print(f"   {h}")
    if failed:
        print(f"\n❌  {len(failed)} conversions failed:")
        for h, e in failed: print(f"   {h}: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Build summary ──────────────────────────────────────────────")
    print(f"  drug_graphs      : {len(preloaded_data['drug_graphs'])}")
    print(f"  protein_graphs   : {len(preloaded_data['protein_graphs'])}")
    print(f"  distance_embeddings : {len(preloaded_data['distance_embeddings'])}")
    print(f"  angular_embeddings  : {len(preloaded_data['angular_embeddings'])}")

    protein_ids  = set(preloaded_data["protein_graphs"].keys())
    distance_ids = set(preloaded_data["distance_embeddings"].keys())
    only_graph   = protein_ids - distance_ids
    only_emb     = distance_ids - protein_ids
    if only_graph:
        print(f"\n⚠️  {len(only_graph)} proteins have a graph but NO embedding: {list(only_graph)[:5]} ...")
    if only_emb:
        print(f"⚠️  {len(only_emb)} proteins have an embedding but NO graph:   {list(only_emb)[:5]} ...")

    # ── Tensor shape sample ───────────────────────────────────────────────────
    if preloaded_data["distance_embeddings"]:
        sample_id = next(iter(preloaded_data["distance_embeddings"]))
        print(f"\n  Sample distance_tensor [{sample_id}]: {preloaded_data['distance_embeddings'][sample_id].shape}")
        print(f"  Sample angular_tensor  [{sample_id}]: {preloaded_data['angular_embeddings'][sample_id].shape}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n💾 Saving...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(preloaded_data, output_path)
    print(f"✅  Saved → {output_path}")


if __name__ == "__main__":
    build_preloaded_tensor(
        drug_graphs_dir    = "drug_graphs_construction/graphs",
        protein_graphs_dir = "graph_construction_proteins/graphproteins",
        embeddings_dir     = "processed_graphs_embeddings",
        output_path        = "final/full_data_preloaded_Tensor.pt",
    )