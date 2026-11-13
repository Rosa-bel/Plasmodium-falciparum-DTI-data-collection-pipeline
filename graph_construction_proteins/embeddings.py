import os
import torch
from tqdm import tqdm


def distance_embeddings(edge_index, edge_attr, embedding_dim=32):
    """
    Compute Radial Basis Function (RBF) embeddings for edges in a PyTorch Geometric graph,
    and return a dictionary with distances for each edge.

    Args:
        edge_index (torch.Tensor): Tensor of shape (2, num_edges) representing edge connections.
        edge_attr (torch.Tensor): Tensor of shape (num_edges,) containing edge distances.
        embedding_dim (int): Number of radial basis functions (embedding dimension).

    Returns:
        dict: Dictionary where keys are (src, dst) tuples and values are the RBF embeddings (torch.Tensor).
    """
    # Extract edge distances
    edge_distances = edge_attr.squeeze()  # Assuming edge_attr contains distances

    # Calculate centers (uniformly spaced between 0 and max distance)
    max_distance = torch.max(edge_distances)
    centers = torch.linspace(0, max_distance, embedding_dim)

    # Calculate sigma (minimum distance between centers)
    sigma = torch.min(centers[1:] - centers[:-1])
    # Compute RBF embeddings
    rbf_embeddings = torch.exp(-((edge_distances.unsqueeze(1) - centers) ** 2) / (2 * sigma ** 2))

    # Create the dictionary with edge keys and distance values
    edge_keys = edge_index.t().tolist()
    distance_dict = {tuple(edge_key): rbf_embeddings[i] for i, edge_key in enumerate(edge_keys)}

    return distance_dict

def angular_embedding_pyg(data, embedding_dim=32):
    """
    Compute angular embeddings for edges in a PyTorch Geometric protein graph.

    Args:
    data (Data): PyTorch Geometric Data object.
    embedding_dim (int): Number of angular basis function (ABF) dimensions.

    Returns:
    dict: Dictionary mapping edges to lists of tuples.
    """
    coords = data.x[:, :3]
    edge_index = data.edge_index

    angular_embeddings = {}

    for edge_id in range(edge_index.size(1)):
        src, dst = edge_index[:, edge_id]

        src_coord = coords[src]
        dst_coord = coords[dst]

        edge_vector = dst_coord - src_coord

        # Batch neighbor calculations
        neighbors = edge_index[0][(edge_index[1] == src)].unique()
        if neighbors.size(0) == 0:
            continue

        neighbor_coords = coords[neighbors]
        neighbor_vectors = src_coord - neighbor_coords
        angles = torch.arccos(torch.clamp(torch.cosine_similarity(edge_vector.unsqueeze(0), neighbor_vectors), -1.0, 1.0))

        # Apply Angular Basis Functions (ABF)
        abf_list = [torch.cos(n * angles) for n in range(embedding_dim)]
        angle_embeddings = torch.stack(abf_list, dim=-1)

        # Preallocate memory for neighbor embeddings
        neighbor_embeddings = [(neighbor.item(), angle_embedding) for neighbor, angle_embedding in zip(neighbors, angle_embeddings)]

        angular_embeddings[(src.item(), dst.item())] = neighbor_embeddings

    return angular_embeddings

def process_and_store_embeddings(input_dir, output_dir ):
    """
    Process graphs to compute and store distance and angle embeddings without edge_index.

    Parameters:
    - input_dir (str): Directory containing graphs.
    - output_dir (str): Directory to save embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(os.listdir(input_dir), desc="Computing embeddings", unit="graph"):
        if file.endswith(".pt"):
            # Check if the embeddings file already exists
            output_file = file.replace(".pt", "_embeddings.pt")
            output_path = os.path.join(output_dir, output_file)

            if os.path.exists(output_path):
                print(f"Embeddings for {file} already exist. Skipping...")
                continue

            graph_path = os.path.join(input_dir, file)
            protein_graph = torch.load(graph_path , weights_only=False)  # Load graph with weights only (no edge_index)

            # Compute distance embeddings
            distance_embd = distance_embeddings(protein_graph.edge_index, protein_graph.edge_attr)

            # Compute angular embeddings
            angular_embeddings = angular_embedding_pyg(protein_graph)

            # Store embeddings without edge_index
            embeddings = {
                "distance_embeddings": distance_embd,
                "angular_embeddings": angular_embeddings,
            }

            # Save embeddings
            torch.save(embeddings, output_path)


if __name__ == "__main__":
    # Directories
    input_dir_processed = "graph_construction_proteins/graphproteins"
    output_dir_embeddings_prss = "processed_graphs_embeddings"
    # Step 1: Compute and store embeddings for  graphs
    #process_and_store_embeddings(input_dir_graphs, output_dir_embeddings)
    #print("Graph embedding computation for protien graphs completed!")

    # Step 2: Compute and store embeddings for processed graphs
    process_and_store_embeddings(input_dir_processed, output_dir_embeddings_prss)
    print("Graph embedding computation for processed protien graphs completed!")
