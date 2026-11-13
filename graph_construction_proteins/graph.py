import torch
from torch_geometric.data import Data
protein = torch.load("graph_construction_proteins/hsa_graphs/hsa_10.pt" , weights_only=False)
protein_generated = torch.load("graphproteins/P11245.pt" , weights_only=False)

graph = protein["protein_graph"]

origin_data = Data(x=graph.x, edge_index=graph.edge_index )
gen_data = Data(x=protein_generated.x, edge_index=protein_generated.edge_index )
print ("orginal Self loop edges: ", origin_data.has_self_loops())
print("Generated Self loop edges: ", gen_data.has_self_loops())

# edges_original = graph.edge_index
# edges_generated = protein_generated.edge_index

# print("Compare between the original and generated edges:")
# for edge in edges_original.t().tolist():
#     if edge not in edges_generated.t().tolist():
#         print(f"Edge {edge} is in the original graph but not in the generated graph.")
