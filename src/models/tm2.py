import torch
import torch.nn as nn
from torch_geometric.nn import GMMConv, dense_mincut_pool
import torch.nn.functional as F
from src.utils.graph_algo import normalize_adj_mx
from src.base.model import BaseModel


class GMGNN(BaseModel):
    def __init__(self, input_dim, h_dim, output_dim, adj_mx, num_clusters):
        super(GMGNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_clusters = num_clusters
        self.adj_mx = normalize_adj_mx(adj_mx, adj_type='symadj', return_type='coo')
        # Define MLP for computing S
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, num_clusters)  # K clusters
        )
        self.gmm1 = GMMConv(input_dim, h_dim, dim=1, kernel_size=12)
        self.gmm2 = GMMConv(h_dim, output_dim, dim=1, kernel_size=12)

    def forward(self, x, edge_index): # (b,t, n, f)
        # First GMMConv layer for message passing
        x = self.gmm1(x, edge_index).relu()

        # Compute the cluster assignment matrix S
        S_logits = self.mlp(x)  # No need to reshape x here, MLP should handle the 3D input
        S = F.softmax(S_logits, dim=-1)  # Apply softmax over the last dimension to compute S

        # Identity matrix creation can be simplified with PyTorch and moved to the right device
        identity_matrix = torch.eye(x.size(1), device=x.device)

        # Use the computed S to pool the graph
        new_x, new_adj, mc_loss, o_loss = dense_mincut_pool(x, self.adj_mx, S)

        # Apply the second GMMConv layer to the pooled features
        new_x = self.gmm2(new_x, edge_index)

        # Return the final output, mc_loss, and o_loss for training purposes
        return new_x, mc_loss, o_loss


