import torch
import torch.nn as nn
from torch_geometric.nn import GMMConv, dense_mincut_pool
import torch.nn.functional as F
from src.utils.graph_algo import normalize_adj_mx
from src.base.model import BaseModel



class GMGNN(BaseModel):
    def __init__(self, init_dim, h_dim, adj_mx, num_clusters, **args):
        super(GMGNN, self).__init__(**args)
        self.gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if not adj_mx.is_sparse:
            adj_mx = adj_mx.to_sparse()
        adj_mx = adj_mx.to(self.gpu)

        self.edge_index = adj_mx.indices()
        self.edge_attr = adj_mx.values().float()

        self.h_dim = h_dim
        self.init_dim = init_dim
        self.num_clusters = num_clusters
        edge_feature_dim = 1  # scalar edge attributes

        #  GMMConv and MLP layers
        self.gmm1 = GMMConv(self.input_dim, h_dim // 2, dim=edge_feature_dim, kernel_size=3)
        self.gmm2 = GMMConv(h_dim // 2, self.output_dim, dim=edge_feature_dim, kernel_size=3)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim // 2, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.num_clusters)
        )

    def forward(self, x, labels=None, edge_index=None, edge_attr=None):
        b, t, n, f = x.shape
        print(f"Initial x shape: {x.shape}")  # [16, 12, 716, 3]

        if edge_index is None:
            edge_index = self.edge_index
        if edge_attr is None:
            edge_attr = self.edge_attr.unsqueeze(-1)

        x = x.permute(0, 3, 1, 2).contiguous()  # [16, 3, 12, 716]
        x = x.reshape(b * n * t, f)  # [137472, 3]
        print(f"Reshaped x for GMM1: {x.shape}")


        # perform  operation
        x = self.gmm1(x, edge_index, edge_attr).relu()  # Applying GMM1
        print(f"Output of GMM1: {x.shape}")

        # output is  [137472, h_dim//2], need to reshape for pooling
        x = x.view(b, t, n, -1).mean(dim=1)  # Reduce back to [b, n, h_dim//2]
        print(f"X reshaped post GMM1: {x.shape}")

        S_logits = self.mlp(x.view(b * n, -1))
        S = F.softmax(S_logits, dim=-1).view(b, n, -1)
        print(f"S_logits and Softmax S shapes: {S_logits.shape}, {S.shape}")

        adj = edge_index_to_adjacency(edge_index, edge_attr.squeeze(), n)
        print(f"Adjacency matrix shape: {adj.shape}")

        new_x, new_adj, mc_loss, o_loss = dense_mincut_pool(x, adj, S)
        print(f"Post-pooling shapes - new_x: {new_x.shape}, new_adj: {new_adj.shape}")

        # flatten for GMMConv compatibility
        new_x = new_x.reshape(-1, new_x.shape[-1])
        print(f"new_x reshaped for GMM2: {new_x.shape}")

        new_x = self.gmm2(new_x, edge_index, edge_attr).relu()
        print(f"Output of GMM2: {new_x.shape}")

        # reshape to match output dimensions (b, t, n, 1)
        new_x = new_x.view(b, t, n, -1)
        print(f"Final new_x shape: {new_x.shape}")

        if labels is not None:
            return new_x, mc_loss, o_loss
        return new_x

def edge_index_to_adjacency(edge_index, edge_attr=None, num_nodes=None):
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    if edge_attr is None:
        edge_attr = torch.ones((edge_index.size(1),), device=edge_index.device)

    # Creating an adjacency matrix
    adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = edge_attr
    return adj

