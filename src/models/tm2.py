import torch
import torch.nn as nn
from torch_geometric.nn import GMMConv, dense_mincut_pool
import torch.nn.functional as F
from src.utils.graph_algo import normalize_adj_mx
from src.base.model import BaseModel


class GMGNN(BaseModel):
    def __init__(self, init_dim, h_dim, adj_mx, num_clusters, end_dim, layer, dropout, device, **args):
        super(GMGNN, self).__init__(**args)
        self.device = device

        if not adj_mx.is_sparse:
            adj_mx = adj_mx.to_sparse()
        adj_mx = adj_mx.to(self.device)

        self.edge_index = adj_mx.indices().to(self.device)
        self.edge_attr = adj_mx.values().float().to(self.device)

        self.h_dim = h_dim
        self.init_dim = init_dim
        self.num_clusters = num_clusters  # Important for dimension calculations

        self.gmm1 = GMMConv(self.init_dim, self.h_dim // 2, dim=1, kernel_size=3).to(self.device)
        self.gmm2 = GMMConv(self.h_dim // 2, self.output_dim, dim=1, kernel_size=3).to(self.device)

        self.mlp = nn.Sequential(
            nn.Linear(self.h_dim // 2, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.num_clusters).to(self.device)
        )
        self.lstm = nn.LSTM(input_size=2, hidden_size=h_dim*self.node_num  , num_layers=layer, batch_first=True, dropout=dropout).to(
            self.device)
        self.lstm.flatten_parameters()  # Ensure parameters are contiguous post-initialization

        self.end_linear1 = nn.Linear(h_dim *self.node_num  , end_dim).to(self.device)
        self.end_linear2 = nn.Linear(end_dim, self.horizon).to(self.device)

    def forward(self, x, edge_index=None, edge_attr=None):
        x = x.to(self.device)
        b, t, n, f = x.shape
        print(f"Initial shape of x: {x.shape}")

        if edge_index is None or edge_attr is None:
            edge_index, edge_attr = self.edge_index, self.edge_attr.unsqueeze(-1)

        # Flatten x for GMM1 processing
        x = x.view(b * t * n, f)
        x = self.gmm1(x, edge_index, edge_attr).to(self.device)
        x = F.relu(x)
        print(f"Shape of x after GMM1: {x.shape}")
        # Reduce dimensionality via standard deviation
        x = x.view(b, t, n, -1).std(dim=1, unbiased=False)
        print(f"Shape of x after std dev reduction: {x.shape}")

        # Generate soft assignment matrix (S)
        S_logits = self.mlp(x.view(b * n, -1))
        S = torch.softmax(S_logits, dim=-1).view(b, n, -1)
        print(f"S logits shape: {S_logits.shape}, S shape: {S.shape}")

        # Compute new adjacency matrix and apply mincut pool
        adj = torch.matmul(x, x.transpose(1, 2))
        print(f"Adjacency matrix shape: {adj.shape}")
        new_x, new_adj, mc_loss, o_loss = dense_mincut_pool(x, adj, S)
        print(f"new_x shape post-pooling: {new_x.shape}, new_adj shape: {new_adj.shape}")
        new_edge_index, new_edge_attr = manual_dense_to_sparse(new_adj)

        # changed from .unsqueeze(-1) which can be an in-place operation
        print(f"new edge index {new_edge_index.shape}")
        print(f"new edge attr {new_edge_attr.shape}")

        # Reshape for GMM2 processing
        new_x = new_x.reshape(-1, new_x.size(-1))
        new_x = self.gmm2(new_x, new_edge_index, new_edge_attr).to(self.device)
        new_x = F.relu(new_x)
        print(f"new_x shape post-GMM2: {new_x.shape}")

        # Reshape for LSTM processing
        new_x = new_x.view(b, t, -1).squeeze()  # Correctly align sequences with num_clusters
        print(f"new_x reshaped for LSTM: {new_x.shape}")

        # LSTM processing
        self.lstm.flatten_parameters()  # Ensure parameters are contiguous post-initialization

        out, _ = self.lstm(new_x)
        print(f"lstm after {out.shape}")
        self.lstm.flatten_parameters()  # Ensure parameters are contiguous post-initialization

        x = out[:, -1, :]  # Taking the last timestep's output
        print(f"After  out shape: {x.shape}")

        x = F.relu(self.end_linear1(x))
        print(f"After end linear 1 shape: {x.shape}")

        x = self.end_linear2(x)
        print(f"After end linear 2 shape: {x.shape}")  

        x = x.view(b, n, t, 1).permute(0, 2, 1, 3)  # Correct output shape

        print(f"After end  shape: {x.shape}")  

        return x


def manual_dense_to_sparse(new_adj):
    b, n, _ = new_adj.shape
    edges = []
    edge_attrs = []

    for i in range(b):
        for j in range(n):
            for k in range(n):
                if new_adj[i, j, k] != 0:
                    edges.append([j, k])
                    edge_attrs.append([new_adj[i, j, k]])  # list to maintain 2nd dim

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(new_adj.device)
    edge_attr = torch.tensor(edge_attrs, dtype=new_adj.dtype).to(new_adj.device)  # Shape will be (E, 1)

    return edge_index, edge_attr
