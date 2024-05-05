import torch
import torch.nn as nn
from torch_geometric.nn import GMMConv, dense_mincut_pool
import torch.nn.functional as F
from src.utils.graph_algo import normalize_adj_mx
from src.base.model import BaseModel
import numpy as np
from scipy.sparse import coo_matrix
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import inv


class GMGNN(BaseModel):
    def __init__(self, init_dim, h_dim, adj_mx, num_clusters, end_dim, layer, dropout, **args):
        super(GMGNN, self).__init__(**args)
        self.gpu = torch.device('cuda:0')
        if not isinstance(adj_mx, coo_matrix):
            adj_mx = coo_matrix(adj_mx)  # Ensure it is in COO format
        adj_mx = coo_matrix(adj_mx)  # Ensure it is in COO format
        # Convert the COO formatted adjacency matrix to PyTorch tensors
        indices = np.vstack((adj_mx.row, adj_mx.col))
        values = adj_mx.data
        self.start_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 1))

        # Create tensor from numpy arrays
        indices = torch.from_numpy(indices).type(torch.LongTensor).to(self.gpu)
        values = torch.from_numpy(values).type(torch.FloatTensor).to(self.gpu)
        self.edge_index = indices
        self.edge_attr = values
        self.adj_mx = normalize_adjacency_matrix(adj_mx)
        self.adj_mx = torch.FloatTensor(adj_mx.toarray()).to(self.gpu)

        self.h_dim = h_dim
        self.init_dim = init_dim
        self.update_func = UpdateFunction(768, 32)  # dimensionality of node features

        self.mlp = nn.Sequential(
            nn.Linear(self.horizon, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=128,
                            num_layers=layer,
                            batch_first=True,
                            dropout=dropout)

        self.end_linear1 = nn.Linear(128, 512)
        self.end_linear2 = nn.Linear(512, self.horizon)

    def forward(self, x, edge_index=None, edge_attr=None):
        b, t, n, f = x.shape
        x = x.transpose(1, 2).reshape(b * n, f, 1, t)
        xt = self.start_conv(x).squeeze().transpose(1, 2)

        S_logits = self.mlp(x)

        S = torch.softmax(S_logits, dim=1).view(b, n, -1)
        # Need to match dimensions for dense_mincut_pool: expected (batch, nodes, features)
        xt = xt.reshape(b, n, -1)

        new_x, new_adj, mc_loss, o_loss = dense_mincut_pool(xt, self.adj_mx, S)

        new_x = probabilistic_aggregation(new_x, new_adj, self.update_func)

        expanded_features = torch.matmul(S, new_x)

        lstm_input = expanded_features.view(-1, 1, expanded_features.size(2))

        lstm_output, _ = self.lstm(lstm_input)

        final_output = lstm_output[:, -1, :]

        x = torch.relu(self.end_linear1(final_output))

        x = self.end_linear2(x)

        x = x.reshape(b, n, t, 1).transpose(1, 2)

        return x, mc_loss, o_loss


class UpdateFunction(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(UpdateFunction, self).__init__()
        self.layer1 = nn.Linear(feature_dim, output_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(output_dim, output_dim)

    def forward(self, x, aggregated_features):
        # combine current features and aggregated features
        combined_features = torch.cat([x, aggregated_features], dim=-1)
        # pass through the layers
        x = self.layer1(combined_features)
        x = self.activation(x)
        x = self.layer2(x)
        return x


def probabilistic_aggregation(x, adj, update_func):
    """
    x: Node features [batch_size, num_nodes, feature_dim]
    adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
    update_func: An instance of UpdateFunction or any nn.Module
    """
    # Verify shapes of input tensors
    assert x.dim() == 3, "x must be a 3D tensor"
    assert adj.dim() == 3, "adj must be a 3D tensor"
    assert x.size(0) == adj.size(0), "Batch size mismatch between x and adj"
    assert x.size(1) == adj.size(1), "Number of nodes mismatch between x and adj"
    assert x.size(1) == adj.size(2), "Number of nodes mismatch between x and adj"

    # aggregate features using the adjacency matrix
    aggregated_features = torch.bmm(adj, x) / adj.sum(dim=2, keepdim=True).clamp(min=1)

    # update node states by considering both current and aggregated features
    updated_features = update_func(x, aggregated_features)

    return updated_features


def manual_dense_to_sparse(new_adj, device):
    b, n, _ = new_adj.shape
    edges = []
    edge_attrs = []

    for i in range(b):
        for j in range(n):
            for k in range(n):
                if new_adj[i, j, k] != 0:
                    edges.append([j, k])
                    edge_attrs.append([new_adj[i, j, k]])  # list to maintain 2nd dim

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    edge_attr = torch.tensor(edge_attrs, dtype=new_adj.dtype).to(device)  # Shape will be (E, 1)

    return edge_index, edge_attr


def normalize_adjacency_matrix(adj):
    """
    Normalize the adjacency matrix using symmetric normalization (D^(-1/2) * A * D^(-1/2)).

    Args:
    adj (scipy.sparse.coo_matrix): Sparse representation of the adjacency matrix.

    Returns:
    scipy.sparse.coo_matrix: Normalized adjacency matrix.
    """
    if not isinstance(adj, coo_matrix):
        adj = coo_matrix(adj)  # Convert to COO format if not already

    # sum the connections for each node: degree of nodes
    rowsum = np.array(adj.sum(1))

    # Compute D^(-1/2)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # handle division by zero in isolated nodes
    d_mat_inv_sqrt = diags(d_inv_sqrt)  # Create a diagonal matrix D^(-1/2)

    # Compute the normalized adjacency matrix: D^(-1/2) * A * D^(-1/2)
    norm_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    return norm_adj.tocoo()  # ensure it returns in COO format
