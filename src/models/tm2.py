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

        # Create tensor from numpy arrays
        indices = torch.from_numpy(indices).type(torch.LongTensor).to(self.gpu)
        values = torch.from_numpy(values).type(torch.FloatTensor).to(self.gpu)
        self.edge_index = indices
        self.edge_attr = values
        self.adj_mx = normalize_adjacency_matrix(adj_mx)
        self.adj_mx = torch.FloatTensor(adj_mx.toarray()).to(self.gpu)  # Ensuring it is a tensor

        self.h_dim = h_dim
        self.init_dim = init_dim

        self.gmm1 = GMMConv(self.init_dim, 192, dim=1, kernel_size=3).to(self.gpu)
        self.gmm2 = GMMConv(192, 96, dim=1, kernel_size=1).to(self.gpu)

        self.mlp = nn.Sequential(
            nn.Linear(192, self.node_num),
            nn.ReLU(),
            nn.Linear(self.node_num, self.node_num)
        )
        self.lstm = nn.LSTM(input_size=96, hidden_size=192,
                            num_layers=layer,
                            batch_first=True,
                            dropout=dropout)

        self.end_linear1 = nn.Linear(192, 1536)
        self.end_linear2 = nn.Linear(1536, self.horizon)

    def forward(self, x, edge_index=None, edge_attr=None):
        b, t, n, f = x.shape
        if edge_index is None or edge_attr is None:
            edge_index, edge_attr = self.edge_index, self.edge_attr.unsqueeze(-1)
        all_pooled_features = []  # This will store features from all batches and timesteps
        # Initialize storage for each timestep's pooled output
        mc_losses = []
        o_losses = []
        for time_step in range(t):
            xt = x[:, time_step, :, :].reshape(b * n, f)
            xt = F.relu(self.gmm1(xt, edge_index, edge_attr))
            xt = xt.view(b, n, -1)

            S_logits = self.mlp(xt.view(b * n, -1))
            S = torch.softmax(S_logits, dim=1).view(b, n, -1)
            batch_pooled_features = []  # To store pooled features for this timestep across all batches

            for i in range(b):
                xt_i = xt[i].unsqueeze(0)
                S_i = S[i].unsqueeze(0)
                adj_i = self.adj_mx.unsqueeze(0)  # Unsqueeze to add batch dimension
                xi_pool, adj_pool, mc_loss, o_loss = dense_mincut_pool(xt_i, adj_i, S_i)
                # print(xi_pool.shape)
                batch_pooled_features.append(xi_pool.squeeze(0))  # Remove batch dim from pooled features
                mc_losses.append(mc_loss)
                o_losses.append(o_loss)
            # print(len(pooled_features_list))
            all_pooled_features.append(torch.stack(batch_pooled_features))

        lstm_input = torch.cat(all_pooled_features, dim=0)  # Shape should be (b*t, n, feature_dim)
        # print(lstm_input.shape)
        # Reshape for GMM2 processing
        lstm_input = lstm_input.reshape(-1, lstm_input.size(-1))  # Flatten for GMM2
        # print(lstm_input.shape)
        lstm_input = self.gmm2(lstm_input, edge_index, edge_attr)
        lstm_input = F.relu(lstm_input).view(-1, 96)
        # print(f"new_x shape post-GMM2: {lstm_input.shape}")
        lstm_input = lstm_input.view(b * n, t, -1).squeeze()  # Correctly align sequences with num_clusters
        # print(f"checking logic: {lstm_input.shape}")

        lstm_output, _ = self.lstm(lstm_input)
        # print(f"lstm output : {lstm_output.shape}")

        final_output = lstm_output[:, -1, :]
        # print(f"lstm no time output : {final_output.shape}")

        x = F.relu(self.end_linear1(final_output))
        # print(f"linear layer 1 t : {x.shape}")

        x = self.end_linear2(x)
        # print(f"linear layer 2 : {x.shape}")

        x = x.reshape(b, n, t, 1).transpose(1, 2)  # Reshape to (b, n, t, 1) and then transpose n and t
        # print(f"final layer : {x.shape}")
        mc_loss = torch.stack(mc_losses).mean() if mc_losses else torch.tensor(0)
        o_loss = torch.stack(o_losses).mean() if o_losses else torch.tensor(0)
        return x ,mc_loss, o_loss  # Include mc_loss and o_loss in the return statement


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

    # Sum the connections for each node: degree of nodes
    rowsum = np.array(adj.sum(1))

    # Compute D^(-1/2)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # handle division by zero in isolated nodes
    d_mat_inv_sqrt = diags(d_inv_sqrt)  # Create a diagonal matrix D^(-1/2)

    # Compute the normalized adjacency matrix: D^(-1/2) * A * D^(-1/2)
    norm_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    return norm_adj.tocoo()  # ensure it returns in COO format
