import numpy as np
import random
import numpy as np
import torch
from gym import Env
from gym.spaces import Box, Discrete
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, dense_mincut_pool, GCNConv
import torch.optim as optim
from abc import abstractmethod
from src.base.model import BaseModel
from src.utils.graph_algo import normalize_adj_mx


class GNNModel(BaseModel):
    def __init__(self, init_dim, hid_dim, end_dim, layer, dropout, num_heads, num_clusters, num_tilings, filter_type,
                 tiles_per_tiling, adj_mx, **args):
        super(GNNModel, self).__init__(**args)
        # Existing GNNModel initialization code
        self.env = TrafficNetworkEnv(num_actions=10, state_space_bounds=[(-1, 1), (-1, 1), (-1, 1)])  # Updated bounds
        self.ddqn_agent = DoubleQLearningAgent(state_size=self.env.observation_space.shape[0],
                                               action_size=self.env.action_space.n)
        # Additional initialization code for the GNNModel
        # Initialize the Adaptive Tile Coding Feature Transformer
        self.tile_coding = AdaptiveTileCodingFeatureTransformer(num_tilings, tiles_per_tiling,
                                                                [(-1, 1), (-1, 1), (-1, 1)])  # Updated bounds

        # self.assignment_matrix = AssignmentMatrix(self.input_dim, num_clusters)
        self.adj_mx = normalize_adj_mx(adj_mx, adj_type='symadj', return_type='coo')
        self.conv = nn.Conv2d(in_channels=self.input_dim, out_channels=self.input_dim, kernel_size=(1, 1))

        self.lin1 = nn.Linear(self.input_dim, self.output_dim)
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, 1), bias=True)

        self.assignment_matrix = nn.Linear(self.output_dim,
                                           num_clusters)  # To compute assignments for dense_minicut_pool
        self.gpu = torch.device('cuda:0')
        self.start_conv = nn.Conv2d(in_channels=1, out_channels=init_dim, kernel_size=(1, 1))

        self.lstm = nn.LSTM(input_size=init_dim, hidden_size=hid_dim, num_layers=layer, batch_first=True,
                            dropout=dropout)

        self.end_linear1 = nn.Linear(hid_dim, end_dim)
        self.end_linear2 = nn.Linear(end_dim, self.horizon)

    def forward(self, _X, _edge_ix=None):  # (b, t, n, f)
        batch_size, features, num_nodes, time_steps = _X.shape

        adj_tensor = None  # Default to None
        # Try initializing adj_tensor from self.adj_mx if _edge_ix is None
        if _edge_ix is None and len(self.adj_mx) > 0:
            try:
                adj_dense = self.adj_mx[1].todense()  # Convert COO to dense
                adj_tensor = torch.Tensor(adj_dense).to(self.gpu)
            except AttributeError:
                print("self.adj_mx may not be in the expected format. Ensure it's a COO matrix.")

        # Fallback or default initialization for adj_tensor
        if adj_tensor is None:
            # print("Using default identity matrix for adj_tensor.")
            identity_matrix = np.eye(_X.size(1))
            adj_tensor = torch.Tensor(identity_matrix).to(self.gpu)
            # print(f"adj_tensor1{adj_tensor.shape}")
        x = _X.transpose(1, 3)
        b, f, n, t = x.shape
        x = x.transpose(1, 2).reshape(b * n, f, 1, t)
        # print(f"x1: {x.shape}") # x1: torch.Size([22912, 3, 1, 12])

        x = self.conv(x).squeeze().transpose(1, 2)  # x2: torch.Size([22912, 12, 3])

        # print(f"x2: {x.shape}")

        h = F.relu(self.lin1(x))

        # print(f"h shape {h.shape}")

        S = F.softmax(self.assignment_matrix(h), dim=1)
        # print(f"S shape {S.shape}")

        # Proceed to call dense_mincut_pool
        new_h, new_adj, mc_loss, o_loss = dense_mincut_pool(h, adj_tensor, S)

        transformed_new_h = new_h.view(-1, 1, 32, 1)  # Adjusting dimensions to fit the conv layer's expectations

        # Now, pass the transformed_new_h to the convolutional layer
        pred = self.end_conv(transformed_new_h)
        num_nodes = pred.shape[0] // batch_size
        pred = pred.reshape(batch_size, num_nodes, -1, pred.shape[3])
        pred = pred.permute(0, 2, 1, 3)
        pred_reshaped = pred.reshape(32, 12, 32, 716, 1)
        pred_updated = pred_reshaped.mean(dim=2)

        # print(f"pred shape {pred_updated.shape}")
        x = pred_updated.transpose(1, 3)
        b, f, n, t = x.shape

        x = x.transpose(1, 2).reshape(b * n, f, 1, t)
        x = self.start_conv(x).squeeze().transpose(1, 2)

        out, _ = self.lstm(x)
        x = out[:, -1, :]

        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)
        x = x.reshape(b, n, t, 1).transpose(1, 2)

        return x,  mc_loss, o_loss

    def prepare_edge_index(self, adj_mx):
      
        row, col = adj_mx.row, adj_mx.col
        edge_index = torch.tensor([row, col], dtype=torch.long)
        return edge_index


class AssignmentMatrix(nn.Module):
    def __init__(self, num_features, num_clusters):
        super(AssignmentMatrix, self).__init__()
        # Linear layer that transforms node features to cluster assignments
        self.assignment = nn.Linear(num_features, num_clusters)

    def forward(self, x):
        # Apply the linear transformation
        assignments = self.assignment(x)
        # Use softmax to get probabilities that sum to 1 across clusters for each node
        S = F.softmax(assignments, dim=1)
        return S


class AdaptiveTileCodingFeatureTransformer:
    def __init__(self, initial_num_tilings, initial_tiles_per_tiling, state_space_bounds):
        self.dim = None
        self.num_tilings = initial_num_tilings
        self.initial_tiles_per_tiling = initial_tiles_per_tiling
        self.state_space_bounds = state_space_bounds
        self.adaptation_rate = 0.01  #  tiles  adapted
        self.tile_width = [(bound[1] - bound[0]) / (initial_tiles_per_tiling - 1) for bound in state_space_bounds]
        self.tiles_per_tiling = [initial_tiles_per_tiling for _ in state_space_bounds]
        self.offsets = self.calculate_offsets()

    def calculate_offsets(self):
        offsets = []
        for i, width in enumerate(self.tile_width):
            offset = width / self.num_tilings * np.arange(self.num_tilings)
            offsets.append(offset)
        return offsets

    def adapt_tiles(self, states):
        if states.size == 0:
            print("Warning: 'states' is empty.")
            return
            #  variance calculation
        state_vars = np.var(states, axis=0)

        for i, var in enumerate(state_vars):
            if var > 0.9:  # thresholds to be defined based on the problem
                self.tiles_per_tiling[i] += self.adaptation_rate
            elif var < 0.2:
                self.tiles_per_tiling[i] = max(self.initial_tiles_per_tiling,
                                               self.tiles_per_tiling[i] - self.adaptation_rate)
        self.tile_width = [(bound[1] - bound[0]) / (tiles - 1) for bound, tiles in
                           zip(self.state_space_bounds, self.tiles_per_tiling)]
        self.offsets = self.calculate_offsets()

    def transform(self, states):
        # verify that states is a numpy array with expected shape
        assert isinstance(states, np.ndarray), "'states' must be a numpy array."
        assert len(states.shape) == 2, "'states' must be 2-dimensional."

        """Transforms states into adaptively tile-coded features."""
        # update tiles based on the current state
        self.adapt_tiles(states)

        # calculate the dimensions of the feature vector
        self.dim = sum(self.tiles_per_tiling) * self.num_tilings

        states = np.array(states)
        features = np.zeros((states.shape[0], self.dim))

        feature_index = 0
        for tiling in range(self.num_tilings):
            for dim, (low, high) in enumerate(self.state_space_bounds):
                scaled_states = (states[:, dim] - low) / self.tile_width[dim]
                scaled_states += self.offsets[dim][tiling]
                tile_indices = np.floor(scaled_states).astype(int)
                tile_indices = np.clip(tile_indices, 0, self.tiles_per_tiling[dim] - 1)
                indices = feature_index + tile_indices
                for i, index in enumerate(indices):
                    features[i, index] = 1
                feature_index += self.tiles_per_tiling[dim]

        return torch.tensor(features, dtype=torch.float32)


class TrafficNetworkEnv(Env):
    def __init__(self, num_actions, state_space_bounds):
        super().__init__()
        self.action_space = Discrete(num_actions)
        self.observation_space = Box(low=np.array([bound[0] for bound in state_space_bounds]),
                                     high=np.array([bound[1] for bound in state_space_bounds]))
        # initalize state
        self.state = self.reset()

    def step(self, action):
        # apply action to the environment
        # need to implement logic to apply the action and update the environment's state

        # obtain the new state as graph data
        new_state_graph_data = self._get_new_state_graph_data()

        # calc reward based on the new state or the action taken
        reward = self._calculate_reward(new_state_graph_data)

        # chk if the episode is done
        done = self._check_done()

        # return additional info
        info = {}

        # instead of returning gnn_output, return the raw graph data
        return new_state_graph_data, reward, done, info

    def reset(self):
        initial_state_graph_data = self._get_initial_state_graph_data()
        return initial_state_graph_data

    def render(self, mode='human'):
        pass

    def _get_new_state_graph_data(self):
        pass

    def _calculate_reward(self, new_state_graph_data):
        pass

    def _check_done(self):
        pass

    def _get_initial_state_graph_data(self):
        pass


class DoubleQLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, gamma=0.95, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        self.learning_rate = learning_rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q1 = np.zeros((state_size, action_size))
        self.Q2 = np.zeros((state_size, action_size))

    def select_action(self, state):
        """Selects an action using an epsilon-greedy policy based on the mean Q-values."""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        else:

            mean_Q = (self.Q1[state] + self.Q2[state]) / 2

            return np.argmax(mean_Q)

    def update_model(self, state, action, reward, next_state, done):
        """Updates the Q-tables using the Double Q-learning algorithm."""
        if np.random.rand() < 0.5:
            # Update Q1
            next_action = np.argmax(self.Q1[next_state])
            td_target = reward + self.gamma * self.Q2[next_state][next_action] * (not done)
            td_error = td_target - self.Q1[state][action]
            self.Q1[state][action] += self.learning_rate * td_error
        else:
            # Update Q2
            next_action = np.argmax(self.Q2[next_state])
            td_target = reward + self.gamma * self.Q1[next_state][next_action] * (not done)
            td_error = td_target - self.Q2[state][action]
            self.Q2[state][action] += self.learning_rate * td_error

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, Q1, Q2):
        self.Q1 = Q1
        self.Q2 = Q2

    def save(self):
        return self.Q1, self.Q2

    def get_q_values(self, state):
        q_values_from_Q1 = self.Q1[state]
        q_values_from_Q2 = self.Q2[state]
        return (q_values_from_Q1 + q_values_from_Q2) / 2

        # self.save()

"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import sys



class GaussianMixtureGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, K):
        super(GaussianMixtureGNN, self).__init__(aggr='add')  # Using add aggregation
        self.K = K  # Number of Gaussian mixtures
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.weight = torch.nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        self.linear.reset_parameters()

    def forward(self, x, edge_index):
        x = self.linear(x)
        x = self.propagate(edge_index, x=x, size=None)  # Message passing
        if x.dim() >2:
            x = x.mean(dim=0)
        return x

    def message(self, x_j):
        return torch.matmul(x_j, self.weight)

    def update(self, aggr_out):
        return F.relu(aggr_out)

    def fit(self, data, epochs=100, info=False):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])

            loss.backward()
            optimizer.step()
            if epoch % 20 == 0 and info:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
                sys.stderr.write(f"\r{epoch:02d}/{epochs:02d} | Loss: {loss:<6.2f} | Tr Acc: {acc*100:3.2f}% | "
                                 f"Val Loss: {val_loss:<6.2f} | Val Acc: {val_acc*100:3.2f}%")
                sys.stderr.flush()
    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc

"""
