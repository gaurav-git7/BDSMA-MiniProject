import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE mean aggregation layer.
    h_v = ReLU(W_self * h_v + W_neigh * Mean_{u in N(v)} h_u)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear_self = nn.Linear(in_features, out_features, bias=bias)
        self.linear_neigh = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # adj_norm is row-normalized adjacency matrix (D^-1 A or similar)
        neigh_x = torch.matmul(adj_norm, x)
        out = self.linear_self(x) + self.linear_neigh(neigh_x)
        return out


class GATLayer(nn.Module):
    """
    Graph Attention Layer with LeakyReLU and softmax attention weights.
    """
    def __init__(self, in_features: int, out_features: int, alpha: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x: torch.Tensor, adj_dense: torch.Tensor) -> torch.Tensor:
        # x: (N, in_features)
        h = self.linear(x)  # (N, out_features)
        N = h.size(0)

        # Self-attention mechanism
        # Concatenate [h_i || h_j] for all pairs
        h_i = h.repeat_interleave(N, dim=0)
        h_j = h.repeat(N, 1)
        a_input = torch.cat([h_i, h_j], dim=1).view(N, N, -1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # Mask non-edges
        mask = (adj_dense == 0)
        e = e.masked_fill(mask, -1e9)
        attention = F.softmax(e, dim=1)
        return torch.matmul(attention, h)


class MembershipGNN(nn.Module):
    """
    2-layer GNN encoder mapping [X | S] -> soft community membership matrix F (N x K, >= 0).
    Supported encoder_type: 'sage' (default) or 'gat'.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        encoder_type: str = "sage",
        dropout: float = 0.2
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.dropout = dropout

        if self.encoder_type == "gat":
            self.layer1 = GATLayer(in_dim, hidden_dim)
            self.layer2 = GATLayer(hidden_dim, out_dim)
        else:  # default sage
            self.layer1 = GraphSAGELayer(in_dim, hidden_dim)
            self.layer2 = GraphSAGELayer(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor, adj_dense: torch.Tensor) -> torch.Tensor:
        adj_to_use = adj_dense if self.encoder_type == "gat" else adj_norm

        h = self.layer1(x, adj_to_use)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.layer2(h, adj_to_use)
        # Non-negative membership matrix F via ReLU (as in NOCD)
        F_mat = F.relu(h)
        return F_mat
