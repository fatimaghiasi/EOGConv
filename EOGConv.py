import torch
import torch.nn as nn

#Newest version with edges represented as custom matrixes, not vectors, to represent anisotropy

class EOGConv(nn.Module):
    """
    Edge-Only Graph Convolution with per-endpoint features.

    Each edge e connects two nodes (u, v) and has features:

        edge_x[e, 0, :] -> feature vector for the endpoint attached to edge_index[0, e]
        edge_x[e, 1, :] -> feature vector for the endpoint attached to edge_index[1, e]

    This allows edges to be anisotropic: the two endpoints of an edge can have
    different feature vectors. Symmetric edges can be represented by setting
    both endpoint vectors equal during tensor generation.

    For each edge e = (u, v):

        - At node u we aggregate features from the endpoints of all edges
          that touch u (using the endpoint that is actually attached to u).
        - At node v we do the same.

        left_sum[e]  = aggregated features of other edge endpoints at node u
        right_sum[e] = aggregated features of other edge endpoints at node v

    The new edge feature is then computed as:

        x_new[e] = W_left  * left_sum[e]
                 + W_self  * [x_src[e] || x_dst[e]]
                 + W_right * right_sum[e]

    where:

        x_src[e] = edge_x[e, 0, :]
        x_dst[e] = edge_x[e, 1, :]

    Shapes
    ------
    edge_x    : [E, 2, C_end]
    edge_index: [2, E]
    output    : [E, C_out]
    """

    def __init__(self,
                 in_channels_per_end,
                 out_channels,
                 aggr="sum",
                 directed=False,
                 symmetric=True):
        super().__init__()

        assert aggr in {"sum", "mean"}, "aggr must be 'sum' or 'mean'"
        self.in_channels_per_end = in_channels_per_end
        self.out_channels = out_channels
        self.aggr = aggr
        self.directed = directed
        self.symmetric = symmetric

        C_end = in_channels_per_end

        # Self term: acts on concatenated [x_src || x_dst]
        self.W_self = nn.Parameter(torch.empty(out_channels, 2 * C_end))

        if (not directed) and symmetric:
            # True symmetry: one shared matrix for both sides
            self.W_side = nn.Parameter(torch.empty(out_channels, C_end))
            self.W_left = None
            self.W_right = None
        else:
            # Asymmetric / directed mode: separate matrices
            self.W_side = None
            self.W_left = nn.Parameter(torch.empty(out_channels, C_end))
            self.W_right = nn.Parameter(torch.empty(out_channels, C_end))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_self)
        if self.W_side is not None:
            nn.init.xavier_uniform_(self.W_side)
        if self.W_left is not None:
            nn.init.xavier_uniform_(self.W_left)
        if self.W_right is not None:
            nn.init.xavier_uniform_(self.W_right)

    def forward(self,
                edge_x,
                edge_index,
                edge_batch=None,
                num_nodes=None):

        device = edge_x.device
        shape = edge_x.shape
        if len(shape) != 3:
            raise ValueError(f"edge_x must have 3 dims [E, 2, C_end], got shape {shape}")

        E, two, C_end = shape
        assert two == 2, "edge_x must have shape [E, 2, C_end]"
        assert C_end == self.in_channels_per_end, "in_channels_per_end mismatch"

        src = edge_index[0]  # first endpoint nodes
        dst = edge_index[1]  # second endpoint nodes

        if num_nodes is None:
            num_nodes = int(edge_index.max()) + 1

        # Per-endpoint features
        x_src = edge_x[:, 0, :]   # [E, C_end], endpoint at src
        x_dst = edge_x[:, 1, :]   # [E, C_end], endpoint at dst

        # ----------------------------------------
        # 1) Node-level sums of incident endpoints
        # ----------------------------------------
        node_sum = torch.zeros(num_nodes, C_end,
                               device=device,
                               dtype=edge_x.dtype)

        node_sum.index_add_(0, src, x_src)
        node_sum.index_add_(0, dst, x_dst)

        if self.aggr == "mean":
            deg = torch.zeros(num_nodes,
                              device=device,
                              dtype=edge_x.dtype)
            one = torch.ones(E,
                             device=device,
                             dtype=edge_x.dtype)
            deg.index_add_(0, src, one)
            deg.index_add_(0, dst, one)

        # ----------------------------------------
        # 2) Neighbor sums at each endpoint
        # ----------------------------------------
        left_sum_incl = node_sum[src]   # includes x_src
        right_sum_incl = node_sum[dst]  # includes x_dst

        left_sum = left_sum_incl - x_src
        right_sum = right_sum_incl - x_dst

        if self.aggr == "mean":
            deg_src = deg[src]
            deg_dst = deg[dst]

            left_deg = (deg_src - 1.0).clamp(min=1.0).unsqueeze(-1)
            right_deg = (deg_dst - 1.0).clamp(min=1.0).unsqueeze(-1)

            left_sum = left_sum / left_deg
            right_sum = right_sum / right_deg

        # ----------------------------------------
        # 3) Select weight matrices
        # ----------------------------------------
        if (not self.directed) and self.symmetric:
            W_left = self.W_side
            W_right = self.W_side
        else:
            W_left = self.W_left
            W_right = self.W_right

        # ----------------------------------------
        # 4) Linear transforms + combine
        # ----------------------------------------
        left_msg = left_sum @ W_left.t()        # [E, C_out]
        right_msg = right_sum @ W_right.t()     # [E, C_out]

        self_feat = torch.cat([x_src, x_dst], dim=-1)  # [E, 2*C_end]
        self_msg = self_feat @ self.W_self.t()         # [E, C_out]

        x_new = left_msg + self_msg + right_msg
        return x_new
