from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax

# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
#         super(DecoderRNN, self).__init__()

#         # define the properties
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.vocab_size = vocab_size

#         # lstm cell
#         self.lstm = nn.Sequential(nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size))
#         for _ in range(num_layers - 1):
#             self.lstm.append(nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size))

#         # output fully connected layer
#         self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

#         # embedding layer
#         self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)

#         # activations
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, features, captions):
#         # batch size
#         batch_size = features.size(0)

#         # init the hidden and cell states to zeros
#         hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda()
#         cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()

#         # define the output tensor placeholder
#         outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).cuda()

#         # embed the captions
#         captions_embed = self.embed(captions)

#         # first step with embedding as input
#         hidden_state, cell_state = self.lstm(features, (hidden_state, cell_state))

#         # for the 2nd+ time step, using teacher forcer
#         for t in range(1, captions.size(1)):
#             hidden_state, cell_state = self.lstm_cell(captions_embed[:, t, :], (hidden_state, cell_state))

#             # output of the attention mechanism
#             out = self.fc_out(hidden_state)

#             # build the output tensor
#             outputs[:, t, :] = out

#         return outputs


class FFNN(nn.Module):
    def __init__(self, input_dim=512, output_dim=128, hidden_dim=256, dropout=0.3):
        super(FFNN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class LSTM(nn.Module):
    def __init__(self, size_vocab=64, hidden_dim=512, layers=2, dropout=0.3):
        super(LSTM, self).__init__()
        self.n_layers = layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            dropout=dropout,
        )
        self.norm_1 = nn.LayerNorm(hidden_dim, eps=0.001)
        self.norm_2 = nn.LayerNorm(hidden_dim, eps=0.001)
        self.fnn = nn.Linear(hidden_dim, size_vocab)

    def reset_parameters(self):
        for m in [
            self.fnn.weight,
            self.lstm.weight_ih_l0,
            self.lstm.weight_ih_l1,
            self.lstm.weight_hh_l0,
            self.lstm.weight_hh_l1,
        ]:
            glorot(m)

    def forward(self, i, h):
        f = self.norm_1(i)
        f, h = self.lstm(f, h)
        f = self.fnn(self.norm_2(f)).squeeze(0)
        return f, h


class GATEConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr="add", node_dim=0)

        self.dropout = dropout
        self.att_l = Parameter(torch.empty(1, out_channels))
        self.att_r = Parameter(torch.empty(1, in_channels))
        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.att_l, self.att_r, self.lin1.weight, self.lin2.weight]:
            glorot(layer)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # edge_updater_type: (x: Tensor, edge_attr: Tensor)
        alpha = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)

        # propagate_type: (x: Tensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out + self.bias
        return out

    def edge_update(
        self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]
    ) -> Tensor:
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha = (x_j @ self.att_l.t()).squeeze(-1) + (x_i @ self.att_r.t()).squeeze(-1)
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        return F.dropout(alpha, p=self.dropout, training=self.training)

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return self.lin2(x_j) * alpha.unsqueeze(-1)


class AttentiveFP(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        out_n: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.out_n = out_n
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim, dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(
                hidden_channels, hidden_channels, dropout=dropout, add_self_loops=False, negative_slope=0.01
            )
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(
            hidden_channels, hidden_channels, dropout=dropout, add_self_loops=False, negative_slope=0.01
        )
        self.mol_conv.explain = False  # Cannot explain global pooling.
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for m in [self.lin1, self.gate_conv, self.gru, self.mol_conv, self.mol_gru, self.lin2]:
            m.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor) -> Tensor:
        """"""
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin2(out)
        return torch.cat([out.unsqueeze(0)] * self.out_n, dim=0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"out_channels={self.out_channels}, "
            f"edge_dim={self.edge_dim}, "
            f"num_layers={self.num_layers}, "
            f"num_timesteps={self.num_timesteps}"
            f")"
        )
