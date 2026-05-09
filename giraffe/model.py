#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax

from giraffe.utils import get_input_dims, read_config_ini

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MeanAggregation(nn.Module):
    """Mean pooling over graph nodes."""
    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return global_mean_pool(x, batch)


class SumAggregation(nn.Module):
    """Sum pooling over graph nodes."""
    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return global_add_pool(x, batch)


class AttentionPooling(nn.Module):
    """Attention-based graph pooling (learnable weights per node)."""
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.att = nn.Linear(hidden_channels, 1)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        att_weights = self.att(x).squeeze(-1)  # [N]
        att_weights = F.softmax(att_weights, dim=0)
        # Expand to match x dim for weighted sum
        out = global_add_pool(x * att_weights.unsqueeze(-1), batch)
        return out


class MABMessagePassing(nn.Module):
    """MAB-style message passing base class (inspired by Chemprop).
    
    Separates initialize → message → update → finalize stages for cleaner code.
    """
    def __init__(self, in_channels: int, hidden_channels: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.edge_dim = edge_dim
        self.dropout = dropout
        
        # Input transformation
        self.W_i = Linear(in_channels + edge_dim, hidden_channels)
        # Message transformation
        self.W_h = Linear(hidden_channels, hidden_channels)
        # Output transformation
        self.W_o = Linear(in_channels + hidden_channels, hidden_channels)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def initialize(self, x: Tensor, edge_attr: Tensor) -> Tensor:
        """Initialize hidden states from node and edge features."""
        # Expand x to match edge count for edge-level messages
        return self.act(self.W_i(x))

    def message(self, H: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """Compute messages along edges."""
        src, dst = edge_index
        # Aggregate messages from neighbors
        return H[src]  # [num_edges, hidden]

    def update(self, M: Tensor, H_0: Tensor) -> Tensor:
        """Update hidden states with messages (with optional GRU)."""
        H_t = self.W_h(M)
        return self.act(H_0 + H_t)  # Residual update like Chemprop

    def finalize(self, V: Tensor, M: Tensor) -> Tensor:
        """Finalize node embeddings by concatenating with messages."""
        H = torch.cat([V, M], dim=-1)
        return self.act(self.W_o(H))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """Full forward pass through MAB layers."""
        H_0 = self.initialize(x, edge_attr)
        H = H_0
        
        # Single message passing step (extend in subclass for multiple layers)
        M = self.message(H, edge_index, edge_attr)
        H = self.update(M, H_0)
        
        # Finalize
        out = self.finalize(x, H)
        return self.dropout_layer(out)


class GRUUpdate(nn.Module):
    """GRU-based update (more expressive than residual)."""
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.gru = GRUCell(hidden_channels, hidden_channels)

    def forward(self, message: Tensor, hidden: Tensor) -> Tensor:
        return self.gru(message, hidden).relu()


class ResidualUpdate(nn.Module):
    """Simple residual update (lighter, like Chemprop)."""
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.W_h = Linear(hidden_channels, hidden_channels)
        self.act = nn.ReLU()

    def forward(self, message: Tensor, hidden: Tensor) -> Tensor:
        return self.act(hidden + self.W_h(message))


def load_models(checkpoint, epoch):
    """Helper class to load encoder and decoder models and corresponding config object
    from trained model checkpoint.
    """
    dim_atom, dim_bond = get_input_dims()
    conf = read_config_ini(checkpoint)
    vae = conf["vae"] == "True"

    # Define models
    rnn = LSTM(
        input_dim=conf["alphabet"],
        embedding_dim=conf["dim_tok"],
        hidden_dim=conf["dim_rnn"],
        layers=conf["layers_rnn"],
        dropout=conf["dropout"],
    )

    GNN_Class = AttentiveFP2 if vae else AttentiveFP
    gnn = GNN_Class(
        in_channels=dim_atom,
        hidden_channels=conf["dim_gnn"],
        out_channels=conf["dim_rnn"],
        edge_dim=dim_bond,
        num_layers=conf["layers_gnn"],
        num_timesteps=conf["kernels_gnn"],
        dropout=conf["dropout"],
    )

    gnn.load_state_dict(
        torch.load(os.path.join(checkpoint, f"atfp_{epoch}.pt"), map_location=DEVICE)
    )
    rnn.load_state_dict(
        torch.load(os.path.join(checkpoint, f"lstm_{epoch}.pt"), map_location=DEVICE)
    )
    gnn = gnn.to(DEVICE)
    rnn = rnn.to(DEVICE)

    return gnn, rnn, conf


class FFNN(nn.Module):
    def __init__(
        self, input_dim=512, output_dim=125, hidden_dim=256, n_layers=3, dropout=0.2
    ):
        super(FFNN, self).__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(input_dim)
        layers = (
            [nn.Linear(input_dim, hidden_dim), nn.Dropout(dropout), nn.LeakyReLU()]
            + int(n_layers - 2)
            * [nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout), nn.LeakyReLU()]
            + [nn.Linear(hidden_dim, output_dim)]
        )
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(0, len(self.mlp), 3):
            glorot(self.mlp[i].weight)

    def forward(self, x):
        x_norm = self.layer_norm(x)
        out = self.mlp(x_norm)
        # Safe residual: pad if input too short, else slice
        if x.size(-1) >= self.output_dim:
            residual = x[..., : self.output_dim]
        else:
            residual = F.pad(x, (0, self.output_dim - x.size(-1)))
        return out + residual


class LSTM(nn.Module):
    def __init__(
        self, input_dim=48, embedding_dim=128, hidden_dim=512, layers=2, dropout=0.2
    ):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim
        self.embedding_dim = embedding_dim
        self.n_layers = layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            dropout=dropout,
        )

        self.norm_in = nn.LayerNorm(embedding_dim, eps=0.001)
        self.norm_out = nn.LayerNorm(hidden_dim, eps=0.001)
        self.fcn = nn.Linear(hidden_dim, input_dim)

        # tie embedding and output weights to increase efficiency
        if embedding_dim == hidden_dim:
            self.embedding.weight = self.fcn.weight

        self.reset_parameters()

    def reset_parameters(self):
        for m in [
            self.fcn.weight,
            self.lstm.weight_ih_l0,
            self.lstm.weight_ih_l1,
            self.lstm.weight_hh_l0,
            self.lstm.weight_hh_l1,
        ]:
            glorot(m)

    def forward(self, input, hiddens):
        features = self.embedding(input)
        features = self.norm_in(features)
        features, hiddens = self.lstm(features, hiddens)
        features = self.norm_out(features)
        features = self.fcn(features)
        return features, hiddens


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
        alpha = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out + self.bias
        return out

    def edge_update(
        self,
        x_j: Tensor,
        x_i: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
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
    graph attention mechanisms."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
        undirected: bool = False,
        return_vertex_embeddings: bool = False,
        aggregator: str = "add",  # "add", "mean", "attention"
        use_gru: bool = True,  # Use GRU updates (True) or residual (False, lighter)
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        self.undirected = undirected
        self.return_vertex_embeddings = return_vertex_embeddings

        # Set up aggregator
        if aggregator == "mean":
            self.aggregator = MeanAggregation()
        elif aggregator == "attention":
            self.aggregator = AttentionPooling(hidden_channels)
        else:  # "add" or default
            self.aggregator = SumAggregation()

        # Pre-compute reverse edge index for undirected message passing
        self._rev_edge_index = None

    def _get_rev_edge_index(self, edge_index: Tensor) -> Tensor:
        """Get reverse edge index for undirected message passing."""
        if self._rev_edge_index is None or self._rev_edge_index.size(0) != edge_index.size(1):
            src, dst = edge_index
            rev_index = torch.zeros(2, edge_index.size(1), dtype=torch.long, device=edge_index.device)
            rev_index[0], rev_index[1] = dst, src
            self._rev_edge_index = rev_index
        return self._rev_edge_index

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
        undirected: bool = False,
        return_vertex_embeddings: bool = False,
        aggregator: str = "add",
        use_gru: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        self.undirected = undirected
        self.return_vertex_embeddings = return_vertex_embeddings
        self.use_gru = use_gru

        # Set up aggregator
        if aggregator == "mean":
            self.aggregator = MeanAggregation()
        elif aggregator == "attention":
            self.aggregator = AttentionPooling(hidden_channels)
        else:
            self.aggregator = SumAggregation()

        # Pre-compute reverse edge index for undirected message passing
        self._rev_edge_index = None

        self.lin1 = Linear(in_channels, hidden_channels)
        self.norm_init = nn.LayerNorm(hidden_channels)
        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim, dropout)
        
        # Use GRU or residual update based on use_gru flag
        if use_gru:
            self.gru = GRUCell(hidden_channels, hidden_channels)
        else:
            self.gru = ResidualUpdate(hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_norms = torch.nn.ModuleList()
        self.atom_updates = torch.nn.ModuleList()

        for _ in range(num_layers - 1):
            conv = GATConv(
                hidden_channels,
                hidden_channels,
                heads=4,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
                negative_slope=0.01,
            )
            self.atom_convs.append(conv)
            self.atom_norms.append(nn.LayerNorm(hidden_channels))
            if use_gru:
                self.atom_updates.append(GRUCell(hidden_channels, hidden_channels))
            else:
                self.atom_updates.append(ResidualUpdate(hidden_channels))

        self.mol_conv = GATConv(
            hidden_channels,
            hidden_channels,
            heads=4,
            concat=False,
            dropout=dropout,
            add_self_loops=False,
            negative_slope=0.01,
        )
        self.mol_conv.explain = False  # Cannot explain global pooling.
        
        # Use GRU or residual for molecule-level updates
        if use_gru:
            self.mol_gru = GRUCell(hidden_channels, hidden_channels)
        else:
            self.mol_gru = ResidualUpdate(hidden_channels)
        
        self.mol_norm = nn.LayerNorm(hidden_channels)

        # Pre-allocated buffer for molecule-level edge_index (batch_size x 2)
        self.register_buffer("_mol_edge_index", None, persistent=False)

        self.lin2 = Linear(hidden_channels, out_channels)

        # Optional: compiled forward for speed
        self._compiled = False
        self._forward_compiled = None

        self.reset_parameters()

    def reset_parameters(self):
        for m in [
            self.lin1,
            self.gate_conv,
            self.gru,
            self.mol_conv,
            self.mol_gru,
            self.lin2,
            self.norm_init,
            self.mol_norm,
        ]:
            m.reset_parameters()
        for conv, gru, ln in zip(self.atom_convs, self.atom_grus, self.atom_norms):
            conv.reset_parameters()
            gru.reset_parameters()
            ln.reset_parameters()

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor
    ) -> Tensor | tuple[Tensor, Tensor]:
        # Atom Embedding
        x = F.leaky_relu_(self.lin1(x))
        x = self.norm_init(x)
        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        # loop through layers
        for conv, update, norm in zip(self.atom_convs, self.atom_updates, self.atom_norms):
            h = conv(x, edge_index)
            
            # Undirected message passing: average forward and backward messages
            if self.undirected:
                rev_edge_index = self._get_rev_edge_index(edge_index)
                h_rev = conv(x, rev_edge_index)
                h = (h + h_rev) * 0.5
            
            h = norm(h)
            h = F.elu_(h, inplace=True)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = update(h, x)

        # Store vertex embeddings before pooling (for return_vertex_embeddings)
        x_final = x  # [num_atoms, hidden_channels]

        # Molecule Embedding using configurable aggregator
        batch_size = batch.size(0)
        buf = self._mol_edge_index
        if buf is None or buf.size(1) != batch_size:
            row = torch.arange(batch_size, device=batch.device)
            buf = torch.stack([row, batch], dim=0)
            self._mol_edge_index = buf
        else:
            buf[0, :].fill_(0)
            buf[1, :].copy_(batch)
        edge_index_mol = buf
        
        out = self.aggregator(x, batch)
        out = F.relu_(out)

        # loop through time steps
        for _ in range(self.num_timesteps):
            h = self.mol_conv((x, out), edge_index_mol)
            h = self.mol_norm(h)
            h = F.elu_(h, inplace=True)
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        mol_embedding = self.lin2(out)

        # Return based on return_vertex_embeddings flag
        if self.return_vertex_embeddings:
            return mol_embedding, x_final  # (mol_embedding, atom_embeddings)
        return mol_embedding

    def compile(self, mode="default"):
        """Wrap forward with torch.compile for GPU speedup.

        Args:
            mode: "default", "reduce-overhead", or "max-autotune"
        """
        try:
            import torch._inductor
        except ImportError:
            return  # torch.compile not available (CPU-only or older torch)

        def forward_wrapper(x, edge_index, edge_attr, batch):
            return self.forward(x, edge_index, edge_attr, batch)

        self._forward_compiled = torch.compile(forward_wrapper, mode=mode, dynamic=False)
        self._compiled = True

    def compiled_forward(self, x, edge_index, edge_attr, batch):
        if self._compiled and self._forward_compiled is not None:
            return self._forward_compiled(x, edge_index, edge_attr, batch)
        return self.forward(x, edge_index, edge_attr, batch)


class AttentiveFP2(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.
    This variant creates two outputs, e.g. for a VAE setting or to train each
    hidden state of the consecutive RNN separately.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
        undirected: bool = False,
        return_vertex_embeddings: bool = False,
        aggregator: str = "add",
        use_gru: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        self.undirected = undirected
        self.return_vertex_embeddings = return_vertex_embeddings
        self.use_gru = use_gru

        # Set up aggregator
        if aggregator == "mean":
            self.aggregator = MeanAggregation()
        elif aggregator == "attention":
            self.aggregator = AttentionPooling(hidden_channels)
        else:
            self.aggregator = SumAggregation()

        self._rev_edge_index = None

        self.lin1 = Linear(in_channels, hidden_channels)
        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim, dropout)
        
        # Use GRU or residual update based on use_gru flag
        if use_gru:
            self.gru = GRUCell(hidden_channels, hidden_channels)
        else:
            self.gru = ResidualUpdate(hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_updates = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(
                hidden_channels,
                hidden_channels,
                heads=4,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
                negative_slope=0.01,
            )
            self.atom_convs.append(conv)
            if use_gru:
                self.atom_updates.append(GRUCell(hidden_channels, hidden_channels))
            else:
                self.atom_updates.append(ResidualUpdate(hidden_channels))

        self.mol_conv = GATConv(
            hidden_channels,
            hidden_channels,
            heads=4,
            concat=False,
            dropout=dropout,
            add_self_loops=False,
            negative_slope=0.01,
        )
        self.mol_conv.explain = False  # Cannot explain global pooling.
        
        # Use GRU or residual for molecule-level updates
        if use_gru:
            self.mol_gru = GRUCell(hidden_channels, hidden_channels)
        else:
            self.mol_gru = ResidualUpdate(hidden_channels)

        # Pre-allocated buffer for molecule-level edge_index (batch_size x 2)
        self.register_buffer("_mol_edge_index", None, persistent=False)

        self.lin_mu = Linear(hidden_channels, out_channels)
        self.lin_var = Linear(hidden_channels, out_channels)

        # Optional: compiled forward for speed
        self._compiled = False
        self._forward_compiled = None

        self.reset_parameters()

    def _get_rev_edge_index(self, edge_index: Tensor) -> Tensor:
        """Get reverse edge index for undirected message passing."""
        if self._rev_edge_index is None or self._rev_edge_index.size(0) != edge_index.size(1):
            src, dst = edge_index
            rev_index = torch.zeros(2, edge_index.size(1), dtype=torch.long, device=edge_index.device)
            rev_index[0], rev_index[1] = dst, src
            self._rev_edge_index = rev_index
        return self._rev_edge_index

    def reset_parameters(self):
        for m in [
            self.lin1,
            self.gate_conv,
            self.gru,
            self.mol_conv,
            self.mol_gru,
            self.lin_mu,
            self.lin_var,
        ]:
            m.reset_parameters()
        for conv, update in zip(self.atom_convs, self.atom_updates):
            conv.reset_parameters()
            update.reset_parameters()

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        # Atom Embedding
        x = F.leaky_relu_(self.lin1(x))
        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, update in zip(self.atom_convs, self.atom_updates):
            h = conv(x, edge_index)
            
            # Undirected message passing: average forward and backward messages
            if self.undirected:
                rev_edge_index = self._get_rev_edge_index(edge_index)
                h_rev = conv(x, rev_edge_index)
                h = (h + h_rev) * 0.5
            
            h = F.elu_(h, inplace=True)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = update(h, x)

        # Store vertex embeddings before pooling
        x_final = x

        # Molecule Embedding using configurable aggregator
        batch_size = batch.size(0)
        buf = self._mol_edge_index
        if buf is None or buf.size(1) != batch_size:
            row = torch.arange(batch_size, device=batch.device)
            buf = torch.stack([row, batch], dim=0)
            self._mol_edge_index = buf
        else:
            buf[0, :].fill_(0)
            buf[1, :].copy_(batch)
        edge_index_mol = buf

        out = self.aggregator(x, batch)
        out = F.relu_(out)

        for _ in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index_mol), inplace=True)
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        mu = self.lin_mu(out)
        logvar = self.lin_var(out)

        if self.return_vertex_embeddings:
            return mu, logvar, x_final  # (mu, logvar, atom_embeddings)
        return mu, logvar

    def compile(self, mode="default"):
        """Wrap forward with torch.compile for GPU speedup.

        Args:
            mode: "default", "reduce-overhead", or "max-autotune"
        """
        try:
            import torch._inductor
        except ImportError:
            return  # torch.compile not available

        def forward_wrapper(x, edge_index, edge_attr, batch):
            return self.forward(x, edge_index, edge_attr, batch)

        self._forward_compiled = torch.compile(forward_wrapper, mode=mode, dynamic=False)
        self._compiled = True

    def compiled_forward(self, x, edge_index, edge_attr, batch):
        if self._compiled and self._forward_compiled is not None:
            return self._forward_compiled(x, edge_index, edge_attr, batch)
        return self.forward(x, edge_index, edge_attr, batch)


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """Reparameterization trick to sample from N(mu, var) from N(0,1).

    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


def vae_loss_func(
    loss_smls, loss_prop, mu, log_var, w_vae, w_prop, wae, lambd=1.0, sigma=1.0
) -> List[Tensor]:
    """Computes the VAE KLD or WAE MMD loss. Clamp and replace possible NaNs."""
    if wae:
        loss_vae = torch.nan_to_num(compute_mmd(mu, lambd, sigma), nan=1e4)
    else:
        loss_vae = torch.nan_to_num(
            torch.mean(
                -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
            ).clamp(max=1e4),
            nan=1e4,
        )
    return (
        torch.add(
            loss_smls,
            torch.multiply(loss_vae, w_vae) + torch.multiply(loss_prop, w_prop),
        ),
        loss_vae,
    )


def compute_kernel(x1, x2, eps=1e-7, sigma=1.0):
    """Compute the inverse multiquadratic kernel between inputs."""
    x1, x2 = x1.unsqueeze(-1), x2.unsqueeze(-1)
    C = 2 * x1.size(-1) * sigma**2
    dist_sq = torch.cdist(x1, x2).pow(2)
    return C / (eps + C + dist_sq)


def compute_mmd(z, lamb=1.0, sigma=None, n_samples=100):
    """Compute the maximum mean discrepancy (MMD) between the input and a Gaussian prior.
    WAE-MMD loss adapted from https://openreview.net/pdf?id=HkL7n1-0b (Algorithm 2)
    """
    batch_size = z.size(0)
    if batch_size > n_samples:
        idx = torch.randperm(batch_size)[:n_samples]
        z = z[idx]
        batch_size = n_samples

    if sigma is None:
        with torch.no_grad():
            distances = torch.cdist(z, z)
            sigma = distances.median().clamp(min=1e-6)

    prior_z = torch.randn_like(z) * sigma
    kern_pz = compute_kernel(prior_z, prior_z, sigma=sigma)
    kern_z = compute_kernel(z, z, sigma=sigma)
    kern_pz_z = compute_kernel(prior_z, z, sigma=sigma)

    mmd_sq = (
        kern_pz.diagonal().mean()
        + kern_z.diagonal().mean()
        - 2 * kern_pz_z.diagonal().mean()
    )
    return lamb * torch.sqrt(mmd_sq.clamp(min=1e-6))


def anneal_cycle_linear(n_steps, start=0.0, stop=1.0, n_cycle=8, n_grow=3, ratio=0.75):
    w = np.ones(n_steps) * stop
    period = int(n_steps / n_cycle)

    for c in range(n_cycle):
        stop_cur = min(stop, stop * (c + 1) / (n_grow + 1)) if n_grow else stop
        step = (stop_cur - start) / (period * ratio)  # linear schedule
        v, i = start, 0
        while v <= stop_cur and (int(i + c * period) < n_steps):
            w[int(i + c * period)] = v
            v += step
            i += 1
        w[int(i + c * period) : int((c + 1) * period + 1)] = v
    return w


def anneal_cycle_sigmoid(n_steps, start=0.0, stop=1.0, n_cycle=8, n_grow=3, ratio=0.75):
    w = np.ones(n_steps)
    period = int(n_steps / n_cycle)

    for c in range(n_cycle):
        stop_cur = min(stop, stop * (c + 1) / (n_grow + 1)) if n_grow else stop
        step = (stop_cur - start) / (period * ratio)
        v, i = start, 0
        while v <= stop_cur and (int(i + c * period) < n_steps):
            w[int(i + c * period)] = 1.0 / (1.0 + np.exp(-(v * 8.0 - 4.0)))
            v += step
            i += 1
        w[int(i + c * period) : int((c + 1) * period + 1)] = 1.0 / (
            1.0 + np.exp(-(v * 8.0 - 4.0))
        )
    return w


def anneal_cycle_sigmoid_lin(
    n_steps, start=0.0, stop=1.0, n_cycle=8, n_grow=3, slope=1.5
):
    w = []
    period = int(n_steps / n_cycle)
    for c in range(n_cycle):
        stop_cur = min(stop, stop * (c + 1) / (n_grow + 1)) if n_grow else stop
        x = np.linspace(0, 1, period)
        curve = start + (stop_cur - start) / (1 + 1000 ** (-slope * (x - 0.5)))
        w.extend(curve.tolist())
    return np.array(w)


def anneal_const_linear(n_iter, start=0.0, stop=1.0):
    return np.linspace(start, stop, n_iter)


def anneal_const_sigmoid(n_iter, start=0.0, stop=1.0, slope=1.5):
    x = np.linspace(0, 1, n_iter)
    return start + (stop - start) / (1 + 1000 ** (-slope * (x - 0.5)))


def create_annealing_schedule(
    epochs,
    epoch_steps,
    anneal_start,
    anneal_stop,
    anneal_cycle,
    anneal_grow,
    anneal_ratio,
    anneal_type,
):
    anneal_stop = min(int(anneal_stop), int(epochs))
    total_steps = (anneal_stop - anneal_start) * epoch_steps
    if anneal_cycle:
        n_cycle = (anneal_stop - anneal_start) // anneal_cycle

    anneal = np.zeros(anneal_start * epoch_steps)
    if anneal_cycle and (total_steps / epoch_steps) % anneal_cycle:
        n_cycle += 1
    if anneal_type == "cyc_linear":
        ann_sched = anneal_cycle_linear(
            total_steps, n_cycle=n_cycle, n_grow=anneal_grow, ratio=anneal_ratio
        )
    elif anneal_type == "cyc_sigmoid":
        ann_sched = anneal_cycle_sigmoid(
            total_steps, n_cycle=n_cycle, n_grow=anneal_grow, ratio=anneal_ratio
        )
    elif anneal_type == "cyc_sigmoid_lin":
        ann_sched = anneal_cycle_sigmoid_lin(
            total_steps, n_cycle=n_cycle, n_grow=anneal_grow
        )
    elif anneal_type == "linear":
        ann_sched = anneal_const_linear(total_steps)
    elif anneal_type == "sigmoid":
        ann_sched = anneal_const_sigmoid(total_steps)
    elif anneal_type == "constant":
        ann_sched = np.ones(total_steps)
    else:
        raise NotImplementedError(f"Annealing type {anneal_type} not implemented.")
    anneal = np.concatenate((anneal, ann_sched)).flatten()
    if anneal_stop < epochs:
        anneal = np.concatenate(
            (anneal, np.ones((epochs - anneal_stop) * epoch_steps))
        ).flatten()
    return anneal


def log_gradients_to_tensorboard(named_params, writer, step):
    for name, param in named_params.items():
        if param.requires_grad and param.grad is not None:
            # 1. Log the Scaled Norm (Magnitude)
            grad_norm = param.grad.norm().item()
            writer.add_scalar(f"Gradients_Norm/{name}", grad_norm, step)

            # 2. Log the Distribution (Histogram)
            writer.add_histogram(f"Gradients_Dist/{name}", param.grad, step)
