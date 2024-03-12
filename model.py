#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, Size, Tensor

from pygmt import GraphMultisetTransformer, weights_init
from utils import DIM_EMBEDDING


class FFNN(nn.Module):
    def __init__(self, input_dim=512, output_dim=128, hidden_dim=256, dropout=0.3):
        super(FFNN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.mlp(x)


class LSTM(nn.Module):
    def __init__(self, input_dim=64, embedding_dim=128, hidden_dim=256, layers=2, dropout=0.3):
        super(LSTM, self).__init__()
        self.n_layers = layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            dropout=dropout,
        )
        self.norm_1 = nn.LayerNorm(embedding_dim, eps=0.001)
        self.norm_2 = nn.LayerNorm(hidden_dim, eps=0.001)
        self.fnn = nn.Linear(hidden_dim, input_dim)

        # initialize
        nn.init.xavier_uniform_(self.fnn.weight)
        nn.init.zeros_(self.fnn.bias)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l1)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l1)
        self.lstm.bias_ih_l0.data.fill_(0.0)
        self.lstm.bias_ih_l0.data[hidden_dim : 2 * hidden_dim].fill_(1.0)
        self.lstm.bias_ih_l1.data.fill_(0.0)
        self.lstm.bias_ih_l1.data[hidden_dim : 2 * hidden_dim].fill_(1.0)
        self.lstm.bias_hh_l0.data.fill_(0.0)
        self.lstm.bias_hh_l0.data[hidden_dim : 2 * hidden_dim].fill_(1.0)
        self.lstm.bias_hh_l1.data.fill_(0.0)
        self.lstm.bias_hh_l1.data[hidden_dim : 2 * hidden_dim].fill_(1.0)

    def forward(self, i, h):
        f = self.norm_1(self.embedding(i))
        f, h = self.lstm(f, h)
        f = self.fnn(self.norm_2(f)).squeeze(0)
        return f, h


class GraphTransformer(nn.Module):
    def __init__(
        self,
        n_kernels=3,
        pooling_heads=8,
        mlp_dim=512,
        mp_dim=64,
        kernel_dim=64,
        embeddings_dim=64,
        rnn_dim=512,
        rnn_layers=2,
        dropout=0.1,
        aggr="add",
    ):
        super(GraphTransformer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(DIM_EMBEDDING, embedding_dim=embeddings_dim)

        self.pre_egnn_mlp = nn.Sequential(
            nn.Linear(embeddings_dim * 4, kernel_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(kernel_dim * 2, kernel_dim),
        )

        self.kernels = nn.ModuleList()
        for _ in range(n_kernels):
            self.kernels.append(
                EGNN_sparse(
                    feats_dim=kernel_dim,
                    m_dim=mp_dim,
                    aggr=aggr,
                )
            )

        self.post_egnn_mlp = nn.Sequential(
            nn.Linear(mp_dim * n_kernels, kernel_dim),
            self.dropout,
            nn.SiLU(),
            nn.Linear(kernel_dim, kernel_dim),
            nn.SiLU(),
            nn.Linear(kernel_dim, kernel_dim),
            nn.SiLU(),
        )

        self.transformers = nn.ModuleList()
        for _ in range(pooling_heads):
            self.transformers.append(
                GraphMultisetTransformer(
                    in_channels=kernel_dim,
                    hidden_channels=kernel_dim,
                    out_channels=kernel_dim,
                    pool_sequences=["GMPool_G", "SelfAtt", "GMPool_I"],
                    num_heads=1,
                    layer_norm=True,
                )
            )

        self.post_pooling_mlps = nn.ModuleList()
        for _ in range(rnn_layers):
            self.post_pooling_mlps.append(
                nn.Sequential(
                    nn.Linear(kernel_dim * pooling_heads, rnn_dim),
                    self.dropout,
                    nn.SiLU(),
                    nn.Linear(rnn_dim, 2 * rnn_dim),
                    self.dropout,
                    nn.SiLU(),
                    nn.Linear(2 * rnn_dim, rnn_dim),
                )
            )
        for m in [self.transformers, self.kernels, self.post_egnn_mlp, self.post_pooling_mlps]:
            m.apply(weights_init)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, g_batch):
        embedded = torch.cat(
            [
                self.embedding(g_batch.atomids),
                self.embedding(g_batch.is_ring),
                self.embedding(g_batch.hyb),
                self.embedding(g_batch.arom),
            ],
            dim=1,
        )
        features = self.pre_egnn_mlp(embedded)

        feature_list = [kernel(x=features, edge_index=g_batch.edge_index) for kernel in self.kernels]
        features = self.post_egnn_mlp(torch.cat(feature_list, dim=1))

        feature_list = [
            trsnfrmr(x=features, batch=g_batch.batch, edge_index=g_batch.edge_index) for trsnfrmr in self.transformers
        ]
        features = torch.cat(feature_list, dim=1)

        feature_list = [mlp(features).unsqueeze(0) for mlp in self.post_pooling_mlps]

        return torch.cat(feature_list, dim=0)


class EGNN_sparse(MessagePassing):
    """PyTorch geometric message-passing layer implementing E(n)-Equivariant Graph Neural Networks
    for 2D molecular graphs."""

    def __init__(self, feats_dim, m_dim=32, dropout=0.1, aggr="add", **kwargs):
        """Initialization of the 2D message passing layer.

        :param feats_dim: Node feature dimension.
        :type feats_dim: int
        :param m_dim: Meessage passing feature dimesnion, defaults to 32
        :type m_dim: int, optional
        :param dropout: Dropout value, defaults to 0.1
        :type dropout: float, optional
        :param aggr: Message aggregation type, defaults to "add"
        :type aggr: str, optional
        """
        assert aggr in {
            "add",
            "sum",
            "max",
            "mean",
        }, "pool method must be a add, sum, max or mean"

        kwargs.setdefault("aggr", aggr)
        super(EGNN_sparse, self).__init__(**kwargs)
        self.edge_norm1 = nn.LayerNorm(m_dim)
        self.edge_norm2 = nn.LayerNorm(m_dim)
        self.node_norm1 = nn.LayerNorm(feats_dim)
        self.node_norm2 = nn.LayerNorm(feats_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(feats_dim * 2, feats_dim * 2 * 2),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(feats_dim * 2 * 2, m_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        )
        self.apply(weights_init)

    def forward(self, x: Tensor, edge_index: Adj):
        """Forward pass in the mesaage passing fucntion.

        :param x: Node features.
        :type x: Tensor
        :param edge_index: Edge indices.
        :type edge_index: Adj
        :return: Updated node features.
        :rtype: Tensor
        """
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        """Message passing.

        :param x_i: Node n_i.
        :type x_i: Tensor
        :param x_j: Node n_j.
        :type x_j: Tensor
        :return: Message m_ji
        :rtype: Tensor
        """
        return self.edge_mlp(torch.cat([x_i, x_j], dim=-1))

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """Overall propagation within the message passing."""
        # get input tensors
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute("message", coll_dict)
        aggr_kwargs = self.inspector.distribute("aggregate", coll_dict)
        update_kwargs = self.inspector.distribute("update", coll_dict)

        # get messages
        m_ij = self.message(**msg_kwargs)
        m_ij = self.edge_norm1(m_ij)

        # aggregate messages
        m_i = self.aggregate(m_ij, **aggr_kwargs)
        m_i = self.edge_norm2(m_i)

        # get updated node features
        hidden_feats = self.node_norm1(kwargs["x"])
        hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
        hidden_out = self.node_norm2(hidden_out)
        hidden_out = kwargs["x"] + hidden_out

        return self.update(hidden_out, **update_kwargs)
