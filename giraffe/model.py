#! /usr/bin/env python
# -*- coding: utf-8

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax

from giraffe.utils import get_input_dims, read_config_ini

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        self, input_dim=512, output_dim=125, hidden_dim=256, n_layers=3, dropout=0.3
    ):
        super(FFNN, self).__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
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
        return self.mlp(x)


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
        features = self.fcn(features).clamp(min=1e-6, max=1e6)
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
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
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
                hidden_channels,
                hidden_channels,
                dropout=dropout,
                add_self_loops=False,
                negative_slope=0.01,
            )
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(
            hidden_channels,
            hidden_channels,
            dropout=dropout,
            add_self_loops=False,
            negative_slope=0.01,
        )
        self.mol_conv.explain = False  # Cannot explain global pooling.
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for m in [
            self.lin1,
            self.gate_conv,
            self.gru,
            self.mol_conv,
            self.mol_gru,
            self.lin2,
        ]:
            m.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor
    ) -> Tensor:
        # Atom Embedding
        x = F.leaky_relu_(self.lin1(x))
        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        # loop through layers
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()

        # Molecule Embedding
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)
        out = global_add_pool(x, batch).relu_()

        # loop through time steps
        for _ in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        return self.lin2(out)


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
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
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
                hidden_channels,
                hidden_channels,
                dropout=dropout,
                add_self_loops=False,
                negative_slope=0.01,
            )
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(
            hidden_channels,
            hidden_channels,
            dropout=dropout,
            add_self_loops=False,
            negative_slope=0.01,
        )
        self.mol_conv.explain = False  # Cannot explain global pooling.
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin_mu = Linear(hidden_channels, out_channels)
        self.lin_var = Linear(hidden_channels, out_channels)
        self.reset_parameters()

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
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor
    ) -> Tensor:
        # Atom Embedding
        x = F.leaky_relu_(self.lin1(x))
        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()

        # Molecule Embedding
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for _ in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        return self.lin_mu(out), self.lin_var(out)


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
    C = 2 * x2.size(-1) * sigma**2
    kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

    # Exclude diagonal elements
    return kernel.sum() - kernel.diag().sum()


def compute_mmd(z, lamb=1.0, sigma=1.0):
    """Compute the maximum mean discrepancy (MMD) between the input and a Gaussian prior.
    WAE-MMD loss adapted from https://openreview.net/pdf?id=HkL7n1-0b (Algorithm 2)
    """
    prior_z = (
        torch.randn_like(z) * sigma
    )  # use normal gaussian as prior (with custom sigma if needed)
    kern_pz = compute_kernel(prior_z, prior_z, sigma=sigma)
    kern_z = compute_kernel(z, z, sigma=sigma)
    kern_pz_z = compute_kernel(prior_z, z, sigma=sigma)
    pre = lamb / (z.size(0) * (z.size(0) - 1))
    return (
        pre * kern_pz.mean()
        + pre * kern_z.mean()
        - (2 * lamb) / (z.size(0) ** 2) * kern_pz_z.mean()
    )


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
