#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time

import click
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.rdBase import DisableLog
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from dataset import MolDataset, tokenizer
from descriptors import rdkit_descirptors
from model import FFNN, LSTM, GraphTransformer

for level in RDLogger._levels:
    DisableLog(level)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.argument("filename")
@click.option("-d", "--delimiter", default="\t", help="Column delimiter of input file.")
@click.option("-c", "--smls_col", default="SMILES", help="Name of column that contains SMILES.")
@click.option("-e", "--epochs", default=100, help="Nr. of epochs to train.")
@click.option("-o", "--dropout", default=0.2, help="Dropout fraction.")
@click.option("-b", "--batch_size", default=128, help="Number of molecules per batch.")
@click.option("-l", "--lr", default=0.001, help="Learning rate.")
@click.option("-f", "--lr_factor", default=0.75, help="Factor for learning rate decay.")
@click.option("-s", "--lr_step", default=5, help="Step size for learning rate decay.")
def main(filename, delimiter, smls_col, epochs, dropout, batch_size, lr, lr_factor, lr_step):
    # Load hyperparameters from config file and define globle variables
    dim_model = 128
    dim_hidden = 512
    n_layers = 2
    n_kernels = 3
    n_pool_heads = 4
    n_props = rdkit_descirptors([Chem.MolFromSmiles('O1CCNCC1')]).shape[1]
    _, t2i = tokenizer()
    alphabet = len(t2i)

    # Define paths
    path_model = f"models/{os.path.basename(filename)[:-4]}/"
    path_loss = f"loss/{os.path.basename(filename)[:-4]}/"
    os.makedirs(path_model, exist_ok=True)
    os.makedirs(path_loss, exist_ok=True)
    print("Paths (model, loss): ", path_model, path_loss)

    # Create models
    egnn = GraphTransformer(
        n_kernels=n_kernels,
        pooling_heads=n_pool_heads,
        mlp_dim=dim_hidden,
        kernel_dim=dim_model,
        embeddings_dim=dim_model,
        dropout=dropout
    )
    lstm = LSTM(
        input_dim=alphabet,
        embedding_dim=dim_model,
        hidden_dim=dim_hidden,
        layers=n_layers,
        dropout=dropout,
    )
    ffnn = FFNN(input_dim=dim_hidden, hidden_dim=dim_model, output_dim=n_props)
    egnn = egnn.to(DEVICE)
    lstm = lstm.to(DEVICE)
    ffnn = ffnn.to(DEVICE)

    # Calculate model parameters
    egnn_parameters = filter(lambda p: p.requires_grad, egnn.parameters())
    egnn_params = sum([np.prod(e.size()) for e in egnn_parameters])
    lstm_parameters = filter(lambda p: p.requires_grad, lstm.parameters())
    lstm_params = sum([np.prod(e.size()) for e in lstm_parameters])
    ffnn_parameters = filter(lambda p: p.requires_grad, ffnn.parameters())
    ffnn_params = sum([np.prod(e.size()) for e in ffnn_parameters])
    print(f"Num EGNN parameters: {egnn_params / 1e3:.2f}k")
    print(f"Num LSTM parameters: {lstm_params / 1e3:.2f}k")
    print(f"Num FFNN parameters: {ffnn_params / 1e3:.2f}k")
    print(f"Total: {(ffnn_params + lstm_params + egnn_params) / 1e6:.2f}M")

    # Define optimizer and criterion
    opt_params = list(lstm.parameters()) + list(egnn.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=lr, betas=(0.9, 0.999))
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step, gamma=lr_factor
    )
    train_dataset = MolDataset(filename=filename, delimiter=delimiter, smls_col=smls_col)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    losses_lstm, losses_prop = [], []
    for epoch in range(epochs):
        print(f" ---------- Epoch {epoch + 1} ---------- ")
        time_start = time.time()
        l_lstm, l_prop = train_one_epoch(egnn, lstm, ffnn, optimizer, criterion1, criterion2, train_loader)
        scheduler.step()
        losses_lstm.append(l_lstm)
        losses_prop.append(l_prop)
        loop_time = time.time() - time_start
        print("Epoch:", epoch, "Loss LSTM:", l_lstm, "Loss Props.:", l_prop, "Time:", loop_time)

        # save loss and models
        if (epoch + 20) % 20 == 0:
            torch.save(egnn.state_dict(), f"{path_model}egnn_{epoch}.pt")
            torch.save(lstm.state_dict(), f"{path_model}lstm_{epoch}.pt")
            torch.save(losses_lstm, f"{path_loss}lstm_training.pt")
            torch.save(losses_prop, f"{path_loss}prop_training.pt")


def train_one_epoch(
    egnn,
    lstm,
    ffnn,
    optimizer,
    criterion1,
    criterion2,
    train_loader
):
    egnn.train()
    lstm.train()
    ffnn.train()
    total_token, total_props, step = 0, 0, 0
    for g in tqdm(train_loader):
        g = g.to(DEVICE)
        trg = g.trg_smi
        trg = torch.transpose(trg, 0, 1)

        optimizer.zero_grad()
        h = egnn(g)
        # initialize LSTM layers with hidden state and others with 0
        hiddens = tuple([h] + [torch.zeros(h.shape).to(h.device) for _ in range(lstm.n_layers - 1)])
        # SMILES loss
        loss_mol = torch.zeros(1).to(DEVICE)
        for j in range(trg.size(0) - 1):
            target = trg[j + 1, :]
            prev_token = trg[j, :].unsqueeze(0)
            forward, hiddens = lstm(prev_token, hiddens)
            loss_forward = criterion1(forward, target)
            loss_mol = torch.add(loss_mol, loss_forward)
        loss_per_token = loss_mol.cpu().detach().numpy()[0] / j
        # Properties loss
        pred_props = ffnn(h.mean(0))
        loss_props = criterion2(pred_props, g.props.reshape(-1, pred_props.size(1)))
        # Combine losses
        loss = loss_mol + loss_props
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(lstm.parameters(), 0.5)  # clip gradient of LSTM
        optimizer.step()
        total_props += loss_props.cpu().detach().numpy()
        total_token += loss_per_token
        if step > 0 and step % 50 == 0:
            print("Loss LSTM:", total_token / step, "Loss Props.:", total_props / step)
        step += 1

    return total_token / len(train_loader), total_props / len(train_loader)


if __name__ == "__main__":
    main()
