#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configparser
import os
import time

import click
import numpy as np
import torch
import torch.nn as nn
from rdkit import RDLogger  # Chem
from rdkit import Chem
from rdkit.rdBase import DisableLog
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from dataset import AttFPDataset, tokenizer
from descriptors import rdkit_descirptors
from model import FFNN, LSTM, AttentiveFP
from utils import get_input_dims

# from descriptors import rdkit_descirptors

for level in RDLogger._levels:
    DisableLog(level)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.argument("filename")
@click.option("-d", "--delimiter", default="\t", help="Column delimiter of input file.")
@click.option("-c", "--smls_col", default="SMILES", help="Name of column that contains SMILES.")
@click.option("-e", "--epochs", default=50, help="Nr. of epochs to train.")
@click.option("-o", "--dropout", default=0.2, help="Dropout fraction.")
@click.option("-b", "--batch_size", default=128, help="Number of molecules per batch.")
@click.option("-l", "--lr", default=0.0005, help="Learning rate.")
@click.option("-f", "--lr_factor", default=0.9, help="Factor for learning rate decay.")
@click.option("-s", "--lr_step", default=3, help="Step size for learning rate decay.")
@click.option("-a", "--save_after", default=2, help="Epoch steps to save model.")
def main(filename, delimiter, smls_col, epochs, dropout, batch_size, lr, lr_factor, lr_step, save_after):
    # Write parameters to config file and define variables
    ini = configparser.ConfigParser()
    config = {
        "filename": filename,
        "epochs": epochs,
        "dropout": dropout,
        "batch_size": batch_size,
        "lr": lr,
        "lr_factor": lr_factor,
        "lr_step": lr_step,
        "save_after": save_after,
    }
    config["dim_model"] = dim_model = 128
    config["dim_hidden"] = dim_hidden = 512
    config["n_layers"] = n_layers = 2
    config["n_kernels"] = n_kernels = 3
    config["dim_rnn"] = dim_rnn = 512
    config["n_props"] = n_props = rdkit_descirptors([Chem.MolFromSmiles("O1CCNCC1")]).shape[1]
    _, t2i = tokenizer()
    config["alphabet"] = alphabet = len(t2i)
    dim_atom, dim_bond = get_input_dims()

    # Define paths
    path_model = f"models/{os.path.basename(filename)[:-4]}/"
    path_loss = f"logs/{os.path.basename(filename)[:-4]}/"
    os.makedirs(path_model, exist_ok=True)
    os.makedirs(path_loss, exist_ok=True)
    print("Paths (model, loss): ", path_model, path_loss)

    # store config file for later sampling
    with open(f"{path_model}config.ini", "w") as configfile:
        ini["CONFIG"] = config
        ini.write(configfile)

    # Create models
    gnn = AttentiveFP(
        in_channels=dim_atom,
        hidden_channels=dim_hidden,
        out_channels=dim_rnn,
        out_n=n_layers,
        dropout=dropout,
        edge_dim=dim_bond,
        num_layers=n_layers,
        num_timesteps=n_kernels,
    )
    # TODO: adapt parameters
    rnn = LSTM(
        size_vocab=alphabet,
        hidden_dim=dim_hidden,
        layers=n_layers,
        dropout=dropout,
    )
    mlp = FFNN(input_dim=dim_rnn, hidden_dim=dim_model, output_dim=n_props)
    gnn = gnn.to(DEVICE)
    rnn = rnn.to(DEVICE)
    mlp = mlp.to(DEVICE)

    # Calculate model parameters
    gnn_parameters = filter(lambda p: p.requires_grad, gnn.parameters())
    gnn_params = sum([np.prod(e.size()) for e in gnn_parameters])
    rnn_parameters = filter(lambda p: p.requires_grad, rnn.parameters())
    rnn_params = sum([np.prod(e.size()) for e in rnn_parameters])
    mlp_parameters = filter(lambda p: p.requires_grad, mlp.parameters())
    mlp_params = sum([np.prod(e.size()) for e in mlp_parameters])
    print(f"Num AtFP parameters:  {gnn_params / 1e6:.2f}M")
    print(f"Num RNN parameters:   {rnn_params / 1e6:.2f}M")
    print(f"Num MLP parameters:   {mlp_params / 1e6:.2f}M")
    print(f"Total Nr parameters:  {(rnn_params + gnn_params) / 1e6:.2f}M")

    # Define optimizer and criterion
    opt_params = list(rnn.parameters()) + list(gnn.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=lr, betas=(0.9, 0.999))
    writer = SummaryWriter(path_loss)
    criterion1 = nn.CrossEntropyLoss(ignore_index=t2i[" "], reduction="sum")
    criterion2 = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_factor)
    train_dataset = AttFPDataset(filename=filename, delimiter=delimiter, smls_col=smls_col)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    for epoch in range(1, epochs + 1):
        print(f" ---------- Epoch {epoch} ---------- ")
        time_start = time.time()
        l_s, l_p = train_one_epoch(gnn, rnn, mlp, optimizer, criterion1, criterion2, train_loader, writer, epoch)
        scheduler.step()
        loop_time = time.time() - time_start
        print("Epoch:", epoch, "Loss SMILES:", l_s, "Loss Props.:", l_p, "Time:", loop_time)

        # save loss and models
        if epoch % save_after == 0:
            torch.save(gnn.state_dict(), f"{path_model}egnn_{epoch}.pt")
            torch.save(rnn.state_dict(), f"{path_model}lstm_{epoch}.pt")


def train_one_epoch(
    gnn,
    rnn,
    mlp,
    optimizer,
    criterion1,
    criterion2,
    train_loader,
    writer,
    epoch,
):
    gnn.train()
    rnn.train()
    mlp.train()
    total_token, total_props, step = 0, 0, 0
    for g in tqdm(train_loader):
        g = g.to(DEVICE)
        trg = g.trg_smi
        trg = torch.transpose(trg, 0, 1)

        optimizer.zero_grad()
        h = gnn(g.atoms, g.edge_index, g.bonds, g.batch)
        # initialize LSTM layers with hidden state and others with 0
        c_0 = tuple([torch.zeros(h.shape).to(h.device) for _ in range(rnn.n_layers - 1)])
        # SMILES loss
        loss_mol = torch.zeros(1).to(DEVICE)
        # start with initial embedding from graph
        forward, hiddens = rnn(h, c_0)
        loss_forward = criterion1(forward, trg[1, :])
        loss_mol = torch.add(loss_mol, loss_forward)
        for j in range(1, trg.size(0) - 1):
            target = trg[j + 1, :]
            prev_token = trg[j, :].unsqueeze(0)
            forward, hiddens = rnn(prev_token, hiddens)
            loss_forward = criterion1(forward, target)
            loss_mol = torch.add(loss_mol, loss_forward)
        loss_per_token = loss_mol.cpu().detach().numpy()[0] / j
        # Properties loss
        pred_props = mlp(h.mean(0))
        loss_props = criterion2(pred_props, g.props.reshape(-1, pred_props.size(1)))
        # Combine losses (weight properties less to balance losses)
        loss = loss_mol + loss_props * 0.1
        loss.backward()
        optimizer.step()
        total_props += loss_props.cpu().detach().numpy() * 0.1
        total_token += loss_per_token
        # write tensorboard summary
        if step > 0:
            writer.add_scalar("loss_train_lstm", total_token / step, (epoch - 1) * len(train_loader) + step)
            writer.add_scalar("loss_train_prop", total_props / step, epoch * len(train_loader) + step)
            if step % 500 == 0:
                print(f"\n\tLoss SMILES: {total_token / step:.3f}, Loss Props.: {total_props / step:.2f}")
        step += 1
    return total_token / step, total_props / step


if __name__ == "__main__":
    main()
