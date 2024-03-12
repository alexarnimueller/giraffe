#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configparser
import os
import time

import click
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.rdBase import DisableLog
from torch.utils.tensorboard import SummaryWriter
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
@click.option("-e", "--epochs", default=50, help="Nr. of epochs to train.")
@click.option("-o", "--dropout", default=0.2, help="Dropout fraction.")
@click.option("-b", "--batch_size", default=128, help="Number of molecules per batch.")
@click.option("-l", "--lr", default=0.0005, help="Learning rate.")
@click.option("-f", "--lr_factor", default=0.75, help="Factor for learning rate decay.")
@click.option("-s", "--lr_step", default=5, help="Step size for learning rate decay.")
@click.option("-a", "--save_after", default=5, help="Epoch steps to save model.")
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
    config["n_pool_heads"] = n_pool_heads = 4
    config["rnn_dim"] = rnn_dim = 512
    config["n_props"] = n_props = rdkit_descirptors([Chem.MolFromSmiles("O1CCNCC1")]).shape[1]
    _, t2i = tokenizer()
    config["alphabet"] = alphabet = len(t2i)

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
    egnn = GraphTransformer(
        n_kernels=n_kernels,
        pooling_heads=n_pool_heads,
        mlp_dim=dim_hidden,
        kernel_dim=dim_model,
        embeddings_dim=dim_model,
        dropout=dropout,
        rnn_dim=rnn_dim,
        rnn_layers=n_layers,
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
    writer = SummaryWriter(path_loss)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_factor)
    train_dataset = MolDataset(filename=filename, delimiter=delimiter, smls_col=smls_col)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    losses_lstm, losses_prop = [], []
    for epoch in range(epochs):
        print(f" ---------- Epoch {epoch + 1} ---------- ")
        time_start = time.time()
        l_l, l_p = train_one_epoch(egnn, lstm, ffnn, optimizer, criterion1, criterion2, train_loader, writer, epoch)
        scheduler.step()
        losses_lstm.append(l_l)
        losses_prop.append(l_p)
        loop_time = time.time() - time_start
        print("Epoch:", epoch, "Loss LSTM:", l_l, "Loss Props.:", l_p, "Time:", loop_time)

        # save loss and models
        if epoch > 0 and epoch % save_after == 0:
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
    train_loader,
    writer,
    epoch,
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
        loss_props = criterion2(pred_props, g.props.reshape(-1, pred_props.size(1))) / train_loader.batch_size
        # Combine losses (weight properties less to balance losses)
        loss = loss_mol + loss_props * 0.1
        loss.backward()
        optimizer.step()
        total_props += loss_props.cpu().detach().numpy() * 0.1
        total_token += loss_per_token
        # write tensorboard summary
        if step > 0:
            writer.add_scalar("loss_train_lstm", total_token / step, epoch * len(train_loader) + step)
            writer.add_scalar("loss_train_prop", total_props / step, epoch * len(train_loader) + step)
            if step % 500 == 0:
                print(f"\n\tLoss LSTM: {total_token / step:.3f}, Loss Props.: {total_props / step:.2f}")
        step += 1

    return total_token / len(train_loader), total_props / len(train_loader)


if __name__ == "__main__":
    main()
