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
from rdkit.Chem.Descriptors import CalcMolDescriptors
from rdkit.rdBase import DisableLog
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from dataset import AttFPDataset, tokenizer
from model import FFNN, LSTM, AttentiveFP
from sampling import temperature_sampling
from utils import get_input_dims

for level in RDLogger._levels:
    DisableLog(level)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.argument("filename")
@click.option("-n", "--run_name", default=None, help="Name of the run for saving (filename if omitted).")
@click.option("-d", "--delimiter", default="\t", help="Column delimiter of input file.")
@click.option("-c", "--smls_col", default="SMILES", help="Name of column that contains SMILES.")
@click.option("-e", "--epochs", default=200, help="Nr. of epochs to train.")
@click.option("-o", "--dropout", default=0.1, help="Dropout fraction.")
@click.option("-b", "--batch_size", default=256, help="Number of molecules per batch.")
@click.option("-r", "--random", is_flag=True, help="Randomly sample molecules in each training step.")
@click.option("-es", "--epoch_steps", default=1000, help="If random, number of batches per epoch.")
@click.option("-v", "--val", default=0.05, help="Fraction of the data to use for validation.")
@click.option("-l", "--lr", default=1e-3, help="Learning rate.")
@click.option("-lf", "--lr_fact", default=0.75, help="Learning rate decay factor.")
@click.option("-ls", "--lr_step", default=10, help="LR Step decay after nr. of epochs.")
@click.option("-a", "--after", default=5, help="Epoch steps to save model.")
@click.option("-nk", "--kernels_gnn", default=2, help="Nr. GNN kernels")
@click.option("-ng", "--layers_gnn", default=2, help="Nr. GNN layers")
@click.option("-nr", "--layers_rnn", default=2, help="Nr. RNN layers")
@click.option("-nm", "--layers_mlp", default=2, help="Nr. MLP layers")
@click.option("-dg", "--dim_gnn", default=512, help="Hidden dimension of GNN layers")
@click.option("-dr", "--dim_rnn", default=512, help="Hidden dimension of RNN layers")
@click.option("-de", "--dim_emb", default=512, help="Dimension of RNN token embedding")
@click.option("-dm", "--dim_mlp", default=512, help="Hidden dimension of MLP layers")
@click.option("-wp", "--prop_weight", default=0.1, help="Factor for weighting property loss vs. SMILES")
@click.option("--scale/--no-scale", default=True, help="Whether to scale all properties from 0 to 1")
@click.option("-p", "--n_proc", default=6, help="Number of CPU processes to use")
def main(
    filename,
    run_name,
    delimiter,
    smls_col,
    epochs,
    dropout,
    batch_size,
    random,
    epoch_steps,
    val,
    lr,
    lr_fact,
    lr_step,
    after,
    kernels_gnn,
    layers_gnn,
    layers_rnn,
    layers_mlp,
    dim_gnn,
    dim_rnn,
    dim_emb,
    dim_mlp,
    prop_weight,
    scale,
    n_proc,
):
    if run_name is None:  # take filename as run name if not specified
        run_name = os.path.basename(filename).split(".")[0]

    _, t2i = tokenizer()
    dim_atom, dim_bond = get_input_dims()

    # Write parameters to config file and define variables
    ini = configparser.ConfigParser()
    config = {
        "filename": filename,
        "run_name": run_name,
        "epochs": epochs,
        "dropout": dropout,
        "batch_size": batch_size,
        "random": random,
        "n_proc": n_proc,
        "epoch_steps": epoch_steps if random else "full",
        "frac_val": val,
        "lr": lr,
        "lr_fact": lr_fact,
        "save_after": after,
        "n_gnn_layers": layers_gnn,
        "n_rnn_layers": layers_rnn,
        "n_mlp_layers": layers_mlp,
        "dim_gnn": dim_gnn,
        "n_kernels": kernels_gnn,
        "dim_embed": dim_emb,
        "dim_rnn": dim_rnn,
        "dim_mlp": dim_mlp,
        "weight_props": prop_weight,
        "scaled_props": scale,
    }
    config["n_props"] = n_props = len(CalcMolDescriptors(Chem.MolFromSmiles("O1CCNCC1")))
    config["alphabet"] = alphabet = len(t2i)

    # Define paths
    path_model = f"models/{run_name}/"
    path_loss = f"logs/{run_name}/"
    os.makedirs(path_model, exist_ok=True)
    os.makedirs(path_loss, exist_ok=True)
    print("\nPaths (model, loss): ", path_model, path_loss)

    # store config file for later sampling, retraining etc.
    with open(f"{path_model}config.ini", "w") as configfile:
        ini["CONFIG"] = config
        ini.write(configfile)

    # Create models
    gnn = AttentiveFP(
        in_channels=dim_atom,
        hidden_channels=dim_gnn,
        out_channels=dim_rnn,
        edge_dim=dim_bond,
        num_layers=layers_gnn,
        num_timesteps=kernels_gnn,
        dropout=dropout,
    ).to(DEVICE)
    rnn = LSTM(
        input_dim=alphabet,
        embedding_dim=dim_emb,
        hidden_dim=dim_rnn,
        layers=layers_rnn,
        dropout=dropout,
    ).to(DEVICE)
    mlp = FFNN(input_dim=dim_rnn, hidden_dim=dim_mlp, n_layers=layers_mlp, output_dim=n_props).to(DEVICE)

    # Calculate model parameters
    gnn_parameters = filter(lambda p: p.requires_grad, gnn.parameters())
    gnn_params = sum([np.prod(e.size()) for e in gnn_parameters])
    rnn_parameters = filter(lambda p: p.requires_grad, rnn.parameters())
    rnn_params = sum([np.prod(e.size()) for e in rnn_parameters])
    mlp_parameters = filter(lambda p: p.requires_grad, mlp.parameters())
    mlp_params = sum([np.prod(e.size()) for e in mlp_parameters])
    print(f"\nNum GNN parameters:  {gnn_params / 1e6:.2f}M")
    print(f"Num RNN parameters:  {rnn_params / 1e6:.2f}M")
    print(f"Num MLP parameters:  {mlp_params / 1e6:.2f}M")
    print(f"  Total parameters: {(mlp_params + rnn_params + gnn_params) / 1e6:.2f}M")

    # Define optimizer and loss criteria
    opt_params = list(rnn.parameters()) + list(gnn.parameters()) + list(mlp.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=lr)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_fact)
    criterion1 = nn.CrossEntropyLoss(reduction="mean")
    criterion2 = nn.MSELoss(reduction="mean")
    dataset = AttFPDataset(
        filename=filename,
        delimiter=delimiter,
        smls_col=smls_col,
        random=random,
        scaled_props=scale,
        steps=int(epoch_steps * batch_size * (1 + val)),
    )
    train_set, val_set = torch.utils.data.random_split(dataset, [1.0 - val, val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_proc, drop_last=False)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=n_proc, drop_last=False)
    writer = SummaryWriter(path_loss)
    print(
        f"Using {len(train_set)}{' random' if random else ''} molecules for training "
        + f"and {len(val_set)}{' random' if random else ''} for validation per epoch."
    )

    for epoch in range(1, epochs + 1):
        print(f"\n---------- Epoch {epoch} ----------")
        time_start = time.time()
        l_s, l_p = train_one_epoch(
            gnn, rnn, mlp, optimizer, criterion1, criterion2, train_loader, writer, epoch, t2i, prop_weight
        )
        l_vs, l_vp = validate_one_epoch(
            gnn,
            rnn,
            mlp,
            criterion1,
            criterion2,
            valid_loader,
            writer,
            (epoch + 1) * len(train_loader),
            t2i,
            prop_weight,
        )
        dur = time.time() - time_start
        schedule.step()
        last_lr = schedule.get_last_lr()[0]
        writer.add_scalar("lr", last_lr, (epoch + 1) * len(train_loader))
        print(
            f"Epoch: {epoch}, Train Loss SMILES: {l_s:.3f}, Train Loss Props.: {l_p:.3f}, "
            + f"Val. Loss SMILES: {l_vs:.3f}, Val. Loss Props.: {l_vp:.3f}, "
            + f"LR: {last_lr:.6f}, Time: {dur//60:.0f}min {dur%60:.0f}sec"
        )

        _, _ = temperature_sampling(gnn, rnn, 0.5, np.random.choice(val_set.dataset.data, 1)[0], 10, 96, True)

        # save loss and models
        if epoch % after == 0:
            torch.save(gnn.state_dict(), f"{path_model}atfp_{epoch}.pt")
            torch.save(rnn.state_dict(), f"{path_model}lstm_{epoch}.pt")
            torch.save(mlp.state_dict(), f"{path_model}ffnn_{epoch}.pt")


def train_one_epoch(gnn, rnn, mlp, optimizer, criterion1, criterion2, train_loader, writer, epoch, t2i, w):
    gnn.train(True)
    rnn.train(True)
    mlp.train(True)
    steps = len(train_loader)
    total_smls, total_props, step = 0, 0, 0
    for g in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        # graph and target smiles
        g = g.to(DEVICE)
        trg = g.trg_smi
        trg = torch.transpose(trg, 0, 1)

        # get GNN embedding as input for RNN (h0) and FFNN
        hn = gnn(g.atoms, g.edge_index, g.bonds, g.batch)
        hn = torch.cat(
            [hn.unsqueeze(0)] + [torch.zeros((1, hn.size(0), hn.size(1))).to(DEVICE)] * (rnn.n_layers - 1), dim=0
        )

        # start with initial embedding from graph and zeros as cell state
        cn = torch.zeros((rnn.n_layers, hn.size(1), rnn.hidden_dim)).to(DEVICE)

        # now get learn tokens using Teacher Forcing
        loss_mol = torch.zeros(1).to(DEVICE)
        finish = (trg == t2i["$"]).nonzero()[-1, 0]
        for j in range(finish - 1):  # only loop until last end-token to prevent nan loss
            nxt, (hn, cn) = rnn(trg[j, :].unsqueeze(0), (hn, cn))
            loss_fwd = torch.nan_to_num(criterion1(nxt.view(trg.size(1), -1), trg[j + 1, :]), nan=1e-6)
            loss_mol = torch.add(loss_mol, loss_fwd)
        loss_per_token = loss_mol.cpu().detach().numpy()[0] / j

        # Properties loss
        pred_props = mlp(hn[0])
        loss_props = torch.nan_to_num(criterion2(pred_props, g.props.reshape(-1, pred_props.size(1))), nan=1e-6)

        # combine losses, apply desired weight to property loss
        loss = loss_mol + torch.multiply(loss_props, w)
        loss.backward(retain_graph=True)
        optimizer.step()

        total_smls += loss_per_token
        total_props += loss_props.cpu().detach().numpy()

        # write tensorboard summary
        step += 1
        writer.add_scalar("loss_train_smiles", total_smls / step, (epoch - 1) * steps + step)
        writer.add_scalar("loss_train_props", total_props / step, (epoch - 1) * steps + step)
        if step % 500 == 0:
            print(f"Step: {step}/{steps}, Loss SMILES: {total_smls / step:.3f}, Loss Props.: {total_props / step:.3f}")

    return (total_smls / step, total_props / step)


@torch.no_grad
def validate_one_epoch(gnn, rnn, mlp, criterion1, criterion2, valid_loader, writer, step, t2i, w):
    gnn.train(False)
    rnn.train(False)
    mlp.train(False)
    steps = len(valid_loader)
    loss_smls, loss_props = 0, 0
    with torch.no_grad():
        for g in tqdm(valid_loader, desc="Validation"):
            # graph and target smiles
            g = g.to(DEVICE)
            trg = g.trg_smi
            trg = torch.transpose(trg, 0, 1)

            # build hidden and cell value inputs
            hn = gnn(g.atoms, g.edge_index, g.bonds, g.batch)
            hn = torch.cat(
                [hn.unsqueeze(0)] + [torch.zeros((1, hn.size(0), hn.size(1))).to(DEVICE)] * (rnn.n_layers - 1), dim=0
            )
            cn = torch.zeros((rnn.n_layers, hn.size(1), rnn.hidden_dim)).to(DEVICE)

            # SMILES
            val_loss = torch.zeros(1).to(DEVICE)
            finish = (trg == t2i["$"]).nonzero()[-1, 0]
            for j in range(finish - 1):  # only loop until last end-token to prevent nan loss
                nxt, (hn, cn) = rnn(trg[j, :].unsqueeze(0), (hn, cn))
                loss_fwd = criterion1(nxt.view(trg.size(1), -1), trg[j + 1, :])
                val_loss = torch.add(val_loss, loss_fwd)
            loss_smls += val_loss.cpu().detach().numpy()[0] / j

            # properties
            pred_props = mlp(hn[0])
            loss_props += criterion2(pred_props, g.props.reshape(-1, pred_props.size(1))).cpu().detach().numpy()

    # write tensorboard summary for this epoch and return validation loss
    writer.add_scalar("loss_val_smiles", loss_smls / steps, step)
    writer.add_scalar("loss_val_props", loss_props / steps, step)
    return (loss_smls / steps, loss_props / steps)


if __name__ == "__main__":
    main()
