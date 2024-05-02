#! /usr/bin/env python
# -*- coding: utf-8

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

from dataset import AttFPDataset, tokenizer
from model import FFNN, LSTM, AttentiveFP2, anneal_cycle_sigmoid
from sampling import temperature_sampling
from train_vae import train_one_epoch, validate_one_epoch
from utils import get_input_dims

for level in RDLogger._levels:
    DisableLog(level)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.argument("filename")
@click.argument("checkpoint")
@click.option("-e", "--epoch_load", help="Epoch of models to load.")
@click.option("-n", "--run_name", default=None, help="Name of the run for saving (filename if omitted).")
@click.option("-d", "--delimiter", default="\t", help="Column delimiter of input file.")
@click.option("-c", "--smls_col", default="SMILES", help="Name of column that contains SMILES.")
@click.option("-et", "--epochs", default=50, help="Nr. of epochs to train.")
@click.option("-o", "--dropout", default=0.1, help="Dropout fraction.")
@click.option("-b", "--batch_size", default=256, help="Number of molecules per batch.")
@click.option("-r", "--random", is_flag=True, help="Randomly sample molecules in each training step.")
@click.option("-es", "--epoch_steps", default=1000, help="If random, number of batches per epoch.")
@click.option("-v", "--val", default=0.05, help="Fraction of the data to use for validation.")
@click.option("-l", "--lr", default=1e-4, help="Learning rate.")
@click.option("-lf", "--lr_fact", default=0.75, help="Learning rate decay factor.")
@click.option("-ls", "--lr_step", default=5, help="LR Step decay after nr. of epochs.")
@click.option("-a", "--after", default=5, help="Epoch steps to save model.")
@click.option("-wp", "--weight_prop", default=5.0, help="Factor for weighting property loss in VAE loss")
@click.option("-wp", "--weight_kld", default=0.25, help="Factor for weighting KL divergence loss in VAE loss")
@click.option("--scale/--no-scale", default=True, help="Whether to scale all properties from 0 to 1")
@click.option("-p", "--n_proc", default=6, help="Number of CPU processes to use")
def main(
    filename,
    checkpoint,
    epoch_load,
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
    weight_prop,
    weight_kld,
    scale,
    n_proc,
):
    if run_name is None:  # take filename as run name if not specified
        run_name = os.path.basename(filename).split(".")[0]

    _, t2i = tokenizer()
    dim_atom, dim_bond = get_input_dims()

    # load config from trained model
    ini = configparser.ConfigParser()
    ini.read(os.path.join(checkpoint, "config.ini"))
    conf = {}
    for k, v in ini["CONFIG"].items():
        try:
            conf[k] = int(v)
        except ValueError:
            try:
                conf[k] = float(v)
            except ValueError:
                conf[k] = v
    del ini

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
        "lr_step": lr_step,
        "save_after": after,
        "n_gnn_layers": conf["n_gnn_layers"],
        "n_rnn_layers": conf["n_rnn_layers"],
        "n_mlp_layers": conf["n_mlp_layers"],
        "dim_gnn": conf["dim_gnn"],
        "n_kernels": conf["n_kernels"],
        "dim_embed": conf["dim_embed"],
        "dim_rnn": conf["dim_rnn"],
        "dim_mlp": conf["dim_mlp"],
        "weight_props": weight_prop,
        "weight_kld": weight_kld,
        "scaled_props": scale,
    }
    config["n_props"] = n_props = len(CalcMolDescriptors(Chem.MolFromSmiles("O1CCNCC1")))
    config["alphabet"] = conf["alphabet"]

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

    # Create models and load weights
    gnn = AttentiveFP2(
        in_channels=dim_atom,
        hidden_channels=conf["dim_gnn"],
        out_channels=conf["dim_rnn"],
        edge_dim=dim_bond,
        num_layers=conf["n_gnn_layers"],
        num_timesteps=conf["n_kernels"],
        dropout=conf["dropout"],
    )
    rnn = LSTM(
        input_dim=conf["alphabet"],
        embedding_dim=conf["dim_embed"],
        hidden_dim=conf["dim_gnn"],
        layers=conf["n_rnn_layers"],
        dropout=conf["dropout"],
    )
    mlp = FFNN(
        input_dim=conf["dim_rnn"], hidden_dim=conf["dim_mlp"], n_layers=conf["n_mlp_layers"], output_dim=n_props
    )

    gnn.load_state_dict(torch.load(os.path.join(checkpoint, f"atfp_{epoch_load}.pt"), map_location=DEVICE))
    rnn.load_state_dict(torch.load(os.path.join(checkpoint, f"lstm_{epoch_load}.pt"), map_location=DEVICE))
    mlp.load_state_dict(torch.load(os.path.join(checkpoint, f"ffnn_{epoch_load}.pt"), map_location=DEVICE))
    gnn = gnn.to(DEVICE)
    rnn = rnn.to(DEVICE)
    mlp = mlp.to(DEVICE)

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
        steps=int(epoch_steps * batch_size * (1 + val)) if random else 0,
    )
    train_set, val_set = torch.utils.data.random_split(dataset, [1.0 - val, val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_proc, drop_last=False)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=n_proc, drop_last=False)
    writer = SummaryWriter(path_loss)
    print(
        f"Using {len(train_set)}{' random' if random else ''} molecules for training "
        + f"and {len(val_set)}{' random' if random else ''} for validation per epoch."
    )

    # KLD weight annealing
    total_steps = epochs * (epoch_steps if random else len(train_loader))
    anneal = anneal_cycle_sigmoid(total_steps, n_cycle=epochs // 2, n_grow=5, ratio=1.0)

    for epoch in range(1, epochs + 1):
        print(f"\n---------- Epoch {epoch} ----------")
        time_start = time.time()

        l_s, l_p, l_k, fk = train_one_epoch(
            gnn,
            rnn,
            mlp,
            optimizer,
            criterion1,
            criterion2,
            train_loader,
            writer,
            epoch,
            t2i,
            weight_prop,
            weight_kld,
            anneal,
        )
        l_vs, l_vp, l_vk = validate_one_epoch(
            gnn,
            rnn,
            mlp,
            criterion1,
            criterion2,
            valid_loader,
            writer,
            epoch * len(train_loader),
            t2i,
            weight_prop,
            weight_kld * fk,
        )
        dur = time.time() - time_start
        schedule.step()
        last_lr = schedule.get_last_lr()[0]
        writer.add_scalar("lr", last_lr, (epoch + 1) * len(train_loader))
        print(
            f"Epoch: {epoch}, Train Loss SMILES: {l_s:.3f}, Train Loss Props.: {l_p:.3f}, Train Loss KLD.: {l_k:.3f}, "
            + f"Val. Loss SMILES: {l_vs:.3f}, Val. Loss Props.: {l_vp:.3f}, Val. Loss KLD.: {l_vk:.3f}, "
            + f"Weight KLD: {fk*weight_kld:.6f}, LR: {last_lr:.6f}, Time: {dur//60:.0f}min {dur%60:.0f}sec"
        )

        _, _ = temperature_sampling(
            gnn, rnn, 0.5, np.random.choice(val_set.dataset.data, 10), 10, dataset.max_len, verbose=True, vae=True
        )

        # save loss and models
        if epoch % after == 0:
            torch.save(gnn.state_dict(), f"{path_model}atfp_{epoch}.pt")
            torch.save(rnn.state_dict(), f"{path_model}lstm_{epoch}.pt")
            torch.save(mlp.state_dict(), f"{path_model}ffnn_{epoch}.pt")


if __name__ == "__main__":
    main()
