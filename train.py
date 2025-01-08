#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configparser
import os
import time

import click
import numpy as np
import torch
import torch.nn as nn
from rdkit import RDLogger
from rdkit.rdBase import DisableLog
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from dataset import AttFPDataset, AttFPTableDataset, load_from_fname, tokenizer
from model import (
    FFNN,
    LSTM,
    AttentiveFP,
    AttentiveFP2,
    create_annealing_schedule,
    reparameterize,
    vae_loss_func,
)
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
@click.option("-e", "--epochs", default=100, help="Nr. of epochs to train.")
@click.option("-o", "--dropout", default=0.1, help="Dropout fraction.")
@click.option("-b", "--batch_size", default=256, help="Number of molecules per batch.")
@click.option("-r", "--random", is_flag=True, help="Randomly sample molecules in each training step.")
@click.option("-p", "--props", default=None, help="Comma-seperated list of descriptors to use. All, if omitted")
@click.option("--epoch_steps", "--es", default=1000, help="If random, number of batches per epoch.")
@click.option("-v", "--val", default=0.05, help="Fraction of the data to use for validation.")
@click.option("-l", "--lr", default=1e-3, help="Learning rate.")
@click.option("--lr_fact", "--lf", default=0.75, help="Learning rate decay factor.")
@click.option("--lr_step", "--ls", default=10, help="LR Step decay after nr. of epochs.")
@click.option("-a", "--after", default=5, help="Epoch steps to save model.")
@click.option("-t", "--temp", default=0.5, help="Temperature to use during SMILES sampling.")
@click.option("--n_sample", "--ns", default=100, help="Nr. SMILES to sample after each trainin epoch.")
@click.option("--kernels_gnn", "--nk", default=2, help="Nr. GNN kernels")
@click.option("--layers_gnn", "--ng", default=2, help="Nr. GNN layers")
@click.option("--layers_rnn", "--nr", default=2, help="Nr. RNN layers")
@click.option("--layers_mlp", "--nm", default=2, help="Nr. MLP layers")
@click.option("--dim_gnn", "--dg", default=512, help="Hidden dimension of GNN layers")
@click.option("--dim_rnn", "--dr", default=512, help="Hidden dimension of RNN layers")
@click.option("--dim_tok", "--dt", default=64, help="Dimension of RNN token embedding")
@click.option("--dim_mlp", "--dm", default=512, help="Hidden dimension of MLP layers")
@click.option("--weight_prop", "--wp", default=20.0, help="Factor for weighting property loss in VAE loss")
@click.option("--weight_vae", "--wk", default=0.2, help="Factor for weighting KL divergence loss in VAE loss")
@click.option(
    "--anneal_type",
    "--at",
    default="sigmoid",
    help="Shape of VAE weight annealing: constant, linear, sigmoid, cyc_linear, cyc_sigmoid, cyc_sigmoid_lin",
)
@click.option("--anneal_start", "--as", default=10, help="Epoch at which to start VAE loss annealing.")
@click.option("--anneal_stop", "--ao", default=None, help="Epoch at which to stop VAE loss annealing (stay const.)")
@click.option("--anneal_cycle", "--ac", default=5, help="Number of epochs for one VAE loss annealing cycle")
@click.option("--anneal_grow", "--ag", default=5, help="Number of annealing cycles with increasing values")
@click.option("--anneal_ratio", "--ar", default=0.75, help="Fraction of annealing vs. constant VAE loss weight")
@click.option("--vae/--no-vae", default=True, help="Whether to train a variational AE or classical AE")
@click.option("--wae/--no-wae", default=False, help="Whether to train a Wasserstein autoencoder using MMD")
@click.option("--scale/--no-scale", default=True, help="Whether to scale all properties from 0 to 1")
@click.option("--n_proc", "--np", default=6, help="Number of CPU processes to use")
def main(
    filename,
    run_name,
    delimiter,
    smls_col,
    epochs,
    dropout,
    batch_size,
    random,
    props,
    epoch_steps,
    val,
    lr,
    lr_fact,
    lr_step,
    after,
    temp,
    n_sample,
    kernels_gnn,
    layers_gnn,
    layers_rnn,
    layers_mlp,
    dim_gnn,
    dim_rnn,
    dim_tok,
    dim_mlp,
    weight_prop,
    weight_vae,
    anneal_type,
    anneal_start,
    anneal_stop,
    anneal_cycle,
    anneal_grow,
    anneal_ratio,
    vae,
    wae,
    scale,
    n_proc,
):
    if run_name is None:  # take filename as run name if not specified
        run_name = os.path.basename(filename).split(".")[0]

    _, t2i = tokenizer()
    dim_atom, dim_bond = get_input_dims()

    # Write parameters to config file and define variables
    weight_vae = weight_vae if (vae or wae) else 0.0
    anneal_cycle = anneal_cycle if (vae or wae) else 0
    anneal_grow = anneal_grow if (vae or wae) else 0.0
    anneal_ratio = anneal_ratio if (vae or wae) else 0
    vae = vae if not wae else False
    ini = configparser.ConfigParser()
    config = {
        "filename": filename,
        "run_name": run_name,
        "epochs": epochs,
        "dropout": dropout,
        "batch_size": batch_size,
        "random": random,
        "porps": props if props else "full",
        "n_proc": n_proc,
        "epoch_steps": epoch_steps if random else "full",
        "frac_val": val,
        "lr": lr,
        "lr_fact": lr_fact,
        "lr_step": lr_step,
        "save_after": after,
        "n_gnn_layers": layers_gnn,
        "n_rnn_layers": layers_rnn,
        "n_mlp_layers": layers_mlp,
        "dim_gnn": dim_gnn,
        "n_kernels": kernels_gnn,
        "dim_embed": dim_tok,
        "dim_rnn": dim_rnn,
        "dim_mlp": dim_mlp,
        "weight_props": weight_prop,
        "weight_vae": weight_vae,
        "anneal_type": anneal_type,
        "anneal_start": anneal_start,
        "anneal_stop": anneal_stop,
        "anneal_cycle": anneal_cycle,
        "anneal_grow": anneal_grow,
        "anneal_ratio": anneal_ratio,
        "scaled_props": scale,
        "vae": vae,
        "wae": wae,
    }
    config["alphabet"] = alphabet = len(t2i)

    # Define paths
    path_model = f"models/{run_name}/"
    path_loss = f"logs/{run_name}/"
    os.makedirs(path_model, exist_ok=True)
    os.makedirs(path_loss, exist_ok=True)
    print("\nPaths (model, loss): ", path_model, path_loss)

    # load data to determine if there are properties
    print("\nReading SMILES dataset...")
    content = load_from_fname(filename, smls_col=smls_col, delimiter=delimiter)
    DatasetClass = AttFPDataset if content.columns.tolist() == [smls_col] else AttFPTableDataset
    dataset = DatasetClass(
        filename=filename,
        delimiter=delimiter,
        smls_col=smls_col,
        props=props,
        random=random,
        scaled_props=scale,
        steps=int(epoch_steps * batch_size * (1 + val)),
    )
    config["n_props"] = n_props = dataset.n_props
    train_set, val_set = torch.utils.data.random_split(dataset, [1.0 - val, val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_proc, drop_last=False)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=n_proc, drop_last=False)

    # store config file for later sampling, retraining etc.
    with open(f"{path_model}config.ini", "w") as configfile:
        ini["CONFIG"] = config
        ini.write(configfile)

    # Create models
    GNN_Class = AttentiveFP2 if vae else AttentiveFP
    gnn = GNN_Class(
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
        embedding_dim=dim_tok,
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
    writer = SummaryWriter(path_loss)
    print(
        f"Using {len(train_set)}{' random' if random else ''} molecules for training "
        + f"and {len(val_set)}{' random' if random else ''} for validation per epoch."
    )

    # VAE loss weight annealing
    anneal_stop = epochs if anneal_stop is None else anneal_stop
    anneal = create_annealing_schedule(
        epochs,
        epoch_steps if random else len(train_loader),
        anneal_start,
        anneal_stop,
        anneal_cycle,
        anneal_grow,
        anneal_ratio,
        anneal_type,
    )

    for epoch in range(1, epochs + 1):
        print(f"\n---------- Epoch {epoch} ----------")
        time_start = time.time()
        l_s, l_p, l_k, fv = train_one_epoch(
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
            weight_vae,
            anneal,
            vae,
            wae,
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
            weight_vae * fv,
            vae,
            wae,
        )
        dur = time.time() - time_start
        schedule.step()
        last_lr = schedule.get_last_lr()[0]
        writer.add_scalar("lr", last_lr, epoch * len(train_loader))

        # sampling
        valids, _, _, _, _, _ = temperature_sampling(
            gnn,
            rnn,
            temp,
            np.random.choice(val_set.dataset.data[smls_col], n_sample),
            n_sample,
            dataset.max_len,
            verbose=True,
            vae=vae,
            wae=wae,
        )
        valid = len(valids) / n_sample
        writer.add_scalar("valid", valid, epoch * len(train_loader))

        print(
            f"Epoch: {epoch}, Train Loss SMILES: {l_s:.3f}, Train Loss Props.: {l_p:.3f}, Train Loss VAE.: {l_k:.3f}, "
            + f"Val. Loss SMILES: {l_vs:.3f}, Val. Loss Props.: {l_vp:.3f}, Val. Loss VAE.: {l_vk:.3f}, "
            + f"Weight VAE: {fv * weight_vae:.6f}, Frac. valid: {valid:.3f}, LR: {last_lr:.6f}, "
            + f"Time: {dur // 60:.0f}min {dur % 60:.0f}sec"
        )

        # save loss and models
        if epoch % after == 0:
            torch.save(gnn.state_dict(), f"{path_model}atfp_{epoch}.pt")
            torch.save(rnn.state_dict(), f"{path_model}lstm_{epoch}.pt")
            torch.save(mlp.state_dict(), f"{path_model}ffnn_{epoch}.pt")


def train_one_epoch(
    gnn, rnn, mlp, optimizer, criterion1, criterion2, train_loader, writer, epoch, t2i, wp, wk, asteps, vae, wae
):
    gnn.train(True)
    rnn.train(True)
    mlp.train(True)
    steps = len(train_loader)
    total_loss, total_smls, total_props, total_vae, f_vae, step = 0, 0, 0, 0, 0, 0
    for g in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        # graph and target smiles
        g = g.to(DEVICE)
        trg = g.trg_smi
        trg = torch.transpose(trg, 0, 1)

        # get GNN embedding as input for RNN (h0) and FFNN
        if vae and not wae:
            mu, var = gnn(g.atoms, g.edge_index, g.bonds, g.batch)
            hn = reparameterize(mu, var)
        else:
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

        # Properties loss (mask unavailable properties)
        pred_props = mlp(hn[0])
        pred_props = pred_props * g.prop_mask.reshape(-1, pred_props.size(1))
        props = g.props.reshape(-1, pred_props.size(1)) * g.prop_mask.reshape(-1, pred_props.size(1))
        loss_props = torch.nan_to_num(criterion2(pred_props, props), nan=1e-6)

        if vae or wae:
            # combine losses in VAE style
            if wae:
                mu, var = hn[0], None  # no reparameterization, output is "mean", no var needed for loss
            f_vae = asteps[(epoch - 1) * steps + step]  # annealing)
            loss, loss_vae = vae_loss_func(loss_mol / j, loss_props, mu, var, wk * f_vae, wp, wae)
            total_vae += loss_vae.cpu().detach().numpy()
        else:
            # combine losses, apply desired weight to property loss
            loss = loss_mol + torch.multiply(loss_props, wp)
        loss.backward(retain_graph=True)
        optimizer.step()

        # calculate total losses
        total_loss += loss.cpu().detach().numpy()
        total_smls += loss_mol.cpu().detach().numpy()[0] / j
        total_props += loss_props.cpu().detach().numpy()

        # write tensorboard summary
        step += 1
        writer.add_scalar("loss_train_total", total_loss / step, (epoch - 1) * steps + step)
        writer.add_scalar("loss_train_smiles", total_smls / step, (epoch - 1) * steps + step)
        writer.add_scalar("loss_train_props", total_props / step, (epoch - 1) * steps + step)
        if vae or wae:
            writer.add_scalar("loss_train_vae", total_vae / step, (epoch - 1) * steps + step)
            writer.add_scalar("vae_weight", wk * f_vae, (epoch - 1) * steps + step)
        if step % 500 == 0:
            print(f"Step: {step}/{steps}, Loss SMILES: {total_smls / step:.3f}, Loss Props.: {total_props / step:.3f}")

    return (total_smls / steps, total_props / steps, total_vae / steps, f_vae)


@torch.no_grad
def validate_one_epoch(gnn, rnn, mlp, criterion1, criterion2, valid_loader, writer, step, t2i, wp, wk, vae, wae):
    gnn.train(False)
    rnn.train(False)
    mlp.train(False)
    steps = len(valid_loader)
    total_total, total_smls, total_props, total_vae = 0, 0, 0, 0
    with torch.no_grad():
        for g in tqdm(valid_loader, desc="Validation"):
            # graph and target smiles
            g = g.to(DEVICE)
            trg = g.trg_smi
            trg = torch.transpose(trg, 0, 1)

            # build hidden and cell value inputs
            if vae and not wae:
                mu, var = gnn(g.atoms, g.edge_index, g.bonds, g.batch)
                hn = reparameterize(mu, var)
            else:
                hn = gnn(g.atoms, g.edge_index, g.bonds, g.batch)
            hn = torch.cat(
                [hn.unsqueeze(0)] + [torch.zeros((1, hn.size(0), hn.size(1))).to(DEVICE)] * (rnn.n_layers - 1), dim=0
            )
            cn = torch.zeros((rnn.n_layers, hn.size(1), rnn.hidden_dim)).to(DEVICE)

            # SMILES
            mol_loss = torch.zeros(1).to(DEVICE)
            finish = (trg == t2i["$"]).nonzero()[-1, 0]
            for j in range(finish - 1):  # only loop until last end-token to prevent nan loss
                nxt, (hn, cn) = rnn(trg[j, :].unsqueeze(0), (hn, cn))
                loss_fwd = criterion1(nxt.view(trg.size(1), -1), trg[j + 1, :])
                mol_loss = torch.add(mol_loss, loss_fwd)

            # loss
            # properties
            pred_props = mlp(hn[0])
            pred_props = pred_props * g.prop_mask.reshape(-1, pred_props.size(1))
            props = g.props.reshape(-1, pred_props.size(1)) * g.prop_mask.reshape(-1, pred_props.size(1))
            loss_props = criterion2(pred_props, props)
            if vae or wae:  # vae
                if wae:
                    mu, var = hn[0], None  # no reparameterization, output is "mean", no var needed for loss
                loss, loss_vae = vae_loss_func(mol_loss / j, loss_props, mu, var, wk, wp, wae)
                total_vae += loss_vae.cpu().detach().numpy()
            else:  # just weighted
                loss = mol_loss + torch.multiply(loss_props, wp)

            # calculate total losses
            total_total += loss.cpu().detach().numpy()
            total_props += loss_props.cpu().detach().numpy()
            total_smls += mol_loss.cpu().detach().numpy()[0] / j

    # write tensorboard summary for this epoch and return validation loss
    writer.add_scalar("loss_val_total", total_total / steps, step)
    writer.add_scalar("loss_val_smiles", total_smls / steps, step)
    writer.add_scalar("loss_val_props", total_props / steps, step)
    if vae or wae:
        writer.add_scalar("loss_val_vae", total_vae / steps, step)
    return (total_smls / steps, total_props / steps, total_vae / steps)


if __name__ == "__main__":
    main()
