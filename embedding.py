#! /usr/bin/env python
# -*- coding: utf-8

import configparser
import os

import click
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from dataset import AttFPDataset
from model import AttentiveFP
from utils import get_input_dims

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.option("-d", "--delimiter", default="\t", help="Column delimiter of input file.")
@click.option("-c", "--smls_col", default="SMILES", help="Name of column that contains SMILES.")
@click.option("-f", "--folder", help="Checkpoint folder to load models from.")
@click.option("-e", "--epoch", help="Epoch of models to load.")
@click.option("-b", "--batch_size", default=512, help="Batch size to use for embedding.")
@click.option("-n", "--n_jobs", default=4, help="Number of cores to use for data loader.")
def main(input_file, output_file, delimiter, smls_col, folder, epoch, batch_size, n_jobs):
    dim_atom, dim_bond = get_input_dims()
    ini = configparser.ConfigParser()
    ini.read(os.path.join(folder, "config.ini"))
    conf = {}
    for k, v in ini["CONFIG"].items():
        try:
            conf[k] = int(v)
        except ValueError:
            try:
                conf[k] = float(v)
            except ValueError:
                conf[k] = v

    # Load data
    print(f"\nReading SMILES from {input_file}")
    data = pd.read_csv(input_file, delimiter=delimiter)
    smiles = data[smls_col].tolist()
    del data

    # Define GNN
    gnn = AttentiveFP(
        in_channels=dim_atom,
        hidden_channels=conf["dim_gnn"],
        out_channels=conf["dim_rnn"],
        dropout=conf["dropout"],
        edge_dim=dim_bond,
        num_layers=conf["n_layers"],
        num_timesteps=conf["n_kernels"],
    )
    path = os.path.join(folder, f"atfp_{epoch}.pt")
    print(f"Loading pretrained model from {path}")
    gnn.load_state_dict(torch.load(path, map_location=DEVICE))
    gnn = gnn.to(DEVICE)

    # Embed molecules
    embs = smiles_embedding(gnn=gnn, smiles=smiles, batch_size=batch_size, n_jobs=n_jobs)

    # Concatenate and save embeddings
    out = np.concatenate((np.asarray(smiles).reshape(-1, 1), embs), axis=1)
    np.savetxt(output_file, out, delimiter=",", fmt="%s")
    print(f"Embeddings saved to {output_file}\n")


@torch.no_grad
def smiles_embedding(gnn, smiles, batch_size, n_jobs):
    gnn.eval()
    dataset = AttFPDataset(smiles)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_jobs)
    embs = torch.empty((0, gnn.out_channels), dtype=torch.float32).to(DEVICE)
    for g in tqdm(loader, desc="Embedding"):
        g = g.to(DEVICE)
        embs = torch.cat((embs, gnn(g.atoms, g.edge_index, g.bonds, g.batch)))
    return embs.detach().cpu().numpy()


if __name__ == "__main__":
    main()
