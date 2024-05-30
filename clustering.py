#! /usr/bin/env python
# -*- coding: utf-8

import configparser
import os

import click
import hdbscan
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from dataset import AttFPDataset
from model import AttentiveFP2
from utils import get_input_dims

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.option("-d", "--delimiter", default="\t", help="Column delimiter of input file.")
@click.option("-c", "--smls_col", default="SMILES", help="Name of column that contains SMILES.")
@click.option("-i", "--id_col", default=None, help="Name of the compound ID column.")
@click.option("-f", "--folder", default="models/pub_vae_lin_final", help="Checkpoint folder to load models from.")
@click.option("-e", "--epoch", default=45, help="Epoch of models to load.")
@click.option("-b", "--batch_size", default=512, help="Batch size to use for embedding.")
@click.option("-n", "--n_mols", default=0, help="Number of molecules to randomly sample. 0 = all")
@click.option("-j", "--n_jobs", default=4, help="Number of cores to use for data loader.")
def main(input_file, output_file, delimiter, smls_col, id_col, folder, epoch, batch_size, n_mols, n_jobs):
    smls, embds, ids = embed_file(input_file, delimiter, smls_col, id_col, folder, epoch, batch_size, n_mols, n_jobs)
    print("HDBSCAN Clustering...")
    clst = hdbscan.HDBSCAN()
    clst.fit(embds)
    print(f"Identified {max(clst.labels_)+1} clusters (plus singletons).")
    if id_col:
        out = np.concatenate((np.asarray(ids).reshape(-1, 1), clst.labels_.reshape(-1, 1)), axis=1)
    else:
        out = np.concatenate((np.asarray(smls).reshape(-1, 1), clst.labels_.reshape(-1, 1)), axis=1)
    np.savetxt(output_file, out, delimiter=",", fmt="%s")
    print(f"HDBSCAN clusters saved to {output_file}\n")


def embed_file(input_file, delimiter, smls_col, id_col, folder, epoch, batch_size, n_mols, n_jobs):
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
    print(f"\nReading {input_file}")
    data = pd.read_csv(input_file, delimiter=delimiter)
    if n_mols:
        data = data.sample(n=int(n_mols), replace=False)
    smiles = data[smls_col].tolist()
    if id_col:
        ids = data[id_col].tolist()
    del data

    # Define GNN
    gnn = AttentiveFP2(
        in_channels=dim_atom,
        hidden_channels=conf["dim_gnn"],
        out_channels=conf["dim_rnn"],
        edge_dim=dim_bond,
        num_layers=conf["n_gnn_layers"],
        num_timesteps=conf["n_kernels"],
        dropout=conf["dropout"],
    )
    path = os.path.join(folder, f"atfp_{epoch}.pt")
    print(f"Loading pretrained model from {path}")
    gnn.load_state_dict(torch.load(path, map_location=DEVICE))
    gnn = gnn.to(DEVICE)

    # Embed molecules
    embs = smiles_embedding(gnn=gnn, smiles=smiles, batch_size=batch_size, n_jobs=n_jobs)

    # Concatenate and save embeddings
    return smiles, embs, ids if id_col else None


@torch.no_grad
def smiles_embedding(gnn, smiles, batch_size, n_jobs):
    gnn.eval()
    dataset = AttFPDataset(smiles)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_jobs)
    embs = torch.empty((0, gnn.out_channels), dtype=torch.float32).to(DEVICE)
    for g in tqdm(loader, desc="Embedding"):
        g = g.to(DEVICE)
        h, _ = gnn(g.atoms, g.edge_index, g.bonds, g.batch)
        embs = torch.cat((embs, h))
    return embs.detach().cpu().numpy()


if __name__ == "__main__":
    main()
