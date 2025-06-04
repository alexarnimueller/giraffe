#! /usr/bin/env python
# -*- coding: utf-8

"""Script to embed molecules using a pretrained GIRAFFE model."""

import os
from time import time

import click
import numpy as np
import pandas as pd
import torch
from openTSNE import TSNE
from pacmap import PaCMAP
from sklearn.decomposition import PCA
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from giraffe.dataset import AttFPDataset
from giraffe.model import AttentiveFP, AttentiveFP2
from giraffe.utils import get_input_dims, read_config_ini

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.option("-d", "--delimiter", default="\t", help="Column delimiter of input file.")
@click.option(
    "-c", "--smls_col", default="SMILES", help="Name of column that contains SMILES."
)
@click.option(
    "-i", "--id_col", default=None, help="Name of column that contains compound IDs."
)
@click.option(
    "-f",
    "--folder",
    default="models/siglin_wae2",
    help="Checkpoint folder to load models from.",
)
@click.option("-e", "--epoch", default=85, help="Epoch of models to load.")
@click.option(
    "-b", "--batch_size", default=256, help="Batch size to use for embedding."
)
@click.option(
    "-n",
    "--n_mols",
    default=0,
    help="Number of molecules to randomly sub-sample. Default: 0 = all",
)
@click.option(
    "-j", "--n_jobs", default=4, help="Number of cores to use for data loader."
)
@click.option("-x", "--prec", default=4, help="Float precision")
@click.option(
    "-p",
    "--pca",
    is_flag=True,
    help="Whether embeddings should be further reduced by PCA",
)
@click.option(
    "-t",
    "--tsne",
    is_flag=True,
    help="Whether embeddings should be further reduced by a t-SNE",
)
@click.option(
    "-a",
    "--pacmap",
    is_flag=True,
    help="Whether embeddings should be further reduced by PacMap",
)
@click.option(
    "-z",
    "--n_comp",
    default=2,
    help="If PCA/t-SNE/PacMap is used, number of components to reduce to",
)
@click.option(
    "-x",
    "--perplex",
    default=30,
    help="Perplexity to be used for t-SNE dim. reduction.",
)
@click.option("-s", "--seed", default=None, help="Random seed for PCA / t-SNE / PacMap")
def main(
    input_file,
    output_file,
    delimiter,
    smls_col,
    id_col,
    folder,
    epoch,
    batch_size,
    n_mols,
    prec,
    pca,
    tsne,
    pacmap,
    n_comp,
    perplex,
    n_jobs,
    seed,
):
    seed = seed if seed else int(time())
    ids, embds = embed_file(
        input_file,
        delimiter,
        smls_col,
        id_col,
        folder,
        epoch,
        batch_size,
        n_mols,
        n_jobs,
        pca,
        tsne,
        pacmap,
        n_comp,
        perplex,
        seed,
    )
    out = np.concatenate(
        (np.asarray(ids).reshape(-1, 1), np.round(embds, int(prec))), axis=1
    )
    np.savetxt(output_file, out, delimiter=",", fmt="%s")
    print(f"Embeddings saved to {output_file}\n")


def embed_file(
    input_file,
    delimiter,
    smls_col,
    id_col,
    folder,
    epoch,
    batch_size,
    n_mols,
    n_jobs,
    pca,
    tsne,
    pacmap,
    n_comp,
    perplex,
    seed,
):
    dim_atom, dim_bond = get_input_dims()
    conf = read_config_ini(folder)
    wae = conf["wae"] == "True"
    vae = False if wae else conf["vae"] == "True"

    # Load data
    print(f"\nReading SMILES from {input_file}")
    data = pd.read_csv(input_file, delimiter=delimiter, engine="python")
    if n_mols:
        data = data.sample(n=int(n_mols), replace=False)
    smiles = data[smls_col].tolist()
    if id_col:
        ids = data[id_col].tolist()
    else:
        ids = smiles  # Use SMILES as IDs if no ID column is provided
    del data

    # Define GNN
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
    path = os.path.join(folder, f"atfp_{epoch}.pt")
    print(f"Loading pretrained model from {path}")
    gnn.load_state_dict(torch.load(path, map_location=DEVICE))
    gnn = gnn.to(DEVICE)

    # Embed molecules
    embs = smiles2embedding(
        gnn=gnn, smiles=smiles, batch_size=batch_size, n_jobs=n_jobs, vae=vae
    )

    if pca and not pacmap and not tsne:
        # PCA
        print("Running PCA on embeddings...")
        embs = PCA(n_components=n_comp, random_state=seed).fit_transform(embs)

    if tsne:
        print(f"Running t-SNE on {n_comp * 12}-PCA-compressed embeddings...")
        embs = TSNE(
            perplexity=perplex,
            initialization="pca",
            metric="cosine",
            n_jobs=n_jobs,
            random_state=seed,
        ).fit(PCA(n_components=n_comp * 12, random_state=seed).fit_transform(embs))

    if pacmap:
        print("Running PacMAP embeddings...")
        embs = PaCMAP(
            n_components=n_comp, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0
        ).fit_transform(embs)

    # Return embeddings for IDs
    return ids, embs


@torch.no_grad
def smiles2embedding(smiles, gnn, batch_size, n_jobs, vae):
    gnn.eval()
    dataset = AttFPDataset(smiles, embed=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=n_jobs
    )
    embs = torch.empty((0, gnn.out_channels), dtype=torch.float32).to(DEVICE)
    for g in loader:
        g = g.to(DEVICE)
        if vae:
            h, _ = gnn(g.atoms, g.edge_index, g.bonds, g.batch)
        else:
            h = gnn(g.atoms, g.edge_index, g.bonds, g.batch)
        embs = torch.cat((embs, h))
    return embs.detach().cpu().numpy()


if __name__ == "__main__":
    main()
