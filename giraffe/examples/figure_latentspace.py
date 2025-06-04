#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""A script to plot the latent space of a VAE model using t-SNE or PCA for dimensionality reduction."""

import os

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from openTSNE import TSNE
from pacmap import PaCMAP
from rdkit.Chem import MolFromSmiles
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from giraffe.dataset import PropertyScaler
from giraffe.embedding import embed_file

WDIR = os.path.dirname(os.path.abspath(__file__).replace("examples/", ""))


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file")
@click.option("-d", "--delim", default="\t")
@click.option("-s", "--smls_col", default="SMILES")
@click.option("-n", "--n_mols", default=25600)
@click.option("-e", "--epoch", default=70)
@click.option(
    "-c",
    "--checkpoint",
    type=click.Path(exists=True),
    default=f"{WDIR}/models/pub_wae",
)
@click.option("-j", "--n_jobs", default=6)
@click.option("-t", "--tsne", is_flag=True, default=False)
@click.option("-p", "--pacmap", is_flag=True, default=False)
@click.option("-x", "--perplex", default=30)
def main(
    input_file,
    output_file,
    delim,
    smls_col,
    n_mols,
    epoch,
    checkpoint,
    n_jobs,
    tsne,
    pacmap,
    perplex,
):
    smls, embs = embed_file(
        input_file=input_file,
        delimiter=delim,
        smls_col=smls_col,
        id_col=None,
        folder=checkpoint,
        epoch=epoch,
        n_mols=n_mols,
        n_jobs=n_jobs,
        batch_size=512,
        pca=True,
        tsne=tsne,
        pacmap=pacmap,
        n_comp=2,
        perplex=perplex,
        seed=123456,
    )
    # properties
    print("Calculating properties")
    sclr = PropertyScaler(
        ["MolWt", "MolLogP", "qed", "NumHDonors", "NumAromaticRings", "FractionCSP3"],
        do_scale=True,
    )
    names = list(sclr.descriptors.keys())
    mols = [MolFromSmiles(s) for s in tqdm(smls, desc="Smiles2Mol")]
    props = pd.DataFrame(
        [sclr.transform(m) for m in tqdm(mols, desc="Calculating properties")],
        columns=names,
    )

    if not pacmap and not tsne:
        # PCA
        print("Running PCA on embeddings...")
        emb_red = PCA(n_components=2, random_state=42).fit_transform(embs)

    if tsne:
        print("Running t-SNE on PCA-compressed embeddings...")
        emb_red = TSNE(
            perplexity=perplex,
            initialization="pca",
            metric="cosine",
            n_jobs=n_jobs,
            random_state=42,
        ).fit(PCA(n_components=16, random_state=42).fit_transform(embs))

    if pacmap:
        print("Running PacMAP embeddings...")
        emb_red = PaCMAP(
            n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0
        ).fit_transform(embs)

    # plot
    print("Plotting...")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
    for i, ax in enumerate(axes.flat):
        ax.set_title(names[i])
        ax.scatter(
            emb_red[:, 0],
            emb_red[:, 1],
            s=5,
            alpha=0.66,
            cmap="gnuplot",
            c=props[names[i]],
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # color bar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap="gnuplot"),
        cax=cbar_ax,
        orientation="vertical",
        label="scaled values",
    )
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
