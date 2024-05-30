#! /usr/bin/env python
# -*- coding: utf-8 -*-

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from openTSNE import TSNE
from rdkit.Chem import MolFromSmiles
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from dataset import PropertyScaler
from embedding import embed_file

WDIR = "~/Code/Generative/GraphGiraffe"


@click.command()
@click.argument("filename")
@click.option("-d", "--delim", default="\t")
@click.option("-s", "--smls_col", default="SMILES")
@click.option("-n", "--n_mols", default=25600)
@click.option("-e", "--epoch", default=100)
@click.option("-c", "--checkpoint", type=click.Path(exists=True), default=f"{WDIR}/models/pubchem_vae_cyc2")
@click.option("-j", "--n_jobs", default=4)
@click.option("-t", "--tsne", is_flag=True, default=False)
@click.option("-x", "--perplex", default=40)
def main(filename, delim, smls_col, n_mols, epoch, checkpoint, n_jobs, tsne, perplex):
    smls, embs = embed_file(filename, delim, smls_col, checkpoint, epoch, 256, n_mols, n_jobs)
    # properties
    print("Calculating properties")
    sclr = PropertyScaler(["MolWt", "MolLogP", "qed", "NumHDonors", "NumAromaticRings", "FractionCSP3"], do_scale=True)
    names = list(sclr.descriptors.keys())
    mols = [MolFromSmiles(s) for s in tqdm(smls, desc="Smiles2Mol")]
    props = pd.DataFrame([sclr.transform(m) for m in tqdm(mols, desc="Calculating properties")], columns=names)

    # PCA
    print("Running PCA on embeddings...")
    emb_red = PCA(n_components=5, random_state=42).fit_transform(embs)

    if tsne:
        print("Running t-SNE on PCA-compressed embeddings...")
        emb_red = TSNE(
            perplexity=perplex,
            initialization="pca",
            metric="cosine",
            n_jobs=n_jobs,
            random_state=42,
        ).fit(emb_red)

    # plot
    print("Plotting...")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
    for i, ax in enumerate(axes.flat):
        ax.set_title(names[i])
        ax.scatter(emb_red[:, 0], emb_red[:, 1], s=5, alpha=0.66, cmap="gnuplot", c=props[names[i]])
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

    fname = f"{WDIR}/paper/figures/latentspace-tsne.png" if tsne else f"{WDIR}/paper/figures/latentspace-pca.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
