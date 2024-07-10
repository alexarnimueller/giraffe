#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os

import click
import matplotlib.pyplot as plt
import pandas as pd
from rdkit.Chem import MolFromSmiles
from tqdm.auto import tqdm

from dataset import PropertyScaler

WDIR = os.path.expanduser("~/Code/Generative/GraphGiraffe")


@click.command()
@click.argument("filename")
@click.option("-d", "--delim", default="\t")
@click.option("-s", "--smls_col", default="SMILES")
@click.option("-t", "--train_data", default=None)
def main(filename, delim, smls_col, train_data):
    data = pd.read_csv(filename, delimiter=delim)
    sclr = PropertyScaler(
        ["MolWt", "MolLogP", "qed", "NumHDonors", "NumAromaticRings", "FractionCSP3"], do_scale=False
    )
    names = list(sclr.descriptors.keys())
    mols = [MolFromSmiles(s) for s in tqdm(data[smls_col], desc="Smiles2Mol")]
    props = pd.DataFrame([sclr.transform(m) for m in tqdm(mols, desc="Calculating properties")], columns=names)

    if train_data:
        data_train = pd.read_csv(train_data)
        mols_train = [MolFromSmiles(s) for s in tqdm(data_train[smls_col], desc="Smiles2Mol")]
        props_train = pd.DataFrame(
            [sclr.transform(m) for m in tqdm(mols_train, desc="Calculating properties")], columns=names
        )

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for i, ax in enumerate(axes.flat):
        ax.hist(props[names[i]], bins=50, color="teal", alpha=0.66, density=True, label="sampled")
        if train_data:
            ax.hist(props_train[names[i]], bins=50, color="orange", alpha=0.66, density=True, label="training")
        ax.set_xlabel(names[i], fontsize=12, fontweight="bold")
        ax.set_yticks([])

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{WDIR}/paper/figures/props-hist.png")
    plt.close(fig)

    print("Properties of sampled compounds:")
    print(props.describe())
    if train_data:
        print("Properties of training compounds:")
        print(props_train.describe())


if __name__ == "__main__":
    main()
