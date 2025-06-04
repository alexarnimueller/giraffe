#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""A script to plot the distribution of ECFP4 similarity of molecules sampled from a single molecule."""

import os
import subprocess

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit.Chem import AllChem, MolFromSmiles
from rdkit.DataStructs import DiceSimilarity

WDIR = os.path.dirname(os.path.abspath(__file__).replace("examples/", ""))


@click.command()
@click.option("-s", "--smls", default="c1ccccc1C2=NCC(=O)N(C)c3ccc(Cl)cc23")
@click.option("-n", "--n_mols", default=1000)
@click.option("-t", "--temp", default=0.5)
@click.option("-o", "--epoch", default=70)
@click.option(
    "-c", "--checkpoint", type=click.Path(exists=True), default=f"{WDIR}/models/wae_pub"
)
@click.option("--col", type=str, default="blue")
def main(smls, n_mols, temp, epoch, checkpoint, col):
    # sample from interpolation
    name = checkpoint.split("/")[-1]
    subprocess.run(
        [
            "python",
            f"{WDIR}/sampling.py",
            "-s",
            f"{smls}",
            "-n",
            str(n_mols),
            "-t",
            f"{temp}",
            "-e",
            str(epoch),
            "-c",
            checkpoint,
            "-o",
            f"{WDIR}/output/similarity_single_{name}.csv",
        ]
    )
    # read sampled compounds and calculate fingerprints
    data = pd.read_csv(f"{WDIR}/output/similarity_single_{name}.csv")
    data["Mol"] = data["SMILES"].apply(
        lambda s: MolFromSmiles(s) if MolFromSmiles(s) else None
    )
    data["FP"] = data["Mol"].apply(
        lambda m: AllChem.GetMorganFingerprint(m, 2) if m else None
    )

    # calculate similarity to start and end
    m_goal = MolFromSmiles(smls)
    fp_goal = AllChem.GetMorganFingerprint(m_goal, 2)
    data["Tanimoto Similarity"] = data["FP"].apply(
        lambda fp: DiceSimilarity(fp, fp_goal) if fp else None
    )

    if len(data) < n_mols:  # if invalid molecules found
        n_mols = len(data)

    data.to_csv(f"{WDIR}/output/similarity_single_{name}.csv", index=False)

    # plot
    sns.displot(
        data["Tanimoto Similarity"], kind="kde", fill=True, color=col, aspect=1.5
    )
    plt.xlim(0.2, 1)
    plt.tight_layout()
    plt.savefig(f"{WDIR}/output/similarity_single_{name}.pdf")
    plt.close()


if __name__ == "__main__":
    main()
