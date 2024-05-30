#! /usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit.Chem import AllChem, MolFromSmiles
from rdkit.DataStructs import DiceSimilarity

WDIR = "~/Code/Generative/GraphGiraffe"


@click.command()
@click.option("-n", "--n_mols", default=10000)
@click.option("-t", "--temp", default=0.01)
@click.option("-o", "--epoch", default=100)
@click.option("-c", "--checkpoint", type=click.Path(exists=True), default=f"{WDIR}/models/pub_vae_lin4_final")
@click.option("--col", type=str, default="black")
def main(n_mols, temp, epoch, checkpoint, col):
    # sample from interpolation
    # subprocess.run(
    #     [
    #         "python",
    #         f"{WDIR}/sampling.py",
    #         "-v",
    #         "-n",
    #         str(n_mols),
    #         "-t",
    #         f"{temp}",
    #         "-e",
    #         str(epoch),
    #         "-c",
    #         checkpoint,
    #         "-p",
    #         "-o",
    #         "similarity_parent",
    #     ]
    # )
    # read sampled compounds and calculate fingerprints
    # subprocess.run(["cp", f"{WDIR}/output/similarity_parent.csv", f"{WDIR}/paper/figures/similarity_parent.csv"])
    data = pd.read_csv(f"{WDIR}/paper/figures/similarity_parent.csv")
    data["Mol"] = data["SMILES"].apply(lambda s: MolFromSmiles(s) if MolFromSmiles(s) else None)
    data["Mol_Parent"] = data["Parent"].apply(lambda s: MolFromSmiles(s) if MolFromSmiles(s) else None)
    data["FP"] = data["Mol"].apply(lambda m: AllChem.GetMorganFingerprint(m, 2) if m else None)
    data["FP_Parent"] = data["Mol_Parent"].apply(lambda m: AllChem.GetMorganFingerprint(m, 2) if m else None)

    # calculate similarity to parent
    data["Tanimoto Similarity"] = data.apply(
        lambda row: DiceSimilarity(row["FP"], row["FP_Parent"]) if row["FP"] else None, axis=1
    )

    if len(data) < n_mols:  # if invalid molecules found
        n_mols = len(data)

    # save
    data[["SMILES", "Parent", "Tanimoto Similarity"]].to_csv(
        f"{WDIR}/paper/figures/similarity_parent.csv", index=False
    )

    # plot
    sns.displot(data["Tanimoto Similarity"], kind="kde", fill=True, color=col, aspect=1.5)
    plt.xlim(0.2, 1.0)
    plt.tight_layout()
    plt.savefig(f"{WDIR}/paper/figures/similarity_parent.pdf")
    plt.close()


if __name__ == "__main__":
    main()
