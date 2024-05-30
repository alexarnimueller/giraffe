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
@click.option("-s", "--smls", default="c1ccccc1C2=NCC(=O)N(C)c3ccc(Cl)cc23")
@click.option("-n", "--n_mols", default=1000)
@click.option("-t", "--temp", default=0.5)
@click.option("-o", "--epoch", default=100)
@click.option("-c", "--checkpoint", type=click.Path(exists=True), default=f"{WDIR}/models/pub_vae_lin4_final")
@click.option("--col", type=str, default="blue")
def main(smls, n_mols, temp, epoch, checkpoint, col):
    # sample from interpolation
    subprocess.run(
        [
            "python",
            f"{WDIR}/sampling.py",
            "-v",
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
            "similarity_single",
        ]
    )
    # read sampled compounds and calculate fingerprints
    subprocess.run(["cp", f"{WDIR}/output/similarity_single.csv", f"{WDIR}/paper/figures/similarity_single.csv"])
    data = pd.read_csv(f"{WDIR}/paper/figures/similarity_single.csv")
    data["Mol"] = data["SMILES"].apply(lambda s: MolFromSmiles(s) if MolFromSmiles(s) else None)
    data["FP"] = data["Mol"].apply(lambda m: AllChem.GetMorganFingerprint(m, 2) if m else None)

    # calculate similarity to start and end
    m_goal = MolFromSmiles(smls)
    fp_goal = AllChem.GetMorganFingerprint(m_goal, 2)
    data["Tanimoto Similarity"] = data["FP"].apply(lambda fp: DiceSimilarity(fp, fp_goal) if fp else None)

    if len(data) < n_mols:  # if invalid molecules found
        n_mols = len(data)
    
    data.to_csv(f"{WDIR}/paper/figures/similarity_single.csv", index=False)

    # plot
    sns.displot(data["Tanimoto Similarity"], kind="kde", fill=True, color=col, aspect=1.5)
    plt.xlim(0.2, 1)
    plt.tight_layout()
    plt.savefig(f"{WDIR}/paper/figures/similarity_single.pdf")
    plt.close()


if __name__ == "__main__":
    main()
