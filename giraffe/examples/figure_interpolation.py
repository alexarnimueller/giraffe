#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""A script to plot the interpolation between two molecules in the latent space as ECFP4 similarity."""

import os
import subprocess

import click
import matplotlib.pyplot as plt
import pandas as pd
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
from rdkit.DataStructs import DiceSimilarity

WDIR = os.path.dirname(os.path.abspath(__file__).replace("examples/", ""))


@click.command()
@click.option(
    "-s",
    "--start",
    default="O=C(O)[C@@H]2N3C(=O)[C@@H](NC(=O)[C@@H](c1ccc(O)cc1)N)[C@H]3SC2(C)C",
)
@click.option("-e", "--end", default="c1ccccc1C2=NCC(=O)N(C)c3ccc(Cl)cc23")
@click.option("-n", "--n_steps", default=100)
@click.option("-t", "--temp", default=0.25)
@click.option("-o", "--epoch", default=70)
@click.option(
    "-c",
    "--checkpoint",
    type=click.Path(exists=True),
    default=f"{WDIR}/models/pub_wae",
)
def main(start, end, n_steps, temp, epoch, checkpoint):
    name = checkpoint.split("/")[-1]
    output = f"{WDIR}/output/interpolation_{name}.csv"
    # sample from interpolation
    subprocess.run(
        [
            "python",
            f"{WDIR}/sampling.py",
            "-i",
            # "-v",
            "-s",
            f"{start},{end}",
            "-n",
            str(n_steps),
            "-t",
            f"{temp}",
            "-e",
            str(epoch),
            "-c",
            checkpoint,
            "-o",
            f"{output}",
        ]
    )

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    data = pd.read_csv(f"{output}")
    data["Mol"] = data["SMILES"].apply(
        lambda s: MolFromSmiles(s) if MolFromSmiles(s) else None
    )
    data["FP"] = data["Mol"].apply(lambda m: fpgen.GetFingerprint(m) if m else None)

    # calculate similarity to start and end
    fp_start = fpgen.GetFingerprint(MolFromSmiles(start))
    fp_end = fpgen.GetFingerprint(MolFromSmiles(end))
    data["Sim_Start"] = data["FP"].apply(
        lambda fp: DiceSimilarity(fp, fp_start) if fp else None
    )
    data["Sim_End"] = data["FP"].apply(
        lambda fp: DiceSimilarity(fp, fp_end) if fp else None
    )

    if len(data) < n_steps:  # if invalid molecules found
        n_steps = len(data)

    # plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(range(n_steps), data["Sim_Start"], label="To Start", color="r", lw=2)
    ax.plot(range(n_steps), data["Sim_End"], label="To End", color="b", lw=2)
    ax.set_xlabel("# Steps", fontsize=16, fontweight="bold")
    ax.set_ylabel("Tanimoto Similarity", fontsize=16, fontweight="bold")
    ax.set_ylim([0.1, 1])
    ax.legend(fontsize=16, shadow=True, loc="upper center")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output[:-4]}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
