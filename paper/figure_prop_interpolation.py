#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess

import click
import matplotlib.pyplot as plt
import pandas as pd
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import CalcMolDescriptors

WDIR = os.path.expanduser("~/Code/Generative/GraphGiraffe")


@click.command()
@click.option("-s", "--start", default="O=C(O)[C@@H]2N3C(=O)[C@@H](NC(=O)[C@@H](c1ccc(O)cc1)N)[C@H]3SC2(C)C")
@click.option("-e", "--end", default="c1ccccc1C2=NCC(=O)N(C)c3ccc(Cl)cc23")
@click.option("-n", "--n_steps", default=100)
@click.option("-t", "--temp", default=0.1)
@click.option("-o", "--epoch", default=45)
@click.option("-c", "--checkpoint", type=click.Path(exists=True), default=f"{WDIR}/models/pub_vae_lin_final")
def main(start, end, n_steps, temp, epoch, checkpoint):
    # sample from interpolation
    # subprocess.run(
    #     [
    #         "python",
    #         f"{WDIR}/sampling.py",
    #         "-i",
    #         # "-v",
    #         "-s",
    #         f"{start},{end}",
    #         "-n",
    #         str(n_steps),
    #         "-t",
    #         f"{temp}",
    #         "-e",
    #         str(epoch),
    #         "-c",
    #         checkpoint,
    #         "-o",
    #         "output/interpolation_props.csv",
    #     ]
    # )
    # # read sampled compounds and calculate properties
    # subprocess.run(["cp", f"{WDIR}/output/interpolation_props.csv", f"{WDIR}/paper/figures/interpolation_props.csv"])
    data = pd.read_csv(f"{WDIR}/paper/figures/interpolation_props.csv")
    data["Mol"] = data["SMILES"].apply(lambda s: MolFromSmiles(s) if MolFromSmiles(s) else None)
    data["Desc"] = data["Mol"].apply(lambda m: CalcMolDescriptors(m) if m else None)

    if len(data) < n_steps:  # if invalid molecules found
        n_steps = len(data)

    # plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    ax1.plot(range(n_steps), data["Desc"].apply(lambda row: row["MolWt"]), "k", label="MW", lw=2)
    ax2.plot(range(n_steps), data["Desc"].apply(lambda row: row["MolLogP"]), "k", label="LogP", lw=2)
    ax3.plot(range(n_steps), data["Desc"].apply(lambda row: row["TPSA"]), "k", label="TPSA", lw=2)
    ax4.plot(range(n_steps), data["Desc"].apply(lambda row: row["FractionCSP3"]), "k", label="fCsp3", lw=2)
    ax1.set_ylabel("MW", fontsize=16, fontweight="bold")
    ax2.set_ylabel("LogP", fontsize=16, fontweight="bold")
    ax3.set_xlabel("# Steps", fontsize=16, fontweight="bold")
    ax3.set_ylabel("TPSA", fontsize=16, fontweight="bold")
    ax4.set_xlabel("# Steps", fontsize=16, fontweight="bold")
    ax4.set_ylabel("fCsp3", fontsize=16, fontweight="bold")
    ax1.legend(loc="best", fontsize=14)
    ax2.legend(loc="best", fontsize=14)
    ax3.legend(loc="best", fontsize=14)
    ax4.legend(loc="best", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{WDIR}/paper/figures/interpolation-props.png")
    plt.close(fig)


if __name__ == "__main__":
    main()