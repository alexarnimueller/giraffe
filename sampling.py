#! /usr/bin/env python
# -*- coding: utf-8

import configparser
import os

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.rdBase import DisableLog
from torch_geometric.loader import DataLoader

from dataset import OneMol, tokenizer
from model import RNN, AttentiveFP
from utils import get_input_dims, is_valid_mol

for level in RDLogger._levels:
    DisableLog(level)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.argument("checkpointfolder")
@click.option("-e", "--epoch", help="Epoch of models to load.")
@click.option("-s", "--smiles", help="Reference SMILES to use as seed for sampling.")
@click.option("-n", "--num", default=100, help="How many molecules to sample.")
@click.option("-t", "--temp", default=0.6, help="Temperature to use for sampling.")
@click.option("-m", "--maxlen", default=100, help="Maximum allowed SMILES string length.")
def main(checkpointfolder, epoch, smiles, num, temp, maxlen):
    assert Chem.MolFromSmiles(smiles), "invalid SMILES string!"
    dim_atom, dim_bond = get_input_dims()
    ini = configparser.ConfigParser()
    ini.read(os.path.join(checkpointfolder, "config.ini"))
    conf = {}
    for k, v in ini["CONFIG"].items():
        try:
            conf[k] = int(v)
        except ValueError:
            try:
                conf[k] = float(v)
            except ValueError:
                conf[k] = v

    # Define models
    rnn = RNN(
        size_vocab=conf["alphabet"],
        hidden_dim=conf["dim_hidden"],
        layers=conf["n_layers"],
        dropout=conf["dropout"],
    )
    gnn = AttentiveFP(
        in_channels=dim_atom,
        hidden_channels=conf["dim_hidden"],
        out_channels=conf["dim_rnn"],
        dropout=conf["dropout"],
        edge_dim=dim_bond,
        num_layers=conf["n_layers"],
        num_timesteps=conf["n_kernels"],
    )

    gnn.load_state_dict(torch.load(os.path.join(checkpointfolder, f"egnn_{epoch}.pt"), map_location=DEVICE))
    rnn.load_state_dict(torch.load(os.path.join(checkpointfolder, f"lstm_{epoch}.pt"), map_location=DEVICE))
    gnn = gnn.to(DEVICE)
    rnn = rnn.to(DEVICE)

    # Sample molecules
    print(f"Sampling {num} molecules:")
    novels, probs_abs = temperature_sampling(gnn=gnn, rnn=rnn, temp=temp, smiles=smiles, num_mols=num, maxlen=maxlen)

    # Save predictions
    df = pd.DataFrame({"SMILES": novels, "log-likelihoodog": probs_abs})
    df_sorted = df.sort_values(by=["log-likelihoodog"], ascending=False)
    os.makedirs("output/", exist_ok=True)
    df_sorted.to_csv("output/sampled.csv", index=False)


def temperature_sampling(gnn, rnn, temp, smiles, num_mols, maxlen):
    gnn.eval()
    rnn.eval()
    softmax = nn.Softmax(dim=1)
    i2t, t2i = tokenizer()

    mol = OneMol(smiles, maxlen)
    loader = DataLoader(mol, batch_size=1)
    g = next(iter(loader)).to(DEVICE)

    smiles_list, score_list = [], []
    for _ in range(num_mols):  # trange(
        # initialize RNN layers with hidden state and others with 0
        feats = gnn(g.atoms, g.edge_index, g.bonds, g.batch).unsqueeze(0)
        hn = torch.zeros((rnn.n_layers, feats.size(1), rnn.hidden_dim)).to(DEVICE)
        score = 0
        step, stop = 0, False
        with torch.no_grad():
            pred_smls_list = []
            while not stop:
                pred, hn = rnn(feats, hn, step=step)

                # calculate propabilities
                prob = softmax(pred)
                prob = np.squeeze(prob.cpu().detach().numpy())
                prob = prob.astype("float64")

                # transform with temperature and get most probable token
                pred = np.exp(prob / temp) / np.sum(np.exp(prob / temp))
                pred = np.random.multinomial(1, pred, size=1)
                pred = np.argmax(pred)
                pred_smls_list.append(pred)
                feats = torch.LongTensor([[pred]]).to(DEVICE)

                # calculate score (the higher the %, the smaller the log-likelihood)
                score += +(-np.log(prob[pred]))

                if i2t[pred] == "$" or len(pred_smls_list) > maxlen:  # stop once the end token is reached
                    stop = True
                step += 1

        s = "".join(i2t[i] for i in pred_smls_list)
        print(s)
        valid, smiles_j = is_valid_mol(s, True)
        if valid:
            smiles_list.append(smiles_j)
            score_list.append(score)

    novels, inchiks, probs_abs = [], [], []
    for idx, smls in enumerate(smiles_list):
        ik = Chem.MolToInchiKey(Chem.MolFromSmiles(smls))
        if ik and ik not in inchiks:
            novels.append(smls)
            inchiks.append(ik)
            probs_abs.append(score_list[idx])

    print(f"Number of valid, unique and novel molecules: {len(novels)}")

    return novels, probs_abs


if __name__ == "__main__":
    main()
