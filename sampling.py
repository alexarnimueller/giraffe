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
from tqdm import trange

from dataset import OneMol, tokenizer
from model import LSTM, GraphTransformer
from utils import is_valid_mol

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
    lstm = LSTM(
        input_dim=conf["alphabet"],
        embedding_dim=conf["dim_model"],
        hidden_dim=conf["dim_hidden"],
        layers=conf["n_layers"],
        dropout=conf["dropout"],
    )
    egnn = GraphTransformer(
        n_kernels=conf["n_kernels"],
        pooling_heads=conf["n_pool_heads"],
        mlp_dim=conf["dim_hidden"],
        kernel_dim=conf["dim_model"],
        embeddings_dim=conf["dim_model"],
        dropout=conf["dropout"],
    )

    egnn.load_state_dict(torch.load(os.path.join(checkpointfolder, f"egnn_{epoch}.pt"), map_location=DEVICE))
    lstm.load_state_dict(torch.load(os.path.join(checkpointfolder, f"lstm_{epoch}.pt"), map_location=DEVICE))
    egnn = egnn.to(DEVICE)
    lstm = lstm.to(DEVICE)

    # Sample molecules
    print(f"Sampling {num} molecules:")
    novels, probs_abs = temperature_sampling(
        egnn=egnn, lstm=lstm, temp=temp, smiles=smiles, num_mols=num, maxlen=maxlen
    )

    # Save predictions
    df = pd.DataFrame({"SMILES": novels, "log-likelihoodog": probs_abs})
    df_sorted = df.sort_values(by=["log-likelihoodog"], ascending=False)
    os.makedirs("output/", exist_ok=True)
    df_sorted.to_csv("output/sampled.csv", index=False)


def temperature_sampling(egnn, lstm, temp, smiles, num_mols, maxlen):
    egnn.eval()
    lstm.eval()
    softmax = nn.Softmax(dim=1)
    i2t, t2i = tokenizer()

    mol = OneMol(smiles, maxlen)
    g = mol[0].to(DEVICE)

    smiles_list, score_list = [], []
    for _ in range(num_mols):  # trange(
        # initialize LSTM layers with hidden state and others with 0
        h = egnn(g)
        hiddens = tuple([h] + [torch.zeros(h.shape).to(h.device) for _ in range(lstm.n_layers - 1)])
        score = 0
        stop = False
        with torch.no_grad():
            pred_smls_list = []
            pred_smls = torch.from_numpy(np.asarray([t2i["^"]])).to(DEVICE).unsqueeze(0)
            while not stop:
                pred, hiddens = lstm(pred_smls, hiddens)

                # calculate propabilities
                prob = softmax(pred)
                prob = np.squeeze(prob.cpu().detach().numpy())
                prob = prob.astype("float64")

                # transform with temperature and get most probable token
                pred = np.exp(prob / temp) / np.sum(np.exp(prob / temp))
                pred = np.random.multinomial(1, pred, size=1)
                pred = np.argmax(pred)
                pred_smls_list.append(pred)
                pred_smls = torch.LongTensor([[pred]]).to(DEVICE)

                # calculate score (the higher the %, the smaller the log-likelihood)
                score += +(-np.log(prob[pred]))

                if i2t[pred] == "$" or len(pred_smls_list) > maxlen:  # stop once the end token is reached
                    stop = True
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
