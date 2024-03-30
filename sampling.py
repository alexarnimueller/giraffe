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
from model import LSTM, AttentiveFP
from utils import get_input_dims, is_valid_mol

for level in RDLogger._levels:
    DisableLog(level)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.argument("checkpointfolder")
@click.option("-e", "--epoch", help="Epoch of models to load.")
@click.option("-s", "--smiles", help="Reference SMILES to use as seed for sampling.")
@click.option("-n", "--num", default=100, help="How many molecules to sample.")
@click.option("-t", "--temp", default=0.5, help="Temperature to use for multinomial sampling.")
@click.option("-l", "--maxlen", default=100, help="Maximum allowed SMILES string length.")
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
    rnn = LSTM(
        input_dim=conf["alphabet"],
        embedding_dim=conf["dim_embed"],
        hidden_dim=conf["dim_gnn"],
        layers=conf["n_layers"],
        dropout=conf["dropout"],
    )
    gnn = AttentiveFP(
        in_channels=dim_atom,
        hidden_channels=conf["dim_gnn"],
        out_channels=conf["dim_rnn"],
        dropout=conf["dropout"],
        edge_dim=dim_bond,
        num_layers=conf["n_layers"],
        num_timesteps=conf["n_kernels"],
    )

    gnn.load_state_dict(torch.load(os.path.join(checkpointfolder, f"atfp_{epoch}.pt"), map_location=DEVICE))
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


def temperature_sampling(gnn, rnn, temp, smiles, num_mols, maxlen, verbose=False):
    gnn.eval()
    rnn.eval()
    softmax = nn.Softmax(dim=1)
    i2t, t2i = tokenizer()

    mol = OneMol(smiles, maxlen)
    loader = DataLoader(mol, batch_size=1)
    g = next(iter(loader)).to(DEVICE)

    if verbose:
        trange = range
    else:
        from tqdm import trange

    smiles_list, score_list = [], []
    for _ in trange(num_mols):
        # initialize RNN hiddens with GNN features and cell states with 0
        score = 0
        step, stop = 0, False
        with torch.no_grad():
            hn = gnn(g.atoms, g.edge_index, g.bonds, g.batch).unsqueeze(0)
            hn = torch.cat([hn] * rnn.n_layers, dim=0)
            cn = torch.zeros((rnn.n_layers, hn.size(1), rnn.hidden_dim)).to(DEVICE)
            nxt = torch.tensor([[t2i["^"]]]).to(DEVICE)  # start token
            pred_smls_list = []
            while not stop:
                # get next prediction and calculate propabilities
                pred, (hn, cn) = rnn(nxt, (hn, cn))
                prob = softmax(pred.squeeze(0)).cpu().detach().numpy()[0].astype("float64")

                # transform with temperature and get most probable token
                pred = prob_to_token_with_temp(prob, temp=temp)
                pred_smls_list.append(pred)
                nxt = torch.LongTensor([[pred]]).to(DEVICE)

                # calculate score (the higher the %, the smaller the log-likelihood)
                score += +(-np.log(prob[pred]))

                if i2t[pred] == "$" or len(pred_smls_list) > maxlen:  # stop once the end token is reached
                    stop = True
                step += 1

        s = "".join(i2t[i] for i in pred_smls_list)
        if verbose:
            print(s.replace("^", "").replace("$", "").replace(" ", ""), f"-- Score: {score / len(s):.3f}%")
        valid, smiles_j = is_valid_mol(s, True)
        if valid:
            smiles_list.append(smiles_j)
            score_list.append(score / len(s))

    ik_ref = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    unique, inchiks, probs_abs, novels = [], [], [], 0
    for idx, smls in enumerate(smiles_list):
        ik = Chem.MolToInchiKey(Chem.MolFromSmiles(smls))
        if ik != ik_ref:
            novels += 1
        if ik and ik not in inchiks:
            unique.append(smls)
            inchiks.append(ik)
            probs_abs.append(score_list[idx])

    print(f"Sampled {len(smiles_list)} valid, {novels} novel and {len(inchiks)} unique molecules")
    return unique, probs_abs


def prob_to_token_with_temp(prob, temp):
    pred = np.exp(prob / temp) / np.sum(np.exp(prob / temp))
    return np.argmax(np.random.multinomial(1, pred, size=1))


if __name__ == "__main__":
    main()
