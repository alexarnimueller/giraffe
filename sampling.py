#! /usr/bin/env python
# -*- coding: utf-8

import os
import random

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.rdBase import DisableLog
from torch_geometric.loader import DataLoader
from tqdm import trange

from dataset import OneMol, tokenizer
from old.net import EGNN, LSTM, GraphTransformer
from utils import (
    ACT,
    AROMDICT,
    ATOMDICT,
    CONFIG_PATH,
    HYBDICT,
    MODEL_PATH,
    RINGDICT,
    clean_molecule,
    decode_ids,
    get_vocab,
    is_valid_mol,
    selfies_vocab,
    transform_temp,
)

for level in RDLogger._levels:
    DisableLog(level)


@click.command()
@click.argument("checkpointfolder")
@click.option("-s", "--smiles", help="Reference SMILES to use as seed for sampling.")
@click.option("-n", "--num", default=100, help="How many molecules to sample.")
@click.option("-t", "--temp", default=0.6, help="Temperature to use for sampling.")
def main(checkpointfolder, smiles, num, temp):
    # Define paths
    MODEL_PATH = os.path.join(MODEL_PATH, CONFIG_NAME)
    CONFIG_NAME = CONFIG_NAME + "/"
    # Define PDB id and args
    smi = args.smi
    epoch = str(args.epoch)
    num_mols = int(args.num_mols)
    T = args.T

    # Define models
    lstm = LSTM(ALPHABET, D_INNER, N_LAYERS, DROPOUT, D_MODEL)
    egnn = GraphTransformer(N_KERNELS, D_INNER, PROP_DIM, POOLING_HEADS)

    egnn_path = os.path.join(checkpointfolder, f"egnn_{epoch}.pt")
    lstm_path = os.path.join(checkpointfolder, f"lstm_{epoch}.pt")

    egnn.load_state_dict(torch.load(egnn_path, map_location="cpu"))
    lstm.load_state_dict(torch.load(lstm_path, map_location="cpu"))

    egnn = egnn.to("gpu")
    lstm = lstm.to("gpu")

    # Load PDB
    g_mol = MolLoader(smi=smi)
    sample_loader = DataLoader(g_mol, batch_size=1, shuffle=True, num_workers=0)

    # Sample molecules
    print(f"Sampling {num_mols} molecules:")
    novels, probs_abs = temperature_sampling(
        egnn=egnn,
        lstm=lstm,
        temp=temp,
        sample_loader=sample_loader,
        num_mols=num_mols,
    )

    # Save predictions
    df = pd.DataFrame({"SMILES": novels, "log-likelihoodog": probs_abs})
    df_sorted = df.sort_values(by=["log-likelihoodog"], ascending=False)
    os.makedirs("output/", exist_ok=True)
    df_sorted.to_csv(f"output/{args.smi_id}.csv", index=False)


def temperature_sampling(egnn, lstm, temp, ref_smiles, num_mols, maxlen):
    egnn.eval()
    lstm.eval()
    softmax = nn.Softmax(dim=2)
    i2t, t2i = tokenizer()

    mol = OneMol(ref_smiles, maxlen)
    g = mol[0].to("cuda")

    smiles_list, score_list = [], []
    for _ in trange(num_mols):
        hiddens = egnn(g)
        score = 0
        stop = False
        with torch.no_grad():
            pred_smls_list = []
            pred_smls = torch.from_numpy(np.asarray([t2i["^"]])).to("cuda")

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
                pred_smls = torch.LongTensor([[pred]]).to("cuda")

                # calculate score (the higher the %, the smaller the log-likelihood)
                score += +(-np.log(prob[pred]))

                if i2t[pred] == "$" or len(pred_smls_list) > maxlen:  # stop once the end token is reached
                    stop = True

        valid, smiles_j = is_valid_mol(''.join(i2t(i) for i in pred_smls_list), True)
        if valid:
            smiles_list.append(smiles_j)
            score_list.append(score)

    novels, inchiks, probs_abs = []
    for idx, smls in enumerate(smiles_list):
        ik = Chem.MolToInchiKey(Chem.MolFromSmiles(smls))
        if ik not in inchiks:
            novels.append(smls)
            inchiks.append(ik)
            probs_abs.append(score_list[idx])

    print(f"Number of valid, unique and novel molecules: {len(novels)}")

    return novels, probs_abs


if __name__ == "__main__":
    main()
