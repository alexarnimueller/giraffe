#! /usr/bin/env python
# -*- coding: utf-8

import configparser
import os

import click
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.rdBase import DisableLog

from dataset import OneMol, tokenizer
from model import LSTM, AttentiveFP, AttentiveFP2, reparameterize
from utils import get_input_dims, is_valid_mol

for level in RDLogger._levels:
    DisableLog(level)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.argument("checkpointfolder")
@click.option("-e", "--epoch", help="Epoch of models to load.")
@click.option("-s", "--smiles", default=None, help="Reference SMILES to use as seed for sampling.")
@click.option("-n", "--num", default=100, help="How many molecules to sample.")
@click.option("-t", "--temp", default=0.6, help="Temperature to transform logits before for multinomial sampling.")
@click.option("-l", "--maxlen", default=100, help="Maximum allowed SMILES string length.")
@click.option("-v", "--vae", is_flag=True, help="Sampling from a VAE model with mu and std.")
def main(checkpointfolder, epoch, smiles, num, temp, maxlen, vae):
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

    if smiles is not None:
        assert Chem.MolFromSmiles(smiles), "invalid SMILES string!"
    else:
        with open(conf["filename"], "r") as f:  # randomly take num SMILES from the training dataset
            smiles = [s.strip() for s in np.random.choice(f.readlines(), num)]

    # Define models
    rnn = LSTM(
        input_dim=conf["alphabet"],
        embedding_dim=conf["dim_embed"],
        hidden_dim=conf["dim_gnn"],
        layers=conf["n_rnn_layers"],
        dropout=conf["dropout"],
    )
    if vae:
        gnn = AttentiveFP2(
            in_channels=dim_atom,
            hidden_channels=conf["dim_gnn"],
            out_channels=conf["dim_rnn"],
            edge_dim=dim_bond,
            num_layers=conf["n_gnn_layers"],
            num_timesteps=conf["n_kernels"],
            dropout=conf["dropout"],
        )
    else:
        gnn = AttentiveFP(
            in_channels=dim_atom,
            hidden_channels=conf["dim_gnn"],
            out_channels=conf["dim_rnn"],
            edge_dim=dim_bond,
            num_layers=conf["n_gnn_layers"],
            num_timesteps=conf["n_kernels"],
            dropout=conf["dropout"],
        )

    gnn.load_state_dict(torch.load(os.path.join(checkpointfolder, f"atfp_{epoch}.pt"), map_location=DEVICE))
    rnn.load_state_dict(torch.load(os.path.join(checkpointfolder, f"lstm_{epoch}.pt"), map_location=DEVICE))
    gnn = gnn.to(DEVICE)
    rnn = rnn.to(DEVICE)

    # Sample molecules
    print(f"Sampling {num} molecules:")
    novels, probs_abs = temperature_sampling(
        gnn=gnn, rnn=rnn, temp=temp, smiles=smiles, num_mols=num, maxlen=maxlen, vae=vae
    )

    # Save predictions
    df = pd.DataFrame({"SMILES": novels, "log-likelihood": probs_abs})
    df_sorted = df.sort_values(by=["log-likelihood"], ascending=False)
    os.makedirs("output/", exist_ok=True)
    df_sorted.to_csv("output/sampled.csv", index=False)


def temperature_sampling(gnn, rnn, temp, smiles, num_mols, maxlen, vae=True, verbose=False):
    gnn.eval()
    rnn.eval()
    i2t, t2i = tokenizer()

    if isinstance(smiles, str):  # single SMILES
        maxlen = max(maxlen, len(smiles))
        dataset = [OneMol(smiles, maxlen)] * num_mols
    else:  # list
        dataset = [OneMol(s, maxlen) for s in smiles]

    if verbose:
        trange = range
    else:
        from tqdm import trange

    smiles_list, score_list = [], []
    for i in trange(num_mols):
        g = dataset[i][0].to(DEVICE)
        g.batch = torch.tensor([0] * g.num_nodes).to(DEVICE)

        # initialize RNN hiddens with GNN features and cell states with 0
        score = 0
        step, stop = 0, False
        with torch.no_grad():
            if vae:
                mu, var = gnn(g.atoms, g.edge_index, g.bonds, g.batch)
                hn = reparameterize(mu, var)
            else:
                hn = gnn(g.atoms, g.edge_index, g.bonds, g.batch)
            hn = torch.cat(
                [hn.unsqueeze(0)] + [torch.zeros((1, hn.size(0), hn.size(1))).to(DEVICE)] * (rnn.n_layers - 1), dim=0
            )
            cn = torch.zeros((rnn.n_layers, hn.size(1), rnn.hidden_dim)).to(DEVICE)
            nxt = torch.tensor([[t2i["^"]]]).to(DEVICE)  # start token
            pred_smls_list = []
            while not stop:
                # get next prediction and calculate propabilities (apply temperature to logits)
                pred, (hn, cn) = rnn(nxt, (hn, cn))
                prob = torch.softmax(pred.squeeze(0) / temp, dim=-1)
                pred = torch.multinomial(prob, num_samples=1).item()
                pred_smls_list.append(pred)
                nxt = torch.LongTensor([[pred]]).to(DEVICE)

                # calculate score (the higher the %, the smaller the log-likelihood)
                score += +(-torch.log(prob[0, pred]).detach().cpu().numpy())
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

    if isinstance(smiles, str):
        ik_ref = [Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))]
    else:
        ik_ref = [Chem.MolToInchiKey(Chem.MolFromSmiles(s)) for s in smiles]
    unique, inchiks, probs_abs, novels = [], [], [], 0
    for idx, smls in enumerate(smiles_list):
        ik = Chem.MolToInchiKey(Chem.MolFromSmiles(smls))
        if ik not in ik_ref:
            novels += 1
        if ik and ik not in inchiks:
            unique.append(smls)
            inchiks.append(ik)
            probs_abs.append(score_list[idx])

    print(f"Sampled {len(smiles_list)} valid, {novels} novel and {len(inchiks)} unique molecules")
    return unique, probs_abs


if __name__ == "__main__":
    main()
