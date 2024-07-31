#! /usr/bin/env python
# -*- coding: utf-8

import os
import time

import click
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.rdBase import DisableLog

from dataset import OneMol, tokenizer
from model import LSTM, AttentiveFP, AttentiveFP2, reparameterize
from utils import get_input_dims, is_valid_mol, read_config_ini

for level in RDLogger._levels:
    DisableLog(level)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.option("-c", "--checkpoint", default="models/pub_vae_sig", help="Checkpoint folder.")
@click.option("-e", "--epoch", default=70, help="Epoch of models to load.")
@click.option("-s", "--smiles", default=None, help="Reference SMILES to use as seed for sampling.")
@click.option("-n", "--num", default=100, help="How many molecules to sample.")
@click.option("-t", "--temp", default=0.5, help="Temperature to transform logits before for multinomial sampling.")
@click.option("-l", "--maxlen", default=128, help="Maximum allowed SMILES string length.")
@click.option("-o", "--out", default="output/sampled.csv", help="Output filename")
@click.option("-i", "--interpolate", is_flag=True, help="Linear interpolation between 2 SMILES (',' separated in -s).")
@click.option("-r", "--random", is_flag=True, help="Randomly sample from latent space.")
@click.option("-p", "--parent", is_flag=True, help="Store parent seed molecule in output file.")
def main(checkpoint, epoch, smiles, num, temp, maxlen, out, interpolate, random, parent):
    dim_atom, dim_bond = get_input_dims()
    conf = read_config_ini(checkpoint)
    vae = conf["vae"] == "True"

    if smiles is not None:
        if "," not in smiles:
            assert Chem.MolFromSmiles(smiles), "invalid SMILES string!"
    else:
        with open(conf["filename"], "r") as f:  # randomly take num SMILES from the training dataset
            smiles = [s.strip() for s in np.random.choice(f.readlines(), num)]

    # Define models
    rnn = LSTM(
        input_dim=conf["alphabet"],
        embedding_dim=conf["dim_embed"],
        hidden_dim=conf["dim_rnn"],
        layers=conf["n_rnn_layers"],
        dropout=conf["dropout"],
    )

    GNN_Class = AttentiveFP2 if vae else AttentiveFP
    gnn = GNN_Class(
        in_channels=dim_atom,
        hidden_channels=conf["dim_gnn"],
        out_channels=conf["dim_rnn"],
        edge_dim=dim_bond,
        num_layers=conf["n_gnn_layers"],
        num_timesteps=conf["n_kernels"],
        dropout=conf["dropout"],
    )

    gnn.load_state_dict(torch.load(os.path.join(checkpoint, f"atfp_{epoch}.pt"), map_location=DEVICE))
    rnn.load_state_dict(torch.load(os.path.join(checkpoint, f"lstm_{epoch}.pt"), map_location=DEVICE))
    gnn = gnn.to(DEVICE)
    rnn = rnn.to(DEVICE)

    # Sample molecules
    print(f"Sampling {num} molecules")
    smls, probs_abs, unique, novels, parents, dur = temperature_sampling(
        gnn=gnn,
        rnn=rnn,
        temp=temp,
        smiles=smiles,
        num_mols=num,
        maxlen=maxlen,
        vae=vae,
        inter=interpolate,
        random=random,
        parent=parent,
    )
    print(f"Sampled {len(smls)} valid, {len(unique)} unique and {novels} novel molecules in {dur:.2f} seconds.")

    # Save predictions
    if parent:
        df = pd.DataFrame({"SMILES": smls, "log-likelihood": probs_abs, "Parent": parents})
    else:
        df = pd.DataFrame({"SMILES": smls, "log-likelihood": probs_abs})
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Sampled molecules saved to {out}")


@torch.no_grad
def temperature_sampling(
    gnn, rnn, temp, smiles, num_mols, maxlen, vae=True, verbose=False, inter=False, random=False, parent=False
):
    gnn.eval()
    rnn.eval()
    i2t, t2i = tokenizer()

    if inter:
        assert "," in smiles, "Provide 2 SMILES separated by a comma for linear interpolation"
        t_start, t_end = smiles.split(",")
        dataset = [OneMol(t_start, maxlen), OneMol(t_end, maxlen)]
    else:
        if isinstance(smiles, str):  # single SMILES
            maxlen = max(maxlen, len(smiles))
            dataset = [OneMol(smiles, maxlen)] * num_mols
        else:  # list
            dataset = [OneMol(s, maxlen) for s in smiles]

    if verbose:
        trange = range
    else:
        from tqdm import trange

    if inter:  # interpolation
        g_start = dataset[0][0].to(DEVICE)
        g_end = dataset[1][0].to(DEVICE)
        g_start.batch = torch.tensor([0] * g_start.num_nodes).to(DEVICE)
        g_end.batch = torch.tensor([0] * g_end.num_nodes).to(DEVICE)
        if vae:
            hn_s, _ = gnn(g_start.atoms, g_start.edge_index, g_start.bonds, g_start.batch)
            hn_e, _ = gnn(g_end.atoms, g_end.edge_index, g_end.bonds, g_end.batch)
        else:
            hn_s = gnn(g_start.atoms, g_start.edge_index, g_start.bonds, g_start.batch)
            hn_e = gnn(g_end.atoms, g_end.edge_index, g_end.bonds, g_end.batch)
        return linear_interpolation(rnn, hn_s, hn_e, num_mols, temp=temp, maxlen=maxlen)
    elif random:  # random sampling
        return random_sampling(rnn, num_mols, temp=temp, maxlen=maxlen)
    else:  # sampling around provided molecules
        smls, scores, parents = [], [], []
        t_start = time.time()
        for i in trange(num_mols):
            g = dataset[i][0].to(DEVICE)
            g.batch = torch.tensor([0] * g.num_nodes).to(DEVICE)

            # initialize RNN hiddens with GNN features and cell states with 0
            score = 0
            step, stop = 0, False
            if vae:
                mu, var = gnn(g.atoms, g.edge_index, g.bonds, g.batch)
                hn = reparameterize(mu, var)
            else:
                hn = gnn(g.atoms, g.edge_index, g.bonds, g.batch)
            hn = torch.cat(
                [hn.unsqueeze(0)] + [torch.zeros((1, hn.size(0), hn.size(1))).to(DEVICE)] * (rnn.n_layers - 1),
                dim=0,
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
                print(s.replace("^", "").replace("$", "").replace(" ", ""))
            valid, smiles_j = is_valid_mol(s, True)
            if valid:
                smls.append(smiles_j)
                scores.append(score / len(s))
                if parent:
                    p = "".join(i2t[i] for i in g.trg_smi.detach().cpu().numpy()[0])
                    parents.append(p.replace("^", "").replace("$", "").strip())
        t_end = time.time()

        if isinstance(smiles, str):
            ik_ref = [Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))]
        else:
            ik_ref = [Chem.MolToInchiKey(Chem.MolFromSmiles(s)) for s in smiles]
        unique, inchiks, probs_abs, novels = [], [], [], 0
        for idx, s in enumerate(smls):
            ik = Chem.MolToInchiKey(Chem.MolFromSmiles(s))
            if ik not in ik_ref:
                novels += 1
            if ik and ik not in inchiks:
                unique.append(s)
                inchiks.append(ik)
            probs_abs.append(scores[idx])
        return smls, probs_abs, unique, novels, parents, t_end - t_start


@torch.no_grad
def linear_interpolation(rnn, start, end, steps, temp=0.5, maxlen=128):
    i2t, t2i = tokenizer()
    # Create a linear path from start to end
    z = torch.linspace(0, 1, steps)[:, None].to(DEVICE) * (end - start) + start

    # Decode the samples along the path
    smls, scores = [], []
    t_start = time.time()
    for hn in z:
        hn = hn.unsqueeze(0)
        step, score, stop = 0, 0, False
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
            score += +(-torch.log(prob[0, pred]).detach().cpu().numpy())
            if i2t[pred] == "$" or len(pred_smls_list) > maxlen:  # stop once the end token is reached
                stop = True
            step += 1

        s = "".join(i2t[i] for i in pred_smls_list)
        valid, smiles_j = is_valid_mol(s, True)
        if valid:
            smls.append(smiles_j)
            scores.append(score / len(s))
    t_end = time.time()

    unique, inchiks = [], []
    for s in smls:
        ik = Chem.MolToInchiKey(Chem.MolFromSmiles(s))
        if ik and ik not in inchiks:
            unique.append(s)
            inchiks.append(ik)

    return smls, scores, unique, len(unique), None, t_end - t_start


@torch.no_grad
def random_sampling(rnn, num_mols, temp=0.5, maxlen=128):
    i2t, t2i = tokenizer()

    # Decode the samples along the path
    smls, scores = [], []
    t_start = time.time()
    for _ in range(num_mols):
        hn = torch.randn(1, rnn.hidden_dim).to(DEVICE)  # sample a random latent space vector
        step, score, stop = 0, 0, False
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
            score += +(-torch.log(prob[0, pred]).detach().cpu().numpy())
            if i2t[pred] == "$" or len(pred_smls_list) > maxlen:  # stop once the end token is reached
                stop = True
            step += 1

        s = "".join(i2t[i] for i in pred_smls_list)
        valid, smiles_j = is_valid_mol(s, True)
        if valid:
            smls.append(smiles_j)
            scores.append(score / len(s))
    t_end = time.time()

    unique, inchiks = [], []
    for s in smls:
        ik = Chem.MolToInchiKey(Chem.MolFromSmiles(s))
        if ik and ik not in inchiks:
            unique.append(s)
            inchiks.append(ik)

    return smls, scores, unique, len(unique), None, t_end - t_start


if __name__ == "__main__":
    main()
