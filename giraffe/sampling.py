#! /usr/bin/env python
# -*- coding: utf-8

import os
import time

import click
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.rdBase import DisableLog

from giraffe.dataset import OneMol, tokenizer
from giraffe.model import load_models, reparameterize
from giraffe.utils import is_valid_mol

for level in RDLogger._levels:
    DisableLog(level)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.option("-c", "--checkpoint", default="models/wae_pub", help="Checkpoint folder.")
@click.option("-e", "--epoch", default=25, help="Epoch of models to load.")
@click.option(
    "-s", "--smiles", default=None, help="Reference SMILES to use as seed for sampling."
)
@click.option("-n", "--num", default=100, help="How many molecules to sample.")
@click.option(
    "-t",
    "--temp",
    default=0.5,
    help="Temperature to transform logits before for multinomial sampling.",
)
@click.option(
    "-l", "--maxlen", default=200, help="Maximum allowed SMILES string length."
)
@click.option("-o", "--out", default="output/sampled.csv", help="Output filename")
@click.option(
    "-i",
    "--interpolate",
    is_flag=True,
    help="Linear interpolation between 2 SMILES (',' separated in -s).",
)
@click.option("-r", "--random", is_flag=True, help="Randomly sample from latent space.")
@click.option(
    "-p", "--parent", is_flag=True, help="Store parent seed molecule in output file."
)
def main(
    checkpoint, epoch, smiles, num, temp, maxlen, out, interpolate, random, parent
):
    gnn, rnn, conf = load_models(checkpoint, epoch)

    vae = conf["vae"] == "True"
    if "wae" not in conf.keys():
        wae = False
    else:
        wae = conf["wae"] == "True"

    if smiles is not None:
        if "," not in smiles:
            assert Chem.MolFromSmiles(smiles), "invalid SMILES string!"
    else:
        data = (
            pd.read_csv(conf["filename"], compression="gzip")
            if conf["filename"].endswith(".gz")
            else pd.read_csv(conf["filename"])
        )
        smls_col = [c for c in data.columns if "SMILES" in c.upper()][0]
        smiles = data[smls_col].dropna().sample(num).tolist()

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
        wae=wae,
        inter=interpolate,
        random=random,
        parent=parent,
    )
    print(
        f"Sampled {len(smls)} valid, {len(unique)} unique and {novels} novel molecules in {dur:.2f} seconds."
    )

    # Save predictions
    if parent:
        df = pd.DataFrame({"SMILES": smls, "Parent": parents})
    else:
        df = pd.DataFrame({"SMILES": smls})
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Sampled molecules saved to {out}")


@torch.no_grad
def temperature_sampling(
    gnn,
    rnn,
    temp,
    smiles,
    num_mols,
    maxlen,
    vae=True,
    wae=False,
    verbose=False,
    inter=False,
    random=False,
    parent=False,
):
    gnn.eval()
    rnn.eval()
    i2t, t2i = tokenizer()

    if inter:
        assert (
            "," in smiles
        ), "Provide 2 SMILES separated by a comma for linear interpolation"
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
        if vae and not wae:
            hn_s, _ = gnn(
                g_start.atoms, g_start.edge_index, g_start.bonds, g_start.batch
            )
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
            step, score = 0, 0
            if vae and not wae:
                mu, var = gnn(g.atoms, g.edge_index, g.bonds, g.batch)
                hn = reparameterize(mu, var)
            else:
                hn = gnn(g.atoms, g.edge_index, g.bonds, g.batch)
            s = embedding2smiles(hn, rnn, temp, maxlen)
            step += 1
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
def linear_interpolation(rnn, start, end, steps, temp=0.5, maxlen=200):
    i2t, t2i = tokenizer()
    # Create a linear path from start to end
    z = torch.linspace(0, 1, steps)[:, None].to(DEVICE) * (end - start) + start

    # Decode the samples along the path
    smls, scores = [], []
    t_start = time.time()
    for hn in z:
        hn = hn.unsqueeze(0)
        step, score = 0, 0
        s = embedding2smiles(hn, rnn, temp, maxlen)
        step += 1
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
def random_sampling(rnn, num_mols, temp=0.5, maxlen=200):
    i2t, t2i = tokenizer()

    # Decode the samples along the path
    smls, scores = [], []
    t_start = time.time()
    for _ in range(num_mols):
        hn = torch.randn(1, rnn.hidden_dim).to(
            DEVICE
        )  # sample a random latent space vector
        step, score = 0, 0
        s = embedding2smiles(hn, rnn, temp, maxlen)
        step += 1
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
def embedding2smiles(hn, rnn, temp=0.5, maxlen=200):
    i2t, t2i = tokenizer()
    hn = torch.cat(
        [hn.unsqueeze(0)]
        + [torch.zeros((1, hn.size(0), hn.size(1))).to(DEVICE)] * (rnn.n_layers - 1),
        dim=0,
    )
    cn = torch.zeros((rnn.n_layers, hn.size(1), rnn.hidden_dim)).to(DEVICE)
    nxt = torch.tensor([[t2i["^"]]]).to(DEVICE)  # start token
    smls_gen, stop = [], False
    while not stop:
        # get next prediction and calculate propabilities (apply temperature to logits)
        pred, (hn, cn) = rnn(nxt, (hn, cn))
        prob = torch.softmax(pred.squeeze(0) / temp, dim=-1)
        pred = torch.multinomial(prob, num_samples=1).item()
        smls_gen.append(pred)
        nxt = torch.LongTensor([[pred]]).to(DEVICE)
        # stop once the end token is reached
        if i2t[pred] == "$" or len(smls_gen) > maxlen:
            stop = True

    return "".join(i2t[i] for i in smls_gen)


if __name__ == "__main__":
    main()
