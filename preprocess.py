#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Molecule preprocessing functions
"""


import click
import numpy as np
import pandas as pd
from rdkit.Chem import CanonSmiles

from dataset import tokenizer


def keep_longest(smls, return_salt=False):
    """function to keep the longest fragment of a smiles string after fragmentation by splitting at '.'

    :param smls: {list} list of smiles strings
    :param return_salt: {bool} whether to return the stripped salts as well
    :return: {list} list of longest fragments
    """
    parents = []
    salts = []
    if isinstance(smls, str):
        smls = [smls]
    for s in smls:
        if "." in s:
            f = s.split(".")
            lengths = [len(m) for m in f]
            n = int(np.argmax(lengths))
            parents.append(f[n])
            f.pop(n)
            salts.append(f)
        else:
            parents.append(s)
            salts.append([""])
    if return_salt:
        return parents, salts
    return parents


def harmonize_sc(mols):
    """harmonize the sidechains of given SMILES strings to a normalized format

    :param mols: {list} molecules as SMILES string
    :return: {list} harmonized molecules as SMILES string
    """
    out = list()
    for mol in mols:
        # TODO: add more problematic sidechain representation that occur
        pairs = [
            ("[N](=O)[O-]", "[N+](=O)[O-]"),
            ("[O-][N](=O)", "[O-][N+](=O)"),
        ]  # (before, after)
        for b, a in pairs:
            mol = mol.replace(b, a)
        out.append(mol)
    return out


def batchify(iterable, batch_size):
    for ndx in range(0, len(iterable), batch_size):
        batch = iterable[ndx : min(ndx + batch_size, len(iterable))]
        yield batch


def preprocess_smiles_file(filename, smls_col, delimiter, max_len, batch_size):
    def canon(s):
        try:
            o = CanonSmiles(s)
            return o if len(o) <= max_len else None
        except Exception:
            return None

    _, t2i = tokenizer()
    if filename.endswith(".gz"):
        data = pd.read_csv(filename, delimiter=delimiter, compression="gzip", engine="python").rename(
            columns={smls_col: "SMILES"}
        )
    else:
        data = pd.read_csv(filename, delimiter=delimiter, engine="python").rename(columns={smls_col: "SMILES"})
    print(f"{len(data)} SMILES strings read")

    print("Keeping longest fragment...")
    smls = keep_longest(data.SMILES.values)
    del data  # cleanup to save memory

    print("Harmonizing side chains...")
    smls = harmonize_sc(smls)

    print("Checking SMILES validity...")
    out = []
    for batch in batchify(smls, batch_size):
        tmp = [o for o in [canon(s) for s in batch] if o is not None]
        out.extend([s for s in tmp if all([c in t2i.keys() for c in s])])

    uniq = list(set(out))
    print(f"{len(uniq)} valid unique SMILES strings obtained")
    return pd.DataFrame({"SMILES": uniq})


@click.command()
@click.argument("filename")
@click.option("-d", "--delimiter", default="\t", help="Column delimiter of input file.")
@click.option("-c", "--smls_col", default="SMILES", help="Name of column that contains SMILES.")
@click.option("-l", "--max_len", default=150, help="Maximum length of SMILES string in characters.")
@click.option("-b", "--batch_size", default=100000, help="Batch size used to chunck up SMILES list for processing.")
def main(filename, smls_col, delimiter, max_len, batch_size):
    data = preprocess_smiles_file(filename, smls_col, delimiter, max_len, batch_size)
    data.to_csv(f"{filename[:-4]}_proc.txt.gz", sep="\t", index=False, compression="gzip")
    print(f"preprocessing completed! Saved {len(data)} SMILES to {filename[:-4]}_proc.txt.gz")


if __name__ == "__main__":
    main()
