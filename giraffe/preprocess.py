#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Molecule preprocessing functions
"""

import multiprocessing
from functools import partial
from time import time

import click
import numpy as np
import pandas as pd
from rdkit.Chem import CanonSmiles, MolFromSmiles

from giraffe.dataset import attentive_fp_features, tokenizer


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


def harmonize_sc(smls):
    """harmonize the sidechains of given SMILES strings to a normalized format

    :param mols: {list} molecules as SMILES string
    :return: {list} harmonized molecules as SMILES string
    """
    out = list()
    for s in smls:
        # TODO: add more problematic sidechain representation that occur
        pairs = [
            ("[N](=O)[O-]", "[N+](=O)[O-]"),
            ("[O-][N](=O)", "[O-][N+](=O)"),
        ]  # (before, after)
        for b, a in pairs:
            s = s.replace(b, a)
        out.append(s)
    return out


def batchify(iterable, n_batches):
    total_size = len(iterable)
    batch_size = max(1, total_size // n_batches)  # Ensure batch_size is at least 1
    remainder = total_size % n_batches

    start = 0
    for i in range(n_batches):
        # Add one more item to the first 'remainder' batches to handle uneven division
        end = start + batch_size + (1 if i < remainder else 0)
        if start < total_size:  # Only yield non-empty batches
            yield iterable[start:end]
        start = end


def canon_smiles(s, max_len):
    try:
        o = CanonSmiles(s)
        return o if len(o) <= max_len else None
    except Exception:
        return None


def afp_check(s):
    try:
        return attentive_fp_features(MolFromSmiles(s)) is not None
    except Exception:
        return False


def process_batch_smls(batch, t2i_keys, max_len):
    tmp = [o for o in [canon_smiles(s, max_len) for s in batch] if o is not None]
    return [s for s in tmp if all([c in t2i_keys for c in s])]


def process_batch_afp(batch):
    afp_checked = [afp_check(s) for s in batch]
    return [s for s, valid in zip(batch, afp_checked) if valid]


def preprocess_smiles_file(
    filename, smls_col, delimiter, max_len, check_afp, rand, n_proc=1
):
    _, t2i = tokenizer()
    t2i_keys = set(t2i.keys())  # Convert to set for faster lookups

    if filename.endswith(".gz"):
        data = pd.read_csv(
            filename, delimiter=delimiter, compression="gzip", engine="python"
        ).rename(columns={smls_col: "SMILES"})
    else:
        data = pd.read_csv(filename, delimiter=delimiter, engine="python").rename(
            columns={smls_col: "SMILES"}
        )
    print(f"{len(data)} SMILES strings read")

    print("Keeping longest fragment...")
    smls = keep_longest(data.SMILES.values)
    del data  # cleanup to save memory

    print("Harmonizing side chains...")
    smls = harmonize_sc(smls)

    # Set up multiprocessing
    if n_proc is None or n_proc < 1:
        n_proc = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {n_proc} processes for parallel processing")

    print("Checking SMILES validity...")
    batches = list(batchify(smls, n_proc))

    # Process batches in parallel
    with multiprocessing.Pool(processes=n_proc) as pool:
        results = pool.map(
            partial(process_batch_smls, t2i_keys=t2i_keys, max_len=max_len), batches
        )
    out = [item for sublist in results for item in sublist]

    uniq = list(set(out))
    print(f"{len(uniq)} valid unique SMILES strings obtained")

    if check_afp:  # check if AttentiveFP features can be computed for the SMILES
        print("Checking AttentiveFP feature validity...")
        batches = list(batchify(uniq, n_proc))
        # Process batches in parallel
        with multiprocessing.Pool(processes=n_proc) as pool:
            results = pool.map(process_batch_afp, batches)
        uniq = [item for sublist in results for item in sublist]

    out = pd.DataFrame({"SMILES": uniq})
    return out.sample(frac=1) if rand else out


@click.command()
@click.argument("filename")
@click.option("-d", "--delimiter", default="\t", help="Column delimiter of input file.")
@click.option(
    "-c", "--smls_col", default="SMILES", help="Name of column that contains SMILES."
)
@click.option(
    "-l",
    "--max_len",
    default=150,
    help="Maximum length of SMILES string in characters.",
)
@click.option(
    "-a",
    "--check_afp",
    is_flag=True,
    default=False,
    help="Only keep SMILES that contain characters in the AFP tokenizer vocabulary.",
)
@click.option(
    "-r",
    "--rand",
    is_flag=True,
    default=False,
    help="Randomize lines of output file.",
)
@click.option(
    "-n",
    "--n_proc",
    default=4,
    help="Number of parallel processes to use for SMILES processing. ",
)
def main(filename, smls_col, delimiter, max_len, check_afp, rand, n_proc):
    start = time()
    data = preprocess_smiles_file(
        filename, smls_col, delimiter, max_len, check_afp, rand, n_proc
    )
    stop = time()
    data.to_csv(
        f"{filename[:-4]}_proc.txt.gz", sep="\t", index=False, compression="gzip"
    )
    print(
        f"preprocessing completed in {stop-start:.1f}s! Saved {len(data)} SMILES to {filename[:-4]}_proc.txt.gz"
    )


if __name__ == "__main__":
    main()
