#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Molecule preprocessing functions
"""

import gzip
import pickle
from multiprocessing import Manager, Process, cpu_count

import click
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.Descriptors import CalcMolDescriptors
from tqdm.auto import tqdm

from dataset import tokenizer
from descriptors import scale_properties


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


def smiles2mol(smiles, n_proc=1):
    """generate RDKit molecules from smiles strings

    :param smiles: {list/array} list of SMILES strings to turn into molecules
    :param preprocess: {bool} whether preprocessing functions should be applied to the SMILES strings
    :return: {array} array of molecules
    """

    def process(smls, n, d):
        smls = keep_longest(smls)
        smls = harmonize_sc(smls)
        mols = list()
        for s in tqdm(smls, desc="Smiles2Mol"):
            try:
                mols.append(MolFromSmiles(s))
            except Exception:
                print("Error! Can not process SMILES string %s" % s)
                mols.append(None)
        d[n] = mols

    manager = Manager()
    d = manager.dict()

    if n_proc == -1:
        n_proc = cpu_count()
    processes = []
    for i, m in enumerate(np.array_split(np.array(smiles), n_proc)):
        p = Process(
            target=process,
            args=(
                m,
                i,
                d,
            ),
        )
        processes.append(p)
        p.start()

    for proc in processes:
        proc.join()

    return [m for v in d.values() for m in v]


def mol2smiles(mols, n_proc=1):
    """generate SMILES from RDKit molecules"""

    def process(mols, n, d):
        smls = list()
        for m in tqdm(mols, desc="Mol2Smiles"):
            try:
                smls.append(MolToSmiles(m))
            except Exception:
                continue
        d[n] = smls

    manager = Manager()
    d = manager.dict()

    if n_proc == -1:
        n_proc = cpu_count()
    processes = []
    for i, m in enumerate(np.array_split(np.array(mols), n_proc)):
        p = Process(
            target=process,
            args=(
                m,
                i,
                d,
            ),
        )
        processes.append(p)
        p.start()

    for proc in processes:
        proc.join()

    return [s for v in tqdm(d.values()) for s in v]


def mol2props(mols, n_proc=1):
    """generate RDKit properites for molecules"""

    def process(mols, n, d):
        props = list()
        for m in tqdm(mols, desc="Mol2Smiles"):
            try:
                props.append(list(CalcMolDescriptors(m, missingVal=0.0).values()))
            except Exception:
                continue
        d[n] = props

    manager = Manager()
    d = manager.dict()

    if n_proc == -1:
        n_proc = cpu_count()
    processes = []
    for i, m in enumerate(np.array_split(np.array(mols), n_proc)):
        p = Process(
            target=process,
            args=(
                m,
                i,
                d,
            ),
        )
        processes.append(p)
        p.start()

    for proc in processes:
        proc.join()

    return [s for v in tqdm(d.values()) for s in v]


def preprocess_df(filename, smls_col, delimiter, n_proc=1):
    _, t2i = tokenizer()
    data = pd.read_csv(filename, delimiter=delimiter).rename(columns={smls_col: "SMILES"})
    print("data read")
    mols = smiles2mol(data.SMILES.values, n_proc=n_proc)
    print("molecules generated")
    smls = mol2smiles(mols, n_proc=n_proc)
    print("molecules filtered")
    data = pd.DataFrame({"SMILES": smls, "MOL": mols})
    data = data[data.SMILES.apply(lambda x: all([c in t2i.keys() for c in x]))]
    print("smiles token checked")
    return data


def preprocess_single_core(filename, smls_col, delimiter):
    def fun(s):
        try:
            m = MolFromSmiles(s)
        except TypeError:
            m = None
        return m

    _, t2i = tokenizer()
    data = pd.read_csv(filename, delimiter=delimiter).rename(columns={smls_col: "SMILES"})
    print("data read")
    smls = keep_longest(data.SMILES.values)
    smls = harmonize_sc(smls)
    mols = [m for m in [fun(s) for s in smls] if m is not None]
    print("molecules generated")
    smls = [MolToSmiles(m) for m in tqdm(mols, desc="Mol2Smiles")]
    print("molecules filtered")
    data = pd.DataFrame({"SMILES": smls, "MOL": mols})
    data = data[data.SMILES.apply(lambda x: all([c in t2i.keys() for c in x]))]
    print("smiles token checked")
    return data


def create_property_scaler(mols: list, filename: str):
    def process(m):
        return list(CalcMolDescriptors(m, missingVal=0).values())

    print("calculating properties")
    props = np.asarray([process(m) for m in mols])
    print("properties calculated, creating scaler")
    _ = scale_properties(props, save_to=filename)


def create_properties(mols: list, filename: str):
    def process(m):
        return list(CalcMolDescriptors(m, missingVal=0).values())

    print("calculating properties")
    props = np.asarray([process(m) for m in mols])
    pickle.dump(props, gzip.open(filename, "wb"))


@click.command()
@click.argument("filename")
@click.option("-d", "--delimiter", default="\t", help="Column delimiter of input file.")
@click.option("-c", "--smls_col", default="SMILES", help="Name of column that contains SMILES.")
@click.option("-n", "--n_proc", default=4, help="Number of parallel processes to use. -1 = all")
def main(filename, smls_col, delimiter, n_proc):
    if n_proc == 1:
        data = preprocess_single_core(filename, smls_col, delimiter)
    else:
        data = preprocess_df(filename, smls_col, delimiter, n_proc=n_proc)
    # create_properties(data, f"{filename[:-4]}_properties.pkl.gz")
    data = data.drop_duplicates()
    data.drop(columns="MOL").to_csv(f"{filename[:-4]}_proc.txt", sep="\t", index=False)
    print("preprocessing completed!")
    # print(f"saved data to {filename[:-4]}_proc.txt and std scaler to {filename[:-4]}_scaler.bin")


if __name__ == "__main__":
    main()
