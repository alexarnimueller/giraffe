#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from rdkit.Chem import AddHs, MolFromSmiles, MolToSmiles, RemoveHs
from rdkit.Chem.Descriptors import descList
from rdkit.rdBase import DisableLog
from torch_geometric.data import Data, Dataset

from utils import atom_features, bond_features

for level in RDLogger._levels:
    DisableLog(level)


class OneMol(Dataset):
    def __init__(self, smiles, maxlen):
        super(OneMol, self).__init__()
        self.i2t, self.t2i = tokenizer()
        self.smiles = smiles
        self.max_len = maxlen

    def __getitem__(self, idx):
        smils = self.smiles
        mol = MolFromSmiles(smils)
        num_nodes, atom_feats, bond_feats, edge_index = attentive_fp_features(mol)
        smils_pad = np.full(self.max_len + 2, self.t2i[" "], dtype="uint8")
        smils_pad[: len(smils) + 2] = [self.t2i["^"]] + [self.t2i[c] for c in smils] + [self.t2i["$"]]

        return Data(
            atoms=torch.FloatTensor(atom_feats),
            bonds=torch.FloatTensor(bond_feats),
            edge_index=torch.LongTensor(edge_index),
            trg_smi=torch.LongTensor(smils_pad.reshape(1, -1)),
            num_nodes=num_nodes,
        )

    def __len__(self):
        return 1

    def len(self) -> int:
        return self.__len__()

    def get(self, idx: int) -> Data:
        return self.__getitem__(idx)


class AttFPDataset(Dataset):

    def __init__(
        self, filename, delimiter="\t", smls_col="SMILES", props=None, scaled_props=True, random=False, steps=128000
    ):
        super(AttFPDataset, self).__init__()
        # tokenizer
        self.i2t, self.t2i = tokenizer()
        self.random = random
        self.scaled_props = scaled_props
        self.scaler = PropertyScaler(props, do_scale=scaled_props)

        # Load smiles dataset
        if isinstance(filename, str):
            print("\nReading SMILES dataset...")
            self.data = pd.read_csv(filename, delimiter=delimiter)
            if smls_col not in self.data.columns and len(self.data.columns) == 1:
                self.data = pd.concat(
                    (
                        pd.DataFrame({"SMILES": self.data.columns.tolist()}),
                        self.data.rename(columns={self.data.columns[0]: "SMILES"}),
                    )
                )
            else:
                self.data = self.data.rename(columns={smls_col: "SMILES"})

        elif isinstance(filename, list) or isinstance(filename, np.ndarray):
            self.data = pd.DataFrame({"SMILES": filename})
        elif isinstance(filename, pd.Series):
            self.data = filename.to_frame()
            self.data.columns = ["SMILES"]
        else:
            raise NotImplementedError(
                f"Can only understand str, list/array or Series as filename! {type(filename)} provided"
            )

        self.max_len = self.data.SMILES.apply(lambda x: len(x)).max()
        self.data = self.data.SMILES.values.flatten()
        if isinstance(filename, str):
            print(f"Loaded {len(self.data)} SMILES")
            print("Max Length: ", self.max_len)
        # if random, set loops
        if random:
            self.loop = list(range(0, steps))

    def __getitem__(self, idx):
        if self.random:  # randomly sample any molecule
            idx = np.random.randint(len(self.data))

        mol = MolFromSmiles(self.data[idx])

        props = np.nan_to_num(self.scaler.transform(mol), 0.0)  # get scaled properties between 0 and 1
        num_nodes, atom_feats, bond_feats, edge_index = attentive_fp_features(mol)

        smils = MolToSmiles(RemoveHs(mol), doRandom=True)
        if len(smils) > self.max_len:
            smils = self.data[idx]
        smils_pad = np.full(self.max_len + 2, self.t2i[" "], dtype="uint8")
        smils_pad[: len(smils) + 2] = (
            [self.t2i["^"]] + [self.t2i[c] if c in self.t2i else self.t2i["*"] for c in smils] + [self.t2i["$"]]
        )

        return Data(
            atoms=torch.FloatTensor(atom_feats),
            bonds=torch.FloatTensor(bond_feats),
            edge_index=torch.LongTensor(edge_index),
            trg_smi=torch.LongTensor(smils_pad.reshape(1, -1)),
            props=torch.FloatTensor(props),
            num_nodes=num_nodes,
        )

    def __len__(self):
        if self.random:
            return len(self.loop)
        return len(self.data)

    def len(self) -> int:
        return self.__len__()

    def get(self, idx: int) -> Data:
        return self.__getitem__(idx)


def attentive_fp_features(mol):
    mol = AddHs(mol)
    # node and edge features
    atom_feats = np.array([atom_features(a) for a in mol.GetAtoms()])
    bond_feats = np.array([bond_features(a) for a in mol.GetBonds()] * 2)
    # edge indices (in the form that pyg Data edge_index needs it)
    edge_indices = []
    for bond in mol.GetBonds():
        edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
        edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]
    return mol.GetNumAtoms(), atom_feats, bond_feats, np.array(edge_indices).T


class PropertyScaler(object):
    def __init__(self, descriptors: Union[List, str, None] = None, do_scale: bool = True):
        if isinstance(descriptors, str):
            self.descriptors = {}
            desc_regex = re.compile(descriptors)
            for descriptor, func in descList:
                if desc_regex.match(descriptor):
                    self.descriptors[descriptor] = func
        elif isinstance(descriptors, list):
            self.descriptors = {descriptor: func for descriptor, func in descList if descriptor in descriptors}
        else:
            self.descriptors = {descriptor: func for descriptor, func in descList}

        self.load_min_max_values()
        self.do_scale = do_scale

    def load_min_max_values(self):
        d = json.loads(open("data/property_scales.json").read())
        d = {k: v for k, v in d.items() if k in self.descriptors.keys()}

        self.min_val = {k: v[0] for k, v in d.items()}
        self.max_val = {k: v[1] for k, v in d.items()}

    def _calc(self, mol, missing_val=0):
        rslt = {}
        for descriptor, func in self.descriptors.items():
            rslt[descriptor] = list()
            try:
                val = func(mol)
            except Exception:
                val = missing_val
            rslt[descriptor] = val
        return rslt

    def scale(self, x, n):
        try:
            return (min(x, self.max_val[n]) - min(x, self.min_val[n])) / (self.max_val[n] - self.min_val[n])
        except KeyError:
            raise KeyError(x, n, self.min_val, self.max_val, self.descriptors)

    def transform(self, mol):
        props = self._calc(mol)
        if self.do_scale:
            return [self.scale(x, n) for n, x in props.items()]
        else:
            return [v for v in props.values()]


def tokenizer():
    """Function to generate all possibly relevant SMILES token and put them into two translation dictionaries"""
    indices_token = {
        0: " ",
        1: "#",
        2: "%",
        3: "(",
        4: ")",
        5: "*",
        6: "+",
        7: "-",
        8: ".",
        9: "/",
        10: "0",
        11: "1",
        12: "2",
        13: "3",
        14: "4",
        15: "5",
        16: "6",
        17: "7",
        18: "8",
        19: "9",
        20: ":",
        21: "=",
        22: "@",
        23: "A",
        24: "B",
        25: "C",
        26: "D",
        27: "E",
        28: "F",
        29: "G",
        30: "H",
        31: "I",
        32: "K",
        33: "L",
        34: "M",
        35: "N",
        36: "O",
        37: "P",
        38: "R",
        39: "S",
        40: "T",
        41: "U",
        42: "V",
        43: "W",
        44: "X",
        45: "Y",
        46: "Z",
        47: "[",
        48: "\\",
        49: "]",
        50: "a",
        51: "b",
        52: "c",
        53: "d",
        54: "e",
        55: "f",
        56: "g",
        57: "h",
        58: "i",
        59: "k",
        60: "l",
        61: "m",
        62: "n",
        63: "o",
        64: "p",
        65: "r",
        66: "s",
        67: "t",
        68: "u",
        69: "y",
        70: "{",
        71: "}",
        72: "^",
        73: "$",
    }
    token_indices = {v: k for k, v in indices_token.items()}
    return indices_token, token_indices
