#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from rdkit.Chem import AddHs, MolFromSmiles, MolToSmiles, RemoveHs
from rdkit.Chem.Descriptors import CalcMolDescriptors
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
    def __init__(self, filename, delimiter="\t", smls_col="SMILES", props=None, random=False, steps=128000):
        super(AttFPDataset, self).__init__()
        # tokenizer
        self.i2t, self.t2i = tokenizer()
        self.random = random
        self.scaler = PropertyScaler(props)

        # Load smiles dataset
        if isinstance(filename, str):
            print("\nReading SMILES dataset...")
            self.data = pd.read_csv(filename, delimiter=delimiter).rename(columns={smls_col: "SMILES"})
        elif isinstance(filename, list):
            self.data = pd.DataFrame({"SMILES": filename})

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
        props = self.scaler.transform(mol)  # get scaled properties between 0 and 1
        num_nodes, atom_feats, bond_feats, edge_index = attentive_fp_features(mol)

        smils = MolToSmiles(RemoveHs(mol), doRandom=True)
        if len(smils) > self.max_len:
            smils = self.data[idx]
        smils_pad = np.full(self.max_len + 2, self.t2i[" "], dtype="uint8")
        smils_pad[: len(smils) + 2] = [self.t2i["^"]] + [self.t2i[c] for c in smils] + [self.t2i["$"]]

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
    def __init__(self, descriptors: Union[List, None] = None):
        self.descriptors = descriptors
        self.load_min_max_values()

    def load_min_max_values(self):
        d = json.loads(open("data/property_scales.json").read())

        if self.descriptors is not None:
            d = {k: v for k, v in d.items() if k in self.descriptors}
        else:
            self.descriptors = list(d.keys())

        self.min_val = {k: v[0] for k, v in d.items()}
        self.max_val = {k: v[1] for k, v in d.items()}

    def scale(self, x, n):
        return (min(x, self.max_val[n]) - min(x, self.min_val[n])) / (self.max_val[n] - self.min_val[n])

    def transform(self, mol):
        props = CalcMolDescriptors(mol, missingVal=0)
        return [self.scale(x, n) for n, x in props.items()]


def tokenizer():
    """Function to generate all possibly relevant SMILES token and put them into two translation dictionaries"""
    indices_token = {
        0: " ",
        1: "C",
        2: "N",
        3: "O",
        4: "S",
        5: "P",
        6: "F",
        7: "B",
        8: "I",
        9: "H",
        10: "l",
        11: "r",
        12: "i",
        13: "h",
        14: "c",
        15: "n",
        16: "o",
        17: "s",
        18: "p",
        19: "b",
        20: ".",
        21: "%",
        22: "(",
        23: ")",
        24: "[",
        25: "]",
        26: "@",
        27: "-",
        28: "=",
        29: "#",
        30: ":",
        31: "\\",
        32: "/",
        33: "0",
        34: "1",
        35: "2",
        36: "3",
        37: "4",
        38: "5",
        39: "6",
        40: "7",
        41: "8",
        42: "9",
        43: "+",
        44: "^",
        45: "$",
    }
    token_indices = {v: k for k, v in indices_token.items()}
    return indices_token, token_indices
