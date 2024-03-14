#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import AddHs, MolFromSmiles, MolToSmiles, RemoveHs
from torch_geometric.data import Data, Dataset

from descriptors import rdkit_descirptors
from utils import atom_features, bond_features


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

        smils = self.data.iloc[idx]["SMILES"] if len(smils) > self.max_len - 2 else smils
        smils_pad = np.full(self.max_len, self.t2i[" "], dtype="uint8")
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
    def __init__(self, filename, delimiter="\t", smls_col="SMILES"):
        super(AttFPDataset, self).__init__()
        # tokenizer
        self.i2t, self.t2i = tokenizer()
        # Load smiles dataset
        print("Reading SMILES dataset...")
        self.data = pd.read_csv(filename, delimiter=delimiter).rename(columns={smls_col: "SMILES"})
        self.max_len = self.data["SMILES"].apply(lambda x: len(x)).max() + 2

    def __getitem__(self, idx):
        mol = MolFromSmiles(self.data.iloc[idx]["SMILES"])
        props = rdkit_descirptors([mol]).values[0]
        num_nodes, atom_feats, bond_feats, edge_index = attentive_fp_features(mol)

        smils = MolToSmiles(RemoveHs(mol), doRandom=True)
        smils = self.data.iloc[idx]["SMILES"] if len(smils) > self.max_len - 2 else smils
        smils_pad = np.full(self.max_len, self.t2i[" "], dtype="uint8")
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
    # edge features (in the form that pyg Data edge_index needs it)
    edge_indices = []
    for bond in mol.GetBonds():
        edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
        edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]
    return mol.GetNumAtoms(), atom_feats, bond_feats, np.array(edge_indices).T


def tokenizer():
    """Function to generate all possibly relevant SMILES token and put them into two translation dictionaries"""
    indices_token = {
        0: "^",
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
        44: "$",
        45: " ",
    }
    token_indices = {v: k for k, v in indices_token.items()}
    return indices_token, token_indices
