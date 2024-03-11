#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import AddHs, MolFromSmiles, MolToSmiles, RemoveHs
from torch_geometric.data import Data, Dataset

from descriptors import rdkit_descirptors

# from pygdataset import Dataset
from utils import AROMATICITY, ATOMTYPES, HYBRIDISATIONS, IS_RING


class OneMol(Dataset):
    def __init__(self, smiles, maxlen):
        super(OneMol, self).__init__()
        self.i2t, self.t2i = tokenizer()
        self.smiles = smiles
        self.max_len = maxlen

    def __getitem__(self, idx):
        smils = self.smiles
        mol = MolFromSmiles(smils)
        atomids, is_ring, hyb, arom, edge_index = graph_from_mol(mol)
        smils = self.data.iloc[idx]["SMILES"] if len(smils) > self.max_len - 2 else smils
        smils_pad = np.full(self.max_len, self.t2i[' '], dtype="uint8")
        smils_pad[:len(smils) + 2] = [self.t2i['^']] + [self.t2i[c] for c in smils] + [self.t2i['$']]

        return Data(
            atomids=torch.LongTensor(atomids),
            is_ring=torch.LongTensor(is_ring),
            hyb=torch.LongTensor(hyb),
            arom=torch.LongTensor(arom),
            edge_index=torch.LongTensor(edge_index),
            trg_smi=torch.LongTensor(smils_pad.reshape(1, -1)),
            num_nodes=len(atomids)
        )

    def __len__(self):
        return 1
    
    def len(self) -> int:
        return self.__len__()

    def get(self, idx: int) -> Data:
        return self.__getitem__(idx)


class MolDataset(Dataset):
    def __init__(self, filename, delimiter="\t", smls_col="SMILES"):
        super(MolDataset, self).__init__()
        # tokenizer
        self.i2t, self.t2i = tokenizer()
        # Load smiles dataset
        print("Reading SMILES dataset...")
        self.data = pd.read_csv(filename, delimiter=delimiter).rename(columns={smls_col: "SMILES"})
        self.max_len = self.data["SMILES"].apply(lambda x: len(x)).max() + 2

    def __getitem__(self, idx):
        mol = MolFromSmiles(self.data.iloc[idx]["SMILES"])
        atomids, is_ring, hyb, arom, edge_index = graph_from_mol(mol)
        props = rdkit_descirptors([mol]).values[0]

        smils = MolToSmiles(RemoveHs(mol), doRandom=True)
        smils = self.data.iloc[idx]["SMILES"] if len(smils) > self.max_len - 2 else smils
        smils_pad = np.full(self.max_len, self.t2i[' '], dtype="uint8")
        smils_pad[:len(smils) + 2] = [self.t2i['^']] + [self.t2i[c] for c in smils] + [self.t2i['$']]

        return Data(
            atomids=torch.LongTensor(atomids),
            is_ring=torch.LongTensor(is_ring),
            hyb=torch.LongTensor(hyb),
            arom=torch.LongTensor(arom),
            edge_index=torch.LongTensor(edge_index),
            trg_smi=torch.LongTensor(smils_pad.reshape(1, -1)),
            props=torch.FloatTensor(props),
            num_nodes=len(atomids)
        )

    def __len__(self):
        return len(self.data)

    def len(self) -> int:
        return self.__len__()

    def get(self, idx: int) -> Data:
        return self.__getitem__(idx)


def graph_from_mol(mol):
    mol = AddHs(mol)
    # node features
    atomids = np.array([ATOMTYPES[a.GetSymbol()] for a in mol.GetAtoms()])
    is_ring = np.array([IS_RING[str(a.IsInRing())] for a in mol.GetAtoms()])
    hyb = np.array([HYBRIDISATIONS[str(a.GetHybridization())] for a in mol.GetAtoms()])
    arom = np.array([AROMATICITY[str(a.GetIsAromatic())] for a in mol.GetAtoms()])
    # edge features (in the form that pyg Data edge_index needs it)
    edge_dir1, edge_dir2 = [], []
    for b in mol.GetBonds():
        a1 = b.GetBeginAtomIdx()
        a2 = b.GetEndAtomIdx()
        edge_dir1.extend([a1, a2])
        edge_dir2.extend([a2, a1])

    return atomids, is_ring, hyb, arom, np.array([edge_dir1, edge_dir2])


def tokenizer():
    """Function to generate all possible SMILES token and put them into two translation dictionaries
    """
    indices_token = {0: 'H', 1: '9', 2: 'D', 3: 'r', 4: 'T', 5: 'R', 6: 'V', 7: '4',
                     8: 'c', 9: 'l', 10: 'b', 11: '.', 12: 'C', 13: 'Y', 14: 's', 15: 'B',
                     16: 'k', 17: '+', 18: 'p', 19: '2', 20: '7', 21: '8', 22: 'O',
                     23: '%', 24: 'o', 25: '6', 26: 'N', 27: 'A', 28: 't', 29: 'm',
                     30: '(', 31: 'u', 32: 'Z', 33: '#', 34: 'M', 35: 'P', 36: 'G',
                     37: 'I', 38: '=', 39: '-', 40: 'X', 41: '@', 42: 'E', 43: ':',
                     44: '\\', 45: ')', 46: 'i', 47: 'K', 48: '/', 49: '{', 50: 'h',
                     51: 'L', 52: 'n', 53: 'U', 54: '[', 55: '0', 56: 'y', 57: 'e',
                     58: '3', 59: 'g', 60: 'f', 61: '}', 62: '1', 63: 'd', 64: 'W',
                     65: '5', 66: 'S', 67: 'F', 68: ']', 69: 'a', 70: '^', 71: '$', 72: ' '}
    token_indices = {v: k for k, v in indices_token.items()}
    return indices_token, token_indices
