#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configparser
import os
from multiprocessing import Process, Queue, cpu_count

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import (
    CanonSmiles,
    MolFromSmiles,
    MolToInchiKey,
    MolToSmiles,
    ReplaceCore,
    ReplaceSidechains,
)
from rdkit.Chem.Scaffolds import MurckoScaffold

from descriptors import (
    cats_descriptor,
    numpy_fps,
    numpy_maccs,
    parallel_pairwise_similarities,
)

ATOMTYPES = {h: i for i, h in enumerate(["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si"])}
i_pre = max(ATOMTYPES.values()) + 1
IS_RING = {h: i + i_pre for i, h in enumerate(["False", "True"])}
i_pre = max(IS_RING.values()) + 1
HYBRIDISATIONS = {h: i + i_pre for i, h in enumerate(["SP3", "SP2", "SP", "S", "SP3D", "SP3D2", "UNSPECIFIED"])}
i_pre = max(HYBRIDISATIONS.values()) + 1
AROMATICITY = {h: i + i_pre for i, h in enumerate(["False", "True"])}
DIM_EMBEDDING = max(AROMATICITY.values()) + 1


def is_valid_mol(smiles, return_smiles=False):
    """function to check a generated SMILES string for validity

    :param smiles: {str} SMILES string to be checked
    :param return_smiles: {bool} whether the checked valid SMILES string should be returned
    :return: {bool} validity
    """
    smiles = smiles.replace("^", "").replace("$", "").strip()
    try:
        m = MolFromSmiles(smiles, 1)
    except Exception:
        m = None
    if return_smiles:
        return m is not None, smiles
    else:
        return m is not None


def transform_temp(pred, temp):
    """transform predicted probabilities with a temperature

    :param pred: {tensor} list of probabilities to transform
    :param temp: {float} temperature to use for transformation
    :return: transformed probabilities
    """
    pred = np.squeeze(pred.cpu().detach().numpy())
    pred = pred.astype("float64")
    pred = np.exp(pred / temp) / np.sum(np.exp(pred / temp))
    pred = np.random.multinomial(1, pred, size=1)
    return np.argmax(pred)


def extract_murcko_scaffolds(mol):
    """Extract Bemis-Murcko scaffolds from a smile string.

    :param mol: {str} smiles string of a molecule.
    :return: smiles string of a scaffold.
    """
    m1 = MolFromSmiles(mol)
    try:
        core = MurckoScaffold.GetScaffoldForMol(m1)
        scaf = MolToSmiles(core, isomericSmiles=True)
    except Exception:
        return ""
    return scaf


def extract_murcko_scaffolds_marked(mol, mark="[*]"):
    """Extract Bemis-Murcko scaffolds from a smile string.

    :param mol: {str} smiles string of a molecule.
    :param mark: character to mark attachment points.
    :return: smiles string of a scaffold, side chains replaced with [R].
    """
    pos = range(0, 20)
    set_pos = ["[" + str(x) + "*]" for x in pos]

    m1 = MolFromSmiles(mol)
    try:
        core = MurckoScaffold.GetScaffoldForMol(m1)
        tmp = ReplaceSidechains(m1, core)
        smi = MolToSmiles(tmp, isomericSmiles=True)  # isomericSmiles adds a number to the dummy atoms.
    except Exception:
        return ""

    for i in pos:
        smi = smi.replace("".join(set_pos[i]), mark)
    return smi


def extract_side_chains(mol, remove_duplicates=False, mark="[*]"):
    """Extract side chains from a smiles string. Core is handled as Murcko scaffold.

    :param mol: {str} smiles string of a molecule.
    :param remove_duplicates: {bool} Keep or remove duplicates.
    :param mark: character to mark attachment points.
    :return: smiles strings of side chains in a list, attachment points replaced by [R].
    """
    pos = range(0, 20)
    set_pos = ["[" + str(x) + "*]" for x in pos]

    m1 = MolFromSmiles(mol)
    try:
        core = MurckoScaffold.GetScaffoldForMol(m1)
        side_chain = ReplaceCore(m1, core)
        smi = MolToSmiles(side_chain, isomericSmiles=True)  # isomericSmiles adds a number to the dummy atoms.
    except Exception:
        return list()
    for i in pos:
        smi = smi.replace("".join(set_pos[i]), mark)
    if remove_duplicates:
        return list(set(smi.split(".")))
    else:
        return smi.split(".")


def decorate_scaffold(scaffold, sidechains, num=10):
    """Decorate a given scaffold containing marked attachment points ([*]) randomly with the given side chains

    :param scaffold: {str} smiles string of a scaffold with attachment points marked as [*]
    :param sidechains: {str} point-separated side chains as smiles strings
    :param num: {int} number of unique molecules to generate
    :return: ``num``-molecules in a list
    """
    # check if side chains contain rings & adapt the ring number to not confuse them with the ones already in the scaff
    try:
        ring_scaff = int(max(list(filter(str.isdigit, scaffold))))  # get highest number of ring in scaffold
        ring_sc = list(filter(str.isdigit, scaffold))  # get number of rings in side chains
        for r in ring_sc:
            sidechains = sidechains.replace(
                r, str(ring_scaff + int(r))
            )  # replace the ring number with the adapted one
    except ValueError:
        pass

    # do the decoration
    mols = list()
    tmp = scaffold.replace("[*]", "*")
    schns = sidechains.split(".")
    invalcntr = 0
    while len(mols) < num and invalcntr < 50:
        scaff = tmp
        while "*" in scaff:
            scafflist = list(scaff)
            scafflist[scafflist.index("*")] = np.random.choice(schns, replace=False)
            scaff = "".join(scafflist)
        if is_valid_mol(scaff) and (scaff not in mols):
            scaff = CanonSmiles(scaff)
            print(sidechains + "." + scaffold + ">>" + scaff)
            mols.append(sidechains + "." + scaffold + ">>" + scaff)
        else:
            invalcntr += 1
    return mols


def compare_mollists(smiles, reference, canonicalize=True):
    """get the molecules from ``smiles`` that are not in ``reference``

    :param smiles: {list} list of SMILES strings to check for known reference in ``reference``
    :param reference: {list} reference molecules as SMILES strings to compare to ``smiles``
    :param canonicalize: {bool} whether SMILES should be canonicalized before comparison
    :return: {list} unique molecules from ``smiles`` as SMILES strings
    """
    smiles = [s.replace("^", "").replace("$", "").strip() for s in smiles]
    reference = [s.replace("^", "").replace("$", "").strip() for s in reference]
    if canonicalize:
        mols = set([CanonSmiles(s, 1) for s in smiles if MolFromSmiles(s)])
        refs = set([CanonSmiles(s, 1) for s in reference if MolFromSmiles(s)])
    else:
        mols = set(smiles)
        refs = set(reference)
    return [m for m in mols if m not in refs]


def compare_inchikeys(target, reference):
    """Compare a list of InChI keys with a list of reference InChI keys and return novel.

    :param target: {list} list of InChI keys of interest
    :param reference: {list} list of reference InChI keys to compare to
    :return: {2 lists} novel InChI keys and their indices in the full list
    """
    idx = [i for i, k in enumerate(target) if k not in reference]
    return [target[i] for i in idx], idx


def get_most_similar(smiles, referencemol, n=10, desc="FCFP4", similarity="tanimoto"):
    """get the n most similar molecules in a list of smiles compared to a reference molecule

    :param smiles: {list} list of SMILES strings
    :param referencemol: {str} SMILES string of reference molecule
    :param n: {int} number of most similar molecules to get
    :param desc: {str} which descriptor / fingerprint to use, choose from ['FCFP4', 'MACCS', 'CATS']
    :param similarity: {str} how to calculate the similarity between two molecules. use 'tanimoto' for FCFP4 & MACCS
        and 'euclidean' for CATS.
    :return: n most similar molecules (SMILES) in a list
    """
    if desc.upper() == "FCFP4":
        d_lib = numpy_fps([MolFromSmiles(s) for s in smiles], 2, True, 1024)
        d_ref = numpy_fps([MolFromSmiles(referencemol)], 2, True, 1024)
    elif desc.upper() == "MACCS":
        d_lib = numpy_maccs([MolFromSmiles(s) for s in smiles])
        d_ref = numpy_maccs([MolFromSmiles(referencemol)])
    elif desc.upper() == "CATS":
        d_lib = cats_descriptor([MolFromSmiles(s) for s in smiles])
        d_ref = cats_descriptor([MolFromSmiles(referencemol)])
    else:
        raise NotImplementedError("Only FCFP4, MACCS or CATS fingerprints are available!")

    sims = parallel_pairwise_similarities(d_lib, d_ref, similarity).flatten()
    if desc == "CATS":
        top_n = np.argsort(sims)[:n][::-1]
    else:
        top_n = np.argsort(sims)[-n:][::-1]
    return np.array(smiles)[top_n].flatten(), sims[top_n].flatten()


def inchikey_from_smileslist(smiles):
    """Create InChI keys for the given SMILES. - Parallelized

    :param smiles: {list} list of smiles strings
    """

    def _one_inchi(smls, q):
        res = list()
        for s in smls:
            res.append(MolToInchiKey(MolFromSmiles(s)))
        q.put(res)

    queue = Queue()
    rslt = []
    for s in np.array_split(np.array(smiles), cpu_count()):
        p = Process(target=_one_inchi, args=(s, queue))
        p.start()
    for _ in range(cpu_count()):
        rslt.extend(queue.get(timeout=10))
    return list(rslt)


def bond_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
        )
    return np.array(bond_feats, dtype=int)


def atom_features(atom, explicit_H=True, use_chirality=True):
    results = (
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "other"],
        )
        + one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 9])
        + [atom.GetFormalCharge(), bool(atom.GetNumRadicalElectrons())]
        + one_of_k_encoding_unk(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
                "other",
            ],
        )
        + [atom.GetIsAromatic()]
    )
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = (
                results
                + one_of_k_encoding_unk(atom.GetProp("_CIPCode"), ["R", "S"])
                + [atom.HasProp("_ChiralityPossible")]
            )
        except Exception:
            results = results + [False, False] + [atom.HasProp("_ChiralityPossible")]
    return np.array(results, dtype=int)


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def get_input_dims():
    return (
        len(atom_features(Chem.MolFromSmiles("CCO").GetAtoms()[0])),
        len(bond_features(Chem.MolFromSmiles("CCO").GetBonds()[0])),
    )


def read_config_ini(folder):
    ini = configparser.ConfigParser()
    ini.read(os.path.join(folder, "config.ini"))
    conf = {}
    for k, v in ini["CONFIG"].items():
        try:
            conf[k] = int(v)
        except ValueError:
            try:
                conf[k] = float(v)
            except ValueError:
                conf[k] = v
    return conf


def mse_with_nans(y_pred, y_true):
    mask = ~torch.isnan(y_true).to(y_pred.device)
    diff = (torch.flatten(y_pred[mask]) - torch.flatten(y_true[mask])) ** 2
    return torch.sum(diff) / mask.sum() if mask.sum() > 0 else 1e-6


def ce_with_nans(y_pred, y_true):
    mask = ~torch.isnan(y_true).to(y_pred.device)
    return torch.nn.functional.cross_entropy(y_pred[mask], y_true[mask]) / mask.sum() if mask.sum() > 0 else 1e-6
