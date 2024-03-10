#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Molecule preprocessing functions
"""

import importlib
from multiprocessing import Pool, Process, Queue, cpu_count

import click
import numpy as np
import pandas as pd
import rdkit.Chem.MolStandardize.standardize as MolVS_standardizer
import rdkit.Chem.MolStandardize.tautomer as tautomer
from rdkit.Chem import CanonSmiles, Kekulize, MolFromSmiles, MolToSmiles
from rdkit.Chem.MolStandardize.tautomer import TautomerTransform

from utils import ATOMTYPES


def keep_longest(smls, return_salt=False):
    """ function to keep the longest fragment of a smiles string after fragmentation by splitting at '.'

    :param smls: {list} list of smiles strings
    :param return_salt: {bool} whether to return the stripped salts as well
    :return: {list} list of longest fragments
    """
    parents = []
    salts = []
    if isinstance(smls, str):
        smls = [smls]
    for s in smls:
        if '.' in s:
            f = s.split('.')
            lengths = [len(m) for m in f]
            n = int(np.argmax(lengths))
            parents.append(f[n])
            f.pop(n)
            salts.append(f)
        else:
            parents.append(s)
            salts.append([''])
    if return_salt:
        return parents, salts
    return parents


def is_valid_mol(smiles, return_smiles=False):
    """ function to check a (generated) SMILES string for validity

    :param smiles: {str} SMILES string to be checked
    :param return_smiles: {bool} whether the checked valid SMILES string should be returned
    :return: {bool} validity
    """
    try:
        m = CanonSmiles(smiles.replace('^', '').replace('$', '').strip(), 1)
    except Exception:
        m = None
    if return_smiles:
        return m is not None, m
    else:
        return m is not None


def harmonize_sc(mols):
    """ harmonize the sidechains of given SMILES strings to a normalized format

    :param mols: {list} molecules as SMILES string
    :return: {list} harmonized molecules as SMILES string
    """
    out = list()
    for mol in mols:
        # TODO: add more problematic sidechain representation that occur
        pairs = [('[N](=O)[O-]', '[N+](=O)[O-]'),  # (before, after)
                 ('[O-][N](=O)', '[O-][N+](=O)')]
        for b, a in pairs:
            mol = mol.replace(b, a)
        out.append(mol)
    return out


def smiles2mol(smiles, preprocess=True):
    """ generate RDKit molecules from smiles strings

    :param smiles: {list/array} list of SMILES strings to turn into molecules
    :param preprocess: {bool} whether preprocessing functions should be applied to the SMILES strings
    :return: {array} array of molecules
    """
    def process(smls, q):
        if preprocess:
            smls = keep_longest(smls)
            smls = harmonize_sc(smls)
        mols = list()
        for s in smls:
            try:
                s = MolFromSmiles(s, 1)
                Kekulize(s)
                mols.append(s)
            except Exception:
                print("Error! Can not process SMILES string %s" % s)
                mols.append(None)
        q.put(mols)

    queue = Queue()
    for m in np.array_split(np.array(smiles), cpu_count()):
        p = Process(target=process, args=(m, queue,))
        p.start()
    rslt = []
    for _ in range(cpu_count()):
        rslt.extend(queue.get(timeout=10))
    return rslt


def standardize(mol, stereo=True, max_atoms=100, isotope=False, mol_out=False):
    """ Molecule Standardizer - standardizes one SMILES string

    :param mol: {str} RDKit molecule or SMILES string to standardize
    :param stereo: {bool} whether to consider stereochemistry or ignore it
    :param max_atoms: {int} maximal number of heavy atoms allowed
    :param isotope {bool} whether to consider isotopes as different atoms
    :param mol_out: {bool} whether to return the standardized molecule as RDKit mol object
    :return: standardized SMILES string or RDKit Mol
    """
    tautomer.TAUTOMER_TRANSFORMS = update_tautomer_rules()
    importlib.reload(MolVS_standardizer)
    if isinstance(mol, str):
        mol = mol.replace('.[XH]', '')  # found as frequent substituent for undefined counterion
        mol = MolFromSmiles(mol)
    try:
        stndrdzr = MolVS_standardizer.Standardizer(max_tautomers=128)
        if mol and mol.GetNumAtoms() <= max_atoms and all(a.GetSymbol() in ATOMTYPES.keys() for a in mol.GetAtoms()):
            m = stndrdzr.charge_parent(mol)
            if not isotope:
                m = stndrdzr.isotope_parent(m)
            if not stereo:
                m = stndrdzr.stereo_parent(m)  # remove stereochemistry
                m = stndrdzr.tautomer_parent(m)
                out = stndrdzr.standardize(m)
                Kekulize(out)
            else:
                m = stndrdzr.tautomer_parent(m)
                out = stndrdzr.standardize(m)
                Kekulize(out)
        else:
            out = None
        if not mol_out:
            out = MolToSmiles(out, isomericSmiles=True, kekuleSmiles=True)
    except (RuntimeError, TypeError, ValueError, AttributeError) as e:
        out = None
        print('Standardization error ' + str(e))
    return out


def _multi(smiles, stereo, max_atoms, isotope, mol):
    return standardize(smiles, stereo, max_atoms, isotope, mol)


def standardize_mols(mols, stereo=True, max_atoms=100, isotope=False, mol_out=False, processes=-1):
    """ Molecule Standardizer - standardizes a list of SMILES strings

    :param mols: {list} list of RDKit molecules or SMILES strings to standardize
    :param stereo: {bool} whether to consider stereochemistry or ignore it
    :param max_atoms: {int} maximal number of heavy atoms allowed
    :param isotope {bool} whether to consider isotopes as different atoms
    :param mol_out: {bool} whether to return the standardized molecules as RDKit mol objects
    :param processes: {int} number of CPU cores to use for processing (-1 = all available)
    :return: list of standardized SMILES strings
    """
    processes = cpu_count() if processes == -1 else processes
    with Pool(processes=processes) as pool:
        standardized = list(pool.starmap(_multi, [(m, stereo, max_atoms, isotope, mol_out) for m in mols]))
        pool.close()
        pool.join()
    return standardized


def standardize_df(data, column, stereo=True, max_atoms=100, drop_dups=False, drop_na=False, isotope=False, mol=False,
                   replace=False, delimiter=','):
    """ Standardize SMILES strings in a given data frame or csv file that are marked by the column header `column`

    :param data: {pandas.DataFrame or str} data frame or filename to read from
    :param column: {str} column header marking SMILES strings
    :param stereo: {bool} whether to consider/keep stereochemistry information or drop it
    :param max_atoms: {int} maximal number of heavy atoms allowed
    :param drop_dups: {bool} whether to drop duplicate molecule entries
    :param drop_na: {bool} whether to drop 'None' molecule entries that were not processed
    :param mol: {bool} whether to also return the standardized molecules as RDKit mol objects
    :param replace: {bool} whether to replace the input SMILES column with the standardized values
    :param delimiter: {str} delimiter to be used for reading `data` when a filename is given
    :return: pandas dataframe with preprocessed SMILES strings in the column SMILES_Std (or MOL_Std if `mol=True`)
    """
    if isinstance(data, str):
        data = pd.read_csv(data, delimiter=delimiter)

    strdz = standardize_mols(data[column], stereo, max_atoms, isotope, mol)

    if not replace:
        if mol:
            column = 'MOL_Std'
            data[column] = strdz
        else:
            column = 'SMILES_Std'
            data[column] = strdz
    else:
        data[column] = strdz

    if drop_dups:
        data.drop(data.index[data.duplicated(subset=column)], inplace=True)  # drop duplicate molecules

    if drop_na:
        data.dropna(subset=[column], inplace=True)
    return data


def update_tautomer_rules():
    """ Update of the RDKit tautomerization rules, especially rules for amide tautomerization.
    Obtained from MELLODDY

    :return: updated list of tautomer rules
    """
    newtfs = (
        TautomerTransform('1,3 (thio)keto/enol f', '[CX4!H0]-[C;!$([C]([CH1])(=[O,S,Se,Te;X1])-[N,O])]=[O,S,Se,Te;X1]'),
        TautomerTransform('1,3 (thio)keto/enol r', '[O,S,Se,Te;X2!H0]-[C;!$(C(=[C])(-[O,S,Se,Te;X2!H0])-[N,O])]=[C]'),
        TautomerTransform('1,5 (thio)keto/enol f',
                          '[CX4,NX3;!H0]-[C]=[C]-[C;!$([C]([C])(=[O,S,Se,Te;X1])-[N,O])]=[O,S,Se,Te;X1]'),
        TautomerTransform('1,5 (thio)keto/enol r',
                          '[O,S,Se,Te;X2!H0]-[C;!$(C(=[C])(-[O,S,Se,Te;X2!H0])-[N,O])]=[C]-[C]=[C,N]'),
        TautomerTransform('aliphatic imine f', '[CX4!H0]-[C;$([CH1](C)=N),$(C(C)([#6])=N)]=[N;$([NX2][#6]),$([NX2H])]'),
        TautomerTransform('aliphatic imine r',
                          '[N!H0;$([NX3H1][#6]C),$([NX3H2])]-[C;$(C(N)([#6])=C),$([CH1](N)=C)]=[CX3]'),
        TautomerTransform('special imine f', '[N!H0]-[C]=[CX3R0]'),
        TautomerTransform('special imine r', '[CX4!H0]-[c]=[n]'),
        TautomerTransform('1,3 aromatic heteroatom H shift f', '[#7!H0]-[#6R1]=[O,#7X2]'),
        TautomerTransform('1,3 aromatic heteroatom H shift r', '[O,#7;!H0]-[#6R1]=[#7X2]'),
        TautomerTransform('1,3 heteroatom H shift', '[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]'),
        TautomerTransform('1,5 aromatic heteroatom H shift', '[#7,#16,#8;!H0]-[#6,#7]=[#6]-[#6,#7]=[#7,#16,#8;H0]'),
        TautomerTransform('1,5 aromatic heteroatom H shift f',
                          '[#7,#16,#8,Se,Te;!H0]-[#6,nX2]=[#6,nX2]-[#6,#7X2]=[#7X2,S,O,Se,Te]'),
        TautomerTransform('1,5 aromatic heteroatom H shift r',
                          '[#7,S,O,Se,Te;!H0]-[#6,#7X2]=[#6,nX2]-[#6,nX2]=[#7,#16,#8,Se,Te]'),
        TautomerTransform('1,7 aromatic heteroatom H shift f',
                          '[#7,#8,#16,Se,Te;!H0]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[#6]-[#6,#7X2]=[#7X2,S,O,Se,Te,CX3]'),
        TautomerTransform('1,7 aromatic heteroatom H shift r',
                          '[#7,S,O,Se,Te,CX4;!H0]-[#6,#7X2]=[#6]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[NX2,S,O,Se,Te]'),
        TautomerTransform('1,9 aromatic heteroatom H shift f',
                          '[#7,O;!H0]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[#7,O]'),
        TautomerTransform('1,11 aromatic heteroatom H shift f',
                          '[#7,O;!H0]-[#6,nX2]=[#6,nX2]-[#6,nX2]=[#6,nX2]-[#6,nX2]=[#6,nX2]-[#6,nX2]=[#6,nX2]-'
                          + '[#6,nX2]=[#7X2,O]'),
        TautomerTransform('furanone f', '[O,S,N;!H0]-[#6r5]=[#6X3r5;$([#6]([#6r5])=[#6r5])]'),
        TautomerTransform('furanone r', '[#6r5!H0;$([#6]([#6r5])[#6r5])]-[#6r5]=[O,S,N]'),
        TautomerTransform('keten/ynol f', '[C!H0]=[C]=[O,S,Se,Te;X1]', bonds='#-'),
        TautomerTransform('keten/ynol r', '[O,S,Se,Te;!H0X2]-[C]#[C]', bonds='=='),
        TautomerTransform('ionic nitro/aci-nitro f', '[C!H0]-[N+;$([N][O-])]=[O]'),
        TautomerTransform('ionic nitro/aci-nitro r', '[O!H0]-[N+;$([N][O-])]=[C]'),
        TautomerTransform('oxim/nitroso f', '[O!H0]-[N]=[C]'),
        TautomerTransform('oxim/nitroso r', '[C!H0]-[N]=[O]'),
        TautomerTransform('oxim/nitroso via phenol f', '[O!H0]-[N]=[C]-[C]=[C]-[C]=[OH0]'),
        TautomerTransform('oxim/nitroso via phenol r', '[O!H0]-[c]=[c]-[c]=[c]-[N]=[OH0]'),
        TautomerTransform('cyano/iso-cyanic acid f', '[O!H0]-[C]#[N]', bonds='=='),
        TautomerTransform('cyano/iso-cyanic acid r', '[N!H0]=[C]=[O]', bonds='#-'),
        TautomerTransform('isocyanide f', '[C-0!H0]#[N+0]', bonds='#', charges='-+'),
        TautomerTransform('isocyanide r', '[N+!H0]#[C-]', bonds='#', charges='-+'),
        TautomerTransform('phosphonic acid f', '[OH]-[PH0]', bonds='='),
        TautomerTransform('phosphonic acid r', '[PH]=[O]', bonds='-'))
    return newtfs


@click.command()
@click.argument("filename")
@click.option("-d", "--delimiter", default="\t", help="Column delimiter of input file.")
@click.option("-c", "--smls_col", default="SMILES", help="Name of column that contains SMILES.")
@click.option("-m", "--max_atoms", default=75, help="Maximum number of heavy atoms per mol.")
def main(filename, delimiter, smls_col, max_atoms):
    print(f"Loading and preprocessing dataset from file {filename}")
    df = standardize_df(filename, smls_col, max_atoms=max_atoms, delimiter=delimiter, drop_na=True)
    df["SMILES"] = df["SMILES_Std"]
    df = df.drop(columns="SMILES_Std")
    df.to_csv(f"{filename[:-4]}_proc.txt", sep="\t", index=False)


if __name__ == "__main__":
    main()
