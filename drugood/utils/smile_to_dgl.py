import dgl
import numpy as np
import rdkit
import torch
from rdkit import Chem


def get_atom_features(atom):
    # The usage of features is along with the Attentive FP.
    feature = np.zeros(39)

    # Symbol
    symbol = atom.GetSymbol()
    symbol_list = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At']
    if symbol in symbol_list:
        loc = symbol_list.index(symbol)
        feature[loc] = 1
    else:
        feature[15] = 1

    # Degree
    degree = atom.GetDegree()
    if degree > 5:
        print("atom degree larger than 5. Please check before featurizing.")
        raise RuntimeError

    feature[16 + degree] = 1

    # Formal Charge
    charge = atom.GetFormalCharge()
    feature[22] = charge

    # radical electrons
    radelc = atom.GetNumRadicalElectrons()
    feature[23] = radelc

    # Hybridization
    hyb = atom.GetHybridization()
    hybridization_list = [rdkit.Chem.rdchem.HybridizationType.SP,
                          rdkit.Chem.rdchem.HybridizationType.SP2,
                          rdkit.Chem.rdchem.HybridizationType.SP3,
                          rdkit.Chem.rdchem.HybridizationType.SP3D,
                          rdkit.Chem.rdchem.HybridizationType.SP3D2]
    if hyb in hybridization_list:
        loc = hybridization_list.index(hyb)
        feature[loc + 24] = 1
    else:
        feature[29] = 1

    # aromaticity
    if atom.GetIsAromatic():
        feature[30] = 1

    # hydrogens
    hs = atom.GetNumImplicitHs()
    feature[31 + hs] = 1

    # chirality, chirality type
    if atom.HasProp('_ChiralityPossible'):
        # TODO what kind of error
        feature[36] = 1

        try:
            chi = atom.GetProp('_CIPCode')
            chi_list = ['R', 'S']
            loc = chi_list.index(chi)
            feature[37 + loc] = 1
        except KeyError:
            feature[37] = 0
            feature[38] = 0

    return feature


def get_bond_features(bond):
    feature = np.zeros(10)

    # bond type
    type = bond.GetBondType()
    bond_type_list = [rdkit.Chem.rdchem.BondType.SINGLE,
                      rdkit.Chem.rdchem.BondType.DOUBLE,
                      rdkit.Chem.rdchem.BondType.TRIPLE,
                      rdkit.Chem.rdchem.BondType.AROMATIC]
    if type in bond_type_list:
        loc = bond_type_list.index(type)
        feature[0 + loc] = 1
    else:
        print("Wrong type of bond. Please check before feturization.")
        raise RuntimeError

    # conjugation
    conj = bond.GetIsConjugated()
    feature[4] = conj

    # ring
    ring = bond.IsInRing()
    feature[5] = ring

    # stereo
    stereo = bond.GetStereo()
    stereo_list = [rdkit.Chem.rdchem.BondStereo.STEREONONE,
                   rdkit.Chem.rdchem.BondStereo.STEREOANY,
                   rdkit.Chem.rdchem.BondStereo.STEREOZ,
                   rdkit.Chem.rdchem.BondStereo.STEREOE]
    if stereo in stereo_list:
        loc = stereo_list.index(stereo)
        feature[6 + loc] = 1
    else:
        print("Wrong stereo type of bond. Please check before featurization.")
        raise RuntimeError

    return feature


def smile2graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if (mol is None):
        return None
    src = []
    dst = []
    atom_feature = []
    bond_feature = []

    try:
        for atom in mol.GetAtoms():
            one_atom_feature = get_atom_features(atom)
            atom_feature.append(one_atom_feature)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            one_bond_feature = get_bond_features(bond)
            src.append(i)
            dst.append(j)
            bond_feature.append(one_bond_feature)
            src.append(j)
            dst.append(i)
            bond_feature.append(one_bond_feature)

        src = torch.tensor(src).long()
        dst = torch.tensor(dst).long()
        atom_feature = np.array(atom_feature)
        bond_feature = np.array(bond_feature)
        atom_feature = torch.tensor(atom_feature).float()
        bond_feature = torch.tensor(bond_feature).float()
        graph_cur_smile = dgl.graph((src, dst), num_nodes=len(mol.GetAtoms()))
        graph_cur_smile.ndata['x'] = atom_feature
        graph_cur_smile.edata['x'] = bond_feature
        return graph_cur_smile

    except RuntimeError:
        return None


def featurize_atoms(mol):
    feats = []
    for atom in mol.GetAtoms():
        feats.append(atom.GetAtomicNum())
    return {'atomic': torch.tensor(feats).reshape(-1).to(torch.int64)}


def featurize_bonds(mol):
    feats = []
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    for bond in mol.GetBonds():
        btype = bond_types.index(bond.GetBondType())
        # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
        feats.extend([btype, btype])
    return {'type': torch.tensor(feats).reshape(-1).to(torch.int64)}
