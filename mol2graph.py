#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mol2graph.py
@Time    :   2023/03/08 09:46:54
@Author  :   Wan Xuhao 
@Version :   1.0
@Contact :   xhwanrm@whu.edu.cn
@License :   (C)Copyright 2022
@Desc    :   None
'''


from rdkit import Chem
import numpy as np
from ase.io import read
import json
import torch
from torch_geometric.data import Data



def Get_Bondlength(mola, i, j):
    BL = mola.get_distance(i, j)
    return Dis_Bondlength(BL, 1.20, 1.70)


def Dis_Bondlength(L, min, max, step=0.025):
    if L < min:
        Ldis = 0
    elif L > max:
        Ldis = 21
    else:
        Ldis = (((L - min) * 1000)//(step*1000)) + 1
    return Ldis

def Get_Angle(mola, i, j):
    if i == 0 or j == 0:
        Angle = 0
    else:
        Angle = mola.get_angle(j, i, 0)
    return round(Angle)


Total_features = {
    'Atomic_number_list' : list(range(1, 119)) + ['other'],
    'Bond_length_list' : list(range(0, 22)),
    'Angle_list' : list(range(0, 181)),
    'Chirality_list' : ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'],
    'Degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'other'],
    'Formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'other'],
    'Valence_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'other'],
    'NumberH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'other'],
    'Radicale_list': [0, 1, 2, 3, 4, 'other'],
    'Hybridization_list' : ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'other'],
    'Aromatic_list': [False, True],
    'Ring_list': [False, True],
    'Bond_type_list' : ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'other'],
    'Bond_stereo_list': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY', 'other'], 
    'Conjugated_list': [False, True],
}

def get_index(lst, ele):
    try:
        return lst.index(ele)
    except:
        return len(lst) - 1


def Atom2Feature(atom):
    Atom_feature = [
            get_index(Total_features['Atomic_number_list'], atom.GetAtomicNum()),
            Total_features['Chirality_list'].index(str(atom.GetChiralTag())),
            get_index(Total_features['Degree_list'], atom.GetTotalDegree()),
            get_index(Total_features['Formal_charge_list'], atom.GetFormalCharge()),
            get_index(Total_features['Valence_list'], atom.GetTotalValence()),
            get_index(Total_features['NumberH_list'], atom.GetTotalNumHs()),
            get_index(Total_features['Radicale_list'], atom.GetNumRadicalElectrons()),
            get_index(Total_features['Hybridization_list'], str(atom.GetHybridization())),
            Total_features['Aromatic_list'].index(atom.GetIsAromatic()),
            Total_features['Ring_list'].index(atom.IsInRing()),
            ]
    return Atom_feature


def get_AFdims():
    return list(map(len, [
        Total_features['Atomic_number_list'],
        Total_features['Chirality_list'],
        Total_features['Degree_list'],
        Total_features['Formal_charge_list'],
        Total_features['Valence_list'],
        Total_features['NumberH_list'],
        Total_features['Radicale_list'],
        Total_features['Hybridization_list'],
        Total_features['Aromatic_list'],
        Total_features['Ring_list']
        ]))

def Bond2Feature(bond, mola, i, j):
    Bond_feature = [
                get_index(Total_features['Bond_type_list'], str(bond.GetBondType())),
                Total_features['Bond_length_list'].index(Get_Bondlength(mola, i, j)),
                Total_features['Angle_list'].index(Get_Angle(mola, i, j)),
                Total_features['Aromatic_list'].index(bond.GetIsAromatic()),
                Total_features['Bond_stereo_list'].index(str(bond.GetStereo())),
                Total_features['Conjugated_list'].index(bond.GetIsConjugated()),

            ]
    return Bond_feature


def get_BFdims():
    return list(map(len, [
        Total_features['Bond_type_list'],
        Total_features['Bond_length_list'],
        Total_features['Angle_list'],
        Total_features['Aromatic_list'],
        Total_features['Bond_stereo_list'],
        Total_features['Conjugated_list']
        ]))

def AF2dict(Atom_feature):
    [atomic_num_idx, 
    chirality_idx,
    degree_idx,
    formal_charge_idx,
    num_h_idx,
    valence_idx,
    number_radical_e_idx,
    hybridization_idx,
    is_aromatic_idx,
    is_in_ring_idx] = Atom_feature

    AF_dict = {
        'atomic_num': Total_features['Atomic_number_list'][atomic_num_idx],
        'chirality': Total_features['Chirality_list'][chirality_idx],
        'degree': Total_features['Degree_list'][degree_idx],
        'formal_charge': Total_features['Formal_charge_list'][formal_charge_idx],
        'valence':Total_features['Valence_list'][valence_idx],
        'num_h': Total_features['NumberH_list'][num_h_idx],
        'num_rad_e': Total_features['Radicale_list'][number_radical_e_idx],
        'hybridization': Total_features['Hybridization_list'][hybridization_idx],
        'is_aromatic': Total_features['Aromatic_list'][is_aromatic_idx],
        'is_in_ring': Total_features['Ring_list'][is_in_ring_idx]
    }

    return AF_dict


def BF2dict(bond_feature):
    [bond_type_idx,
     bond_length_idx,
     angle_idx,
     is_aromatic_idx,
     bond_stereo_idx,
     is_conjugated_idx] = bond_feature

    BF_dict = {
        'bond_type': Total_features['Bond_type_list'][bond_type_idx],
        'bond_length': Total_features['Bond_length_list'][bond_length_idx],
        'angle': Total_features['Angle_list'][angle_idx],
        'is_aromatic': Total_features['Aromatic_list'][is_aromatic_idx],
        'bond_stereo': Total_features['Bond_stereo_list'][bond_stereo_idx],
        'is_conjugated': Total_features['Conjugated_list'][is_conjugated_idx]
    }

    return BF_dict


def mol2graph(xyzname, jsonname):


    with open(jsonname, 'r', encoding='utf-8') as f:
        inf = json.load(f)
    smiles = inf["pubchem"]["Isomeric SMILES"]
    TDM = inf["pubchem"]["PM6"]["properties"]["total dipole moment"]
    Eg = inf["pubchem"]["PM6"]["properties"]["energy"]["alpha"]["gap"]
    Charge = inf["pubchem"]["PM6"]["properties"]["partial charges"]["mulliken"]


    mola = read(xyzname)
    mols = Chem.MolFromSmiles(smiles)

    # atoms
    Atom_feature = []
    for atom in mols.GetAtoms():
        Atom_feature.append(Atom2Feature(atom))
    x = np.array(Atom_feature, dtype = np.int64)

    # bonds
    num_bond_features = 6
    if len(mols.GetBonds()) > 0: # mol has bonds
        edges_list = []
        Edge_feature = []
        for bond in mols.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = Bond2Feature(bond, mola, i, j)

            # add edges in both directions
            edges_list.append((i, j))
            Edge_feature.append(edge_feature)
            edges_list.append((j, i))
            Edge_feature.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(Edge_feature, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['node_feat'] = x
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['num_nodes'] = len(x)
    graph['g_properties_1'] = TDM
    graph['g_properties_2'] = Eg
    graph['n_properties_1'] = Charge[0:len(x)]

    return graph


if __name__ == '__main__':
    ID = '000000001'
    xyzname = ID + '.PM6.S0.xyz'
    jsonname = ID + '.PM6.S0.json'
    # with open(jsonname, 'r', encoding='utf-8') as f:
    #     content = json.load(f)
    # print(content["pubchem"]["Isomeric SMILES"])
    # print(content["pubchem"]["PM6"]["properties"]["total dipole moment"])
    # print(content["pubchem"]["PM6"]["properties"]["energy"]["alpha"]["gap"])
    # print(content["pubchem"]["PM6"]["properties"]["partial charges"]["mulliken"])
    # print(content["pubchem"]["PM6"].keys())
    graph = mol2graph(xyzname, jsonname)

    data = Data()

    data.__num_nodes__ = int(graph['num_nodes'])
    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
    data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
    data.y1 = torch.Tensor([graph['g_properties_1']])
    data.y2 = torch.Tensor([graph['g_properties_2']])
    data.y3 = torch.Tensor([graph['n_properties_1']])


    print(data)

    torch.save(data, '1.pt')
    data_new = torch.load('1.pt')

    print(data_new)







