#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: xhwan

from rdkit import Chem
from ase.io import read
import ase
from ogb.utils.mol import smiles2graph
import numpy as np
from mol2graph import mol2graph
import os.path as osp
import os
from torch_geometric.data import Data
import torch
from tqdm import tqdm

root = osp.join(os.getcwd())
molidx = 0
Pi2moli = []

for j in tqdm(range(1)): #4680
    downlimit = 25000*j + 1
    uplimit = 25000*(j+1)
    dis_data_dir = 'Compound_' + (str(0)*(9-len(str(downlimit)))) +str(downlimit) + '_' + (str(0)*(9-len(str(uplimit)))) +str(uplimit)
    if osp.exists(dis_data_dir):
        for k in tqdm(range(0, 25000)):

            i = downlimit + k
            nl = len(str(i))
            ID = str(0)*(9-nl)+str(i)

            data_dir = osp.join(root, dis_data_dir, ID)

            if osp.exists(data_dir):

                xyzname = osp.join(data_dir, ID+ '.PM6.S0.xyz')
                jsonname = osp.join(data_dir, ID + '.PM6.S0.json')

                graph = mol2graph(xyzname, jsonname)

                data = Data()

                data.__num_nodes__ = int(graph['num_nodes'])
                data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                data.y1 = torch.Tensor([graph['g_properties_1']])
                data.y2 = torch.Tensor([graph['g_properties_2']])
                data.y3 = graph['n_properties_1']

                savefile = osp.join(root, 'datasaved1', f'PQCM_processed{molidx}.pt')


                torch.save(data, savefile)

                # data_new = torch.load('1.pt')
                #
                # print(data_new)
                idx_rel = (i, molidx)
                Pi2moli.append(idx_rel)
                molidx += 1


    Pi2moli = np.array(Pi2moli)
    np.savetxt('idx_dict.txt', Pi2moli, fmt='%d', encoding='utf-8')





