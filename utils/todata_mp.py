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
from multiprocessing import Process

root = osp.join(os.getcwd())



def data_hand(arg, corei):
    step = 25000
    ID2idx = []
    # outname = 'ID2idx_' + str(corei) + '.txt'
    save_dir = osp.join(root, 'datasaved', 'part_' + str(corei))
    if not osp.exists(save_dir):
        os.mkdir(save_dir)
    for j in tqdm(arg):

        downlimit = step*j + 1
        uplimit = step*(j+1)
        dis_data_dir = 'Compound_' + (str(0)*(9-len(str(downlimit)))) +str(downlimit) + '_' + (str(0)*(9-len(str(uplimit)))) +str(uplimit)

        save_main_dir = osp.join(save_dir, dis_data_dir)
        if not osp.exists(save_main_dir):
            os.mkdir(save_main_dir)
        # molidx = 0
        if osp.exists(dis_data_dir):
            for k in range(0, step):

                i = downlimit + k
                nl = len(str(i))
                ID = str(0)*(9-nl)+str(i)

                data_dir = osp.join(root, dis_data_dir, ID)

                if osp.exists(data_dir):

                    xyzname = osp.join(data_dir, ID+ '.PM6.S0.xyz')
                    jsonname = osp.join(data_dir, ID + '.PM6.S0.json')

                    try:
                        graph = mol2graph(xyzname, jsonname)

                        data = Data()

                        data.__num_nodes__ = int(graph['num_nodes'])
                        data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                        data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                        data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                        data.y1 = torch.Tensor([graph['g_properties_1']])
                        data.y2 = torch.Tensor([graph['g_properties_2']])
                        # data.y3 = torch.Tensor([graph['n_properties_1']])


                        savefile = osp.join(save_main_dir, f'PQCM_processed_{ID}.pt')


                        torch.save(data, savefile)

                        # data_new = torch.load('1.pt')
                        #
                        # print(data_new)
                        # IDwithidx = ID + ' '+ str(downlimit)+ ' '+str(molidx) + '\n'
                        # ID2idx.append(IDwithidx)
                        # molidx += 1
                    except:
                        continue
    print('saving completed of number', corei, j)
    # with open(outname, 'w') as f:
    #     f.writelines(ID2idx)


if __name__ == '__main__':
    pool_num = 60
    args_list = []
    for i in range(60):
        down = i * 50
        up = down + 50
        L = list(range(down, up))
        args_list.append(L)

    # pool_num = 3
    # args_list = [[0], [1], [2]]

    processes = [Process(target=data_hand, args=(args_list[corei], corei)) for corei in range(pool_num)]
    # start all processes
    for process in processes:
        process.start()
    # wait for all processes to complete
    for process in processes:
        process.join()
    # report that all tasks are completed
    print('Done', flush=True)







