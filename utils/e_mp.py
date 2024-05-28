#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: xhwan


import os.path as osp
import os
from tqdm import tqdm
from multiprocessing import Process
import tarfile

root = osp.join(os.getcwd())



def exp(arg, corei):
    for j in tqdm(arg):
        downlimit = 25000*j + 1
        uplimit = 25000*(j+1)
        dis_data_dir = 'Compound_' + (str(0)*(9-len(str(downlimit)))) +str(downlimit) + '_' + (str(0)*(9-len(str(uplimit)))) +str(uplimit) + '_CHNOPSFCl500noSalt.tar.xz'

        if osp.exists(dis_data_dir):
            with tarfile.open(dis_data_dir, 'r') as tar_obj:
                tar_obj.extractall(path = '.')
        print('saving completed of number', corei)



if __name__ == '__main__':
    pool_num = 16
    args_list = []
    for i in range(16):
        down = i * 300
        up = down + 300
        L = list(range(down, up))
        args_list.append(L)

    # pool_num = 3
    # args_list = [[0], [1], [2]]

    processes = [Process(target=exp, args=(args_list[corei], corei)) for corei in range(pool_num)]
    # start all processes
    for process in processes:
        process.start()
    # wait for all processes to complete
    for process in processes:
        process.join()
    # report that all tasks are completed
    print('Done', flush=True)







