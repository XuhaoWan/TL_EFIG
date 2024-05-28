#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: xhwan

import os
import os.path as osp
import tarfile
from tqdm import tqdm

for j in tqdm(range(4860)):
    downlimit = 25000*j + 1
    uplimit = 25000*(j+1)
    dis_data_dir = 'Compound_' + (str(0)*(9-len(str(downlimit)))) +str(downlimit) + '_' + (str(0)*(9-len(str(uplimit)))) +str(uplimit) + '_CHNOPSFCl500noSalt.tar.xz'

    if osp.exists(dis_data_dir):
        with tarfile.open(dis_data_dir, 'r') as tar_obj:
            tar_obj.extractall(path = '.')
