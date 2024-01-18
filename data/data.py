import os
import os.path as osp
import torch
from torch.utils.data import random_split
from typing import Any, Callable, List, Optional, Tuple, Union
from collections.abc import Sequence
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader


class PQCdataset(Dataset):
    def __init__(self, path, num, transform=None, pre_transform=None, pre_filter=None):
        self.original_path = path
        self.datanum = num
        self.folder = osp.join(os.getcwd())

        super().__init__(self.folder, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        return []

    @property
    def processed_paths(self) -> List[str]:
        return [self.original_path]

    def process(self):
        return FileNotFoundError

    def len(self) -> int:
        return int(self.datanum)

    def get(self, idx: int) -> Data:
        data = torch.load(osp.join(self.processed_paths[0], 'part'+str(idx//5000), f'PQCM_processed{idx}.pt'))
        return data


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def get_ds(data_path, Total_num, splitpara):
    dataset = PQCdataset(path = data_path, num = Total_num)

    train_dataset, val_dataset = random_split(dataset=dataset, lengths=splitpara)
    # train_dataset, val_dataset = random_split(dataset=dataset, lengths=splitpara,
    #                                           generator=torch.Generator().manual_seed(0))

    return train_dataset, val_dataset


def get_dl(config, data_path, Total_num, splitpara):
    train_dataset, val_dataset = get_ds(data_path, Total_num, splitpara)
    batch_size = config.batch_size
    num_workers = config.nw

    train_dl = DataLoader(train_dataset, batch_size=batch_size,
                          num_workers=num_workers, shuffle=True,
                          drop_last=True)
    valid_dl = DataLoader(val_dataset, batch_size=batch_size,
                          num_workers=num_workers, shuffle=False,
                          drop_last=False)

    return train_dl, valid_dl


def gety(dl):
    ys = []
    for batch in dl:
        ys.append(batch.y)
    y = torch.cat(ys, dim=0)
    return y


if __name__ == '__main__':
    dataset = PQCdataset(path='datasaved', num=16000)

    # print(len(dataset))
    # for i in tqdm(range(1,45280)):
    #     data_selected = dataset[i]
    #     bs = list(data_selected.edge_attr[:, 4])
    #     if 1 in bs:
    #         print(i)
    # data_selected = dataset[2]
    # print(data_selected)
    # print(data_selected.x.shape)
    # print(data_selected.edge_index.shape)
    # print(data_selected.edge_attr.shape)
    # print(data_selected.y1.shape)
    # print(data_selected.y2.shape)

    train_dl = DataLoader(dataset, batch_size=2, num_workers=1)
    for batch in train_dl:
        break
    print(batch.x.shape)
    # print(batch.edge_index.shape)
    print(batch.y3)
    # num_feas = batch.x.shape[1]
    # print(f'num_feas: {num_feas}')
    # print(dataset[100].y)
    #
    print(len(train_dl))
    # a = gety(train_dl)
    # print(a)
