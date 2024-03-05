import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import random


class BSD500(Dataset):

    def __init__(self, data_file):
        super(Dataset, self).__init__()
        self.data_file = data_file
        self.dataset = None
        with h5py.File(self.data_file, 'r') as file:
            self.keys_list = list(file.keys())
            random.shuffle(self.keys_list)


    def __len__(self):
        return len(self.keys_list)


    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.data_file, 'r')
        data = torch.Tensor(np.array(self.dataset[self.keys_list[idx]]))
        return data




if __name__ == '__main__':
    train_dataset = BSD500('../data/train.h5')
    train_dataloader = DataLoader(train_dataset, num_workers=8, batch_size=128)
    print(len(train_dataloader))
    for batch_idx, data in enumerate(train_dataloader):
        if (batch_idx % 250 == 0):
            print(batch_idx)
