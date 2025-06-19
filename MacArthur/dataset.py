import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from simulator import getSimTraj


class Dataset_(Dataset):
    def __init__(self, conf, traj=None):
        super().__init__()
        self.conf = conf

        try:
            self.load_data(traj)
        except:
            getSimTraj(conf)
            self.load_data(traj)

    def load_data(self, traj):
        if traj is None:
            traj = np.load(self.conf.data_dir + 'trajectory.npy', allow_pickle=True) # (N, T, 2^L, L)

        self.X = torch.tensor(traj, dtype=torch.float32).to(self.conf.device)

    def __getitem__(self, index):
        return self.X[index] 

    def __len__(self):
        return len(self.X)
    
    def getLoader(self, batch_size, shuffle=True, drop_last=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)