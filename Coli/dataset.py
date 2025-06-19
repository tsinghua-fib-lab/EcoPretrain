import os
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


drug_dict = {'TET': 0, 'KM': 1, 'NFLX': 2, 'SS': 3, 'PLM': 4, 'NQO': 5, 'SDC': 6, 'MMC': 7}

def getTraj():
    try:
        traj = np.load('data/trajectorys.npy', allow_pickle=True)
    except:
        traj = np.zeros((43, 27, 8))
        df = pd.read_excel('coli.xlsx', sheet_name='S1ABCDEFGHIJKL')
        for i in range(df.shape[0]):
            row = df.iloc[i]
            
            if 'Parent in M9' in row['strain']:
                pass
            elif 'NFLXE4 in KM 2' in row['strain']:
                pass
            elif 'Parent in TET' in row['strain']:
                repeat = int(row['strain'][-1]) - 1
                for day in range(1, 27+1):
                    traj[0+repeat, day-1, drug_dict[row['drug']]] = row[f'day{day}']
            elif 'KME1 in TET' in row['strain']:
                repeat = int(row['strain'][-1]) - 1
                for day in range(1, 27+1):
                    traj[4+repeat, day-1, drug_dict[row['drug']]] = row[f'day{day}']
            elif 'KME5 in TET' in row['strain']:
                repeat = int(row['strain'][-1]) - 1
                for day in range(1, 27+1):
                    traj[8+repeat, day-1, drug_dict[row['drug']]] = row[f'day{day}']
            elif 'Parent in KM' in row['strain']:
                repeat = int(row['strain'][-1]) - 1
                for day in range(1, 27+1):
                    traj[12+repeat, day-1, drug_dict[row['drug']]] = row[f'day{day}']
            elif 'TETE4 in KM' in row['strain']:
                repeat = int(row['strain'][-1]) - 1
                for day in range(1, 27+1):
                    traj[16+repeat, day-1, drug_dict[row['drug']]] = row[f'day{day}']
            elif 'TETE6 in KM' in row['strain']:
                repeat = int(row['strain'][-1]) - 1
                for day in range(1, 27+1):
                    traj[20+repeat, day-1, drug_dict[row['drug']]] = row[f'day{day}']
            elif 'NFLXE4 in KM' in row['strain']:
                repeat = int(row['strain'][-1]) - 1 if int(row['strain'][-1])==1 else int(row['strain'][-1]) - 2
                for day in range(1, 27+1):
                    traj[24+repeat, day-1, drug_dict[row['drug']]] = row[f'day{day}']
            elif 'NFLXE6 in KM' in row['strain']:
                repeat = int(row['strain'][-1]) - 1
                for day in range(1, 27+1):
                    traj[28-1+repeat, day-1, drug_dict[row['drug']]] = row[f'day{day}']
            elif 'Parent in NFLX' in row['strain']:
                repeat = int(row['strain'][-1]) - 1
                for day in range(1, 27+1):
                    traj[32-1+repeat, day-1, drug_dict[row['drug']]] = row[f'day{day}']
            elif 'KME1 in NFLX' in row['strain']:
                repeat = int(row['strain'][-1]) - 1
                for day in range(1, 27+1):
                    traj[36-1+repeat, day-1, drug_dict[row['drug']]] = row[f'day{day}']
            elif 'KME5 in NFLX' in row['strain']:
                repeat = int(row['strain'][-1]) - 1
                for day in range(1, 27+1):
                    traj[40-1+repeat, day-1, drug_dict[row['drug']]] = row[f'day{day}']
            else:
                print(row['strain'])
                raise ValueError('Unknown strain')
        
        os.makedirs('data/', exist_ok=True)
        np.save('data/trajectorys.npy', traj)
    
    return traj

class Dataset_coli(Dataset):
    def __init__(self, conf, traj=None, test=False, fitness=False):
        super().__init__()
        self.conf = conf

        traj = getTraj() # N, T, L
        self.test_data = self.process_data(traj, test=test, fitness=fitness)

    def process_data(self, original_traj, test, fitness):
        if original_traj is None:
            original_traj = np.load('data/trajectorys.npy', allow_pickle=True)

        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        N, T, L = original_traj.shape
        original_traj = self.scaler.fit_transform(original_traj.reshape(-1, L)).reshape(N, T, L)
        
        if not fitness:
            mask_variable = self.conf.coli.mask_variable
            index = drug_dict[mask_variable]
            original_traj = np.delete(original_traj, index, axis=-1)
        
        test_idx = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 26, 28, 30, 34, 38, 42]
        if test:
            return original_traj[test_idx]
        else:
            traj_idx = [i for i in range(43) if i not in test_idx]
        
        self.X = original_traj[traj_idx] # (N, T, L)

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return len(self.X)
    
    def getLoader(self, batch_size, shuffle=True, drop_last=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


if __name__ == '__main__':
    
    from omegaconf import OmegaConf
    conf = OmegaConf.load('coli.yaml')
    trajectorys = Dataset_coli(conf)