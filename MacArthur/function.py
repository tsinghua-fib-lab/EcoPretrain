import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings; warnings.filterwarnings("ignore")

from simulator import population_function
from model import GenotypeEncoder
from utilis import seed_everything


conf = OmegaConf.load('config.yaml')

# samplenum_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
samplenum_list = [1000]
# size_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
size_list = [10]
L = 8

for sample_num in samplenum_list:
    for size in size_list:
        for K in [8, 16, 32, 64, 2**L]:

            conf = OmegaConf.load('config.yaml')
            conf.log_dir += f'L{L}/seed{conf.seed}/'
            conf.data_dir += f'L{L}/'
            os.makedirs(conf.log_dir+'function', exist_ok=True)

            seed_everything(conf.seed)

            from utilis import random_matrix
            trait_inter_matrix = random_matrix(
                (L, L),
                'tikhonov_sigmoid',
                args={'J_0': 0.5, 'n_star': 4, 'delta': 1},
                triangular=True,
                diagonal=0,
                seed=100
            )
            
            if K != 2**L:
                with open(conf.log_dir + f'coarsen/strains_K{K}.pkl', 'rb') as f:
                    strains_pool_our = pickle.load(f)
                with open(conf.log_dir + f'coarsen/strains_prx22_K{K}.pkl', 'rb') as f:
                    strains_pool_prx22 = pickle.load(f)

                try:
                    read = np.load(conf.log_dir + f'function/dataset_size{size}_N{sample_num}_K{K}.npz')
                    X, y = read['X'], read['y']
                    X_coarsen_our, X_coarsen_prx22 = read['X_coarsen_our'], read['X_coarsen_prx22']
                except:
                    gene_space = np.array([[int(i) for i in format(j, '0'+str(L)+'b')] for j in range(2**L)], dtype=np.float32) # (2^L-1, L)
                    X, y = np.zeros((sample_num, 2**L)), np.zeros((sample_num))
                    X_coarsen_our, X_coarsen_prx22 = np.zeros((sample_num, K)), np.zeros((sample_num, K))
                    for i in tqdm(range(sample_num)):
                        idxs = np.random.choice(gene_space.shape[0], size, replace=False)
                        traits = gene_space[idxs] 
                        y[i] = population_function(traits, L, trait_inter_matrix, total_t=1e5, dt=1e2)
                        X[i, idxs] = 1
                        
                        for trait in traits:
                            idx = [ii for ii, strains in enumerate(strains_pool_our) if np.any(np.all(strains == trait, axis=1))][0]
                            X_coarsen_our[i, idx] = 1
                            idx = [ii for ii, strains in enumerate(strains_pool_prx22) if np.any(np.all(strains == trait, axis=1))][0]
                            X_coarsen_prx22[i, idx] = 1
                    np.savez(conf.log_dir + f'function/dataset_size{size}_N{sample_num}_K{K}.npz', X=X, y=y, X_coarsen_our=X_coarsen_our, X_coarsen_prx22=X_coarsen_prx22)
                
                    if K == 8:
                        plt.hist(y, bins=100, rwidth=0.8)
                        plt.savefig(conf.log_dir+f'function/yhist_size{size}_N{sample_num}_K{K}.png', dpi=300)
                        plt.close()
            else:
                read = np.load(conf.log_dir + f'function/dataset_size{size}_N{sample_num}_K{64}.npz')
                X, y = read['X'], read['y']
            
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=10, shuffle=True, random_state=conf.seed)
            
            scaler = StandardScaler()
            y = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
            
            for ml in ['SVM', 'LR']:
                
                if K == 2**L:
                    r2__, nmse__ = [], []
                    for train_index, test_index in kf.split(X):
                        if ml == 'SVM':
                            predictor = SVR(kernel='rbf')
                        elif ml == 'LR':
                            predictor = LinearRegression()
                        
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test_ = y[train_index], y[test_index]
                        
                        predictor.fit(X_train, y_train)
                        y_pred_ = predictor.predict(X_test)
                        r2__.append(r2_score(y_test_, y_pred_))
                        nmse__.append(np.mean((y_test_-y_pred_)**2)/np.mean((y_test_-np.mean(y_test_))**2))
                    print(f'No coarsen [{ml}]: {np.mean(r2__), np.mean(nmse__)}')
                    continue
                    
                    
                if os.path.exists(conf.log_dir + f'function/metrics_size{size}_N{sample_num}_K{K}_{ml}.npy'):
                    continue
                else:
                    r2_, nmse_ = [], []

                for train_index, test_index in kf.split(X_coarsen_prx22):
                    if ml == 'SVM':
                        predictor = SVR(kernel='rbf')
                    elif ml == 'LR':
                        predictor = LinearRegression()
                    
                    X_train, X_test = X_coarsen_prx22[train_index], X_coarsen_prx22[test_index]
                    y_train, y_test_ = y[train_index], y[test_index]
                    
                    predictor.fit(X_train, y_train)
                    y_pred_ = predictor.predict(X_test)
                    r2_.append(r2_score(y_test_, y_pred_))
                    nmse_.append(np.mean((y_test_-y_pred_)**2)/np.mean((y_test_-np.mean(y_test_))**2))
                    
                    with open(conf.log_dir + f'function/{ml}_size{size}_N{sample_num}_K{K}_prx22.pkl', 'wb') as f:
                        pickle.dump(predictor, f)

                nmse, r2 = [], []
                for train_index, test_index in kf.split(X_coarsen_our):
                    if ml == 'SVM':
                        predictor2 = SVR(kernel='rbf')
                    elif ml == 'LR':
                        predictor2 = LinearRegression()
                    
                    X_train, X_test = X_coarsen_our[train_index], X_coarsen_our[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    predictor2.fit(X_train, y_train)
                    y_pred = predictor2.predict(X_test)
                    r2.append(r2_score(y_test, y_pred))
                    nmse.append(np.mean((y_test-y_pred)**2)/np.mean((y_test-np.mean(y_test))**2))
                    
                    with open(conf.log_dir + f'function/{ml}_size{size}_N{sample_num}_K{K}.pkl', 'wb') as f:
                        pickle.dump(predictor2, f)
                
                
                plt.figure(figsize=(6, 1.7))
                plt.subplot(1, 2, 1)
                plt.boxplot([r2_, r2], tick_labels=['Prefix', 'Embedding'])
                plt.ylabel('R2')
                plt.ylim(-0.1, 1.1)
                plt.subplot(1, 2, 2)
                plt.boxplot([nmse_, nmse], tick_labels=['Prefix', 'Embedding'])
                plt.ylabel('NMSE')
                plt.ylim(-0.1, 1.1)
                plt.tight_layout()
                plt.savefig(conf.log_dir + f'function/metrics_size{size}_N{sample_num}_{ml}_K{K}.png', dpi=300)
                plt.close()
                
                np.save(conf.log_dir + f'function/metrics_size{size}_N{sample_num}_{ml}_K{K}.npy', [r2_, nmse_, r2, nmse])
                
                print(f'Finish size{size}_N{sample_num}_{ml}_K{K}')