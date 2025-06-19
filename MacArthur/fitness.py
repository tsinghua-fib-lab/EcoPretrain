import os
import torch
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler

from utilis import seed_everything
from model import GenotypeEncoder


conf = OmegaConf.load('config.yaml')
conf.log_dir += f'L{conf.system.num_resources}/seed{conf.seed}/'
conf.data_dir += f'L{conf.system.num_resources}/'

trajectories = np.load(conf.data_dir+'trajectory.npy')[:1000]

n_traj, n_step, N, num_resources = trajectories.shape
count = np.zeros((N,)).astype(np.int32) # (2^L-1,)
gene_space = np.array([[int(i) for i in format(j, '0'+str(num_resources)+'b')] for j in range(2**num_resources)], dtype=np.float32) # (2^L-1, L)
for i in range(n_traj):
    for j in range(n_step):
        population = trajectories[i,j] # N, num_resources
        exist_species = np.where(population[:,0]!=-1)[0]
        for species in exist_species:
            gene = population[species]
            index = np.where((gene_space==gene).all(axis=1))[0]
            count[index] += 1
        
fitness = np.log(count/np.sum(count)+1e-6) # (2^L-1,)

T = conf.system.max_mutations
L = conf.system.num_resources
N = 2**L
D = conf.model.feature_dim
h = conf.model.hidden_dim
K = conf.model.K

model2 = GenotypeEncoder(L, conf.model.feature_dim).to(conf.device)
model2.load_state_dict(torch.load(conf.log_dir + f'checkpoints/distill_{500}.pth', map_location=conf.device))

with torch.no_grad():
    model2.eval()
    gene_space_ = torch.from_numpy(gene_space).to(conf.device)
    z = model2(gene_space_).cpu().numpy()
    
    
z_scaler = StandardScaler()
z_ = z_scaler.fit_transform(z)

for ratio in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    baseline_nmse, baseline_r2 = [], []
    our_nmse, our_r2 = [], []
    seed = 1
    r2_last = 0.0
    while len(baseline_nmse)<10:
        seed_everything(seed)
        seed += 1

        shuffle_idx = np.random.permutation(len(gene_space))
        if len(shuffle_idx) % 2 == 1:
            shuffle_idx = shuffle_idx[:-1]

        train_idx = shuffle_idx[:int(len(shuffle_idx)*ratio)]
        test_idx = shuffle_idx[int(len(shuffle_idx)*ratio):]

        X_train = gene_space[train_idx]
        X_test = gene_space[test_idx]
        y_train = fitness[train_idx]
        y_test = fitness[test_idx]



        predictor = SVR(kernel='rbf')
        predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)

        nmse = mean_squared_error(y_test, y_pred) / np.var(y_test)
        r2 = r2_score(y_test, y_pred)
        if r2 < -0.1:
            continue
        baseline_nmse.append(nmse)
        baseline_r2.append(r2)


        z_train = z_[train_idx]
        z_test = z_[test_idx]

        predictor = SVR(kernel='rbf')
        predictor.fit(z_train, y_train)
        y_pred = predictor.predict(z_test)

        nmse_ = mean_squared_error(y_test, y_pred) / np.var(y_test)
        r2_ = r2_score(y_test, y_pred)
        our_nmse.append(nmse_)
        our_r2.append(r2_)
        
        if r2_ > r2_last and ratio == 0.5:
            r2_last = r2_
            print(f'ratio: {ratio}, seed: {seed-1}, r2: {r2_:.4f}, nmse: {nmse_:.4f}')
            os.makedirs(conf.log_dir+f'fitness/', exist_ok=True)
            np.savez(conf.log_dir+f'fitness/r{ratio}.npz', y_test=y_test, y_pred=y_pred, r2=r2_, nmse=nmse_)
    
    print(f'\nratio: {ratio}')
    print(f'baseline_nmse: {np.mean(baseline_nmse):.4f}±{np.std(baseline_nmse):.4f}')
    print(f'nmse: {np.mean(our_nmse):.4f}±{np.std(our_nmse):.4f}')
    print(f'baseline_r2: {np.mean(baseline_r2):.4f}±{np.std(baseline_r2):.4f}')
    print(f'r2: {np.mean(our_r2):.4f}±{np.std(our_r2):.4f}')
    print('-----------------------------------')

    os.makedirs(conf.log_dir+f'fitness/', exist_ok=True)
    np.savez(conf.log_dir+f'fitness/r-{ratio}.npz', baseline_nmse=baseline_nmse, baseline_r2=baseline_r2, our_nmse=our_nmse, our_r2=our_r2)