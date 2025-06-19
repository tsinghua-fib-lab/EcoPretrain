import os
import torch
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from omegaconf import OmegaConf

from utilis import seed_everything
from model import EvolutionDynamicsModel, GenotypeEncoder


conf = OmegaConf.load('config.yaml')
conf.log_dir += f'L{conf.system.num_resources}/seed{conf.seed}/'
conf.data_dir += f'L{conf.system.num_resources}/'

trajectories = np.load(conf.data_dir+'trajectory.npy')[:100]


n_traj, n_step, N, num_resources = trajectories.shape
X, y, flag = [], [], []
for i in range(n_traj):
    for j in range(n_step-1):
        population = trajectories[i,j] # N, num_resources
        next_population = trajectories[i,j+1] # N, num_resources
        
        diff = next_population - population
        diff = diff[diff.sum(axis=1)>0]
        diff = diff + np.ones_like(diff)*-1
        index = np.where(np.isin(next_population, diff).all(axis=1))[0]
        if len(diff) == 0:
            continue
        else:
            X.append(population) 
            y.append(diff[0]) 
            flag.append(index[0])

X = np.array(X)
X_flatten = X.reshape(-1, N*num_resources)
y = np.array(y)
flag = np.array(flag)


T = conf.system.max_mutations
L = conf.system.num_resources
N = 2**L
D = conf.model.feature_dim
h = conf.model.hidden_dim
K = conf.model.K

model2 = GenotypeEncoder(L, conf.model.feature_dim).to(conf.device)
model2.load_state_dict(torch.load(conf.log_dir + f'checkpoints/distill_{500}.pth', map_location=conf.device))


X_tensor = torch.tensor(X).float().to(conf.device) # (sample_num, N, L)
y_tensor = torch.tensor(y).float().to(conf.device) # (sample_num, L)
with torch.no_grad():
    model2.eval()
    z = model2(X_tensor.reshape(-1, L)).reshape(X.shape[0], N, D).cpu().numpy()
    y_z = model2(y_tensor).cpu().numpy()

z_scaler = StandardScaler()
z_flatten = z_scaler.fit_transform(z.reshape(-1, D)).reshape(X.shape[0], N, D).reshape(X.shape[0], N*D)
y_z = z_scaler.transform(y_z)


model = EvolutionDynamicsModel(
    input_dim=L, 
    channel_dim=N, 
    embed_dim=D, 
    temporal_dim=h, 
    output_dim=L, 
    num_heads=conf.model.num_heads, 
    num_layers=conf.model.num_layers, 
    pos_encoding_type=conf.model.pos_encoding_type, 
    codebook_size=16
).to(conf.device)

model.load_state_dict(torch.load(conf.log_dir + f'checkpoints/pretrain_{conf.train.max_epoch}.pth', map_location=conf.device))


class GeneTypePredictor:
    def __init__(self, N, num_resources, ml='SVM'):
        self.N = N
        self.num_resources = num_resources 
        if ml == 'SVM':
            self.classifiers = MultiOutputClassifier(SVC(kernel='rbf', random_state=42))
        elif ml == 'RF':
            self.classifiers = MultiOutputClassifier(RandomForestClassifier(random_state=42))

    def fit(self, X, y):
        # X: (N, N*num_resources)
        # y: (N, num_resources)
        self.classifiers.fit(X, y)
    
    def predict(self, X):
        # X: (B, N*num_resources)
        predictions = self.classifiers.predict(X)
        return predictions


class LatentGeneTypePredictor:
    def __init__(self, N, num_resources, ml='SVR'):
        self.N = N 
        self.num_resources = num_resources 
        if ml == 'SVM':
            self.classifiers = MultiOutputRegressor(SVR(kernel='rbf'))

    def fit(self, X, y):
        # X: (N, N*num_resources)
        # y: (N, num_resources)
        self.classifiers.fit(X, y)
    
    def predict(self, X):
        # X: (B, N*num_resources)
        predictions = self.classifiers.predict(X)
        return predictions


baseline_result = np.zeros((10, 10, 3))
our_result = np.zeros((10, 10, 3))
for r_i, ratio in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    baseline_nmse, baseline_r2 = [], []
    our_nmse, our_r2 = [], []
    X_flatten_ = X_flatten[:int(len(X_flatten)*ratio)]
    z_flatten_ = z_flatten[:int(len(z_flatten)*ratio)]
    y_ = y[:int(len(y)*ratio)]
    y_z_ = y_z[:int(len(y_z)*ratio)]
    print(f'ratio: {ratio}, X_flatten: {X_flatten_.shape}, z_flatten: {z_flatten_.shape}, y: {y_.shape}, y_z: {y_z_.shape}')
    for seed in range(1, 10+1):
        seed_everything(seed)

        shuffle_idx = np.random.permutation(len(X_flatten_))
        sample_num = int(len(shuffle_idx)*0.7)
        X_train, X_test = X_flatten_[shuffle_idx][:sample_num], X_flatten_[shuffle_idx][sample_num:]
        y_train, y_test = y_[shuffle_idx][:sample_num], y_[shuffle_idx][sample_num:]

        predictor = GeneTypePredictor(N, num_resources, ml='SVM')
        predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)

        P = np.mean([precision_score(y_test[i, :], y_pred[i, :], average='binary', zero_division=0) for i in range(len(y_test))])
        R = np.mean([recall_score(y_test[i, :], y_pred[i, :], average='binary', zero_division=0) for i in range(len(y_test))])
        F1 = np.mean([f1_score(y_test[i, :], y_pred[i, :], average='binary', zero_division=0) for i in range(len(y_test))])
        baseline_result[r_i, seed-1] = [P, R, F1]

        z_train, z_test = z_flatten_[shuffle_idx][:sample_num], z_flatten_[shuffle_idx][sample_num:]
        z_y_train, z_y_test = y_z_[shuffle_idx][:sample_num], y_z_[shuffle_idx][sample_num:]

        predictor = LatentGeneTypePredictor(N, num_resources, ml='SVM')
        predictor.fit(z_train, z_y_train)
        z_y_pred = predictor.predict(z_test)

        z_y_pred = z_scaler.inverse_transform(z_y_pred)
        with torch.no_grad():
            model2.eval()
            y_pred = model.population_decoder.net(torch.tensor(z_y_pred).float().to(conf.device)).cpu().numpy() # (B, L, logits)
            y_pred = np.argmax(y_pred, axis=-1)

        P_ = np.mean([precision_score(y_test[i, :], y_pred[i, :], average='binary', zero_division=0) for i in range(len(y_test))])
        R_ = np.mean([recall_score(y_test[i, :], y_pred[i, :], average='binary', zero_division=0) for i in range(len(y_test))])
        F1_ = np.mean([f1_score(y_test[i, :], y_pred[i, :], average='binary', zero_division=0) for i in range(len(y_test))])
        our_result[r_i, seed-1] = [P_, R_, F1_]
        
        print(f'ratio: {ratio}, seed: {seed}, baseline: {P:.4f}, {R:.4f}, {F1:.4f}, our: {P_:.4f}, {R_:.4f}, {F1_:.4f}')

os.makedirs(conf.log_dir+'evolution/', exist_ok=True)
np.savez(conf.log_dir+'evolution/result.npz', baseline_result=baseline_result, our_result=our_result)