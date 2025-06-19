import os
import torch
import numpy as np
from sklearn import svm
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import warnings; warnings.filterwarnings("ignore")

from model import EvolutionDynamicsModel
from utilis import compute_metrics


class GeneTypePredictor:
    def __init__(self, L, b, ml='SVM'):
        self.L = L 
        self.b = b 
        if ml == 'SVM':
            self.classifiers = [svm.SVC(kernel='rbf', random_state=42) for _ in range(L)]

    def fit(self, X, y):
        # X: (N, L*b)
        # y: (N, L)
        for i in range(self.L):
            y_i= y[:, i].reshape(-1)
            self.classifiers[i].fit(X, y_i)
        

    def predict(self, X):
        # X: (N, L*b)
        predictions = np.ones((X.shape[0], self.L)) * -1  # N, L
        for i in range(self.L):
            predictions[:, i] = self.classifiers[i].predict(X)
        return predictions
    
    def random_predict(self, N, T):
        predictions = np.random.randint(self.b, size=(N, T-1, self.L))
        return predictions

    def autoregressive_predict(self, initial_state, T):
        # initial_state: (N, L)
        predictions = np.ones((initial_state.shape[0], T-1, self.L), dtype=int) * -1  # N, T-1, L
        
        current_state = initial_state
        for t in range(T-1):
            X = np.zeros((current_state.shape[0], self.L * self.b))
            for i in range(self.L):
                X[:, i*self.b:(i+1)*self.b] = np.eye(self.b)[current_state[:, i]]
            predictions[:, t] = self.predict(X).astype(int)
            current_state = predictions[:, t]
        return predictions
    

DEVICE = 'cuda:0'

conf = OmegaConf.load('config/eqFP611.yaml')

traj_num = 1000

traj = np.load(conf.data_dir + 'evolution_trajectories.npy', allow_pickle=True)[:traj_num] # (N, T, L)
traj_onehot = np.eye(conf.eqFP611.states)[traj.astype(int)]  # (N, T, L, nstates)

dim_list = [4, 8, 12, 16, 20, 24, 28]
metrics_baseline, metrics_our, metrics_random = [], [[] for _ in range(len(dim_list))], []

shuffle_idx = np.random.permutation(traj_onehot.shape[0])
X_train, X_test = np.split(traj_onehot[shuffle_idx], 2) # (N, T, L, nstates)
y_train, y_test = np.split(traj[shuffle_idx], 2) # (N, T, L)

predictor = GeneTypePredictor(conf.eqFP611.loci, conf.eqFP611.states, ml='SVM')

L, b, T = conf.eqFP611.loci, conf.eqFP611.states, conf.eqFP611.steps

X_train_ = X_train[:, :-1].reshape(-1, L*b)
y_train_ = y_train[:, 1:].reshape(-1, L)
predictor.fit(X_train_, y_train_)

initial_state = y_test[:, 0]
ml_predictions = predictor.autoregressive_predict(initial_state, T)

metrics = compute_metrics(y_test[:, 1:], ml_predictions, num_classes=conf.eqFP611.states) # N×(T-1)×(accuracy, precision, recall, f1)
metrics_baseline.append(metrics)






predictor = GeneTypePredictor(conf.eqFP611.loci, conf.eqFP611.states, ml='SVM')
random_predictions = predictor.random_predict(initial_state.shape[0], conf.eqFP611.steps)

random_metrics = compute_metrics(y_test[:, 1:], random_predictions, num_classes=conf.eqFP611.states)
metrics_random.append(random_metrics)






for dim_idx, dim in enumerate(dim_list):
    print(f'Processing dim={dim}...')

    log_dir = conf.log_dir + f'd{dim}/'

    L = conf.eqFP611.loci
    b = conf.eqFP611.states
    input_channels = b
    feature_dim = dim
    num_heads = conf.model.num_heads
    num_layers = conf.model.num_layers
    codebook_size = conf.model.K
    pos_encoding_type = conf.model.pos_encoding_type
    use_rotation = conf.model.use_rotation
    code_init = conf.model.code_init
    ae = EvolutionDynamicsModel(input_channels, feature_dim, L, b, num_heads, num_layers, pos_encoding_type=pos_encoding_type, codebook_size=codebook_size, use_rotation=use_rotation, code_init=code_init).to(DEVICE)
    ae.load_state_dict(torch.load(log_dir + f'checkpoints/pretrain_{conf.train.max_epoch}.pth', map_location=DEVICE))


    batch_size = conf.train.batch_size
    for i in range(0, X_train.shape[0]//batch_size+1):
        x = torch.tensor(X_train[i*batch_size:(i+1)*batch_size]).float().to(DEVICE)
        x_ = torch.tensor(X_test[i*batch_size:(i+1)*batch_size]).float().to(DEVICE)
        with torch.no_grad():
            z = ae.encode(x)[0] # (N, T, D)
            z_ = ae.encode(x_)[0]
            z = z.cpu().numpy()
            z_ = z_.cpu().numpy()
            if i == 0:
                z_train = z
                z_test = z_
            else:
                z_train = np.concatenate([z_train, z], axis=0) # (N, T, D)
                z_test = np.concatenate([z_test, z_], axis=0)

    # StandardScaler
    scaler = StandardScaler()
    scaler.fit(z_train.reshape(-1, feature_dim))
    z_train_norm = scaler.transform(z_train.reshape(-1, feature_dim)).reshape(-1, conf.eqFP611.steps, feature_dim)
    z_test_norm = scaler.transform(z_test.reshape(-1, feature_dim)).reshape(-1, conf.eqFP611.steps, feature_dim)

    # SVR
    predictor = MultiOutputRegressor(SVR(kernel='rbf'))
    # # LR
    # predictor = MultiOutputRegressor(LinearRegression())

    z_train_x = z_train_norm[:, :-1].reshape(-1, feature_dim)
    z_train_y = z_train_norm[:, 1:].reshape(-1, feature_dim)
    predictor.fit(z_train_x, z_train_y)

    ml_predictions_z = np.zeros_like(z_test_norm)
    ml_predictions_z[:, 0] = z_test_norm[:, 0]
    for t in range(1, conf.eqFP611.steps):
        ml_predictions_z[:, t] = predictor.predict(ml_predictions_z[:, t-1])

    ml_predictions_z = scaler.inverse_transform(ml_predictions_z.reshape(-1, feature_dim)).reshape(-1, conf.eqFP611.steps, feature_dim)
    z_test_y = scaler.inverse_transform(z_test_norm.reshape(-1, feature_dim)).reshape(-1, conf.eqFP611.steps, feature_dim)

    with torch.no_grad():
        for i in range(0, X_train.shape[0]//batch_size+1):
            z = torch.tensor(ml_predictions_z[i*batch_size:(i+1)*batch_size]).float().to(DEVICE)
            x = ae.decode(z)
            x = x.cpu().numpy()
            if i == 0:
                x_pred = x
            else:
                x_pred = np.concatenate([x_pred, x], axis=0)

    x_pred = np.argmax(x_pred, axis=-1)

    metrics = compute_metrics(y_test[:, 1:], x_pred[:, 1:], num_classes=conf.eqFP611.states) # N×(T-1)×(accuracy, precision, recall, f1)
    metrics_our[dim_idx].append(metrics)
    
    os.makedirs('figs/fig3/', exist_ok=True)
    np.save(f'figs/fig3/eqFP611_z_pred_{dim}.npy', ml_predictions_z)
    np.save(f'figs/fig3/eqFP611_x_pred_{dim}.npy', x_pred)
    np.save(f'figs/fig3/eqFP611_z_gt_{dim}.npy', z_test_y)
    np.save(f'figs/fig3/eqFP611_x_gt.npy', y_test)

np.save('figs/fig3/eqFP611_baseline.npy', metrics_baseline)
np.save('figs/fig3/eqFP611_our.npy', metrics_our)
np.save('figs/fig3/eqFP611_random.npy', metrics_random)


# from sklearn.decomposition import TruncatedSVD
# method = TruncatedSVD(n_components=2)
# z_true_trajs_ = method.fit_transform(z_test.reshape(-1, z_test.shape[-1])).reshape(z_test.shape[0], z_test.shape[1], -1) # (N, T, 2)
# z_pred_trajs_ml_ = method.transform(ml_predictions_z.reshape(-1, ml_predictions_z.shape[-1])).reshape(ml_predictions_z.shape[0], ml_predictions_z.shape[1], -1)

