import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings; warnings.filterwarnings("ignore")

from simulator import invasion
from model import GenotypeEncoder
from utilis import seed_everything


conf = OmegaConf.load('config.yaml')
samplenum_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
size_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
# samplenum_list = [10000]
# size_list = range(1, 26)


for sample_num in samplenum_list:
    for size in size_list:

        conf = OmegaConf.load('config.yaml')
        conf.log_dir += f'L{conf.system.num_resources}/seed{conf.seed}/'
        conf.data_dir += f'L{conf.system.num_resources}/'
        os.makedirs(conf.log_dir+'invasion', exist_ok=True)

        seed_everything(conf.seed)

        num_resources=conf.system.num_resources
        from utilis import random_matrix
        trait_inter_matrix = random_matrix(
            (num_resources, num_resources),
            'tikhonov_sigmoid',
            args={'J_0': 0.5, 'n_star': 4, 'delta': 1},
            triangular=True,
            diagonal=0,
            seed=100
        )

        try:
            X, y = np.load(conf.log_dir + f'invasion/dataset_size{size}_N{sample_num}.npz')['X'], np.load(conf.log_dir + f'invasion/dataset_size{size}_N{sample_num}.npz')['y']
            X_flatten = X.reshape(-1, (size+1)*num_resources)
        except:
            gene_space = np.array([[int(i) for i in format(j, '0'+str(num_resources)+'b')] for j in range(2**num_resources)], dtype=np.float32) # (2^L-1, L)
            X, y = np.zeros((sample_num, size+1, num_resources)), np.zeros((sample_num,))
            for i in tqdm(range(sample_num)):
                idxs = np.random.choice(gene_space.shape[0], size+1, replace=False)
                X[i] = gene_space[idxs] 
                y[i] = invasion(X[i, :-1], X[i, -1], num_resources, trait_inter_matrix, total_t=200000., dt=200.)
            X_flatten = X.reshape(-1, (size+1)*num_resources)
            y = y.astype(np.int32)
            np.savez(conf.log_dir + f'invasion/dataset_size{size}_N{sample_num}.npz', X=X, y=y)

        
            plt.hist(y, bins=2, rwidth=0.8)
            plt.xticks([0.25, 0.75], ['extinct', 'persist'])
            plt.savefig(conf.log_dir+f'invasion/yhist_size{size}_N{sample_num}', dpi=300)
            plt.close()

        
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=10, shuffle=True, random_state=conf.seed)
        
        
        # for ml in ['SVM', 'LR', 'GBDT']:
        for ml in ['SVM']:
            
            if os.path.exists(conf.log_dir + f'invasion/metrics_size{size}_N{sample_num}_{ml}.npy'):
                continue
            else:
                precision_, recall_, f1_ = [], [], []            
            
            
            for train_index, test_index in tqdm(kf.split(X_flatten)):
                if ml == 'SVM':
                    predictor = SVC(kernel='rbf')
                elif ml == 'LR':
                    predictor = LogisticRegression(max_iter=1000)
                elif ml == 'GBDT':
                    predictor = GradientBoostingClassifier(n_estimators=100)
                
                X_train, X_test = X_flatten[train_index], X_flatten[test_index]
                y_train, y_test_ = y[train_index], y[test_index]
                
                predictor.fit(X_train, y_train)
                y_pred_ = predictor.predict(X_test)
                precision_.append(precision_score(y_test_, y_pred_, average='binary'))
                recall_.append(recall_score(y_test_, y_pred_, average='binary'))
                f1_.append(f1_score(y_test_, y_pred_, average='binary'))
                
                with open(conf.log_dir + f'invasion/{ml}_size{size}_N{sample_num}_no_embed.pkl', 'wb') as f:
                    pickle.dump(predictor, f)
            
            
            T = conf.system.max_mutations
            L = conf.system.num_resources
            N = 2**L
            D = conf.model.feature_dim
            h = conf.model.hidden_dim
            K = conf.model.K
            model2 = GenotypeEncoder(L, conf.model.feature_dim).to(conf.device)
            model2.load_state_dict(torch.load(conf.log_dir + f'checkpoints/distill_{500}.pth', map_location=conf.device))

            X_tensor = torch.tensor(X).float().to(conf.device) # (sample_num, size+1, L)
            with torch.no_grad():
                model2.eval()
                z = model2(X_tensor.reshape(-1, L)).reshape(sample_num, size+1, D).reshape(sample_num, (size+1)*D).cpu().numpy()

            z_scaler = StandardScaler()
            z = z_scaler.fit_transform(z)
            np.save(conf.log_dir + f'invasion/scaler_size{size}_N{sample_num}.npy', [z_scaler.mean_, z_scaler.var_, z_scaler.scale_])

            mape, r2 = [], []
            precision, recall, f1 = [], [], []
            for train_index, test_index in tqdm(kf.split(z)):
                if ml == 'SVM':
                    predictor2 = SVC(kernel='rbf')
                elif ml == 'LR':
                    predictor2 = LogisticRegression(max_iter=1000)
                elif ml == 'GBDT':
                    predictor2 = GradientBoostingClassifier(n_estimators=100)
                
                X_train, X_test = z[train_index], z[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                predictor2.fit(X_train, y_train)
                y_pred = predictor2.predict(X_test)
                precision.append(precision_score(y_test, y_pred, average='binary'))
                recall.append(recall_score(y_test, y_pred, average='binary'))
                f1.append(f1_score(y_test, y_pred, average='binary'))
                
                with open(conf.log_dir + f'invasion/{ml}_size{size}_N{sample_num}.pkl', 'wb') as f:
                    pickle.dump(predictor2, f)
            
            
            plt.figure(figsize=(6, 1.7))
            plt.subplot(1, 3, 1)
            plt.boxplot([precision_, precision], tick_labels=['Genotype', 'Encoded'])
            plt.ylabel('Precision')
            plt.ylim(-0.1, 1.1)
            plt.subplot(1, 3, 2)
            plt.boxplot([recall_, recall], tick_labels=['Genotype', 'Encoded'])
            plt.ylabel('Recall')
            plt.ylim(-0.1, 1.1)
            plt.subplot(1, 3, 3)
            plt.boxplot([f1_, f1], tick_labels=['Genotype', 'Encoded'])
            plt.ylabel('F1')
            plt.ylim(-0.1, 1.1)
            plt.tight_layout()
            plt.savefig(conf.log_dir + f'invasion/metrics_size{size}_N{sample_num}_{ml}.png', dpi=300)
            plt.close()
            
            np.save(conf.log_dir + f'invasion/metrics_size{size}_N{sample_num}_{ml}.npy', [precision_, recall_, f1_, precision, recall, f1])
            
            print(f'Finish size{size}_N{sample_num}_{ml}')