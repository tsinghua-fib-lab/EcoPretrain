import os
import sys
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scienceplots


plt.style.use(['nature'])
plt.rcParams['text.usetex'] = False
path = '../calibri.ttf'
fm.fontManager.addfont(path)
prop = fm.FontProperties(fname=path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['mathtext.fontset'] = 'dejavusans'


def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def error(message, trigger_exit=True):
    print("\n"+message+"\n")
    if(trigger_exit):
        sys.exit()


def random_matrix(shape, mode, args={}, sparsity=0.0, symmetric=False, triangular=False, diagonal=None, ordered=False, order_power=0, scale_range=None, seed=None):
    if(seed is not None):
        np.random.seed(seed)
    #--------------------------------
    # Generate random values according to one of the following random models:
    #--------------------------------
    if(mode == 'tikhonov_sigmoid'):
        J_0    = args['J_0'] if 'J_0' in args else 0.2
        n_star = args['n_star'] if 'n_star' in args else 10
        delta  = args['delta'] if 'delta' in args else 3
        M = np.zeros(shape=shape)
        for i, j in np.ndindex(M.shape):
            if(i >= j):
                continue
            M[i,j] = np.random.normal( loc=0, scale=J_0*(1/(1 + np.exp((max(i+1, j+1) - n_star)/delta))) )   # +1s because i,j indices start at 0
    else:
        error(f"Error in random_matrix(): generator mode '{mode}' is not recognized.")
    #--------------------------------
    # Apply specified sparsity:
    # num_ = range(len(np.triu_indices(M.shape[0], k=0 if diagonal is not None else 1)[0])) if triangular else M.shape[1]*M.shape[0]
    # zeroed_indices = np.random.choice(, replace=False, size=int(M.shape[1]*M.shape[0]*sparsity))        
    if(triangular):
        active_indices   = np.triu_indices(M.shape[0], k=0 if diagonal is not None and diagonal != 0 else 1)
        zeroed_indices_i = np.random.choice(range(len(active_indices[0])), replace=False, size=int(len(active_indices[0])*sparsity))
        zeroed_indices   = (active_indices[0][zeroed_indices_i], active_indices[1][zeroed_indices_i])
        M[zeroed_indices] = 0
    else:
        zeroed_indices = np.random.choice(M.shape[1]*M.shape[0], replace=False, size=int(M.shape[1]*M.shape[0]*sparsity))
        M[np.unravel_index(zeroed_indices, M.shape)] = 0 
    #--------------------------------
    # Make symmetric, if applicable:
    if(symmetric):
        if(shape[0] != shape[1]):
            error(f"Error in random_matrix(): shape {shape} is not square and cannot be made symmetric.")
        M = np.tril(M) + np.triu(M.T, 1)
    #--------------------------------
    # Make triangular, if applicable:
    if(triangular):
        if(shape[0] != shape[1]):
            error(f"Error in random_matrix(): shape {shape} is not square and cannot be made triangular.")
        M *= 1 - np.tri(*M.shape, k=-1, dtype=np.bool_)
    #--------------------------------
    # Set diagonal, if applicable:
    if(diagonal is not None):
        np.fill_diagonal(M, diagonal)
    #--------------------------------
    # Make ordered, if applicable:
    if(ordered):
        vals = np.array(sorted(M[M!=0], key=abs, reverse=True))

        def weighted_shuffle(items, weights):
            order = sorted(range(len(items)), key=lambda i: np.random.uniform(low=0, high=1) ** (1.0 / weights[i]))
            return [items[i] for i in order]

        vals = weighted_shuffle(items=vals, weights=((vals**order_power)/np.sum(vals**order_power))[::-1])

        c = 0
        for j in range(M.shape[1]):
            for i in range(M.shape[0]):
                if(M[i,j] != 0):
                    M[i,j] = vals[c]
                    c += 1


    #--------------------------------
    # Scale values to desired range, if applicable:
    if(scale_range is not None):
        M[M != 0] = np.interp(M[M != 0], (M[M != 0].min(), M[M != 0].max()), (scale_range[0], scale_range[1]))
    #--------------------------------
    return M


def plot_abundance_dynamics(abundance_history, save_path=f'./'):
    abundance_transposed = abundance_history.T
    time_steps = np.arange(abundance_transposed.shape[1])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.stackplot(time_steps, abundance_transposed, labels=[f'Species {i}' for i in range(abundance_transposed.shape[0])])

    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Abundance", fontsize=14)
    ax.set_title("Eco-evolutionary dynamics of species abundance", fontsize=16)
    
    ax.set_yscale('log')

    # ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False)

    plt.tight_layout()
    plt.savefig(save_path+'abundance_dynamics.png', dpi=300)
    
    
def plot_relative_abundance_dynamics(abundance_history, save_path=f'./'):
    relative_abundance = abundance_history / abundance_history.sum(axis=1, keepdims=True)
    relative_abundance_transposed = relative_abundance.T
    time_steps = np.arange(relative_abundance_transposed.shape[1])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.stackplot(time_steps, relative_abundance_transposed, labels=[f'Species {i}' for i in range(relative_abundance_transposed.shape[0])])

    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Relative Abundance", fontsize=14)
    ax.set_title("Eco-evolutionary dynamics of relative species abundance", fontsize=16)
    
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path+'relative_abundance_dynamics.png', dpi=300)
    

def plot_trait_interactions(trait_inter_matrix, save_path=f'./'):
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(trait_inter_matrix, cmap='bwr', interpolation='nearest', vmin=-0.5, vmax=0.5)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Trait interactions", fontsize=14)
    ax.set_xlabel("Trait Index", fontsize=12)
    ax.set_ylabel("Trait Index", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path+'trait_interactions.png', dpi=300)
    
    
    

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def compute_metrics_bi(x, x_hat, mask=True):
    N, T, num_species, L = x.shape 
    C = 4 
    metrics = np.zeros((N, T, C))

    for i in range(N):
        for j in range(T):
            y_true_all = x[i, j]
            y_pred_all = x_hat[i, j]

            sum_metrics = np.zeros(C)
            valid_count = 0 

            for k in range(num_species):
                if np.any(y_true_all[k] == -1) and mask:
                    continue

                y_true = y_true_all[k]
                y_pred = y_pred_all[k]

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

                sum_metrics += np.array([accuracy, precision, recall, f1])
                valid_count += 1

            if valid_count > 0:
                metrics[i, j] = sum_metrics / valid_count

    return metrics