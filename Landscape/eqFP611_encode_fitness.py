import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error


def evaluate_model(genotype_space, fitness_space, available_ratio, degree=1, n_repeats=10, vis=False, ml='LR'):
    results = []

    for i in range(n_repeats):
        np.random.seed(i)
        
        n_train = int(len(genotype_space) * available_ratio)
        indices = np.arange(len(genotype_space))
        np.random.shuffle(indices)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        X_train, X_test = genotype_space[train_indices], genotype_space[test_indices]
        y_train, y_test = fitness_space[train_indices], fitness_space[test_indices]

        poly = PolynomialFeatures(degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        if ml == 'LR':
            model = LinearRegression()
        elif ml == 'SVR':
            model = SVR(kernel='rbf')
        elif ml == 'RF':
            model = RandomForestRegressor(random_state=42)
        else:
            raise ValueError('Invalid ML model specified. Please choose from "LR" or "SVR".')
        model.fit(X_train_poly, y_train)

        y_pred = model.predict(X_test_poly)

        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        nmse = np.mean((y_test - y_pred) ** 2) / np.mean((y_test - np.mean(y_test)) ** 2)
        
        if r2 > 0:
            results.append([r2, mape, nmse])

    results = np.array(results)
    
    return results, (y_test, y_pred)


def train(dim, device):
    
    from omegaconf import OmegaConf

    conf = OmegaConf.load('config/eqFP611.yaml')
    conf.device = device
    conf.model.feature_dim = dim
    conf.log_dir = conf.log_dir + f'd{conf.model.feature_dim}/'

    from simulate.eqFP611 import Dataset_eqFP611
    batch_size = conf.train.batch_size
    train_dataset = Dataset_eqFP611(conf)
    train_loader = train_dataset.getLoader(batch_size, shuffle=True)

    from model.model import EvolutionDynamicsModel

    L = conf.eqFP611.loci
    b = conf.eqFP611.states
    input_channels = b
    feature_dim = conf.model.feature_dim
    num_heads = conf.model.num_heads
    num_layers = conf.model.num_layers
    codebook_size = conf.model.K
    pos_encoding_type = conf.model.pos_encoding_type
    use_rotation = conf.model.use_rotation
    code_init = conf.model.code_init
    model = EvolutionDynamicsModel(input_channels, feature_dim, L, b, num_heads, num_layers, pos_encoding_type=pos_encoding_type, codebook_size=codebook_size, use_rotation=use_rotation, code_init=code_init).to(conf.device)


    import os
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import ExponentialLR
    import warnings; warnings.filterwarnings("ignore")

    optimizer = optim.Adam(model.parameters(), lr=conf.train.lr)
    scheduler = ExponentialLR(optimizer, conf.train.lr_decay)
    criterion = torch.nn.CrossEntropyLoss()

    try:
        model.load_state_dict(torch.load(conf.log_dir + f'checkpoints/pretrain_{conf.train.max_epoch}.pth', map_location=device))
    except:
        loss_ema_list = []
        for epoch in range(conf.train.max_epoch):
            model.train()
            loss_ema = None
            for i, (x, x_label) in enumerate(train_loader):
                # x: (batch_size, T, L, b)
                # x_label: (batch_size, T, L)
                x, x_label = x.to(conf.device), x_label.long().to(conf.device)
                
                optimizer.zero_grad()
                x_logit, z_e_x, z_q_x, indices = model(x, vq=False)
                
                # Reconstruction loss
                loss_recons = criterion(x_logit.view(-1, b), x_label.view(-1))
                
                loss = loss_recons
                loss_ema = loss_recons.item() if loss_ema is None else 0.95*loss_ema + 0.05*loss.item()
                
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f'Epoch {epoch:03d}, batch {i:03d} | recons: {loss_recons.item():.5f}')
            scheduler.step()
            loss_ema_list.append(loss_ema)
        
        os.makedirs(conf.log_dir + 'checkpoints/', exist_ok=True)
        torch.save(model.state_dict(), conf.log_dir + f'checkpoints/pretrain_{epoch+1}.pth')

    from simulate.eqFP611 import getSimTraj
    import numpy as np

    trajectorys = getSimTraj(conf, vis=False, save=False, traj_num=100, force=True)
    trajectorys = np.array(trajectorys)  # (N, T, L)
    trajectorys_onehot = np.eye(conf.eqFP611.states)[trajectorys.astype(int)]  # (N, T, L, nstates)
    trajectorys_onehot = torch.tensor(trajectorys_onehot).float().to(conf.device)

    import pickle
    with open('data/eqFP611/system.pkl', 'rb') as f:
        system = pickle.load(f)
    trajectorys_fitness = np.array([system.get_fitness(genotype) for genotype in trajectorys.reshape(-1, conf.eqFP611.loci)]).reshape(trajectorys.shape[:2])

    model.eval()
    # x_logit, z_e_x, z_q_x, indices = model(trajectorys_onehot, vq=True)
    x_logit, z_e_trajectorys, z_q_trajectorys, indices = model(trajectorys_onehot, vq=False)
    trajectorys_hat = x_logit.argmax(-1).cpu().numpy() # (N, T, L)

    from utilis import compute_metrics
    metrics = compute_metrics(trajectorys, trajectorys_hat, num_classes=conf.eqFP611.states) # N×T×(accuracy, precision, recall, f1)

    from umap import UMAP
    import matplotlib.pyplot as plt
    import numpy as np

    # Umap
    umap = UMAP(n_components=2)
    z_e_x_pca = umap.fit_transform(z_e_trajectorys.view(-1, z_e_trajectorys.size(-1)).detach().cpu().numpy()) # (N*T, 2)
    z_q_x_pca = umap.transform(z_q_trajectorys.view(-1, z_e_trajectorys.size(-1)).detach().cpu().numpy()) # (N*T, 2)
    z_e_x_pca, z_q_x_pca = z_e_x_pca.reshape(z_e_trajectorys.size(0), z_e_trajectorys.size(1), -1), z_q_x_pca.reshape(z_q_trajectorys.size(0), z_q_trajectorys.size(1), -1) # (N, T, 2)

    # Plotting in 2D
    fig = plt.figure(figsize=(16, 6.5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # 2D Scatter plots with color maps
    for i in range(trajectorys.shape[0]):
        sc1 = ax1.scatter(z_e_x_pca[i, :, 0], z_e_x_pca[i, :, 1],
                        c=trajectorys_fitness[i], cmap='viridis', alpha=0.5)
        # sc2 = ax2.scatter(z_q_x_pca[i, :, 0], z_q_x_pca[i, :, 1],
        #                   c=trajectorys_fitness[i], cmap='viridis', alpha=0.5)
        sc2 = ax2.scatter(z_e_x_pca[i, :, 0], z_e_x_pca[i, :, 1],
                        c=np.arange(z_e_x_pca.shape[1]), cmap='viridis', alpha=0.5)
        
    # Adding colorbars
    cbar1 = plt.colorbar(sc1, ax=ax1, pad=0.1)
    cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.1)
    cbar1.set_label('Fitness')
    cbar2.set_label('Time step')

    # Titles and layout
    ax1.set_title('Continuous Space')
    ax2.set_title('Code Space')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    plt.tight_layout()  
    plt.savefig(conf.log_dir + 'latent_space_2D.png', dpi=300)


    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    N, T, C = metrics.shape
    metric_names = ['accuracy', 'precision', 'recall', 'f1']

    data = []
    for t in range(T):
        for c in range(C):
            values = metrics[:, t, c]  
            data.extend([[t, metric_names[c], val] for val in values])

    df = pd.DataFrame(data, columns=["Time Step", "Metric", "Value"])

    plt.figure(figsize=(12, 4))
    sns.barplot(x="Time Step", y="Value", hue="Metric", data=df, palette="Set2", ci="sd")
    plt.title("Mean Distribution of Selected Metrics across Time Steps")
    plt.xlabel("Time Steps")
    plt.ylabel("Mean Metric Value")
    plt.legend(title="Metric", bbox_to_anchor=(1.005, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(conf.log_dir + 'bar_plot.png', dpi=300)


    from model.model import GenotypeEncoder

    L = conf.eqFP611.loci
    b = conf.eqFP611.states
    feature_dim = conf.model.feature_dim
    model2 = GenotypeEncoder(L, b, feature_dim).to(conf.device)

    optimizer = optim.Adam(model2.parameters(), lr=1e-3)
    scheduler = ExponentialLR(optimizer, conf.train.lr_decay)
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.CrossEntropyLoss()

    loss_ema_list = []
    for epoch in range(200):
        model2.train()
        loss_ema = None
        for i, (x, x_label) in enumerate(train_loader):
            # x: (batch_size, T, L, b)
            # x_label: (batch_size, T, L)
            x = x.to(conf.device)
            x_label = x_label.long().to(conf.device)
            
            optimizer.zero_grad()
            z, _ = model2(x)
            
            with torch.no_grad():
                z_e_x, z_q_x = model.encode(x)
            
            # Reconstruction loss
            loss_recons = criterion(z, z_e_x)        
            loss = loss_recons
            loss_ema = loss_recons.item() if loss_ema is None else 0.95*loss_ema + 0.05*loss.item()
            
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {epoch:03d}, batch {i:03d} | recons: {loss_recons.item():.5f}')
        scheduler.step()
        if (1+epoch) % 10 == 0:
            os.makedirs(conf.log_dir + 'checkpoints/', exist_ok=True)
            torch.save(model2.state_dict(), conf.log_dir + f'checkpoints/distill_{epoch+1}.pth')
        
        loss_ema_list.append(loss_ema)

    model2.eval()
    z_e_trajectorys_distill, _ = model2(trajectorys_onehot)

    # Umap
    z_e_x_pca_distill = umap.transform(z_e_trajectorys_distill.view(-1, z_e_trajectorys_distill.size(-1)).detach().cpu().numpy()) # (N*T, 2)
    z_e_x_pca_distill = z_e_x_pca_distill.reshape(z_e_trajectorys_distill.size(0), z_e_trajectorys_distill.size(1), -1) # (N, T, 2)

    # Plotting in 2D
    fig = plt.figure(figsize=(16, 6.5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # 2D Scatter plots with color maps
    for i in range(trajectorys.shape[0]):
        sc1 = ax1.scatter(z_e_x_pca_distill[i, :, 0], z_e_x_pca_distill[i, :, 1],
                        c=trajectorys_fitness[i], cmap='viridis', alpha=0.5)
        sc2 = ax2.scatter(z_e_x_pca_distill[i, :, 0], z_e_x_pca_distill[i, :, 1],
                        c=np.arange(z_e_x_pca_distill.shape[1]), cmap='viridis', alpha=0.5)
        
    # Adding colorbars
    cbar1 = plt.colorbar(sc1, ax=ax1, pad=0.1)
    cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.1)
    cbar1.set_label('Fitness')
    cbar2.set_label('Time step')

    # Titles and layout
    ax1.set_title('Continuous Space')
    ax2.set_title('Code Space')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    plt.tight_layout()  
    plt.savefig(conf.log_dir + 'latent_space_2D_distill.png', dpi=300)
    


def test(dim, device):

    from omegaconf import OmegaConf

    conf = OmegaConf.load('config/eqFP611.yaml')
    conf.device = device
    conf.model.feature_dim = dim
    conf.log_dir = conf.log_dir + f'd{conf.model.feature_dim}/'

    import pickle
    import itertools
    import numpy as np

    # load system
    with open('data/eqFP611/system.pkl', 'rb') as f:
        system = pickle.load(f)
    
    def generate_genotype_space(L, b):
        genotype_space = list(itertools.product(range(b), repeat=L))
        return np.array(genotype_space)

    # Genotype space
    loci, states = conf.eqFP611.loci, conf.eqFP611.states
    genotype_space = generate_genotype_space(loci, states)
    # onehot
    genotype_space_onehot = np.eye(states)[genotype_space]
    genotype_space_onehot = genotype_space_onehot.reshape(-1, loci * states)

    # Fitness
    fitness_space = np.array([system.get_fitness(genotype) for genotype in genotype_space])

    import torch

    from model.model import GenotypeEncoder

    L = conf.eqFP611.loci
    b = conf.eqFP611.states
    feature_dim = conf.model.feature_dim
    encoder = GenotypeEncoder(L, b, feature_dim).to(conf.device)
    encoder.load_state_dict(torch.load(conf.log_dir + f'checkpoints/distill_200.pth', map_location=device))

    genotype_space_onehot_ = np.eye(b)[genotype_space]
    genotype_space_onehot_ = torch.tensor(genotype_space_onehot_, dtype=torch.float32).to(conf.device)
    encoder.eval()
    with torch.no_grad():
        z_latent = encoder(genotype_space_onehot_.unsqueeze(0))[0].squeeze(0)
    genotype_space_encoded = z_latent.cpu().numpy()

    genotype_space_encoded_norm = (genotype_space_encoded - genotype_space_encoded.mean(axis=0)) / genotype_space_encoded.std(axis=0)

    for available_ratio in [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]:
        for ml in ['LR', 'SVR']:
            results, (y_test, y_pred) = evaluate_model(genotype_space_encoded_norm, fitness_space, available_ratio=available_ratio, degree=1, n_repeats=50, vis=False, ml=ml)
            
            np.save(conf.log_dir + f'result_{ml}_{available_ratio}.npy', results)
            np.save(conf.log_dir + f'y_test_{ml}_{available_ratio}.npy', y_test)
            np.save(conf.log_dir + f'y_pred_{ml}_{available_ratio}.npy', y_pred)

def test_origin():

    from omegaconf import OmegaConf
    conf = OmegaConf.load('config/eqFP611.yaml')

    import pickle
    import itertools
    import numpy as np

    # load system
    with open('data/eqFP611/system.pkl', 'rb') as f:
        system = pickle.load(f)

    def generate_genotype_space(L, b):
        genotype_space = list(itertools.product(range(b), repeat=L))
        return np.array(genotype_space)

    # Genotype space
    loci, states = conf.eqFP611.loci, conf.eqFP611.states
    genotype_space = generate_genotype_space(loci, states)
    # onehot
    genotype_space_onehot = np.eye(states)[genotype_space]
    genotype_space_onehot = genotype_space_onehot.reshape(-1, loci * states)

    # Fitness
    fitness_space = np.array([system.get_fitness(genotype) for genotype in genotype_space])

    for available_ratio in [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]:
        for ml in ['LR', 'SVR']:
            results, (y_test, y_pred) = evaluate_model(genotype_space_onehot, fitness_space, available_ratio=available_ratio, degree=1, n_repeats=50, vis=False, ml=ml)
            
            np.save(conf.log_dir + f'result_origin_{ml}_{available_ratio}.npy', results)
            np.save(conf.log_dir + f'y_test_origin_{ml}_{available_ratio}.npy', y_test)
            np.save(conf.log_dir + f'y_pred_origin_{ml}_{available_ratio}.npy', y_pred)


def vis(max_dim=60): 
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import os
    import pandas as pd

    from omegaconf import OmegaConf
    conf = OmegaConf.load('config/eqFP611.yaml')
    
    available_ratios = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    dims = range(4, max_dim+1, 4)
    ml_list = ['LR', 'SVR']

    cmap = matplotlib.colormaps.get_cmap('viridis')
    
    for ml in ml_list:

        data = []
        for dim in dims:
            for available_ratio in available_ratios:
                file_path = os.path.join(conf.log_dir, f'd{dim}/result_{ml}_{available_ratio}.npy')
                if os.path.exists(file_path):
                    results = np.load(file_path, allow_pickle=True).item()
                    r2_mean, r2_std = results['R2_mean'], results['R2_std']
                    mape_mean, mape_std = results['MAPE_mean'], results['MAPE_std']
                    data.append([dim, available_ratio, r2_mean, r2_std, mape_mean, mape_std])

        df = pd.DataFrame(data, columns=['Dim', 'Available Ratio', 'R2 Mean', 'R2 Std', 'MAPE Mean', 'MAPE Std'])

        reference_data = []
        for available_ratio in available_ratios:
            reference_file_path = os.path.join(conf.log_dir, f'result_origin_{ml}_{available_ratio}.npy')
            if os.path.exists(reference_file_path):
                reference_results = np.load(reference_file_path, allow_pickle=True).item()
                reference_r2_mean = reference_results['R2_mean']
                reference_r2_std = reference_results['R2_std']
                reference_mape_mean = reference_results['MAPE_mean']
                reference_mape_std = reference_results['MAPE_std']
                
                if 0 <= reference_r2_mean <= 1:
                    reference_data.append([available_ratio, reference_r2_mean, reference_r2_std, reference_mape_mean, reference_mape_std])

        reference_df = pd.DataFrame(reference_data, columns=['Available Ratio', 'R2 Mean', 'R2 Std', 'MAPE Mean', 'MAPE Std'])

        plt.figure(figsize=(12, 10))

        plt.subplot(2, 1, 1)
        norm = plt.Normalize(vmin=min(dims), vmax=max(dims))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        for i, dim in enumerate(dims):
            df_dim = df[df['Dim'] == dim]
            color = cmap(norm(dim)) 
            plt.errorbar(df_dim['Available Ratio'], df_dim['R2 Mean'], yerr=df_dim['R2 Std'],
                        fmt='-o', color=color, capsize=3)

        plt.errorbar(reference_df['Available Ratio'], reference_df['R2 Mean'], yerr=reference_df['R2 Std'],
                    fmt='-o', color='red', linewidth=2, label="Not encoding")

        plt.title(f"R2 Across Different Latent Dimensions and Available Ratios ({ml})")
        plt.xlabel("Available Ratio")
        plt.ylabel("R2 Mean")

        sm.set_array([]) 
        cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical')
        cbar.set_label('Latent Dimension (Dim)')

        plt.subplot(2, 1, 2)
        for i, dim in enumerate(dims):
            df_dim = df[df['Dim'] == dim]
            color = cmap(norm(dim)) 
            plt.errorbar(df_dim['Available Ratio'], df_dim['MAPE Mean'], yerr=df_dim['MAPE Std'],
                        fmt='-o', color=color, capsize=3)
        
        plt.errorbar(reference_df['Available Ratio'], reference_df['MAPE Mean'], yerr=reference_df['MAPE Std'],
                    fmt='-o', color='red', linewidth=2, label="Not encoding")

        plt.title(f"MAPE Across Different Latent Dimensions and Available Ratios ({ml})")
        plt.xlabel("Available Ratio")
        plt.ylabel("MAPE Mean")

        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical')
        cbar.set_label('Latent Dimension (Dim)')

        plt.tight_layout()
        plt.savefig(conf.log_dir + f'result_{ml}.png', dpi=300)



if __name__ == '__main__':
    
    device = 'cuda:0'
    
    # Figure 1 (num_heads=1)
    for dim in range(2, 20+1, 2):
        train(dim, device)
    
    # Figure 2 (num_heads=4)
    for dim in range(4, 24+1, 4):
        train(dim, device)
        test(dim, device)
    test_origin()
    vis(max_dim=24)