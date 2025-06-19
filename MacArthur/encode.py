import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import warnings; warnings.filterwarnings("ignore")
from omegaconf import OmegaConf
import numpy as np

from utilis import compute_metrics_bi
from simulator import getSimTraj
from model import EvolutionDynamicsModel
from dataset import Dataset_



conf = OmegaConf.load('config.yaml')
conf.log_dir += f'L{conf.system.num_resources}/seed{conf.seed}/'
conf.data_dir += f'L{conf.system.num_resources}/'

batch_size = conf.train.batch_size
train_dataset = Dataset_(conf)
train_loader = train_dataset.getLoader(batch_size, shuffle=True)

try:
    test_trajs = np.load(conf.data_dir + 'test_trajs.npy')
except:
    test_trajs = getSimTraj(conf, traj_num=int(0.1*conf.system.num_traj), test=True)
    np.save(conf.data_dir + 'test_trajs.npy', test_trajs)

test_trajs = torch.tensor(test_trajs, dtype=torch.float32).to(conf.device)



metrics_list = []
for feature_dim in [1, 2, 3, 4, 5, 6, 7, 8]:
    T = conf.system.max_mutations
    L = conf.system.num_resources
    N = 2**L
    D = feature_dim
    h = conf.model.hidden_dim
    K = conf.model.K

    model = EvolutionDynamicsModel(
        input_dim=L, 
        channel_dim=N, 
        embed_dim=D, 
        temporal_dim=h, 
        output_dim=L, 
        num_heads=conf.model.num_heads, 
        num_layers=conf.model.num_layers, 
        pos_encoding_type=conf.model.pos_encoding_type, 
        codebook_size=K
    ).to(conf.device)

    optimizer = optim.Adam(model.parameters(), lr=conf.train.lr)
    scheduler = ExponentialLR(optimizer, conf.train.lr_decay)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-1)

    try:
        model.load_state_dict(torch.load(conf.log_dir + f'dim/checkpoints/pretrain_{conf.train.max_epoch}_d{feature_dim}.pth', map_location=conf.device))
    except RuntimeError:
        pass
    except:
        loss_ema_list = []
        for epoch in range(conf.train.max_epoch):
            model.train()
            loss_ema = None
            for i, (x) in enumerate(train_loader):
                # x: (batch_size, T, 2^L, L)            
                optimizer.zero_grad()
                z, zz, z_hat, x_logits, _, _, _ = model(x, vq=False)
                
                # Reconstruction loss
                label = x.long()
                loss_recons = ce(x_logits.view(-1, 3), label.view(-1))
                
                loss = loss_recons
                loss_ema = loss_recons.item() if loss_ema is None else 0.95*loss_ema + 0.05*loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                if i % 10 == 0:
                    print(f'Epoch {epoch:03d}, batch {i:03d} | recons: {loss_recons.item():.5f}')

            scheduler.step()
            if (1+epoch) % conf.train.max_epoch == 0:
                os.makedirs(conf.log_dir + 'dim/checkpoints/', exist_ok=True)
                torch.save(model.state_dict(), conf.log_dir + f'dim/checkpoints/pretrain_{epoch+1}_d{feature_dim}.pth')
            
            loss_ema_list.append(loss_ema)


    batchsize = conf.train.batch_size
    z_hat, test_trajs_hat = [], []
    for i in range(int(np.ceil(0.1*conf.system.num_traj/batchsize))):
        x = test_trajs[i*batchsize:(i+1)*batchsize]
        _, _, z_hat_, x_logits, _, _, _ = model(x, vq=False)
        z_hat.append(z_hat_.reshape(-1, 2**conf.system.num_resources, feature_dim).cpu().detach().numpy())
        test_trajs_hat.append(x_logits.argmax(dim=-1).cpu().numpy().astype(int))
    torch.cuda.empty_cache()

    z_hat = np.concatenate(z_hat, axis=0)
    test_trajs_hat = np.concatenate(test_trajs_hat, axis=0)


    metrics = compute_metrics_bi(test_trajs.cpu().numpy().astype(int), test_trajs_hat, mask=True) # N×T×(accuracy, precision, recall, f1)
    metrics_list.append(metrics)
    print(f'Feature dim: {feature_dim}, Accuracy: {metrics[..., 0].mean():.5f}, Precision: {metrics[..., 1].mean():.5f}, Recall: {metrics[..., 2].mean():.5f}, F1: {metrics[..., 3].mean():.5f}')

# np.save(conf.log_dir + f'dim/metrics_list.npy', metrics_list)