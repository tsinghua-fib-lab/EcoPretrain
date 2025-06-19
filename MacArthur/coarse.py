import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from simulator import evolution_test
from utilis import seed_everything

conf = OmegaConf.load('config.yaml')
seed_everything(conf.seed)

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from model import EvolutionDynamicsModel

T = conf.system.max_mutations
D = conf.model.feature_dim
h = conf.model.hidden_dim

tmp = conf.data_dir
# for L in [7, 8, 9]:
for L in [8]:
    N = 2**L
    # conf.train.max_epoch = 30
    for K in [8, 16, 32, 64]:
        log_dir = conf.log_dir + f'L{L}/seed{conf.seed}/'
        conf.data_dir = tmp + f'L{L}/'
        conf.model.num_resources = L
        conf.model.K = K
        os.makedirs(log_dir + 'coarsen', exist_ok=True)
        
        model = EvolutionDynamicsModel(
            input_dim=L, 
            channel_dim=N, 
            embed_dim=D, 
            temporal_dim=h, 
            output_dim=L, 
            num_heads=conf.model.num_heads, 
            num_layers=conf.model.num_layers, 
            pos_encoding_type=conf.model.pos_encoding_type, 
            codebook_size=16 if L==7 else 64
        ).to(conf.device)
        model.load_state_dict(torch.load(log_dir + f'checkpoints/pretrain_{conf.train.max_epoch}.pth', map_location=conf.device))
        
        from dataset import Dataset_

        batch_size = conf.train.batch_size
        train_dataset = Dataset_(conf)
        train_loader = train_dataset.getLoader(batch_size, shuffle=True)
            
        model.init_codebook(K, D)
        try:
            model.load_state_dict(torch.load(log_dir + f'checkpoints/discrete_K{K}_{conf.train.max_epoch}.pth', map_location=conf.device))
        except:
            model.re_init_codebook(train_loader)
            
            optimizer = optim.Adam(model.parameters(), lr=conf.train.lr)
            scheduler = ExponentialLR(optimizer, conf.train.lr_decay)
            ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
            
            for param in model.population_attention.parameters():
                param.requires_grad = False
            for param in model.temporal_transformer.parameters():
                param.requires_grad = False
            
            loss_ema_list = []
            for epoch in range(conf.train.max_epoch):
                model.train()
                loss_ema = None
                for i, x in enumerate(train_loader):
                    # x: (batch_size, T, 2^L, L)
                    optimizer.zero_grad()
                    z, zz, z_hat, x_logits, sg_z_e_x, z_q_x, indices = model(x, vq=True)
                    
                    # Reconstruction loss
                    label = x.long()
                    loss_recons = ce(x_logits.view(-1, 3), label.view(-1))
                    # Vector quantization objective
                    loss_vq = ((z_q_x-z_hat.detach()).pow(2).sum(-1)).mean()
                    
                    loss = loss_recons + loss_vq
                    loss_ema = loss_recons.item() if loss_ema is None else 0.95*loss_ema + 0.05*loss.item()
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    if i % 10 == 0:                
                        K = model.codebook.embedding.weight.size(0)
                        count = torch.zeros(K, device=conf.device)
                        for j in range(K):
                            count[j] = (indices == j).sum()
                        hit_num = (count > 0).sum()
                        hit_rate = hit_num.item() / K
                        print(f'Epoch {epoch:03d}, batch {i:03d} | recons: {loss_recons.item():.5f}, vq: {loss_vq.item():.5f}, hit: {hit_rate*100:.2f}%')
                scheduler.step()
                if (1+epoch) % conf.train.max_epoch == 0:
                    os.makedirs(log_dir + 'checkpoints/', exist_ok=True)
                    torch.save(model.state_dict(), log_dir + f'checkpoints/discrete_K{K}_{epoch+1}.pth')
                
                loss_ema_list.append(loss_ema)

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
        model.load_state_dict(torch.load(log_dir + f'checkpoints/discrete_K{K}_{conf.train.max_epoch}.pth', map_location=conf.device))

        from model import GenotypeEncoder

        model2 = GenotypeEncoder(L, conf.model.feature_dim).to(conf.device)
        model2.load_state_dict(torch.load(log_dir + f'checkpoints/distill_{500}.pth', map_location=conf.device))

        gene_space = torch.tensor([[int(i) for i in format(j, '0'+str(L)+'b')] for j in range(2**L)], dtype=torch.float32).to(conf.device) # (2^L-1, L)
        model.eval()
        model2.eval()

        embedding_space = model2(gene_space)
        _, indices = model.coarse(embedding_space)

        strains_pool = []
        gene_space = gene_space.cpu().numpy().astype(int)
        for i in range(K):
            strains = []
            for j in range(2**L):
                if indices[j] == i:
                    strains.append(gene_space[j])
            
            strains_pool.append(strains)
        strains_pool_vaild = [strains for strains in strains_pool if len(strains) > 0]
        with open(log_dir + f'coarsen/strains_K{K}.pkl', 'wb') as f:
            pickle.dump(strains_pool_vaild, f)

        if K < 16+1:
            plt.figure(figsize=(14, 5))
            for i in range(K):
                plt.subplot(2, 8, i+1)
                if len(strains_pool[i]) == 0:
                    plt.text(0.5, 0.5, 'Empty', fontsize=12, ha='center', va='center')
                else:
                    plt.imshow(np.array(strains_pool[i]), cmap='binary', aspect='auto')
                plt.xticks([])
                plt.yticks([])
                plt.title(f'Strain {i}')
            plt.tight_layout()
            plt.savefig(log_dir + f'coarsen/strains_K{K}.png', dpi=300)

        from utilis import random_matrix
        trait_inter_matrix = random_matrix(
            (L, L),
            'tikhonov_sigmoid',
            args={'J_0': 0.5, 'n_star': 4, 'delta': 1},
            triangular=True,
            diagonal=0,
            seed=100
        )

        try:
            q_evo = np.load(log_dir + f'coarsen/coarse_K{K}.npy')
        except:            
            q_evo = []
            for _ in range(10):
                q_evo.append(evolution_test(strains_pool_vaild, L, trait_inter_matrix, total_t=1e5, dt=1e2, mutation_num=100, repeat=50))
            
            np.save(log_dir + f'coarsen/coarse_K{K}.npy', q_evo)

        L_ = int(np.log2(K))
        strains_pool_prx22 = [[] for _ in range(2**L_)]
        indices_prx22 = []
        for gene in gene_space:
            gene_prx = gene[:L_]
            index = int(''.join([str(int(i)) for i in gene_prx]), 2)
            strains_pool_prx22[index].append(gene)
            indices_prx22.append(index)
        with open(log_dir + f'coarsen/strains_prx22_K{K}.pkl', 'wb') as f:
            pickle.dump(strains_pool_prx22, f)
        
        try:
            q_evo_prx22 = np.load(log_dir + f'coarsen/coarse_prx22_K{K}.npy')
        except:            
            q_evo_prx22 = []
            for _ in range(10):
                q_evo_prx22.append(evolution_test(strains_pool_prx22, L, trait_inter_matrix, total_t=1e5, dt=1e2, mutation_num=100, repeat=50))
            
            np.save(log_dir + f'coarsen/coarse_prx22_K{K}.npy', q_evo_prx22)

        if K < 16+1:
            plt.figure(figsize=(14, 4))
            for i in range(K):
                plt.subplot(2, 8, i+1)
                if len(strains_pool_prx22[i]) == 0:
                    plt.text(0.5, 0.5, 'Empty', fontsize=12, ha='center', va='center')
                else:
                    plt.imshow(np.array(strains_pool_prx22[i]), cmap='binary', aspect='auto')
                plt.xticks([])
                plt.yticks([])
                plt.title(f'Strain {i}')
            plt.tight_layout()
            plt.savefig(log_dir + f'coarsen/strains_L{int(np.log2(K))}.png', dpi=300)

        # L_ = int(np.log2(K))
        # strains_pool_prx22 = [[] for _ in range(2**L_)]
        # indices_prx22 = []
        # for gene in gene_space:
        #     gene_prx = gene[:L_]
        #     index = int(''.join([str(int(i)) for i in gene_prx]), 2)
        #     strains_pool_prx22[index].append(gene)
        #     indices_prx22.append(index)


        # plt.figure(figsize=(8, 1.5))
        # plt.hist(indices_prx22, bins=K, color='orange', range=(0, K), align='left', rwidth=0.8, histtype='bar', label='PRX22', alpha=0.5)
        # plt.hist(indices.cpu().numpy(), bins=K, color='skyblue', edgecolor='black', range=(0, K), align='left', rwidth=0.8, histtype='bar', label='Our')
        # plt.xlabel('Strain Index')
        # plt.ylabel('Number of Strains')
        # plt.legend(frameon=False)
        # plt.tight_layout()
        # plt.savefig(log_dir + f'coarsen/strain_distribution_K{K}.png', dpi=300)

        # try:
        #     q_evo_rand = np.load(log_dir + f'coarsen/coarse_randm_K{K}.npz')['q_evo']
        # except:
        #     strains_pool_randm = [[] for _ in range(K)]
        #     for gene in gene_space:
        #         index = np.random.randint(K)
        #         strains_pool_randm[index].append(gene)

        #     strains_pool_randm = [strains for strains in strains_pool_randm if len(strains) > 0]
            
        #     q_evo_rand = evolution_test(strains_pool_randm, L, trait_inter_matrix, total_t=conf.system.total_time, dt=conf.system.dt, mutation_num=conf.system.max_mutations, repeat=100)

        #     np.save(log_dir + f'coarsen/coarse_randm_K{K}.npy', q_evo_rand)

        print(f'L={L}, K={K}, PRX22: {np.mean(q_evo_prx22):.4f}, Our: {np.mean(q_evo):.4f}')