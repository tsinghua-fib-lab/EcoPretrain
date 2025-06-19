import torch
import torch.nn as nn
from torchdiffeq import odeint
import math
from sklearn.cluster import KMeans


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def efficient_rotation(e, q):
    q = q.detach()

    e_norm = torch.norm(e, dim=-1, keepdim=True)
    q_norm = torch.norm(q, dim=-1, keepdim=True)

    e_hat = e / e_norm
    q_hat = q / q_norm

    r = ((e_hat + q_hat) / torch.norm(e_hat + q_hat, dim=-1, keepdim=True)).detach()

    r_e_dot = torch.sum(r * e, dim=-1, keepdim=True)
    reflect_r = 2 * r * r_e_dot

    q_hat_e_dot = torch.sum(q_hat * e, dim=-1, keepdim=True)
    rotate_q = 2 * q_hat * q_hat_e_dot

    transformed_q = e - reflect_r + rotate_q

    return transformed_q

class CodeBook(nn.Module):
    def __init__(self, K, D, init='uniform'):
        super(CodeBook, self).__init__()
        self.embedding = nn.Embedding(K, D)
        self.initialize_embeddings(init)

    def initialize_embeddings(self, init):
        if init == 'normal':
            self.embedding.weight.data.normal_(0, 1)
        elif init == 'uniform':
            self.embedding.weight.data.uniform_(-1 / self.embedding.num_embeddings, 1 / self.embedding.num_embeddings)
        elif init == 'xavier':
            nn.init.xavier_uniform_(self.embedding.weight)
        elif init == 'kaiming':
            nn.init.kaiming_uniform_(self.embedding.weight)
        elif init == 'orthogonal':
            nn.init.orthogonal_(self.embedding.weight)
        elif init == 'sparse':
            nn.init.sparse_(self.embedding.weight, sparsity=0.1)
        elif init == 'eye':
            nn.init.eye_(self.embedding.weight)
        else:
            raise ValueError("Unknown init method")

    def forward(self, z_e_x):
        with torch.no_grad():
            B, K = z_e_x.size(0), self.embedding.weight.size(0)
            codebook_vectors = self.embedding.weight.unsqueeze(0).expand(B, -1, -1)
            z_e_x_expanded = z_e_x.unsqueeze(1).expand(-1, K, -1)

            distances = torch.sum((z_e_x_expanded - codebook_vectors) ** 2, dim=-1)
            index = torch.argmin(distances, dim=-1)
        return index

    def straight_through(self, z_e_x, rotation=False):
        index = self.forward(z_e_x)
        z_q_x = self.embedding(index)

        if not rotation:
            sg_z_e_x = z_e_x + (z_q_x - z_e_x).detach()
        else:
            sg_z_e_x = efficient_rotation(z_e_x, z_q_x)

        return sg_z_e_x, z_q_x, index

    def lookup(self, index):
        return self.embedding(index)

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=5000, learnable=False):
        super(PositionalEncoding, self).__init__()
        self.dim_model = dim_model
        self.learnable = learnable

        if learnable:
            self.position_embedding = nn.Parameter(torch.randn(1, max_len, dim_model))
        else:
            position = torch.arange(0, max_len).unsqueeze(1)
            if dim_model % 2 == 0:
                div_term = torch.exp(torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model))
                pe = torch.zeros(max_len, dim_model)
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
            else:
                div_term = torch.exp(torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model))
                pe = torch.zeros(max_len, dim_model+1)
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe[:, :-1]
            pe = pe.unsqueeze(0)
            self.register_buffer('position_embedding', pe)

    def forward(self, x):
        seq_len = x.size(1)
        if self.learnable:
            return x + self.position_embedding[:, :seq_len, :]
        else:
            return x + self.position_embedding[:, :seq_len, :]


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, feature_dim, max_len=5000, pos_encoding_type="sine"):
        super(TemporalTransformer, self).__init__()
        self.position_encoding = PositionalEncoding(
            dim_model=input_dim, max_len=max_len, learnable=(pos_encoding_type == "learnable")
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.position_encoding(x)
        out = self.transformer_encoder(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim, num_heads, num_layers, max_len=5000, pos_encoding_type="sine"):
        super(Encoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.temporal_transformer = TemporalTransformer(
            input_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            feature_dim=feature_dim,
            max_len=max_len,
            pos_encoding_type=pos_encoding_type
        )

    def forward(self, x):
        batch_size, T, L = x.size()
        x = x.reshape(batch_size * T, L)
        x = self.mlp(x)
        x = x.reshape(batch_size, T, -1)
        out = self.temporal_transformer(x)
        return out

class Decoder(nn.Module):
    def __init__(self, input_dim, L, num_heads, num_layers, feature_dim, max_len=5000, pos_encoding_type="sine"):
        super(Decoder, self).__init__()
        self.position_encoding = PositionalEncoding(
            dim_model=input_dim, max_len=max_len, learnable=(pos_encoding_type == "learnable")
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(input_dim, L)
        self.L = L

    def forward(self, x):
        x = self.position_encoding(x)
        x = self.transformer_decoder(x)
        x = self.output_layer(x)
        x = x.view(-1, x.size(1), self.L)
        return x

class EvolutionDynamicsModel(nn.Module):
    def __init__(self, feature_dim, L, num_heads, num_layers, max_len=5000, pos_encoding_type="sine", codebook_size=512, use_rotation=True, code_init='uniform'):
        super(EvolutionDynamicsModel, self).__init__()
        self.use_rotation = use_rotation
        
        self.encoder = Encoder(
            input_dim=L,
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_len=max_len,
            pos_encoding_type=pos_encoding_type
        )
        
        self.codebook = CodeBook(K=codebook_size, D=feature_dim, init=code_init)
        
        self.decoder = Decoder(
            input_dim=feature_dim,
            L=L,
            num_heads=num_heads,
            num_layers=num_layers,
            feature_dim=feature_dim,
            max_len=max_len,
            pos_encoding_type=pos_encoding_type
        )
        
        self.apply(initialize_weights)
        
    def re_init_codebook(self, train_loader):
        with torch.no_grad():
            z_e_x = []
            for x, _ in train_loader:
                x = x.to(self.encoder.conv_extractor.conv1.weight.device)
                z_e_x.append(self.encoder(x))
                
                if len(z_e_x) == 20000: # 20000
                    break
            z_e_x = torch.cat(z_e_x, dim=1) # 20000 x T x D
        
        z_e_x = z_e_x.reshape(-1, z_e_x.size(-1)).cpu().numpy()
        cluster = KMeans(n_clusters=self.codebook.embedding.num_embeddings, random_state=42, n_init='auto').fit(z_e_x)
        # cluster = AgglomerativeClustering(n_clusters=self.codebook.embedding.num_embeddings).fit(z_e_x)
        self.codebook.embedding.weight.data = torch.tensor(cluster.cluster_centers_).to(self.codebook.embedding.weight.device)

    def forward(self, x, vq=True):
        z_e_x = self.encoder(x)
        
        z_e_x_ = z_e_x.reshape(-1, z_e_x.size(-1))
        z_q_x_st, z_q_x, indices = self.codebook.straight_through(z_e_x_, rotation=self.use_rotation)
        z_q_x_st = z_q_x_st.view(x.size(0), -1, z_q_x_st.size(-1)) 
        z_q_x = z_q_x.view(x.size(0), -1, z_q_x.size(-1)) 
        indices = indices.view(x.size(0), -1) 
        
        if vq:
            x_logit = self.decoder(z_q_x_st) 
        else:
            x_logit = self.decoder(z_e_x)
        return x_logit, z_q_x_st, z_q_x, indices
    
    def encode(self, x):
        z_e_x = self.encoder(x)
        z_e_x_ = z_e_x.reshape(-1, z_e_x.size(-1)) 
        z_q_x_st, _, _ = self.codebook.straight_through(z_e_x_, rotation=self.use_rotation) 
        z_q_x_st = z_q_x_st.view(x.size(0), -1, z_q_x_st.size(-1)) 
        return z_e_x, z_q_x_st
    
    def decode(self, z):
        return self.decoder(z)


class GenotypeEncoder(nn.Module):
    def __init__(self, L, D):
        super(GenotypeEncoder, self).__init__()        
        self.net = nn.Sequential(
            nn.Linear(L, 64),
            nn.ReLU(),
            nn.Linear(64, D),
        )
        
        self.apply(initialize_weights)
    
    def forward(self, x):
        # x: (Batch, T, L)
        B, T, L = x.size()
        x = x.view(B*T, L) # (Batch * T, L)
        z = self.net(x) # (Batch * T, D)
        z = z.view(B, T, -1) # (Batch, T, D)
        return z


class NeuralODE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralODE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        # normal initialization
        self.net[0].weight.data.normal_(0, 0.01)
        self.net[2].weight.data.normal_(0, 0.01)

    def dynamics(self, t, x):
        return self.net(x)

    def forward(self, x0, T):
        x0 = x0.squeeze(1) 

        time_points = torch.linspace(0, 10, T) 

        pred = odeint(self.dynamics, x0, time_points, method='dopri5') # e.g. 'dopri5', 'euler', 'rk4'
        pred = pred.permute(1, 0, 2)

        return pred[:, 1:] 


    
if __name__ == '__main__':
    from omegaconf import OmegaConf
    conf = OmegaConf.load('coli.yaml')

    L = conf.coli.loci
    feature_dim = conf.model.feature_dim
    num_heads = conf.model.num_heads
    num_layers = conf.model.num_layers
    codebook_size = conf.model.K
    pos_encoding_type = conf.model.pos_encoding_type
    use_rotation = conf.model.use_rotation
    code_init = conf.model.code_init
    model = EvolutionDynamicsModel(feature_dim, L, num_heads, num_layers, pos_encoding_type=pos_encoding_type, codebook_size=codebook_size, use_rotation=use_rotation, code_init=code_init).to(conf.device)
    
    dummy_input = torch.randn(16, 10, L).to(conf.device)
    x_hat, z_q_x_st, z_q_x, indices = model(dummy_input)
    print(x_hat.size(), z_q_x_st.size(), z_q_x.size(), indices.size())