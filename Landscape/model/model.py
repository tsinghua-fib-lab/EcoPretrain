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

class ConvExtractor(nn.Module):
    def __init__(self, input_channels, output_dim, kernel_size=3):
        super(ConvExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv3 = nn.Conv1d(128, output_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        out = self.global_pooling(x)
        return out.squeeze(-1)

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
    def __init__(self, input_channels, feature_dim, num_heads, num_layers, max_len=5000, pos_encoding_type="sine"):
        super(Encoder, self).__init__()
        self.conv_extractor = ConvExtractor(input_channels=input_channels, output_dim=feature_dim)
        self.temporal_transformer = TemporalTransformer(
            input_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            feature_dim=feature_dim,
            max_len=max_len,
            pos_encoding_type=pos_encoding_type
        )

    def forward(self, x):
        batch_size, T, L, b = x.size()
        x = x.reshape(batch_size * T, L, b)
        x = self.conv_extractor(x)
        x = x.reshape(batch_size, T, -1)
        out = self.temporal_transformer(x)
        return out

class Decoder(nn.Module):
    def __init__(self, input_dim, L, b, num_heads, num_layers, feature_dim, max_len=5000, pos_encoding_type="sine"):
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
        self.output_layer = nn.Linear(input_dim, L * b)
        self.L = L
        self.b = b

    def forward(self, x):
        x = self.position_encoding(x)
        x = self.transformer_decoder(x)
        x = self.output_layer(x)
        x = x.view(-1, x.size(1), self.L, self.b)
        return x

class EvolutionDynamicsModel(nn.Module):
    def __init__(self, input_channels, feature_dim, L, b, num_heads, num_layers, max_len=5000, pos_encoding_type="sine", codebook_size=512, use_rotation=True, code_init='uniform'):
        super(EvolutionDynamicsModel, self).__init__()
        self.use_rotation = use_rotation
        
        self.encoder = Encoder(
            input_channels=input_channels,
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
            b=b,
            num_heads=num_heads,
            num_layers=num_layers,
            feature_dim=feature_dim,
            max_len=max_len,
            pos_encoding_type=pos_encoding_type
        )
        # self.decoder = nn.Sequential(
        #     nn.Linear(feature_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, L * b),
        #     nn.Unflatten(-1, (L, b))
        # )
        
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
    def __init__(self, L, b, D):
        super(GenotypeEncoder, self).__init__()        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(L * b, 128),
            nn.ReLU(),
            nn.Linear(128, D),
        )
        self.net2 = nn.Sequential(
            nn.Linear(D, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        self.apply(initialize_weights)
    
    def forward(self, x):
        # x: (Batch, T, L, b)
        B, T, L, b = x.size()
        x = x.view(B*T, L, b) # (Batch * T, L, b)
        
        z = self.net(x) # (Batch * T, D)
        fitness = self.net2(z) # (Batch*T, 1)
        
        fitness = fitness.view(B, T, 1)
        z = z.view(B, T, -1) # (Batch, T, D)
        return z, fitness


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


def gumbel_softmax(pi, device, tau=1.0):
    """
    Gumbel-Softmax distribution.
    Implementation from https://github.com/ericjang/gumbel-softmax.
    pi: [B, ..., n_classes] class probs of categorical z
    tau: temperature
    Returns [B, ..., n_classes] as a one-hot vector
    """
    y = gumbel_softmax_sample(pi, tau, device)
    shape = y.size()
    _, ind = y.max(dim=-1)  # [B, ...]
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return y_hard - y.detach() + y

def gumbel_softmax_sample(pi, tau, device, epsilon=1e-12):
    y = torch.log(pi + epsilon) + gumbel_sample(pi.size(), device)
    return torch.nn.functional.softmax(y / tau, dim=-1)

def gumbel_sample(shape, device, epsilon=1e-20):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + epsilon) + epsilon)


class NeuralMJP(nn.Module):
    def __init__(self, input_dim, nhidden, nstates):
        super().__init__()
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(nstates, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, input_dim),
        )
        
        # Neural Master Equation
        self.nstates = nstates
        self.qfunc = nn.Sequential(
            nn.Linear(input_dim, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, self.nstates),
            nn.Softmax(dim=1),
        )
        self.gfunc_in = nn.Sequential(
            nn.Linear(input_dim, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, self.nstates),
            nn.Softmax(dim=1),
        )
        self.gfunc_out = nn.Sequential(
            nn.Linear(input_dim, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, self.nstates),
            nn.Softmax(dim=1),
        )
        
        self.apply(initialize_weights)

    def forward(self, h):
        # h: (B, D)

        q = self.qfunc(h) # (B, N)

        q_self_mask = q.unsqueeze(1).repeat(1, self.nstates, 1)
        q_self_mask[:, torch.arange(self.nstates), torch.arange(self.nstates)] = 0. # z': (B, N, N)
        q_self = q.unsqueeze(1).repeat(1, self.nstates, 1) * torch.eye(self.nstates).to(h.device) # z: (B, N, N)
        
        g_in = self.gfunc_in(h) # (B, N)
        g_out = self.gfunc_out(h) # (B, N)
        
        drift_term = (g_in.unsqueeze(2) * q_self_mask).sum(-1) # (B, N)
        diffusion_term = (g_out.unsqueeze(2) * q_self).sum(-1) # (B, N)
        
        dqdt = drift_term + diffusion_term # (B, N)
        q_next = q + dqdt # (B, N)

        sample_latent = gumbel_softmax(q_next, h.device) # (B, N)
        
        y_hat = self.decoder(sample_latent) # (B, D)

        return y_hat
    
    def autoregressive_predict(self, initial_state, T):
        """
        Perform autoregressive prediction for T-1 steps based on the initial state.

        :param initial_state: Tensor of shape (B, 1, D) - Initial state
        :param T: Total time steps (including initial state)
        :return: Predicted state sequence of shape (B, T-1, D)
        """
        B, _, C = initial_state.shape
        predicted_states = []  # Store predicted states

        # Get initial prediction (t=0)
        current_state = self.forward(initial_state.squeeze(1))  # (B, 1, nstates)
        predicted_states.append(current_state.unsqueeze(1))

        # Start the autoregressive loop
        for t in range(1, T-1):  # T-1 steps after the initial state
            # Predict the next state based on the current state
            current_state = self.forward(current_state)  # (B, 1, nstates)
            predicted_states.append(current_state.unsqueeze(1))

        # Stack all predicted states (B, T-1, nstates)
        predicted_states = torch.cat(predicted_states, dim=1)
        return predicted_states  # Shape: (B, T-1, nstates)