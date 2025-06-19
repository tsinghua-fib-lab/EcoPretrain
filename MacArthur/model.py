import math
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight, 0, 0.01)
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
            div_term = torch.exp(torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model))
            pe = torch.zeros(max_len, dim_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('position_embedding', pe)

    def forward(self, x):
        seq_len = x.size(1)
        if self.learnable:
            return x + self.position_embedding[:, :seq_len, :]
        else:
            return x + self.position_embedding[:, :seq_len, :]

class PopulationAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super(PopulationAttention, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim) 
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, x, mask=None):
        # x: (B, T, N, L)
        B, T, N, L = x.shape

        # Step 1: Project (B, T, N, L) to (B, T, N, D)
        x = self.linear(x)  # (B, T, N, D)
        
        # Step 2: Reshape to (B*T, N, D) for attention over N dimension
        x = x.view(B * T, N, -1)

        # Step 3: Adjust mask to (B*T, N) for MultiheadAttention
        if mask is not None:
            mask = mask.view(B * T, N)  # Reshape to (B*T, N)
        
        # Step 4: Apply Multihead Attention with mask
        attn_output, _ = self.multihead_attn(x, x, x, key_padding_mask=mask)  # Self-attention over N dimension
        
        # Step 5: Reshape output back to (B, T, N, D)
        attn_output = attn_output.view(B, T, N, -1)
        return attn_output

class TemporalTransformer(nn.Module):
    def __init__(self, n_d, h, num_layers=2, num_heads=4, max_len=5000, pos_encoding_type="sine"):
        super(TemporalTransformer, self).__init__()
        
        # Positional encoding
        self.position_encoding = PositionalEncoding(dim_model=h, max_len=max_len, learnable=(pos_encoding_type == "learnable"))
        
        # Transformer encoder for T dimension with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=h, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Linear projection from N * D to h
        self.projection = nn.Linear(n_d, h)
        
        # Projection back to (B, T, N * D)
        self.output_projection = nn.Linear(h, n_d)


    def forward(self, x):
        # x shape: (B, T, N, D)
        B, T, N, D = x.shape
        
        # Step 1: Reshape (B, T, N, D) to (B, T, N * D)
        x = x.reshape(B, T, N * D)
        
        # Step 2: Project (B, T, N * D) to (B, T, h)
        z = self.projection(x)  # (B, T, h)
        
        # Step 3: Apply positional encoding
        z = self.position_encoding(z)
        
        # Step 4: Apply transformer on T dimension
        z = self.transformer_encoder(z)  # (B, T, h)
        
        # Step 5: Project back to (B, T, N * D)
        x_hat = self.output_projection(z)
        
        # Step 6: Reshape back to (B, T, N, D)
        x_hat = x_hat.view(B, T, N, D)
        
        return z, x_hat

class PopulationDecoder(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, n_class, num_heads):
        super(PopulationDecoder, self).__init__()
        
        # Linear layer to project from D to embed_dim
        self.linear_in = nn.Linear(input_dim, embed_dim)
        
        # Multihead attention for N dimension with batch_first=True
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Output projection back to L dimensions
        self.n_class = n_class
        self.linear_out = nn.Linear(embed_dim, output_dim*n_class)

    def forward(self, x, mask=None):
        # x shape: (B, T, N, D)
        B, T, N, D = x.shape

        # Step 1: Project (B, T, N, D) to (B, T, N, embed_dim)
        x = self.linear_in(x)  # (B, T, N, embed_dim)
        
        # Step 2: Reshape to (B*T, N, embed_dim) for attention over N dimension
        x = x.view(B * T, N, -1)

        # Step 3: Adjust mask to (B*T, N) for MultiheadAttention
        if mask is not None:
            mask = mask.view(B * T, N)  # Reshape mask to (B*T, N)
        
        # Step 4: Apply Multihead Attention with mask
        attn_output, _ = self.multihead_attn(x, x, x, key_padding_mask=mask)  # Self-attention over N dimension

        # Step 5: Project attention output back to L dimensions
        attn_output = self.linear_out(attn_output)  # (B*T, N, L*n_class)

        # Step 6: Reshape output back to (B, T, N, L)
        output = attn_output.view(B, T, N, -1, self.n_class)
        
        return output


class PopulationDecoderMLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_class):
        super(PopulationDecoderMLP, self).__init__()
        
        self.n_class = n_class
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim*n_class),
            nn.Unflatten(-1, (output_dim, n_class))
        )

    def forward(self, x, mask=None):
        # x shape: (B, T, N, D)
        B, T, N, D = x.shape
        x = x.view(B * T * N, -1)

        x_logits = self.net(x) # (B*T*N, L, n_class)
        x_logits = x_logits.view(B, T, N, -1, self.n_class)
        return x_logits

class EvolutionDynamicsModel(nn.Module):
    def __init__(self, input_dim, channel_dim, embed_dim, temporal_dim, output_dim, n_class=3, num_heads=4, num_layers=2, max_len=5000, pos_encoding_type="sine", codebook_size=None):
        super(EvolutionDynamicsModel, self).__init__()
        
        # Population Attention for encoding (B, T, N, L) -> (B, T, N, D)
        self.population_attention = PopulationAttention(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads)
        
        # Temporal Transformer for time modeling (B, T, N, D) -> (B, T, h)
        self.temporal_transformer = TemporalTransformer(n_d=channel_dim * embed_dim, h=temporal_dim, num_layers=num_layers,
                                                        num_heads=num_heads, max_len=max_len, pos_encoding_type=pos_encoding_type)
        
        # Codebook for VQ
        codebook_size = codebook_size if codebook_size is not None else 2**input_dim
        self.codebook = CodeBook(K=codebook_size, D=embed_dim)
        
        # Population Decoder to reconstruct the original evolutionary trajectory (B, T, h) -> (B, T, N, L)
        # self.population_decoder = PopulationDecoder(input_dim=embed_dim, embed_dim=embed_dim, output_dim=output_dim, n_class=n_class, num_heads=num_heads)
        self.population_decoder = PopulationDecoderMLP(input_dim=embed_dim, output_dim=output_dim, n_class=n_class)
        
        # Initialize weights
        self.apply(initialize_weights)

    def encoder(self, x):
        # Step 1: Create mask for PopulationAttention based on -1 values in x
        # mask = (x.sum(dim=-1) == -L)  # mask shape: (B, T, N), True where all L dimensions are -1
        mask = None
        
        # Step 2: Apply PopulationAttention to encode (B, T, N, L) -> (B, T, N, D)
        z = self.population_attention(x, mask=mask)
        
        # Step 3: Apply TemporalTransformer for time modeling (B, T, N, D) -> (B, T, N, h)
        zz, z_hat = self.temporal_transformer(z)
        
        return z_hat
    
    def encoder2(self, x):
        # Step 1: Create mask for PopulationAttention based on -1 values in x
        # mask = (x.sum(dim=-1) == -L)  # mask shape: (B, T, N), True where all L dimensions are -1
        mask = None
        
        # Step 2: Apply PopulationAttention to encode (B, T, N, L) -> (B, T, N, D)
        z = self.population_attention(x, mask=mask)
        
        # Step 3: Apply TemporalTransformer for time modeling (B, T, N, D) -> (B, T, N, h)
        zz, z_hat = self.temporal_transformer(z)
        
        return zz
    
    def coarse(self, z):
        # z: (N, L)
        sg_z_e_x, _, indices = self.codebook.straight_through(z)
        return sg_z_e_x, indices
    
    def init_codebook(self, codebook_size, embed_dim):
        self.codebook = CodeBook(K=codebook_size, D=embed_dim)
    
    def re_init_codebook(self, train_loader):
        with torch.no_grad():
            z_e_x = []
            for x in train_loader:
                z_e_x.append(self.encoder(x))
                
                if len(z_e_x) == 100: # 100
                    break
            z_e_x = torch.cat(z_e_x, dim=1) # 100 x T x N x D
        
        z_e_x = z_e_x.reshape(-1, z_e_x.size(-1)).cpu().numpy()
        cluster = KMeans(n_clusters=self.codebook.embedding.num_embeddings, random_state=42, n_init='auto').fit(z_e_x)
        self.codebook.embedding.weight.data = torch.tensor(cluster.cluster_centers_).to(x.device)
    
    def forward(self, x, vq=False):
        # x shape: (B, T, N, L)
        
        # Step 1: Create mask for PopulationAttention based on -1 values in x
        # mask = (x.sum(dim=-1) == -L)  # mask shape: (B, T, N), True where all L dimensions are -1
        mask = None
        
        # Step 2: Apply PopulationAttention to encode (B, T, N, L) -> (B, T, N, D)
        z = self.population_attention(x, mask=mask)
        
        # Step 3: Apply TemporalTransformer for time modeling (B, T, N, D) -> (B, T, N, h)
        zz, z_hat = self.temporal_transformer(z)
        
        # Step 4: Apply VQ if enabled
        # Flatten (B, T, N, h) to (B*T*N, h) for codebook lookup
        z_hat_flat = z_hat.view(-1, z_hat.size(-1))  # Shape: (B*T*N, h)
        # Apply VQ using CodeBook
        sg_z_e_x, z_q_x, indices = self.codebook.straight_through(z_hat_flat)
        
        # Reshape back to (B, T, N, h) after VQ
        z_q_x = z_q_x.view(z_hat.size(0), z_hat.size(1), z_hat.size(2), -1)
        sg_z_e_x = sg_z_e_x.view(z_hat.size(0), z_hat.size(1), z_hat.size(2), -1)
        
        # Step 5: Apply PopulationDecoder to decode back to (B, T, N, L, n_class)
        if vq:
            x_logits = self.population_decoder(sg_z_e_x, mask=mask)
        else:
            x_logits = self.population_decoder(z_hat, mask=mask)
        
        return z, zz, z_hat, x_logits, sg_z_e_x, z_q_x, indices


class GenotypeEncoder(nn.Module):
    def __init__(self, L, D):
        super(GenotypeEncoder, self).__init__()        
        self.net = nn.Sequential(
            nn.Linear(L, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, D),
        )
        
        self.apply(initialize_weights)
    
    def forward(self, x):
        # x: (N, L)        
        z = self.net(x) # (N, D)
        return z



if __name__ == '__main__':

    B, T, N, L = 128, 100, 128, 7 
    D = 4 
    h = 128 

    x = torch.randint(-1, 2, (B, T, N, L)).float().cuda()

    model = EvolutionDynamicsModel(input_dim=L, channel_dim=N, embed_dim=D, temporal_dim=h, output_dim=L, num_heads=4, num_layers=2).cuda()

    zz, z_hat, x_hat = model(x)

    print("zz shape:", zz.shape)       # (B, T, h)
    print("z_hat shape:", z_hat.shape) # (B, T, N, D)
    print("x_hat shape:", x_hat.shape) # (B, T, N, L, n_class)