import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        
        device = time.device
        half_dim = self.dim // 2
        
        # 'embeddings' will be (B, half_dim)
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :] 
        # broadcasting (B, half_dim)
        
        # Sine and Cosine
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # (B, dim)
        
        # if dim is odd, pad with one zero vector
        if self.dim % 2 != 0:
            embeddings = F.pad(embeddings, (0, 1), mode='constant', value=0)
            
        return embeddings