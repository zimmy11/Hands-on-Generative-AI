import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time è un vettore (B,) di interi
        device = time.device
        half_dim = self.dim // 2
        
        # Calcola gli argomenti per seno e coseno
        # 'embeddings' sarà (B, half_dim)
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :] # broadcasting (B, half_dim)
        
        # Alterna seno e coseno
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # (B, dim)
        
        # Gestisce il caso in cui dim è dispari
        if self.dim % 2 != 0:
            embeddings = F.pad(embeddings, (0, 1), mode='constant', value=0)
            
        return embeddings