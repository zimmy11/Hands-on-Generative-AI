import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class AttributeEncoder(nn.Module):
    """
    Projects multi-hot CelebA attributes into a latent space 
    compatible with the U-Net's time embedding.
    """
    def __init__(self, num_attributes=40, embed_dim=128, out_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_attributes, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, out_dim),
            nn.SiLU()
        )

    def forward(self, attributes, drop_prob=0.1):
        """
        Args:
            attributes: Tensor of shape (B, 40) containing 0s and 1s.
            drop_prob: Probability of nullifying the condition for CFG.
        """
        # Multi-hot projection
        c_emb = self.mlp(attributes.float())
        
        # Classifier-Free Guidance: Randomly drop the condition
        if self.training and drop_prob > 0:
            mask = torch.bernoulli(torch.full((attributes.shape[0], 1), 1 - drop_prob)).to(attributes.device)
            c_emb = c_emb * mask
            
        return c_emb