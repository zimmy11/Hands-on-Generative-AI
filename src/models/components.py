import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# ==========================================
# 1. SINUSOIDAL TIME EMBEDDINGS
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Generates sinusoidal embeddings for timestep t.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: Tensor of shape (B,) with timesteps

        Returns:
            Tensor of shape (B, dim) with sinusoidal embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        # Compute frequency range
        freq = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=device) * -freq)
        # Outer product: time * frequency
        embeddings = time[:, None] * freq[None, :]
        # Concatenate sin and cos embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ==========================================
# 2. ATTENTION BLOCK (DDPM++ / NCSN++)
# ==========================================
class AttentionBlock(nn.Module):
    """
    Multi-Head Self-Attention block with Pre-Norm.
    """
    def __init__(self, channels, num_heads=4, num_groups=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups, channels)
        # Linear projections for Q, K, V (Conv1x1)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        # Output projection
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        h_ = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(h_)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention: (B, Heads, D, N)
        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)

        # Scaled dot-product attention
        attn_weights = torch.einsum('bhdn,bhdm->bhnm', q, k) * (head_dim ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Aggregate values
        h_attn = torch.einsum('bhnm,bhdm->bhdn', attn_weights, v)

        # Reshape back to (B, C, H, W)
        h_attn = h_attn.contiguous().view(b, c, h, w)

        # Output projection and residual connection
        return x + self.proj_out(h_attn)


# ==========================================
# 3. RESIDUAL BLOCK WITH TIME EMBEDDING
# ==========================================
class ResBlock(nn.Module):
    """
    Residual Block with optional attention and FiLM modulation by timestep embedding.
    """
    def __init__(self, in_channels, out_channels=None, time_embed_dim=512, num_groups=8, dropout=0.1, use_attention=False):
        super().__init__()
        self.out_channels = out_channels or in_channels

        # First convolutional block
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)

        # Time embedding projection for FiLM: outputs scale and shift
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, self.out_channels * 2)
        )

        # Second convolutional block
        self.norm2 = nn.GroupNorm(num_groups, self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        # Shortcut connection for residual
        if in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # Optional attention block
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = AttentionBlock(self.out_channels, num_groups=num_groups)

    def forward(self, x, t_emb):
        """
        Args:
            x: Input features (B, C, H, W)
            t_emb: Time embeddings (B, time_embed_dim)
        Returns:
            Residual block output (B, out_channels, H, W)
        """
        h = x

        # First block
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)

        # FiLM modulation using time embedding
        t_vec = self.time_proj(t_emb)  # (B, 2*out_channels)
        t_vec = t_vec[:, :, None, None]  # (B, 2*C, 1, 1)
        scale, shift = t_vec.chunk(2, dim=1)

        # Second block with FiLM
        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Residual connection
        h = h + self.shortcut(x)

        # Optional attention
        if self.use_attention:
            h = self.attention(h)

        return h


# ==========================================
# 4. EMA (Exponential Moving Average) MODEL
# ==========================================
class EMAModel(nn.Module):
    """
    Maintains an Exponential Moving Average of model weights.
    EMA weights produce more stable and sharper images than live weights.
    """
    def __init__(self, model, decay=0.999):
        super().__init__()
        self.decay = decay

        # Copy the model structure
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()  # EMA model is always in evaluation mode

        # Disable gradients for EMA parameters
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        """
        Update EMA weights using live model weights.
        Formula: ema = decay * ema + (1 - decay) * current
        """
        for ema_param, current_param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(current_param.data, alpha=1 - self.decay)

    def forward(self, x, t):
        """
        Forward pass through EMA model
        """
        return self.ema_model(x, t)
