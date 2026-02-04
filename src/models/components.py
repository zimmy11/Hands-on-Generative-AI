import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# ==========================================
# 1. SINUSOIDAL POSITION EMBEDDINGS
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Generates sinusoidal embeddings for time steps.
    Standard style used in DDPM / Transformers.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: tensor of shape (B,) representing time steps
        Returns:
            embeddings: (B, dim) sinusoidal embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ==========================================
# 2. ATTENTION BLOCK
# ==========================================
class AttentionBlock(nn.Module):
    """
    Multi-Head Self-Attention block with Pre-Norm.
    Structure: GroupNorm -> QKV -> Scaled Dot-Product Attention -> Residual
    """
    def __init__(self, channels, num_heads=4, num_groups=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)  # Q, K, V projections
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)  # Output projection

    def forward(self, x):
        b, c, h, w = x.shape
        h_ = self.norm(x)
        qkv = self.qkv(h_)  # (B, 3*C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head: (B, Heads, D, N)
        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h*w)
        k = k.view(b, self.num_heads, head_dim, h*w)
        v = v.view(b, self.num_heads, head_dim, h*w)

        # Scaled Dot-Product Attention
        attn_weights = torch.einsum('bhdn,bhdm->bhnm', q, k) * (head_dim ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        h_attn = torch.einsum('bhnm,bhdm->bhdn', attn_weights, v)
        h_attn = h_attn.contiguous().view(b, c, h, w)

        # Output projection + residual
        h_attn = self.proj_out(h_attn)
        return x + h_attn


# ==========================================
# 3. RESIDUAL BLOCK WITH TIME EMBEDDING (FiLM)
# ==========================================
class ResBlock(nn.Module):
    """
    Residual block with optional attention and time embeddings.
    Time embeddings are injected using Feature-wise Linear Modulation (FiLM)
    """
    def __init__(self, in_channels, out_channels=None, time_embed_dim=512, num_groups=8, dropout=0.1, use_attention=False):
        super().__init__()
        self.out_channels = out_channels or in_channels

        # First conv block
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)

        # Time embedding projection (scale & shift for FiLM)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, self.out_channels * 2)  # scale + shift
        )

        # Second conv block
        self.norm2 = nn.GroupNorm(num_groups, self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        # Shortcut connection
        if in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # Optional attention
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = AttentionBlock(self.out_channels, num_groups=num_groups)

    def forward(self, x, t_emb):
        """
        Args:
            x: input feature map (B, C, H, W)
            t_emb: time embedding (B, time_embed_dim)
        Returns:
            output: transformed feature map
        """
        h = x

        # First normalization + activation + conv
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)

        # Time embedding injection via FiLM
        t_vec = self.time_proj(t_emb)  # (B, 2*C)
        t_vec = t_vec[:, :, None, None]
        scale, shift = t_vec.chunk(2, dim=1)

        # Second normalization + FiLM + activation + conv
        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Add residual connection
        h = h + self.shortcut(x)

        # Optional attention
        if self.use_attention:
            h = self.attention(h)

        return h


# ==========================================
# 4. EMA MODEL
# ==========================================
class EMAModel(nn.Module):
    """
    Maintains Exponential Moving Average (EMA) of model weights.
    EMA weights produce more stable and sharper outputs than live weights.
    """
    def __init__(self, model, decay=0.999):
        super().__init__()
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        # Disable gradients for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        """
        Updates EMA weights from the live model.
        Formula: ema = decay * ema + (1 - decay) * model
        """
        for ema_param, current_param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(current_param.data, alpha=1 - self.decay)

    def forward(self, x, t, labels=None):
        """
        Forward pass using EMA weights (pass-through)
        """
        return self.ema_model(x, t, labels=labels)
