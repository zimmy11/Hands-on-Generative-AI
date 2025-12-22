import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Genera embedding sinusoidali per il tempo t.
    Stile standard DDPM/Transformer.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    

    
class AttentionBlock(nn.Module):
    """
    Blocco di Attenzione simile a quello usato in DDPM++ / NCSN++ (Yang Song).
    Struttura: GroupNorm -> MultiHead Self-Attention -> Residual.
    Usa 'Pre-Norm' architecture.
    """
    def __init__(self, channels, num_heads=4, num_groups=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups, channels)
        
        # Proiezione Q, K, V
        # Song usa spesso conv 1x1 (NIN - Network in Network) per le proiezioni
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        
        # Proiezione di output
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # x shape: (B, C, H, W)
        b, c, h, w = x.shape
        
        # 1. Normalizzazione (Pre-Norm)
        h_ = self.norm(x)
        
        # 2. Calcolo Q, K, V
        # (B, 3*C, H, W)
        qkv = self.qkv(h_)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 3. Reshape per Multi-Head Attention
        # Da (B, C, H, W) a (B, Heads, H*W, C/Heads)
        head_dim = c // self.num_heads
        
        q = q.view(b, self.num_heads, head_dim, h*w) # (B, Heads, D, N)
        k = k.view(b, self.num_heads, head_dim, h*w)
        v = v.view(b, self.num_heads, head_dim, h*w)
        
        # 4. Scaled Dot-Product Attention
        # (B, Heads, N, N) dove N = H*W
        # Trasponiamo Q e K per fare il prodotto scalare
        # q: (B, H, D, N) -> permute -> (B, H, N, D) per compatibilità logica standard, 
        # ma qui facciamo einsum o matmul diretto.
        # Song implementation: torch.einsum('bhdn,bhdm->bhnm', q, k)
        
        attn_weights = torch.einsum('bhdn,bhdm->bhnm', q, k) * (head_dim ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 5. Aggregate Values
        # (B, Heads, N, N) * (B, Heads, D, N) -> Attenzione alle dimensioni!
        # v è (B, H, D, N)
        h_attn = torch.einsum('bhnm,bhdm->bhdn', attn_weights, v)
        
        # 6. Reshape back to Spatial
        h_attn = h_attn.contiguous().view(b, c, h, w)
        
        # 7. Output Projection & Residual
        h_attn = self.proj_out(h_attn)
        
        return x + h_attn

# ==========================================
# 2. RESNET BLOCK (Stile DDPM++/Song con FiLM)
# ==========================================

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, time_embed_dim=512, num_groups=8, dropout=0.1, use_attention=False):
        super().__init__()
        self.out_channels = out_channels if out_channels else in_channels
        
        # Block 1
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)
        
        # FiLM (Feature-wise Linear Modulation) Projection
        # Song usa spesso scale & shift derivati dal time embedding
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, self.out_channels * 2) # *2 per scale e shift
        )
        
        # Block 2
        self.norm2 = nn.GroupNorm(num_groups, self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # Shortcut (Skip connection)
        if in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # Attenzione opzionale integrata nel blocco (stile DDPM originale) o esterna
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = AttentionBlock(self.out_channels, num_groups=num_groups)

    def forward(self, x, t_emb):
        """
        x: (B, C, H, W) input image/feature
        t_emb: (B, time_embed_dim) time embedding
        """
        h = x
        
        # --- Part 1 ---
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        # --- Time Injection (FiLM) ---
        # Proiettiamo t_emb
        t_vec = self.time_proj(t_emb) # (B, 2*out_channels)
        # Reshape per broadcasting (B, 2*C, 1, 1)
        t_vec = t_vec[:, :, None, None]
        scale, shift = t_vec.chunk(2, dim=1)
        
        # --- Part 2 ---
        h = self.norm2(h)
        # Applico FiLM: Normalize -> Scale -> Shift -> Act
        h = h * (1 + scale) + shift 
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # --- Combine ---
        h = h + self.shortcut(x)
        
        if self.use_attention:
            h = self.attention(h)
            
        return h
    


class EMAModel(nn.Module):
    """
    Mantiene una media mobile esponenziale (EMA) dei pesi del modello.
    I pesi EMA producono immagini molto più stabili e nitide rispetto ai pesi 'live'.
    """
    def __init__(self, model, decay=0.999):
        super().__init__()
        self.decay = decay
        
        # 1. Clona la struttura del modello originale
        self.ema_model = copy.deepcopy(model)
        
        # 2. Imposta in modalità eval (niente dropout, batchnorm fissa)
        self.ema_model.eval()
        
        # 3. Disabilita il calcolo dei gradienti per risparmiare memoria
        # (L'EMA non viene allenato con backprop, ma aggiornato con formula matematica)
        for param in self.ema_model.parameters():
            param.requires_grad = False
            
    @torch.no_grad()
    def update(self, model):
        """
        Aggiorna i pesi EMA usando i pesi attuali del modello in training.
        Formula: ema_weight = decay * ema_weight + (1 - decay) * new_weight
        """
        # Itera su tutti i parametri (pesi e bias) accoppiando EMA e Modello Live
        for ema_param, current_param in zip(self.ema_model.parameters(), model.parameters()):
            
            # Update in-place per velocità
            ema_param.data.mul_(self.decay).add_(current_param.data, alpha=1 - self.decay)

    def forward(self, x, t):
        """Pass-through per usare l'EMA come un modello normale"""
        return self.ema_model(x, t)