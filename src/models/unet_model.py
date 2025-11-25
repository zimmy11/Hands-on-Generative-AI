import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import SinusoidalPositionEmbeddings

# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels = None, time_embed_dim=128, num_groups=32, self_attention=False, dropout=0.1):
        super().__init__()

        self.out_channels = out_channels if out_channels else in_channels
        # --- Normalization + Conv Block 1 ---
        # Use GroupNorm (LDM standard: 32 groups)
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)
        
        # --- Normalization + Conv Block 2 ---
        self.norm2 = nn.GroupNorm(num_groups, self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        
        # Activation and dropout
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        # --- Time Embedding Projection ---
        # Projects the time embedding into scale & shift vectors for FiLM modulation
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, self.out_channels * 2)
        )

        if in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

            
        # --- Attention (optional) ---
        self.use_attention = self_attention
        if self_attention:
            self.norm_attn = nn.GroupNorm(num_groups, self.out_channels)
            self.attn = nn.MultiheadAttention(self.out_channels, num_heads=4, batch_first=True)

    
    def forward(self, x, t_emb):
        h = x
        
        # --- Block 1: Pre-Normalization + Conv ---
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        # --- Inject Time Embedding via FiLM ---
        # Compute scale and shift from time embedding

        emb = self.time_proj(t_emb)[:, :, None, None]  # [B, 2C, 1, 1]
        scale, shift = emb.chunk(2, dim=1)
        
        # --- Block 2: Pre-Norm + FiLM modulation + Conv ---
        h = self.norm2(h)
        h = h * (1 + scale) + shift  # Apply FiLM (Feature-wise Linear Modulation)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # --- Residual connection ---
        out = h + self.shortcut(x)
        
        # --- Optional Self-Attention ---
        if self.use_attention:
            residual = out
            out = self.norm_attn(out)
            
            # Reshape [B, C, H, W] -> [B, Seq, C] for MultiheadAttention
            b, c, h_dim, w_dim = out.shape
            out = out.view(b, c, h_dim * w_dim).transpose(1, 2)  # [B, Seq, C]
            
            # Apply multi-head self-attention
            out, _ = self.attn(out, out, out)
            
            # Reshape back to [B, C, H, W] and add residual
            out = out.transpose(1, 2).view(b, c, h_dim, w_dim)
            out = out + residual
            
        return out



# UNet Model
class UNet(nn.Module):
    """
    LDM UNet model skeleton.
    """

    def __init__(self, in_channels = 128, out_channels = 4, num_blocks = 2, time_emb_dim = 128, features=[128, 256, 512]):
        super(UNet, self).__init__()



        #The Variational autoencoder reduces the input image of size 3x128x128 to a latent representation of size 4 x 32 x 32.
        # Input
        # 4 x 32 x 32
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_proj = SinusoidalPositionEmbeddings(dim=time_emb_dim)

        # Time Embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4), 
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim) 
        )
        self.time_emb_dim = time_emb_dim



        # Initial Convolution out = [H_in + 2*padding - dilation*(kernel_size-1) -1]/stride +1
        # Output
        # 128 x 32 x 32
        self.init_conv = nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1)

        # Encoder
        self.enc_layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()


        current_channels = features[0]

        for next_channels in features[1:]:
            level_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                use_attn = (current_channels >= 256)
                block = ResBlock(current_channels, time_embed_dim=time_emb_dim, num_groups = min(current_channels//32, 32), self_attention=use_attn)
                level_blocks.append(block)

            # Output size halved (DownSampling Layer)
            downsample = nn.Conv2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1)
            self.downsamples.append(downsample)
            self.enc_layers.append(level_blocks)
            current_channels = next_channels

        # Bottleneck
        self.bottleneck = nn.ModuleList([
        ResBlock(features[-1], time_embed_dim=time_emb_dim, num_groups = min(features[-1]//32, 32), self_attention=True),
        ResBlock(features[-1], time_embed_dim=time_emb_dim, num_groups = min(features[-1]//32, 32), self_attention=True)])


        # Decoder
        self.dec_layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        reversed_features = features[::-1]
        
        # Modified to Allow Skip Connections
        for i in range(len(reversed_features) - 1):
            level_blocks = nn.ModuleList()
            
            out_channels_level = reversed_features[i+1]

            in_channels_up = reversed_features[i]

            # UpSampling: Reduces channels while doubling spatial size.
            # We use Nearest Neighbor Upsampling
            upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels_up, out_channels_level, kernel_size=3, padding=1)
            )
            
            self.upsamples.append(upsample)
            
            # To solve the channel mismatch after concatenation with skip connections,
            # we introduce an adapter convolutional layer.
            # adapter_conv = nn.Conv2d(block_in_channels_after_skip, out_channels_level, kernel_size=1)
            # level_blocks.append(adapter_conv) # Adapter is the first operation in this decoder level

            for j in range(num_blocks):
                if j == 0:
                    block_in = out_channels_level * 2
                else: 
                    block_in = out_channels_level
                
                use_attn = (out_channels_level >= 256)
                block = ResBlock(in_channels = block_in, out_channels = out_channels_level, time_embed_dim=time_emb_dim, num_groups = min(out_channels_level//32, 32), self_attention=use_attn)
                level_blocks.append(block)
            
            self.dec_layers.append(level_blocks)


        in_channels = features[0]
        self.out_conv = nn.Sequential(
        nn.GroupNorm(num_groups= min(in_channels//32, 32), num_channels=in_channels),
        nn.SiLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        


    def forward(self, x, time):

        if time.dtype == torch.float32 and time.max() <= 1.0:
            time = time * 1000.0

        time_sin = self.time_proj(time)         
        t_emb = self.time_mlp(time_sin)


        x = self.init_conv(x)
        # skip connections
        skips = [x]

        # encoder
        for blocks, down in zip(self.enc_layers, self.downsamples):
            for blk in blocks:
                x = blk(x, t_emb)
            skips.append(x)
            x = down(x)
        
        # skip = [Encoder 1 --> 128 x 32 x 32 , Encoder 2 --> 256 x 16 x 16]
        # bottleneck
        for layer in self.bottleneck:
            x = layer(x, t_emb)

        # decoder for skip connections
        skip_iterator = reversed(skips)
        for up, blocks in zip(self.upsamples, self.dec_layers):
            x = up(x)
            
            skip = next(skip_iterator)
            
            # Interpolazione di sicurezza (raramente serve se le dimensioni sono potenze di 2)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)            
            
            # Concatenazione
            x = torch.cat([x, skip], dim=1)
            
            # Passaggio nei blocchi (ora gestiscono loro i canali)
            for blk in blocks:
                x = blk(x, t_emb)

        out = self.out_conv(x)
        return out