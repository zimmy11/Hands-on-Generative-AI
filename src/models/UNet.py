import torch.nn as nn
import torch
from .components import ResBlock, SinusoidalPositionEmbeddings
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, 
                 in_channels=4, 
                 model_channels=16,
                 out_channels=4, 
                 channel_mults=(1, 2),
                 num_res_blocks=2,
                 dropout=0.0,
                 num_attributes=40):
        super().__init__()
        
        self.model_channels = model_channels
        self.time_embed_dim = model_channels * 4
        
        # 1. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )


        # # Null embedding for unconditional case MODIFICA
        self.null_embedding = nn.Parameter(torch.zeros(1, num_attributes))

        # Attribute encoding
        self.attribute_encoder = nn.Sequential(
            nn.Linear(num_attributes, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

  
        # 2. Input Convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # 3. Downsampling (Encoder)
        self.down_blocks = nn.ModuleList()
        current_channels = model_channels
        # Skip connections storage logic
        self.skips_config = [] 
        
        
        # We do not need to concatenate in self.down_blocks also the self.attributes because they are passed later in the forwards
        ds = 1
        for level, mult in enumerate(channel_mults):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):

                is_attn = (level >= len(channel_mults) - 2) 
                
                self.down_blocks.append(ResBlock(
                    in_channels=current_channels,
                    out_channels=out_ch,
                    time_embed_dim=self.time_embed_dim,
                    use_attention=is_attn,
                    dropout=dropout
                ))
                self.skips_config.append(out_ch)
                current_channels = out_ch

                
            # Downsample 
            if level != len(channel_mults) - 1:
                self.down_blocks.append(nn.Conv2d(current_channels, current_channels, 3, stride=2, padding=1))
                ds *= 2

        # 4. Bottleneck (Mid Block)
        self.mid_block1 = ResBlock(current_channels, current_channels, self.time_embed_dim, use_attention=True, dropout=dropout)
        self.mid_block2 = ResBlock(current_channels, current_channels, self.time_embed_dim, use_attention=True, dropout=dropout) # Modficato!!
        
        # 5. Upsampling (Decoder)
        self.up_blocks = nn.ModuleList()
        reversed_mults = list(reversed(channel_mults))
        
        for level, mult in enumerate(reversed_mults):
            out_ch = model_channels * mult
            
            
            for _ in range(num_res_blocks):
                skip_ch = self.skips_config.pop()
                
                is_attn = (level <= 1) 
                
                # Input channels = current + skip
                self.up_blocks.append(ResBlock(
                    in_channels=current_channels + skip_ch,
                    out_channels=out_ch,
                    time_embed_dim=self.time_embed_dim,
                    use_attention=is_attn,
                    dropout=dropout
                ))
                current_channels = out_ch
                
            # Upsample 
            if level != len(channel_mults) - 1:
                self.up_blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))
                ds //= 2

        # 6. Final Output
        self.out_norm = nn.GroupNorm(8, current_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t, labels = None, cond_mask = None):
        # Time Embedding
        t_emb = self.time_mlp(t)

        # Attribute embedding
        y_emb = self.attribute_encoder(labels.float())
        if cond_mask is not None:
            null_emb = self.attribute_encoder(self.null_embedding)
            y_emb = cond_mask[:, None] * y_emb + (1 - cond_mask[:, None]) * null_emb
        
        cond_emb = t_emb + y_emb

        # Initial Conv
        x = self.input_conv(x)
        
        # Store Skips
        skips = []
        
        # --- Encoder ---
        for i, layer in enumerate(self.down_blocks):
            if isinstance(layer, ResBlock):
                x = layer(x, cond_emb)
                skips.append(x)

            else: # Downsample Conv
                x = layer(x)

        
        # --- Bottleneck ---
        x = self.mid_block1(x, cond_emb)
        x = self.mid_block2(x, cond_emb)
        
        
        # --- Decoder ---
        for i, layer in enumerate(self.up_blocks):
            if isinstance(layer, ResBlock):
                # Recover skip
                skip = skips.pop()
                
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

                x = torch.cat([x, skip], dim=1)

                x = layer(x, cond_emb)
            else: # Upsample
                x = layer(x)


        # Final
        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.out_conv(x)

        
        return x
