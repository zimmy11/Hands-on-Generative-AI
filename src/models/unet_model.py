import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import SinusoidalPositionEmbeddings

# Residual Block
class ResBlock(nn.Module):
    """
    A simple Residual Block with two convolutional layers.
    """

    def __init__(self, channels, time_embed_dim = 128, num_groups=8, self_attention=False):
        super(ResBlock, self).__init__()

        # Time Embedding
        self.time_proj1 = nn.Linear(time_embed_dim, channels * 2)
        self.time_proj2 = nn.Linear(time_embed_dim, channels * 2)

        # Input Number of channels (128, 256, 512) x h (16, 32) x w (16, 32)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.act = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups = num_groups, num_channels = channels)
        
        # Output channels = Input channels 

        # Self-attention layer (optional)
        if self_attention:
            self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=4)


    def forward(self, x, t_emb):
        identity = x
        # Adaptive Normalization with Time Embedding

        # Time embedding 1 projection 
        t_proj_1 = self.time_proj1(t_emb).chunk(2, dim=-1)
        gamma1, beta1 = t_proj_1[0].unsqueeze(-1).unsqueeze(-1), t_proj_1[1].unsqueeze(-1).unsqueeze(-1)




        out = self.conv1(x)
        out = self.norm1(out) * (1 + gamma1) + beta1
        out = self.act(out)

        # Time embedding 2 projection 
        t_proj_2 = self.time_proj2(t_emb).chunk(2, dim=-1)
        gamma2, beta2 = t_proj_2[0].unsqueeze(-1).unsqueeze(-1), t_proj_2[1].unsqueeze(-1).unsqueeze(-1)

        out = self.conv2(out)
        out = self.norm2(out) * (1 + gamma2) + beta2
        out = self.act(out)

     
        
        if hasattr(self, "attention"):
            b, c, h, w = out.size()
            out_reshaped = out.view(b, c, h * w).permute(2, 0, 1)  # (h*w, b, c)
            out_attended, _ = self.attention(out_reshaped, out_reshaped, out_reshaped)
            out = out_attended.permute(1, 2, 0).view(b, c, h, w)

        out += identity # Residual connection

        return out



# UNet Model
class UNet(nn.Module):
    """
    LDM UNet model skeleton.
    """

    def __init__(self, in_channels, out_channels, num_blocks = 2, time_emb_dim = 128, features=[128, 256, 512]):
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
                block = ResBlock(current_channels, time_embed_dim=time_emb_dim, num_groups = min(current_channels//32, 32))
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
            upsample = nn.ConvTranspose2d(in_channels_up, out_channels_level, kernel_size=4, stride=2, padding=1)
            self.upsamples.append(upsample)
            
            # After UpSampling + Skip Connection, the channel count will be N_current * 2
            block_in_channels_after_skip = out_channels_level * 2

            
            # To solve the channel mismatch after concatenation with skip connections,
            # we introduce an adapter convolutional layer.
            adapter_conv = nn.Conv2d(block_in_channels_after_skip, out_channels_level, kernel_size=1)
            level_blocks.append(adapter_conv) # Adapter is the first operation in this decoder level

            for _ in range(num_blocks):

                block = ResBlock(out_channels_level, time_embed_dim=time_emb_dim, num_groups = min(out_channels_level//32, 32))
                level_blocks.append(block)
            
            self.dec_layers.append(level_blocks)


        in_channels = features[0]
        self.out_conv = nn.Sequential(
        nn.GroupNorm(num_groups= min(in_channels//32, 32), num_channels=in_channels),
        nn.SiLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        


    def forward(self, x, time):

        time_sin = self.time_proj(time)         
        t_emb = self.time_mlp(time_sin)
        x = self.init_conv(x)

        # skip connections
        skips = []

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
        for up, blocks, skip in zip(self.upsamples, self.dec_layers, reversed(skips)):
            x = up(x)
            # # # if shapes mismatch due to odd sizes, center-crop skip
            if x.shape[-2:] != skip.shape[-2:]:
                    # simple interpolate to match
                    #_, _, h, w = x.shape
                x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
            
            #concat along channels
            x = torch.cat([x, skip], dim=1)
            for blk in blocks:
                if isinstance(blk, nn.Conv2d):
                    # Adapter conv
                    x = blk(x)
                else:
                    x = blk(x, t_emb)

        # final conv
        out = self.out_conv(x)


        return out