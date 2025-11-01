import torch
import torch.nn as nn



class ResBlock(nn.Module):
    """
    A simple Residual Block with two convolutional layers.
    """

    def __init__(self, channels, num_groups=8, self_attention=False):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, channels=channels)
        self.act1 = nn.SELU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups = num_groups, channels = channels)
        self.act2 = nn.SELU(inplace=True)

        # Self-attention layer (optional)
        if self_attention:
            self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=4)


    def forward(self, x, self_attention=False):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self_attention:
            b, c, h, w = out.size()
            out_reshaped = out.view(b, c, h * w).permute(2, 0, 1)  # (h*w, b, c)
            out_attended, _ = self.attention(out_reshaped, out_reshaped, out_reshaped)
            out = out_attended.permute(1, 2, 0).view(b, c, h, w)

        out += identity # Residual connection
        out = self.act2(out)
        return out


class UNet(nn.Module):
    """
    UNet model skeleton.

    This file contains only imports and the class skeleton without layer definitions.
    Fill in the encoder/decoder blocks and the forward method as needed.
    """

    def __init__(self, in_channels, out_channels, num_blocks, features=[128, 256, 512]):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels


        self.init_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Encoder
        self.enc_layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for feature in features:
            level_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                block = nn.Sequential(
                ResBlock(feature, num_groups = min(32, feature)))
                level_blocks.append(block)
            
            downsample = nn.Conv2d(feature, feature, kernel_size=3, stride=2, padding=1)
            self.downsamples.append(downsample)
            self.enc_layers.append(level_blocks)
            in_channels = feature

        # Bottleneck
        self.bottleneck = nn.Sequential(
        ResBlock(features[-1], num_groups= min(32, feature), self_attention=True),
        ResBlock(features[-1], num_groups = min(32, feature), self_attention=True))


        # Decoder
        self.dec_layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for feature in reversed(features):
            level_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                block = nn.Sequential(
                ResBlock(feature, num_groups = min(32, feature)))
                level_blocks.append(block)
            
            upsample = nn.Conv2d(feature, feature, kernel_size=4, stride=2, padding=1)
            self.upsamples.append(upsample)
            self.dec_layers.append(level_blocks)
            in_channels = feature


        self.out_conv = nn.Sequential(
        nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels),
        nn.SiLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        


    def forward(self, x):
        """
        Forward pass.

        """

        x = self.init_conv(x)


        # encoder
        skips = []

        for blocks, down in zip(self.enc_layers, self.downsamples):
            for blk in blocks:
                x = blk(x)
            skips.append(x)
            x = down(x)

        # bottleneck
        for layer in self.bottleneck:
            x = layer(x)

        # decoder (use reversed skips) # for skip connections
        for up, blocks, skip in zip(self.upsamples, self.dec_layers, reversed(skips)):
            x = up(x)
            # if shapes mismatch due to odd sizes, center-crop skip
            # if x.shape[-2:] != skip.shape[-2:]:
            #     # simple crop/pad to match
            #     _, _, h, w = x.shape
            #     skip = center_crop(skip, h, w)
            # concat along channels
            x = torch.cat([x, skip], dim=1)
            for blk in blocks:
                x = blk(x)

        # final conv
        out = self.out_conv(x)
        return out
