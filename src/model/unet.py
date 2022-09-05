import torch
import torch.nn as nn

from src.model.layers import ConvDownBlock, AttentionDownBlock, TransformerPositionalEmbedding, Bottleneck, ConvUpBlock


class UNet(nn.Module):
    """
    Model architecture as described in the DDPM paper, Appendix, section B
    """

    def __init__(self):
        super().__init__()
        # 1. We replaced weight normalization with group normalization
        # 2. Our 32x32 models use four feature map resolutions (32x32 to 4x4), and our 256x256 models use six (I made 5)
        # 3. Two convolutional residual blocks per resolution level and self-attention blocks at the 16x16 resolution
        # between the convolutional blocks [https://arxiv.org/pdf/1712.09763.pdf]
        # 4. Diffusion time t is specified by adding the Transformer sinusoidal position embedding into
        # each residual block [https://arxiv.org/pdf/1706.03762.pdf]
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.positional_encoding = nn.Sequential(
            TransformerPositionalEmbedding(dimension=128),
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4)
        )

        self.downsample_blocks = nn.ModuleList([
            ConvDownBlock(in_channels=128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128*4),                     # 256x256x128 -> 128x128x128
            ConvDownBlock(in_channels=128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128*4),                     # 128x128x128 -> 64x64x128
            ConvDownBlock(in_channels=128, out_channels=256, num_layers=2, num_groups=32, time_emb_channels=128*4),                     # 64x64x128 -> 32x32x256
            # TODO implement Attention layer
            #AttentionDownBlock(in_channels=256, out_channels=256, num_layers=2, num_groups=32, time_emb_channels=128*4),                # 32x32x256 -> 16x16x256
            ConvDownBlock(in_channels=256, out_channels=256, num_layers=2, num_groups=32, downsample=True, time_emb_channels=128*4)     # 16x16x256 -> 16x16x256
        ])

        # TODO implement Bottleneck with Attention layer
        self.bottleneck = ConvDownBlock(in_channels=256, out_channels=256, num_layers=1, num_groups=32, downsample=False, time_emb_channels=128*4) #Bottleneck()                                                                                                  # 16x16x256 -> 16x16x256

        self.upsample_blocks = nn.ModuleList([
            ConvUpBlock(in_channels=256 + 256, out_channels=256, num_layers=2, num_groups=32, time_emb_channels=128 * 4, upsample=True),
            ConvUpBlock(in_channels=256 + 256, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4, upsample=True),
            ConvUpBlock(in_channels=128 + 128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4, upsample=True),
            # AttentionDownBlock(in_channels=256, out_channels=256, num_layers=2, num_groups=32, time_emb_channels=128*4),
            ConvUpBlock(in_channels=128 + 128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4, upsample=True)
        ])

        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_channels=256, num_groups=32),
            nn.SiLU(),
            nn.Conv2d(256, 3, 3, padding=1)
        )

    def forward(self, input_tensor, time):
        time_encoded = self.positional_encoding(time)

        initial_x = self.initial_conv(input_tensor)

        states_for_skip_connections = [initial_x]

        x = initial_x
        for i, block in enumerate(self.downsample_blocks):
            x = block(x, time_encoded)
            states_for_skip_connections.append(x)
        states_for_skip_connections = list(reversed(states_for_skip_connections))

        x = self.bottleneck(x, time_encoded)

        for i, (block, skip) in enumerate(zip(self.upsample_blocks, states_for_skip_connections)):
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_encoded)

        # Concat initial_conv with tensor
        x = torch.cat([x, states_for_skip_connections[-1]], dim=1)
        # Get initial shape with convolutions
        out = self.output_conv(x)

        return out
