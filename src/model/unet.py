import torch
import torch.nn as nn

from src.model.layers import ConvDownBlock, \
    AttentionDownBlock, \
    AttentionUpBlock, \
    TransformerPositionalEmbedding, \
    ConvUpBlock


class UNet(nn.Module):
    """
    Model architecture as described in the DDPM paper, Appendix, section B
    """

    def __init__(self, image_size=256, input_channels=3):
        super().__init__()
        # 1. We replaced weight normalization with group normalization
        # 2. Our 32x32 models use four feature map resolutions (32x32 to 4x4), and our 256x256 models use six (I made 5)
        # 3. Two convolutional residual blocks per resolution level and self-attention blocks at the 16x16 resolution
        # between the convolutional blocks [https://arxiv.org/pdf/1712.09763.pdf]
        # 4. Diffusion time t is specified by adding the Transformer sinusoidal position embedding into
        # each residual block [https://arxiv.org/pdf/1706.03762.pdf]

        self.initial_conv = nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.positional_encoding = nn.Sequential(
            TransformerPositionalEmbedding(dimension=128),
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4)
        )

        self.downsample_blocks = nn.ModuleList([
            ConvDownBlock(in_channels=128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            ConvDownBlock(in_channels=128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            ConvDownBlock(in_channels=128, out_channels=256, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            AttentionDownBlock(in_channels=256, out_channels=256, num_layers=2, num_att_heads=4, num_groups=32, time_emb_channels=128 * 4),
            ConvDownBlock(in_channels=256, out_channels=256, num_layers=2, num_groups=32, time_emb_channels=128 * 4)
        ])

        self.bottleneck = AttentionDownBlock(in_channels=256, out_channels=256, num_layers=2, num_att_heads=4, num_groups=32, time_emb_channels=128*4, downsample=False)                                                                                                  # 16x16x256 -> 16x16x256

        self.upsample_blocks = nn.ModuleList([
            ConvUpBlock(in_channels=256 + 256, out_channels=256, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            AttentionUpBlock(in_channels=256 + 256, out_channels=256, num_layers=2, num_att_heads=4, num_groups=32, time_emb_channels=128 * 4),
            ConvUpBlock(in_channels=256 + 256, out_channels=256, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            ConvUpBlock(in_channels=256 + 128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            ConvUpBlock(in_channels=128 + 128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4)
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
        # Get initial shape [3, 256, 256] with convolutions
        out = self.output_conv(x)

        return out
