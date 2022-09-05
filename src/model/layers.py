import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerPositionalEmbedding(nn.Module):
    """
    From paper "Attention Is All You Need", section 3.5
    """
    def __init__(self, dimension, max_timesteps=1000):
        super(TransformerPositionalEmbedding, self).__init__()
        assert dimension % 2 == 0, "Embedding dimension must be even"
        self.dimension = dimension
        self.pe_matrix = torch.zeros(max_timesteps, dimension)
        # Gather all the even dimensions across the embedding vector
        even_indices = torch.arange(0, self.dimension, 2)
        # Calculate the term using log transforms for faster calculations
        # (https://stackoverflow.com/questions/17891595/pow-vs-exp-performance)
        log_term = torch.log(torch.tensor(10000.0)) / self.dimension
        div_term = torch.exp(even_indices * -log_term)

        # Precompute positional encoding matrix based on odd/even timesteps
        timesteps = torch.arange(max_timesteps).unsqueeze(1)
        self.pe_matrix[:, 0::2] = torch.sin(timesteps * div_term)
        self.pe_matrix[:, 1::2] = torch.cos(timesteps * div_term)

    def forward(self, timestep):
        # [bs, d_model]
        print("PE:", self.pe_matrix[timestep].shape)
        return self.pe_matrix[timestep]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=padding)

    def forward(self, input_tensor):
        x = self.conv(input_tensor)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2.0):
        super(UpsampleBlock, self).__init__()

        self.scale = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, input_tensor):
        # align_corners=True for potential convertibility to ONNX
        x = F.interpolate(input_tensor, scale_factor=self.scale, mode="bilinear", align_corners=True)
        x = self.conv(x)
        return x

class ConvDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, time_emb_channels, num_groups, downsample=True):
        super(ConvDownBlock, self).__init__()
        resnet_blocks = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnet_block = ResNetBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       time_emb_channels=time_emb_channels,
                                       num_groups=num_groups)
            resnet_blocks.append(resnet_block)

        self.resnet_blocks = nn.ModuleList(resnet_blocks)

        self.downsample = DownsampleBlock(in_channels=out_channels, out_channels=out_channels, stride=2, padding=1) \
            if downsample \
            else None

    def forward(self, input_tensor, time_embedding):
        x = input_tensor
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, time_embedding)
        if self.downsample:
            x = self.downsample(x)
        return x


class ConvUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, time_emb_channels, num_groups, upsample=True):
        super(ConvUpBlock, self).__init__()
        resnet_blocks = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnet_block = ResNetBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       time_emb_channels=time_emb_channels,
                                       num_groups=num_groups)
            resnet_blocks.append(resnet_block)

        self.resnet_blocks = nn.ModuleList(resnet_blocks)

        self.upsample = UpsampleBlock(in_channels=out_channels, out_channels=out_channels) \
            if upsample \
            else None

    def forward(self, input_tensor, time_embedding):
        x = input_tensor
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, time_embedding)
        if self.upsample:
            x = self.upsample(x)
        return x


class AttentionDownBlock(nn.Module):
    def __init__(self, **args):
        super(AttentionDownBlock, self).__init__()
        raise NotImplementedError


class Bottleneck(nn.Module):
    def __init__(self, **args):
        super(Bottleneck, self).__init__()
        raise NotImplementedError


class ResNetBlock(nn.Module):
    """
    In the original DDPM paper Wide ResNet was used
    (https://arxiv.org/pdf/1605.07146.pdf).
    """
    def __init__(self, in_channels, out_channels, *, time_emb_channels=None, num_groups=8):
        super(ResNetBlock, self).__init__()
        self.time_embedding_projectile = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_channels, out_channels))
            if time_emb_channels
            else None
        )

        self.block1 = ConvBlock(in_channels, out_channels, groups=num_groups)
        self.block2 = ConvBlock(out_channels, out_channels, groups=num_groups)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_embedding=None):
        input_tensor = x
        h = self.block1(x)
        # According to authors implementations, they inject timestep embedding into the network
        # using MLP after the first conv block in all the ResNet blocks
        # (https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L49)

        time_emb = self.time_embedding_projectile(time_embedding)
        time_emb = time_emb[:, :, None, None]
        x = time_emb + h

        x = self.block2(x)
        return x + self.residual_conv(input_tensor)


class SelfAttentionBlock(nn.Module):
    def __init__(self):
        super(SelfAttentionBlock, self).__init__()
        raise NotImplementedError
