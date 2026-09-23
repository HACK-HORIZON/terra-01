"""
OrbitalHybridNet: CNN and Transformer Hybrid Neural Network for Orbital Image Enhancement.

Architecture Highlights:
1. Shallow Feature Extractor (CNN):
   - Convolutional layers capturing local textures, edges, and high-frequency orbital features.
2. Deep Hybrid Backbone:
   - Residual Channel Attention Blocks (RCAB - CNN): Models localized spatial-channel correlations.
   - Multi-Dconv Head Transposed Attention (MDTA - Transformer): Self-attention across channel dimensions
     with linear O(H*W) complexity, allowing high-resolution satellite imagery tiles to process efficiently.
   - Gated Feed-Forward Networks (GFFN): Depthwise convolution + GELU gating for localized feature mixture.
3. Multi-Scale Detail Fusion:
   - Long skip connections preserving low-level satellite geometry.
4. Reconstruction & Super-Resolution Head:
   - PixelShuffle sub-pixel convolution + Conv refinement layers with residual connection to bicubic baseline.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention module."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResidualCNNBlock(nn.Module):
    """Residual CNN Block with Channel Attention for local orbital feature extraction."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.ca = ChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv1(x)
        res = self.act1(res)
        res = self.conv2(res)
        res = self.ca(res)
        return x + res


class MultiDconvHeadTransposedAttention(nn.Module):
    """
    MDTA: Multi-Dconv Head Transposed Self-Attention (inspired by Restormer).
    Applies attention across channels instead of spatial tokens, yielding
    linear computational complexity O(H * W) rather than quadratic O(H^2 * W^2).
    """
    def __init__(self, dim: int, num_heads: int = 4, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head transposed attention: (B, num_heads, C/heads, H*W)
        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Cross-covariance attention map: (B, num_heads, C/heads, C/heads)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.view(b, c, h, w)
        out = self.project_out(out)
        return out


class GatedFeedForwardNetwork(nn.Module):
    """GFFN: Depthwise convolution and GELU gating for localized feature transformation."""
    def __init__(self, dim: int, ffn_expansion_factor: float = 2.0, bias: bool = False):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1, groups=hidden_dim * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1_x2 = self.dwconv(self.project_in(x))
        x1, x2 = x1_x2.chunk(2, dim=1)
        # Gated activation: x1 * GELU(x2)
        x_gate = F.gelu(x1) * x2
        out = self.project_out(x_gate)
        return out


class TransformerBlock(nn.Module):
    """Transformer block combining MDTA and GFFN with LayerNorm and residual connections."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)  # LayerNorm equivalent for 4D tensors
        self.attn = MultiDconvHeadTransposedAttention(dim, num_heads=num_heads)
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = GatedFeedForwardNetwork(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HybridGroup(nn.Module):
    """A hybrid group uniting CNN local feature extraction and Transformer global context."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.cnn_block1 = ResidualCNNBlock(dim)
        self.transformer_block = TransformerBlock(dim, num_heads=num_heads)
        self.cnn_block2 = ResidualCNNBlock(dim)
        self.fusion = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.cnn_block1(x)
        res = self.transformer_block(res)
        res = self.cnn_block2(res)
        res = self.fusion(res)
        return x + res


class PixelShuffleUpsampler(nn.Module):
    """Upsampling block utilizing sub-pixel convolution (PixelShuffle)."""
    def __init__(self, in_channels: int, out_channels: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        if scale == 2:
            self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif scale == 4:
            self.conv1 = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1)
            self.ps1 = nn.PixelShuffle(2)
            self.conv2 = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
            self.ps2 = nn.PixelShuffle(2)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            return self.pixel_shuffle(self.conv(x))
        elif self.scale == 4:
            x = F.leaky_relu(self.ps1(self.conv1(x)), 0.1)
            return self.ps2(self.conv2(x))
        return self.conv(x)


class OrbitalHybridNet(nn.Module):
    """
    OrbitalHybridNet: CNN-Transformer Hybrid Super-Resolution & Enhancement Network.

    Args:
        in_channels: Number of input spectral bands (default 3 for RGB orbital imagery).
        out_channels: Number of output spectral bands (default 3).
        dim: Intermediate feature dimension (default 48).
        num_groups: Number of stacked Hybrid Groups (default 3).
        num_heads: Transformer attention heads (default 4).
        scale: Resolution enhancement factor (default 2 for 2x super-resolution).
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_groups: int = 3,
        num_heads: int = 4,
        scale: int = 2
    ):
        super().__init__()
        self.scale = scale

        # 1. Shallow Feature Extraction (CNN)
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )

        # 2. Deep Hybrid Backbone (CNN + Transformer)
        self.groups = nn.ModuleList([
            HybridGroup(dim=dim, num_heads=num_heads) for _ in range(num_groups)
        ])
        self.conv_after_backbone = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

        # 3. High-Resolution Reconstruction Head
        self.upsampler = PixelShuffleUpsampler(dim, dim, scale=scale)
        self.conv_final = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base bicubic upsample for global residual learning
        if self.scale > 1:
            base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        else:
            base = x

        # Shallow feature extraction
        f_shallow = self.shallow_conv(x)

        # Deep hybrid processing with long skip connection
        f_deep = f_shallow
        for group in self.groups:
            f_deep = group(f_deep)
        f_deep = self.conv_after_backbone(f_deep) + f_shallow

        # Upsampling and detail reconstruction
        f_up = self.upsampler(f_deep)
        residual_detail = self.conv_final(f_up)

        # Enhanced output = Bicubic Base + Learned Orbital Residual Detail
        out = base + residual_detail
        return torch.clamp(out, 0.0, 1.0)


def create_model(scale: int = 2, pretrained: bool = False, device: str = "cpu") -> OrbitalHybridNet:
    """Factory function to instantiate and initialize OrbitalHybridNet."""
    model = OrbitalHybridNet(scale=scale)
    model = model.to(device)
    model.eval()
    return model
