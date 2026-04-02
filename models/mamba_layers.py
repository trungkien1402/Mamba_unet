"""
Patch operations cho Mamba-UNet
"""

import torch
import torch.nn as nn


class PatchPartition(nn.Module):
    """
    Patch Partition - Chia ảnh thành patches

    Input:  (B, C, H, W)
    Output: (B, embed_dim, H/patch_size, W/patch_size)
    """

    def __init__(self, in_chans=1, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_size = patch_size

        # Conv stride = patch_size để chia patch
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        x = self.proj(x)                       # (B, embed_dim, H/P, W/P)
        B, C, H, W = x.shape

        # reshape để áp dụng LayerNorm theo channel
        x = x.flatten(2).transpose(1, 2)       # (B, H*W, C)
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)

        return x


class PatchMerging(nn.Module):
    """
    Patch Merging - Downsampling

    Input:  (B, C, H, W)
    Output: (B, 2C, H/2, W/2)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.reduction = nn.Conv2d(
            dim,
            2 * dim,
            kernel_size=2,
            stride=2
        )

        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        x = self.reduction(x)                  # (B, 2C, H/2, W/2)
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)       # (B, H*W, C)
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)

        return x


class PatchExpanding(nn.Module):
    """
    Patch Expanding - Upsampling

    Input:  (B, C, H, W)
    Output: (B, C/2, H*2, W*2)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.expand = nn.ConvTranspose2d(
            dim,
            dim // 2,
            kernel_size=2,
            stride=2
        )

        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        x = self.expand(x)                     # (B, C/2, H*2, W*2)
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)       # (B, H*W, C)
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)

        return x


class LinearProjection(nn.Module):
    """
    Linear Projection - Giảm channels trước khi expand
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        return self.proj(x)