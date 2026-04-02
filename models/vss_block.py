"""
Visual State Space (VSS) Block - FIXED VERSION
Theo Figure 3 trong paper với 3 skip connections đầy đủ
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange


# =========================================================
# Selective Scan 2D
# =========================================================
class SS2D(nn.Module):
    """
    Selective Scan 2D - Xử lý ảnh 2D với Mamba

    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2):
        super().__init__()

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Flatten spatial dimensions
        x_flat = rearrange(x, 'b c h w -> b (h w) c')

        # Apply Mamba
        x_out = self.mamba(x_flat)  # (B, H*W, C)

        # Reshape back to image
        x_out = rearrange(x_out, 'b (h w) c -> b c h w', h=H, w=W)

        return x_out


# =========================================================
# DropPath (Stochastic Depth)
# =========================================================
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


# =========================================================
# VSS Block
# =========================================================
class VSSBlock(nn.Module):
    """
    Visual State Space Block với 3 skip connections

    Architecture (theo Figure 3):

    Input
      ↓
    LN → SS2D → (+) ← Input (Skip 1)
      ↓
    DWCNN → (+) ← Previous (Skip 2)
      ↓
    LN → MLP → (+) ← Previous (Skip 3)
      ↓
    Output
    """

    def __init__(self, hidden_dim, drop_path=0., d_state=16):
        super().__init__()

        # -------------------------
        # Branch 1: LN + SS2D
        # -------------------------
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ss2d = SS2D(d_model=hidden_dim, d_state=d_state)

        # -------------------------
        # Branch 2: Depthwise CNN
        # -------------------------
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
                bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

        # -------------------------
        # Branch 3: LN + MLP
        # -------------------------
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # -------------------------
        # DropPath
        # -------------------------
        if isinstance(drop_path, (list, tuple)):
            drop_path = float(drop_path[0])

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        # =====================================================
        # Branch 1: LN + SS2D + Skip
        # =====================================================
        shortcut_1 = x

        x = rearrange(x, 'b c h w -> b h w c')
        x = self.ln_1(x)
        x = rearrange(x, 'b h w c -> b c h w')

        x = self.ss2d(x)

        x = shortcut_1 + self.drop_path(x)

        # =====================================================
        # Branch 2: DWCNN + Skip
        # =====================================================
        shortcut_2 = x

        x = self.dwconv(x)

        x = shortcut_2 + self.drop_path(x)

        # =====================================================
        # Branch 3: LN + MLP + Skip
        # =====================================================
        shortcut_3 = x

        x = rearrange(x, 'b c h w -> b h w c')
        x = self.ln_2(x)
        x = self.mlp(x)
        x = rearrange(x, 'b h w c -> b c h w')

        x = shortcut_3 + self.drop_path(x)

        return x