"""
Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation
FIXED VERSION - Hoàn chỉnh theo paper
"""
import torch
import torch.nn as nn
from .vss_block import VSSBlock
from .mamba_layers import PatchPartition, PatchMerging, PatchExpanding, LinearProjection

class EncoderStage(nn.Module):
    """
    Encoder Stage: Linear Embedding → VSS Block ×N → Patch Merging
    """
    def __init__(self, dim, depth=2, drop_path=0., downsample=True):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # Linear Embedding (ở đầu mỗi stage)
        self.linear_embed = nn.Conv2d(dim, dim, kernel_size=1)
        
        # VSS Blocks
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim, drop_path=drop_path)
            for _ in range(depth)
        ])
        
        # Patch Merging (Downsample)
        self.downsample = PatchMerging(dim) if downsample else None
        
    def forward(self, x):
        """x: (B, C, H, W)"""
        # Linear Embedding
        x = self.linear_embed(x)
        
        # VSS Blocks
        for blk in self.blocks:
            x = blk(x)
        
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x


class DecoderStage(nn.Module):
    """
    Decoder Stage: Patch Expanding → Concat → Linear Projection → VSS Blocks
    """
    def __init__(self, in_dim, skip_dim, out_dim, depth=2, drop_path=0.):
        super().__init__()

        self.upsample = PatchExpanding(in_dim)

        self.linear_proj = LinearProjection(
            in_dim=skip_dim + in_dim // 2,
            out_dim=out_dim
        )

        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=out_dim, drop_path=drop_path)
            for _ in range(depth)
        ])

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.linear_proj(x)

        for blk in self.blocks:
            x = blk(x)

        return x


class MambaUNet(nn.Module):
    """
    Mamba-UNet Architecture
    """
    def __init__(
        self,
        img_size=512,
        in_chans=1,
        num_classes=2,
        embed_dim=32,
        depths=[2, 2, 2, 1],
        drop_path_rate=0.2,
        patch_size=4,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        
        # PATCH PARTITION
        self.patch_partition = PatchPartition(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        
        # ENCODER
        self.encoder_stages = nn.ModuleList()
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(self.num_stages):
            dim = int(embed_dim * 2 ** i)
            stage = EncoderStage(
                dim=dim,
                depth=depths[i],
                drop_path=dpr[cur:cur + depths[i]],
                downsample=(i < self.num_stages - 1)
            )
            self.encoder_stages.append(stage)
            cur += depths[i]
        
        # BOTTLENECK
        bottleneck_dim = int(embed_dim * 2 ** (self.num_stages - 1))
        self.bottleneck = nn.Sequential(
            VSSBlock(hidden_dim=bottleneck_dim, drop_path=0.),
            VSSBlock(hidden_dim=bottleneck_dim, drop_path=0.)
        )
        
        # DECODER
        self.decoder_stages = nn.ModuleList()
        encoder_dims = [int(embed_dim * 2 ** i) for i in range(self.num_stages)]
        decoder_dims = encoder_dims[::-1]

        for i in range(len(decoder_dims) - 1):
            in_dim = decoder_dims[i]
            skip_dim = decoder_dims[i + 1]
            out_dim = decoder_dims[i + 1]

            stage = DecoderStage(
                in_dim=in_dim,
                skip_dim=skip_dim,
                out_dim=out_dim,
                depth=depths[self.num_stages - 2 - i],
                drop_path=0.
            )
            self.decoder_stages.append(stage)
        
        # FINAL LAYERS
        self.final_expand = nn.ConvTranspose2d(
            embed_dim, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        self.seg_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        x: (B, 1, H, W)
        Returns: (B, num_classes, H, W)
        """
        # PATCH PARTITION
        x = self.patch_partition(x)
        
        # ENCODER
        skip_connections = []
        
        for stage in self.encoder_stages:
            x_skip, x = stage(x)
            skip_connections.append(x_skip)
        
        # BOTTLENECK
        x = self.bottleneck(x)
        
        # DECODER
        skip_connections = skip_connections[:-1]
        
        for i, stage in enumerate(self.decoder_stages):
            skip = skip_connections[-(i + 1)]
            x = stage(x, skip)
        
        # FINAL OUTPUT
        x = self.final_expand(x)
        x = self.seg_head(x)
        
        return x


def create_mamba_unet(
    in_chans=1, 
    num_classes=2, 
    img_size=512,
    embed_dim=96,         
    depths=[2, 2, 2, 2],   
    drop_path_rate=0.2,
    patch_size=4
    ):
    """Factory function"""
    model = MambaUNet(
        img_size=img_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depths=depths,
        drop_path_rate=drop_path_rate,
        patch_size= patch_size
    )
    return model
