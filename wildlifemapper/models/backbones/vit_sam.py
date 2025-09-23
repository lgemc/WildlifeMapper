"""
ViT-based SAM backbone implementation
"""

import torch
import torch.nn as nn
from functools import partial
from typing import Dict, Any

from .base import BaseBackbone
from wildlifemapper.segment_anything.modeling.image_encoder import ImageEncoderViT


class ViTSAMBackbone(BaseBackbone):
    """Vision Transformer backbone from SAM"""

    def __init__(self,
                 variant: str = "vit_h",
                 img_size: int = 1024,
                 patch_size: int = 16,
                 out_chans: int = 256,
                 **kwargs):
        super().__init__(**kwargs)

        self.variant = variant
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_chans = out_chans

        # Define variant configurations
        variant_configs = {
            "vit_h": {
                "embed_dim": 1280,
                "depth": 32,
                "num_heads": 16,
                "global_attn_indexes": [7, 15, 23, 31],
            },
            "vit_l": {
                "embed_dim": 1024,
                "depth": 24,
                "num_heads": 16,
                "global_attn_indexes": [5, 11, 17, 23],
            },
            "vit_b": {
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12,
                "global_attn_indexes": [2, 5, 8, 11],
            },
        }

        if variant not in variant_configs:
            raise ValueError(f"Unknown ViT variant: {variant}. Available: {list(variant_configs.keys())}")

        config = variant_configs[variant]

        self.encoder = ImageEncoderViT(
            depth=config["depth"],
            embed_dim=config["embed_dim"],
            img_size=img_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=config["num_heads"],
            patch_size=patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=config["global_attn_indexes"],
            window_size=14,
            out_chans=out_chans,
        )

    def forward(self, x):
        return self.encoder(x, x_hfc=None)

    @property
    def output_channels(self) -> int:
        return self.out_chans

    @property
    def output_stride(self) -> int:
        return self.patch_size

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ViTSAMBackbone':
        return cls(**config)