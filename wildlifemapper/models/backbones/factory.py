"""
Backbone factory for creating different backbone architectures
"""

from typing import Dict, Any
from .base import BaseBackbone
from .vit_sam import ViTSAMBackbone
from .resnet_fpn import ResNetFPNBackbone


# Registry of available backbones
BACKBONE_REGISTRY = {
    # ViT-SAM variants
    "vit_h": ViTSAMBackbone,
    "vit_l": ViTSAMBackbone,
    "vit_b": ViTSAMBackbone,

    # ResNet-FPN variants
    "resnet18_fpn": ResNetFPNBackbone,
    "resnet34_fpn": ResNetFPNBackbone,
    "resnet50_fpn": ResNetFPNBackbone,
    "resnet101_fpn": ResNetFPNBackbone,
    "resnet152_fpn": ResNetFPNBackbone,
}

AVAILABLE_BACKBONES = list(BACKBONE_REGISTRY.keys())


def get_backbone(backbone_type: str, **kwargs) -> BaseBackbone:
    """
    Factory function to create backbone models

    Args:
        backbone_type: Type of backbone to create
        **kwargs: Additional arguments for backbone initialization

    Returns:
        Configured backbone model

    Raises:
        ValueError: If backbone_type is not supported
    """
    if backbone_type not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone type: {backbone_type}. "
            f"Available backbones: {AVAILABLE_BACKBONES}"
        )

    backbone_class = BACKBONE_REGISTRY[backbone_type]

    # For ViT variants, pass the variant name
    if backbone_type.startswith("vit_"):
        kwargs["variant"] = backbone_type
    elif backbone_type.endswith("_fpn"):
        # Extract ResNet variant from name (e.g., "resnet50_fpn" -> "resnet50")
        resnet_variant = backbone_type.replace("_fpn", "")
        kwargs["variant"] = resnet_variant

    return backbone_class.from_config(kwargs)


def register_backbone(name: str, backbone_class: type):
    """
    Register a new backbone class

    Args:
        name: Name to register the backbone under
        backbone_class: Backbone class that inherits from BaseBackbone
    """
    if not issubclass(backbone_class, BaseBackbone):
        raise ValueError(f"Backbone class must inherit from BaseBackbone")

    BACKBONE_REGISTRY[name] = backbone_class
    if name not in AVAILABLE_BACKBONES:
        AVAILABLE_BACKBONES.append(name)