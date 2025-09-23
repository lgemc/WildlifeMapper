"""
ResNet with FPN backbone implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
import torchvision.models as models

from .base import BaseBackbone


class ResNetFPNBackbone(BaseBackbone):
    """ResNet with Feature Pyramid Network backbone"""

    def __init__(self,
                 variant: str = "resnet50",
                 pretrained: bool = True,
                 out_channels: int = 256,
                 fpn_levels: List[int] = [1, 2, 3, 4],
                 **kwargs):
        super().__init__(**kwargs)

        self.variant = variant
        self.out_channels = out_channels
        self.fpn_levels = fpn_levels

        # Get ResNet backbone
        if variant == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            backbone_channels = [64, 128, 256, 512]
        elif variant == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            backbone_channels = [64, 128, 256, 512]
        elif variant == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            backbone_channels = [256, 512, 1024, 2048]
        elif variant == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
            backbone_channels = [256, 512, 1024, 2048]
        elif variant == "resnet152":
            resnet = models.resnet152(pretrained=pretrained)
            backbone_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown ResNet variant: {variant}")

        # Extract feature extraction layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # FPN components
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i, channels in enumerate(backbone_channels):
            if i in fpn_levels:
                lateral_conv = nn.Conv2d(channels, out_channels, 1)
                fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
                self.lateral_convs.append(lateral_conv)
                self.fpn_convs.append(fpn_conv)

        # Initialize FPN weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Extract features from ResNet backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x = self.layer1(x)
        features.append(x)  # Level 1
        x = self.layer2(x)
        features.append(x)  # Level 2
        x = self.layer3(x)
        features.append(x)  # Level 3
        x = self.layer4(x)
        features.append(x)  # Level 4

        # Apply FPN
        fpn_features = []
        for i, (feature, lateral_conv, fpn_conv) in enumerate(
                zip(reversed(features), reversed(self.lateral_convs), reversed(self.fpn_convs))):
            lateral = lateral_conv(feature)
            if i > 0:
                # Upsample and add
                prev_shape = lateral.shape[-2:]
                upsampled = F.interpolate(fpn_features[-1], size=prev_shape, mode='nearest')
                lateral = lateral + upsampled
            fpn_out = fpn_conv(lateral)
            fpn_features.append(fpn_out)

        # Return the highest resolution feature map (typically for dense prediction)
        # For SAM-like usage, we typically want the highest resolution feature
        return fpn_features[-1]

    @property
    def output_channels(self) -> int:
        return self.out_channels

    @property
    def output_stride(self) -> int:
        # ResNet typically has stride 32, but FPN can provide different strides
        # For the highest resolution FPN level, it's typically 4
        return 4

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ResNetFPNBackbone':
        return cls(**config)