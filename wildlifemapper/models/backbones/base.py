"""
Base backbone interface for WildlifeMapper
"""

from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict, Any, Tuple


class BaseBackbone(ABC, nn.Module):
    """Base class for all backbone implementations"""

    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs

    @abstractmethod
    def forward(self, x):
        """Forward pass through the backbone

        Args:
            x: Input tensor

        Returns:
            Feature map(s) from the backbone
        """
        pass

    @property
    @abstractmethod
    def output_channels(self) -> int:
        """Number of output channels from the backbone"""
        pass

    @property
    @abstractmethod
    def output_stride(self) -> int:
        """Output stride of the backbone (input_size / output_size)"""
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseBackbone':
        """Create backbone from configuration dictionary"""
        pass