"""
Focal Loss implementation for WildlifeMapper.

Adapted from the original Focal Loss paper (https://arxiv.org/abs/1708.02002)
and HerdNet implementation for classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in object detection.

    This implementation is adapted for multi-class classification tasks
    in WildlifeMapper, focusing on the classification component of the
    detection pipeline.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        """
        Args:
            alpha (torch.Tensor, optional): Weighting factor for rare class (default: None).
                If provided, should be a tensor of shape (num_classes,) with weights for each class.
            gamma (float): Focusing parameter (default: 2.0). Higher gamma puts more focus on hard examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            ignore_index (int): Specifies a target value that is ignored and does not contribute to the gradient.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Predicted logits of shape (N, C) where N is batch size and C is number of classes
            targets (torch.Tensor): Ground truth labels of shape (N,)

        Returns:
            torch.Tensor: Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)

        # Compute probabilities
        pt = torch.exp(-ce_loss)

        # Compute focal weight (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            # Create alpha tensor for current batch
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SigmoidFocalLoss(nn.Module):
    """
    Sigmoid Focal Loss for multi-label classification or when using sigmoid activation.

    This is useful for cases where multiple classes can be present simultaneously.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha (float): Weighting factor for positive examples (default: 0.25)
            gamma (float): Focusing parameter (default: 2.0)
            reduction (str): Specifies the reduction to apply to the output
        """
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Predicted logits
            targets (torch.Tensor): Ground truth labels (same shape as inputs)

        Returns:
            torch.Tensor: Sigmoid focal loss value
        """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)

        # Compute focal loss components
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)

        # Apply focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss