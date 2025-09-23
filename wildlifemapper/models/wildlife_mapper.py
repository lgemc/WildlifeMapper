"""
Main WildlifeMapper model that can use different architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .backbones import get_backbone
from wildlifemapper.segment_anything.modeling import PromptEncoder, MaskDecoder, TwoWayTransformer
from wildlifemapper.segment_anything.network import MedSAM


class WildlifeMapperModel(nn.Module):
    """
    Main WildlifeMapper model that can use either SAM-based or ResNet-based architecture
    """

    def __init__(self, backbone_config, num_classes=6, prompt_embed_dim=256, image_size=1024):
        super().__init__()

        self.backbone_type = backbone_config.get('backbone_type', 'vit_h')
        self.num_classes = num_classes
        self.image_size = image_size

        # Create backbone
        backbone_type = backbone_config['backbone_type']
        backbone_kwargs = {k: v for k, v in backbone_config.items() if k != 'backbone_type'}
        self.backbone = get_backbone(backbone_type, **backbone_kwargs)

        # Register pixel normalization buffers
        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1))

        if self.backbone_type.startswith('vit_'):
            # Use SAM-based architecture for ViT backbones
            self._build_sam_architecture(prompt_embed_dim, image_size)
        else:
            # Use detection head for ResNet backbones (like DETR/Faster R-CNN style)
            self._build_detection_architecture()

    def _build_sam_architecture(self, prompt_embed_dim, image_size):
        """Build SAM-style architecture with prompt encoder and mask decoder"""
        vit_patch_size = self.backbone.output_stride
        image_embedding_size = image_size // vit_patch_size

        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )

        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=50,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes=self.num_classes,
        )

    def _build_detection_architecture(self):
        """Build detection head for ResNet backbones"""
        backbone_channels = self.backbone.output_channels

        # Classification head
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_classes + 1)  # +1 for background
        )

        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4)  # x, y, w, h
        )

        # Query embeddings for transformer-style detection (similar to DETR)
        self.num_queries = 50  # Number of object queries
        self.query_embed = nn.Embedding(self.num_queries, backbone_channels)

        # Multi-head detection transformer
        self.detection_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=backbone_channels,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=3
        )

        # Final prediction heads
        self.final_class_head = nn.Linear(backbone_channels, self.num_classes + 1)
        self.final_bbox_head = nn.Linear(backbone_channels, 4)

    def forward(self, images, boxes=None):
        # Extract tensor from NestedTensor if needed
        if hasattr(images, 'tensors'):
            input_tensor = images.tensors
        else:
            input_tensor = images

        # Normalize pixel values
        input_tensor = (input_tensor - self.pixel_mean) / self.pixel_std

        # Get backbone features
        features = self.backbone(input_tensor)

        if self.backbone_type.startswith('vit_'):
            return self._forward_sam(features, boxes)
        else:
            return self._forward_detection(features)

    def _forward_sam(self, image_embeddings, boxes):
        """Forward pass for SAM-style architecture"""
        # Get prompt embeddings
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )

        # Decode masks
        low_res_masks, iou_predictions, logits = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )

        return {
            'pred_logits': logits,
            'pred_boxes': iou_predictions,
            'pred_masks': low_res_masks
        }

    def _forward_detection(self, features):
        """Forward pass for ResNet detection architecture"""
        batch_size = features.shape[0]

        # Flatten features for transformer
        # features: [B, C, H, W] -> [H*W, B, C]
        h, w = features.shape[-2:]
        features_flat = features.flatten(2).permute(2, 0, 1)

        # Query embeddings
        query_embeds = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # Apply transformer decoder
        outputs = self.detection_transformer(query_embeds, features_flat)

        # Final predictions
        pred_logits = self.final_class_head(outputs).transpose(0, 1)  # [B, num_queries, num_classes+1]
        pred_boxes = self.final_bbox_head(outputs).transpose(0, 1).sigmoid()  # [B, num_queries, 4]

        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }


def build_model(backbone_config, **kwargs):
    """Factory function to build WildlifeMapper model"""
    return WildlifeMapperModel(backbone_config, **kwargs)