# WildlifeMapper Configurable Backbone Guide

## Overview

WildlifeMapper now supports configurable backbones that can **completely replace** the SAM architecture with ResNet-based detection models. This system provides two distinct architectural paths:

1. **ViT-based SAM Architecture**: Uses the original SAM approach with prompt encoder and mask decoder
2. **ResNet-based Detection Architecture**: Uses ResNet+FPN backbone with transformer-based detection heads (similar to DETR)

## Key Architecture Differences

### ViT-SAM Architecture (vit_h, vit_l, vit_b)
```
Input Image → ViT Backbone → SAM Prompt Encoder → SAM Mask Decoder → Predictions
```

**Components:**
- Vision Transformer backbone (from SAM)
- Prompt encoder for box/point prompts
- Mask decoder with transformer
- Designed for segmentation and detection

### ResNet Detection Architecture (resnet50_fpn, resnet101_fpn, etc.)
```
Input Image → ResNet+FPN → Detection Transformer → Classification & BBox Heads → Predictions
```

**Components:**
- ResNet backbone with Feature Pyramid Network
- Object query embeddings (like DETR)
- Multi-layer transformer decoder
- Separate classification and bounding box regression heads
- **No SAM components at all**

## Available Backbones

### ViT-SAM Family
- `vit_h`: Vision Transformer Huge (best performance, largest model)
- `vit_l`: Vision Transformer Large (balanced performance/size)
- `vit_b`: Vision Transformer Base (fastest, smallest)

### ResNet-FPN Family
- `resnet50_fpn`: ResNet-50 with Feature Pyramid Network
- `resnet101_fpn`: ResNet-101 with Feature Pyramid Network
- `resnet152_fpn`: ResNet-152 with Feature Pyramid Network

## Usage Examples

### 1. Using ViT-SAM Backbone

```yaml
# conf/config.yaml
defaults:
  - backbone: vit_l

# Automatically uses SAM architecture
backbone_type: "vit_l"
```

### 2. Using ResNet Backbone

```yaml
# conf/config.yaml
defaults:
  - backbone: resnet50_fpn

# Automatically uses detection architecture
backbone_type: "resnet50_fpn"
```

### 3. Training with Different Backbones

```bash
# Train with ViT-Large SAM
python wildlifemapper/train_hydra_backbone.py backbone=vit_l

# Train with ResNet-50 FPN (completely different architecture)
python wildlifemapper/train_hydra_backbone.py backbone=resnet50_fpn

# Train with ResNet-101 FPN
python wildlifemapper/train_hydra_backbone.py backbone=resnet101_fpn
```

## Configuration Files

### Backbone Configurations

Each backbone has its own config file in `conf/backbone/`:

**ViT Example (`conf/backbone/vit_l.yaml`):**
```yaml
backbone_type: "vit_l"
backbone_out_channels: 256

backbone_config:
  variant: "vit_l"
  img_size: 1024
  patch_size: 16
  out_chans: 256
```

**ResNet Example (`conf/backbone/resnet50_fpn.yaml`):**
```yaml
backbone_type: "resnet50_fpn"
backbone_out_channels: 256

backbone_config:
  variant: "resnet50"
  pretrained: true
  out_channels: 256
  fpn_levels: [1, 2, 3, 4]
```

## Code Architecture

### New Directory Structure

```
wildlifemapper/
├── models/
│   ├── __init__.py
│   ├── wildlife_mapper.py       # Main model that switches architectures
│   └── backbones/
│       ├── __init__.py
│       ├── factory.py           # Backbone factory
│       ├── base.py             # Base backbone interface
│       ├── vit_sam.py          # ViT-SAM backbone
│       └── resnet_fpn.py       # ResNet-FPN backbone
├── train/
│   ├── __init__.py
│   ├── trainer.py              # Enhanced trainer
│   └── train_utils.py          # Training utilities
└── inference/
    ├── __init__.py
    └── evaluator.py            # Evaluation utilities
```

### Key Classes

**`WildlifeMapperModel`**: Main model class that automatically chooses architecture based on backbone type

**`BaseBackbone`**: Abstract base class for all backbones

**`ViTSAMBackbone`**: Wraps SAM's ViT encoder

**`ResNetFPNBackbone`**: ResNet with Feature Pyramid Network

**`Trainer`**: Enhanced trainer that handles both architectures

## Technical Details

### Parameter Optimization

The system automatically adjusts optimization strategy based on backbone:

**ViT-SAM Models:**
- Different learning rates for backbone vs. decoder components
- Backbone: 0.0001 (fine-tuning pretrained ViT)
- Decoder: standard learning rate

**ResNet Models:**
- Single learning rate for all parameters
- Standard learning rate throughout

### Forward Pass Differences

**ViT-SAM:**
```python
# Requires box prompts
outputs = model(images, box_prompts)
```

**ResNet Detection:**
```python
# No prompts needed
outputs = model(images)
```

### Output Format

Both architectures produce compatible outputs:
```python
{
    'pred_logits': torch.Tensor,  # [batch, num_queries, num_classes+1]
    'pred_boxes': torch.Tensor,   # [batch, num_queries, 4]
    'pred_masks': torch.Tensor    # [batch, num_queries, H, W] (SAM only)
}
```

## Benefits

1. **True Architecture Flexibility**: ResNet completely replaces SAM, not just the backbone
2. **Minimal Code Changes**: Existing training/inference code works with both architectures
3. **Easy Experimentation**: Switch backbones with a single config change
4. **Performance Comparison**: Compare SAM vs. traditional detection approaches
5. **Resource Optimization**: Choose architecture based on computational constraints

## Migration from Original Code

The new system is designed to be backward compatible:

1. Existing SAM-based training works with ViT backbones
2. Original `train_hydra.py` still works
3. New `train_hydra_backbone.py` adds backbone flexibility
4. No changes needed to dataset loading or evaluation

## Example: Complete Training Command

```bash
# Train ResNet-50 FPN model (non-SAM architecture)
python wildlifemapper/train_hydra_backbone.py \
    backbone=resnet50_fpn \
    num_epochs=100 \
    lr=1e-4 \
    batch_size=8 \
    work_dir=./exp/resnet50_fpn \
    use_wandb=true
```

This will train a completely ResNet-based model without any SAM components!