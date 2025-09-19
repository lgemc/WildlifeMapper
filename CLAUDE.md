# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WildlifeMapper is a PyTorch-based computer vision model for detecting, locating, and identifying multiple animal species in aerial imagery. It's based on a modified Segment Anything Model (SAM) architecture with custom modules for wildlife detection and identification. This is the official implementation from the CVPR 2024 paper.

## Requirements

- Python >= 3.7 (supported: 3.8-3.10)
- PyTorch >= 1.7.0
- CUDA >= 10.0 with compatible cuDNN
- Linux or macOS

## Installation

```bash
pip install -r requirements.txt
```

## Key Commands

### Training
```bash
# Basic training run
cd wildlifemapper
./run.sh

# Training with custom parameters
CUDA_VISIBLE_DEVICES=1 python train.py --coco_path /path/to/coco/data --output_dir ./exp/box_model --batch_size 1 --num_workers 0

# Distributed training
./distributed_run.sh
```

### Inference/Visualization
```bash
# Run inference and visualization
./infer.sh

# Custom inference
CUDA_VISIBLE_DEVICES=5 python visualize_prediction.py --coco_path /path/to/coco/data --pretrain_model_path ./exp/box_model/best_checkpoint.pth --num_workers 2

# Direct inference
python inference.py --model_path ./exp/box_model/best_checkpoint.pth --input_path /path/to/images
```

### Debugging
```bash
./debug.sh
```

## Project Architecture

### Core Structure
```
wildlifemapper/
├── train.py              # Main training script
├── inference.py          # Inference and evaluation
├── dataloader_coco.py    # COCO dataset loading
├── visualize_prediction.py # Prediction visualization
├── train_utils.py        # Training utilities
├── segment_anything/     # Modified SAM architecture
│   ├── modeling/        # Core model components
│   │   ├── sam.py       # Main SAM model
│   │   ├── image_encoder.py # Vision Transformer encoder
│   │   ├── box_decoder.py   # Bounding box decoder
│   │   ├── matcher.py       # Bipartite matching
│   │   └── prompt_encoder.py # Prompt encoding
│   ├── build_sam.py     # Model factory functions
│   └── network.py       # MedSAM network wrapper
└── exp/                 # Experiment outputs
    ├── box_model/       # Trained model checkpoints
    └── checkpoint/      # Additional checkpoints
```

### Key Components

1. **SAM-based Architecture**: The model extends Meta's Segment Anything Model with wildlife-specific modifications
2. **COCO Format Data**: Uses COCO-style annotations with custom JSON format for masks and bounding boxes
3. **Multi-GPU Training**: Supports distributed training across multiple GPUs
4. **Custom Decoders**: Specialized box decoder and matcher for wildlife detection

### Data Format

The project expects COCO-format data with custom JSON mask files:
```python
{
    "image": {
        "image_id": int,
        "width": int,
        "height": int,
        "file_name": str
    },
    "annotations": [{
        "id": int,
        "bbox": [x, y, w, h],  # XYWH format
        "predicted_iou": float,
        "stability_score": float
    }]
}
```

### Model Checkpoints

- Trained models are saved in `./exp/box_model/`
- Download pre-trained weights before running inference (see `./exp/box_model/README.md`)
- Best model typically saved as `best_checkpoint.pth`

### Important Notes

- The codebase modifies the original SAM for wildlife detection tasks
- GPU memory requirements are significant; adjust batch size accordingly
- Model paths and data paths need to be adjusted in shell scripts for your environment
- The project includes custom CUDA operations that may need compilation