# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WildlifeMapper is a PyTorch-based computer vision project for detecting, locating, and identifying multiple animal species in aerial imagery. It extends the Segment Anything Model (SAM) architecture for wildlife detection and identification tasks.

## Architecture

The codebase follows a modular structure:

- **Core Model**: Built on top of Segment Anything Model (SAM) architecture in `wildlifemapper/segment_anything/`
  - `modeling/`: Contains core model components (image encoder, prompt encoder, transformer, etc.)
  - `network.py`: Main MedSAM network implementation
  - `predictor.py`: Inference predictor interface
  - `build_sam.py`: Model factory functions

- **Training Pipeline**:
  - `train.py`: Main training script with distributed training support
  - `train_utils.py`: Training utilities and helper functions
  - `dataloader_coco.py`: COCO format dataset loader

- **Inference & Evaluation**:
  - `inference.py`: Model evaluation and COCO metrics computation
  - `visualize_prediction.py`: Visualization tools for predictions

## Common Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Training
```bash
python wildlifemapper/train.py [arguments]
```

### Inference/Evaluation
```bash
python wildlifemapper/inference.py [arguments]
```

### Run Main Script
```bash
python main.py
```

## Development Notes

- **Python Version**: Requires Python 3.7+ (README states 3.7, pyproject.toml specifies 3.12+)
- **Framework**: PyTorch-based with CUDA support
- **Dataset Format**: Uses COCO format for annotations with custom JSON structure for masks
- **Model Weights**: Pre-trained weights are .pth files (ignored in git via .gitignore)
- **Dependencies**: Key dependencies include PyTorch, torchvision, pycocotools, opencv-python, scikit-image

## Key Configuration

- Training uses environment variables for threading control (OMP_NUM_THREADS, etc.)
- Random seed set to 2023 for reproducibility
- Supports distributed training setup
- COCO evaluation metrics for bounding box and segmentation tasks