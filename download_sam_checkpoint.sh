#!/bin/bash

# Download SAM ViT-B checkpoint
echo "Downloading SAM ViT-B checkpoint..."

# Create checkpoint directory if it doesn't exist
mkdir -p ./exp/checkpoint

# Download the checkpoint
wget -O ./exp/checkpoint/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

echo "SAM checkpoint downloaded to ./exp/checkpoint/sam_vit_b_01ec64.pth"