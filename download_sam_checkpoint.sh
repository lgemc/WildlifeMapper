#!/bin/bash

# Download SAM ViT-L checkpoint
echo "Downloading SAM ViT-L checkpoint..."

# Create checkpoint directory if it doesn't exist
mkdir -p ./exp/checkpoint

# Download the checkpoint
wget -O ./exp/checkpoint/sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

echo "SAM checkpoint downloaded to ./exp/checkpoint/sam_vit_l_0b3195.pth"