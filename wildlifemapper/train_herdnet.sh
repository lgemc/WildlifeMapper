#!/bin/bash

# Training script for HerdNet data converted to COCO format
# Make sure you have:
# 1. Converted CSV files to COCO JSON using csv_to_coco.py
# 2. Created ./coco_annotations/train.json and ./coco_annotations/val.json

echo "Starting WildlifeMapper training with HerdNet data..."

# Set GPU (change as needed)
export CUDA_VISIBLE_DEVICES=0

# Training parameters
COCO_PATH="./coco_annotations"
OUTPUT_DIR="./exp/herdnet_model"
BATCH_SIZE=2
NUM_WORKERS=4
NUM_EPOCHS=100

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
python train.py \
  --coco_path $COCO_PATH \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --num_epochs $NUM_EPOCHS \
  --lr 1e-4 \
  --weight_decay 1e-4

echo "Training completed. Models saved in $OUTPUT_DIR"