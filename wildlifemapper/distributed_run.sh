#!/bin/bash
# Single GPU training example
# CUDA_VISIBLE_DEVICES=2 python train.py --coco_path /data/home/bowen/projects/sam/Medical-SAM-Adapter/data/mara_coco_1024 --output_dir ./exp/box_model --batch_size 1 --use_wandb --wandb_project "wildlifemapper"

# Multi-GPU distributed training examples
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port 8087 train.py --coco_path /mnt/mara/coco_1024_fixed --output_dir ./exp/box_model --batch_size 1 --num_workers 4 --use_wandb --wandb_project "wildlifemapper" --wandb_run_name "distributed_8gpu"

# Current distributed training command
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=7 train.py --coco_path /mnt/mara/coco_1024_fixed --output_dir ./exp/box_model --batch_size 2 --num_workers 4 --use_wandb --wandb_project "wildlifemapper" --wandb_run_name "distributed_7gpu"