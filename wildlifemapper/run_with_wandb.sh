#!/bin/bash

# WildlifeMapper Training Script with W&B Logging
# This script provides various training configurations with wandb logging enabled

# Configuration variables - modify as needed
COCO_PATH="/mnt/mara/coco_1024_fixed"
OUTPUT_DIR="./exp/box_model"
WANDB_PROJECT="wildlifemapper"
BATCH_SIZE=1
NUM_WORKERS=0
EPOCHS=550

# W&B Configuration
WANDB_ENTITY=""  # Your W&B username/team (optional)
WANDB_TAGS="wildlife_detection,SAM"  # Comma-separated tags

# Training configurations
CONFIG=${1:-"single_gpu"}  # Default to single GPU

case $CONFIG in
    "single_gpu")
        echo "Running single GPU training with W&B logging..."
        CUDA_VISIBLE_DEVICES=1 python train.py \
            --coco_path $COCO_PATH \
            --output_dir $OUTPUT_DIR \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --num_epochs $EPOCHS \
            --use_wandb \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_name "single_gpu_$(date +%Y%m%d_%H%M%S)"
        ;;

    "resume")
        echo "Resuming training with W&B logging..."
        CHECKPOINT_PATH="./exp/box_model/checkpoint_epoch_240.pth"
        CUDA_VISIBLE_DEVICES=1 python train.py \
            --coco_path $COCO_PATH \
            --output_dir $OUTPUT_DIR \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --num_epochs $EPOCHS \
            --resume $CHECKPOINT_PATH \
            --use_wandb \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_name "resume_training_$(date +%Y%m%d_%H%M%S)"
        ;;

    "distributed")
        echo "Running distributed training with W&B logging..."
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
            --nproc_per_node=7 \
            --master_port 8087 \
            train.py \
            --coco_path $COCO_PATH \
            --output_dir $OUTPUT_DIR \
            --batch_size 2 \
            --num_workers 4 \
            --num_epochs $EPOCHS \
            --use_wandb \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_name "distributed_7gpu_$(date +%Y%m%d_%H%M%S)"
        ;;

    "debug")
        echo "Running debug training (few epochs) with W&B logging..."
        CUDA_VISIBLE_DEVICES=1 python train.py \
            --coco_path $COCO_PATH \
            --output_dir $OUTPUT_DIR \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --num_epochs 5 \
            --use_wandb \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_name "debug_run_$(date +%Y%m%d_%H%M%S)"
        ;;

    "high_lr")
        echo "Running training with high learning rate and W&B logging..."
        CUDA_VISIBLE_DEVICES=1 python train.py \
            --coco_path $COCO_PATH \
            --output_dir $OUTPUT_DIR \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --num_epochs $EPOCHS \
            --lr 0.0005 \
            --use_wandb \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_name "high_lr_experiment_$(date +%Y%m%d_%H%M%S)"
        ;;

    "custom")
        echo "Running custom configuration with W&B logging..."
        # Modify these parameters as needed for your experiment
        CUDA_VISIBLE_DEVICES=1 python train.py \
            --coco_path $COCO_PATH \
            --output_dir $OUTPUT_DIR \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --num_epochs $EPOCHS \
            --lr 0.0001 \
            --weight_decay 0.001 \
            --lr_drop 40 \
            --clip_max_norm 0.1 \
            --bbox_loss_coef 5 \
            --giou_loss_coef 2 \
            --use_wandb \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_name "custom_experiment_$(date +%Y%m%d_%H%M%S)"
        ;;

    *)
        echo "Usage: $0 [single_gpu|resume|distributed|debug|high_lr|custom]"
        echo ""
        echo "Configurations:"
        echo "  single_gpu  - Single GPU training (default)"
        echo "  resume      - Resume training from checkpoint"
        echo "  distributed - Multi-GPU distributed training"
        echo "  debug       - Quick debug run (5 epochs)"
        echo "  high_lr     - Experiment with higher learning rate"
        echo "  custom      - Custom hyperparameters (modify script)"
        echo ""
        echo "Examples:"
        echo "  $0 single_gpu"
        echo "  $0 distributed"
        echo "  $0 debug"
        exit 1
        ;;
esac

echo "Training completed!"
echo "Check your W&B dashboard at: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"