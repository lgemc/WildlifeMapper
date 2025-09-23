"""
Enhanced training script with configurable backbone support - Hydra version
"""

import numpy as np
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from datetime import datetime
import torch

from dataloader_coco import build_dataset
from train.trainer import Trainer
from train import train_utils
from inference import evaluate, get_coco_api_from_dataset
from segment_anything import utils


def create_datasets_and_loaders(cfg):
    """Create datasets and data loaders from Hydra config"""
    class Args:
        def __init__(self, cfg):
            for key, value in cfg.items():
                if key == 'loss' and isinstance(value, DictConfig):
                    for loss_key, loss_value in value.items():
                        setattr(self, loss_key, loss_value)
                elif key == 'training' and isinstance(value, DictConfig):
                    for train_key, train_value in value.items():
                        if train_key == 'loss' and isinstance(train_value, DictConfig):
                            for loss_key, loss_value in train_value.items():
                                setattr(self, loss_key, loss_value)
                        else:
                            setattr(self, train_key, train_value)
                else:
                    setattr(self, key, value)

    args = Args(cfg)

    # Create datasets
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    # Distributed training dataset setup
    if cfg.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, cfg.batch_size, drop_last=True)

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train,
        collate_fn=utils.custom_collate, num_workers=cfg.num_workers
    )
    data_loader_val = DataLoader(
        dataset_val, cfg.batch_size, sampler=sampler_val,
        drop_last=False, collate_fn=utils.custom_collate, num_workers=cfg.num_workers
    )

    # For evaluation, coco_api
    base_ds = get_coco_api_from_dataset(dataset_val)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, base_ds, sampler_train


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Convert config to namespace for compatibility
    class Args:
        def __init__(self, cfg):
            for key, value in cfg.items():
                if key == 'loss' and isinstance(value, DictConfig):
                    for loss_key, loss_value in value.items():
                        setattr(self, loss_key, loss_value)
                elif key == 'training' and isinstance(value, DictConfig):
                    for train_key, train_value in value.items():
                        if train_key == 'loss' and isinstance(train_value, DictConfig):
                            for loss_key, loss_value in train_value.items():
                                setattr(self, loss_key, loss_value)
                        else:
                            setattr(self, train_key, train_value)
                else:
                    setattr(self, key, value)

    args = Args(cfg)
    args.trained_model = cfg.work_dir
    args.dist_url = 'env://'

    # Create datasets and loaders
    dataset_train, dataset_val, data_loader_train, data_loader_val, base_ds, sampler_train = create_datasets_and_loaders(cfg)

    # Initialize trainer with configurable backbone
    trainer = Trainer(cfg)

    print("Number of training samples:", dataset_train.__len__())
    print("Using backbone:", getattr(cfg, 'backbone_type', 'vit_h'))

    # Print trainable parameters
    train_params_list = [n for n, p in trainer.model_without_ddp.named_parameters() if p.requires_grad]
    print("Trainable parameters:", train_params_list)

    # Training loop
    num_epochs = cfg.num_epochs
    best_loss = 1e10

    start_epoch = 0
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("RESUMING TRAINING")
            checkpoint = torch.load(cfg.resume, map_location=trainer.device)
            start_epoch = checkpoint["epoch"] + 1
            trainer.model_without_ddp.load_state_dict(checkpoint['model'])

    if cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        print(f"Training epoch {epoch}/{num_epochs}")

        # Train for one epoch
        train_stats, epoch_loss = trainer.train_epoch(epoch, data_loader_train, sampler_train)

        # Update learning rate
        trainer.lr_scheduler.step()

        # Evaluation after each epoch
        test_stats, coco_evaluator = evaluate(
            trainer.model, trainer.criterion, trainer.postprocessors,
            data_loader_val, base_ds, trainer.device, args
        )

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch
        }

        # Log to wandb if enabled
        if cfg.use_wandb and train_utils.is_main_process():
            import wandb
            wandb.log(log_stats)

        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}')

        # Save checkpoint every 40 epochs
        checkpoint = {
            "model": trainer.model_without_ddp.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "epoch": epoch,
        }
        if epoch % 40 == 0:
            checkpoint_path = f"{cfg.work_dir}/checkpoint_epoch_{epoch}.pth"
            train_utils.save_on_master(checkpoint, checkpoint_path)

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_checkpoint_path = f"{cfg.work_dir}/best_checkpoint.pth"
            train_utils.save_on_master(checkpoint, best_checkpoint_path)


if __name__ == "__main__":
    main()