"""
Enhanced trainer with configurable backbone support
"""

import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from datetime import datetime
import random

from ..models import build_model
from ..segment_anything.build_sam import SetCriterion, PostProcess
from ..segment_anything.modeling.matcher import build_matcher
from ..segment_anything import utils
from . import train_utils




class Trainer:
    """
    Enhanced trainer with configurable backbone support
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Initialize distributed training if needed
        train_utils.init_distributed_mode(cfg)

        # Set random seeds
        seed = cfg.seed + train_utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Initialize model
        self.model = self._build_model()

        # Initialize criterion and postprocessors
        self.criterion, self.postprocessors = self._build_criterion()

        # Initialize optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_scheduler()

        # Initialize wandb if enabled
        self._init_wandb()

    def _build_model(self):
        """Build the model with configurable backbone"""

        # Get backbone configuration
        backbone_config = {
            'backbone_type': getattr(self.cfg, 'backbone_type', 'vit_h'),
        }

        # Add variant-specific configurations
        if hasattr(self.cfg, 'backbone_config'):
            backbone_config.update(self.cfg.backbone_config)

        num_classes = getattr(self.cfg, 'num_classes', 6)

        # Build the unified model (either SAM-based or ResNet-based)
        model = build_model(
            backbone_config=backbone_config,
            num_classes=num_classes,
            prompt_embed_dim=256,
            image_size=getattr(self.cfg, 'img_size', 1024)
        ).to(self.device)

        # Handle distributed training
        model_without_ddp = model
        if self.cfg.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.cfg.gpu], find_unused_parameters=True
            )
            model_without_ddp = model.module

        self.model_without_ddp = model_without_ddp
        return model

    def _build_criterion(self):
        """Build criterion and postprocessors"""
        num_classes = getattr(self.cfg, 'num_classes', 6)

        # Build matcher
        matcher = build_matcher(self.cfg)

        # Get loss configuration
        focal_loss_weight = getattr(self.cfg, 'focal_loss_weight', 0.0)
        focal_loss_alpha = getattr(self.cfg, 'focal_loss_alpha', None)
        focal_loss_gamma = getattr(self.cfg, 'focal_loss_gamma', 2.0)
        use_focal_loss = focal_loss_weight > 0

        # Set up weight dictionary
        ce_weight = focal_loss_weight if use_focal_loss else 3
        weight_dict = {
            'loss_ce': ce_weight,
            'loss_bbox': getattr(self.cfg, 'bbox_loss_coef', 5),
            'loss_giou': getattr(self.cfg, 'giou_loss_coef', 2)
        }

        losses = ['labels', 'boxes', 'cardinality']
        class_weights = getattr(self.cfg, 'class_weights', None)

        criterion = SetCriterion(
            num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=getattr(self.cfg, 'eos_coef', 0.1),
            losses=losses,
            class_weights=class_weights,
            focal_loss_alpha=focal_loss_alpha,
            focal_loss_gamma=focal_loss_gamma,
            use_focal_loss=use_focal_loss
        )
        criterion.to(self.device)

        confidence_threshold = getattr(self.cfg, 'confidence_threshold', 0.1)
        postprocessors = {'bbox': PostProcess(confidence_threshold=confidence_threshold)}

        return criterion, postprocessors

    def _build_optimizer(self):
        """Build optimizer"""
        backbone_type = getattr(self.cfg, 'backbone_type', 'vit_h')

        if backbone_type.startswith('vit_'):
            # SAM-based model: different learning rates for different components
            if hasattr(self.model_without_ddp, 'mask_decoder') and hasattr(self.model_without_ddp, 'prompt_encoder'):
                mask_prompt_params = (list(self.model_without_ddp.mask_decoder.parameters()) +
                                    list(self.model_without_ddp.prompt_encoder.parameters()))
            else:
                mask_prompt_params = []

            # Handle backbone parameters
            backbone = self.model_without_ddp.backbone
            if hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'hfc_embed'):  # ViT-SAM backbone
                backbone_params = (list(backbone.encoder.hfc_embed.parameters()) +
                                 list(backbone.encoder.patch_embed.parameters()) +
                                 list(backbone.encoder.hfc_attn.parameters()))
            else:
                backbone_params = list(backbone.parameters())

            param_groups = []
            if mask_prompt_params:
                param_groups.append({"params": mask_prompt_params})
            if backbone_params:
                param_groups.append({"params": backbone_params, "lr": 0.0001})
        else:
            # ResNet-based model: single learning rate for all parameters
            param_groups = [{"params": self.model_without_ddp.parameters()}]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )

        return optimizer

    def _build_scheduler(self):
        """Build learning rate scheduler"""
        if hasattr(self.cfg, 'scheduler') and getattr(self.cfg.scheduler, 'type', None) == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            warmup_epochs = getattr(self.cfg.scheduler, 'warmup_epochs', 5)
            min_lr = getattr(self.cfg.scheduler, 'min_lr', 1e-6)

            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.num_epochs - warmup_epochs, eta_min=min_lr
            )
            lr_scheduler = SequentialLR(
                self.optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.cfg.lr_drop)

        return lr_scheduler

    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        if self.cfg.use_wandb and train_utils.is_main_process():
            import wandb
            from omegaconf import OmegaConf

            wandb.login()
            wandb.init(
                project="aerial_detection_project",
                name=f'hydra_{self.cfg.backbone_type}_{self.cfg.model_type}',
                config=OmegaConf.to_container(self.cfg, resolve=True),
            )

    def train_epoch(self, epoch, data_loader_train, sampler_train):
        """Train for one epoch"""
        self.model.train()
        self.criterion.train()

        epoch_loss = 0
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 10

        if self.cfg.distributed:
            sampler_train.set_epoch(epoch)

        for data in metric_logger.log_every(data_loader_train, print_freq, header):
            self.optimizer.zero_grad()
            image = data[0]
            targets = data[1]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            image = image.to(self.device)

            # Handle different model architectures
            backbone_type = getattr(self.cfg, 'backbone_type', 'vit_h')
            if backbone_type.startswith('vit_'):
                # SAM-based model needs box prompts
                b, c, h, w = image.tensors.shape
                boxes_np = np.repeat(np.array([[0, 0, h, w]]), self.cfg.batch_size, axis=0)
                outputs = self.model(image, boxes_np)
            else:
                # ResNet-based model doesn't need box prompts
                outputs = self.model(image)

            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # Reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                      for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                raise RuntimeError("Loss became infinite")

            losses.backward()

            if self.cfg.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_max_norm)

            self.optimizer.step()

            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

            epoch_loss += losses.item()

        # Gather stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, epoch_loss