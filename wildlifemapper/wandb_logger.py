"""
Weights & Biases (wandb) logger for WildlifeMapper
Provides comprehensive experiment tracking and visualization
"""

import os
import wandb
import torch
import numpy as np
from typing import Dict, Any, Optional, Union
import matplotlib.pyplot as plt
from datetime import datetime

class WandbLogger:
    """
    A comprehensive W&B logger for WildlifeMapper experiments
    """

    def __init__(self,
                 project: str = "wildlifemapper",
                 name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 tags: Optional[list] = None,
                 notes: Optional[str] = None,
                 dir: Optional[str] = None,
                 enabled: bool = True):
        """
        Initialize W&B logger

        Args:
            project: W&B project name
            name: Run name (auto-generated if None)
            config: Configuration dictionary to log
            tags: Tags for the run
            notes: Notes for the run
            dir: Directory for W&B files
            enabled: Whether logging is enabled
        """
        self.enabled = enabled

        if not self.enabled:
            return

        # Set up run name
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"wildlife_detection_{timestamp}"

        # Initialize wandb
        wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            dir=dir
        )

        self.run = wandb.run

    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None):
        """Log scalar metrics"""
        if not self.enabled:
            return

        wandb.log(metrics, step=step)

    def log_training_metrics(self,
                           train_stats: Dict[str, float],
                           test_stats: Dict[str, float],
                           epoch: int,
                           lr: float):
        """
        Log training and validation metrics

        Args:
            train_stats: Training metrics dictionary
            test_stats: Test/validation metrics dictionary
            epoch: Current epoch
            lr: Current learning rate
        """
        if not self.enabled:
            return

        # Prepare metrics dictionary
        metrics = {
            'epoch': epoch,
            'learning_rate': lr,
        }

        # Add training metrics
        for key, value in train_stats.items():
            metrics[f'train/{key}'] = value

        # Add test metrics
        for key, value in test_stats.items():
            metrics[f'val/{key}'] = value

        wandb.log(metrics, step=epoch)

    def log_model_graph(self, model: torch.nn.Module, input_shape: tuple = (1, 3, 1024, 1024)):
        """
        Log model architecture graph

        Args:
            model: PyTorch model
            input_shape: Input tensor shape for the model
        """
        if not self.enabled:
            return

        try:
            # Create dummy input
            dummy_input = torch.randn(input_shape)

            # Watch the model
            wandb.watch(model, log="all", log_freq=100)

        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")

    def log_hyperparameters(self, config: Dict[str, Any]):
        """Log hyperparameters to wandb config"""
        if not self.enabled:
            return

        wandb.config.update(config)

    def log_images(self,
                   images: Dict[str, Union[np.ndarray, torch.Tensor, plt.Figure]],
                   step: Optional[int] = None):
        """
        Log images to wandb

        Args:
            images: Dictionary of images (name -> image)
            step: Step number
        """
        if not self.enabled:
            return

        wandb_images = {}

        for name, image in images.items():
            if isinstance(image, torch.Tensor):
                # Convert tensor to numpy
                if image.dim() == 4:  # Batch dimension
                    image = image[0]  # Take first image
                if image.dim() == 3 and image.shape[0] in [1, 3]:  # Channel first
                    image = image.permute(1, 2, 0)
                image = image.detach().cpu().numpy()

            if isinstance(image, np.ndarray):
                # Ensure proper format
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)

                wandb_images[name] = wandb.Image(image)

            elif isinstance(image, plt.Figure):
                wandb_images[name] = wandb.Image(image)

        if wandb_images:
            wandb.log(wandb_images, step=step)

    def log_detection_results(self,
                            predictions: Dict[str, Any],
                            targets: Dict[str, Any],
                            images: torch.Tensor,
                            step: Optional[int] = None,
                            max_images: int = 8):
        """
        Log detection results with bounding boxes

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            images: Input images
            step: Step number
            max_images: Maximum number of images to log
        """
        if not self.enabled:
            return

        try:
            # Limit number of images
            batch_size = min(images.shape[0], max_images)

            wandb_images = []

            for i in range(batch_size):
                # Convert image to numpy
                img = images[i].detach().cpu()
                if img.dim() == 3 and img.shape[0] == 3:  # RGB
                    img = img.permute(1, 2, 0)
                img_np = img.numpy()

                # Normalize to [0, 1] if needed
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0

                # Create wandb image with bounding boxes
                boxes = []

                # Add prediction boxes
                if 'pred_boxes' in predictions and i < len(predictions['pred_boxes']):
                    pred_boxes = predictions['pred_boxes'][i].detach().cpu().numpy()
                    pred_scores = predictions['pred_logits'][i].detach().cpu().numpy()

                    # Convert scores to probabilities
                    pred_scores = torch.softmax(torch.tensor(pred_scores), dim=-1).numpy()

                    for j, box in enumerate(pred_boxes):
                        if len(pred_scores) > j:
                            score = pred_scores[j].max()
                            if score > 0.5:  # Confidence threshold
                                # Convert from center format to corner format if needed
                                x_center, y_center, width, height = box
                                x_min = x_center - width / 2
                                y_min = y_center - height / 2

                                boxes.append({
                                    "position": {
                                        "minX": float(x_min),
                                        "minY": float(y_min),
                                        "maxX": float(x_min + width),
                                        "maxY": float(y_min + height)
                                    },
                                    "class_id": int(pred_scores[j].argmax()),
                                    "box_caption": f"pred: {score:.2f}",
                                    "domain": "pixel"
                                })

                # Add ground truth boxes
                if 'boxes' in targets and i < len(targets['boxes']):
                    gt_boxes = targets['boxes'][i].detach().cpu().numpy()

                    for box in gt_boxes:
                        x_center, y_center, width, height = box
                        x_min = x_center - width / 2
                        y_min = y_center - height / 2

                        boxes.append({
                            "position": {
                                "minX": float(x_min),
                                "minY": float(y_min),
                                "maxX": float(x_min + width),
                                "maxY": float(y_min + height)
                            },
                            "class_id": 0,  # Ground truth class
                            "box_caption": "ground_truth",
                            "domain": "pixel"
                        })

                # Create wandb image
                wandb_img = wandb.Image(
                    img_np,
                    boxes=boxes,
                    caption=f"Detection Results - Image {i}"
                )
                wandb_images.append(wandb_img)

            # Log images
            wandb.log({"detection_results": wandb_images}, step=step)

        except Exception as e:
            print(f"Warning: Could not log detection results: {e}")

    def log_histogram(self, name: str, data: Union[torch.Tensor, np.ndarray], step: Optional[int] = None):
        """Log histogram of data"""
        if not self.enabled:
            return

        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        wandb.log({name: wandb.Histogram(data)}, step=step)

    def log_table(self, name: str, data: Dict[str, list], step: Optional[int] = None):
        """Log tabular data"""
        if not self.enabled:
            return

        table = wandb.Table(data=list(zip(*data.values())), columns=list(data.keys()))
        wandb.log({name: table}, step=step)

    def log_system_metrics(self):
        """Log system metrics like GPU usage"""
        if not self.enabled:
            return

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            wandb.log({"gpu_memory_gb": gpu_memory})

    def finish(self):
        """Finish the wandb run"""
        if not self.enabled:
            return

        wandb.finish()

    def save_model(self, model_path: str, aliases: Optional[list] = None):
        """
        Save model as wandb artifact

        Args:
            model_path: Path to the model file
            aliases: Aliases for the model version (e.g., ['latest', 'best'])
        """
        if not self.enabled:
            return

        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} does not exist")
            return

        artifact = wandb.Artifact(
            name="model",
            type="model",
            description="WildlifeMapper trained model"
        )

        artifact.add_file(model_path)

        wandb.log_artifact(artifact, aliases=aliases)

    def log_code(self, code_dir: str = "."):
        """Log code as artifact"""
        if not self.enabled:
            return

        wandb.run.log_code(root=code_dir)

def setup_wandb_logger(args, model_config: Optional[Dict] = None) -> WandbLogger:
    """
    Setup wandb logger from training arguments

    Args:
        args: Training arguments
        model_config: Additional model configuration

    Returns:
        WandbLogger instance
    """
    if not hasattr(args, 'use_wandb') or not args.use_wandb:
        return WandbLogger(enabled=False)

    # Prepare config
    config = {
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'model_type': args.model_type,
        'weight_decay': args.weight_decay,
        'lr_drop': args.lr_drop,
        'clip_max_norm': args.clip_max_norm,
        'task_name': args.task_name,
    }

    # Add loss coefficients
    loss_config = {
        'bbox_loss_coef': args.bbox_loss_coef,
        'giou_loss_coef': args.giou_loss_coef,
        'mask_loss_coef': getattr(args, 'mask_loss_coef', 1.0),
        'dice_loss_coef': getattr(args, 'dice_loss_coef', 1.0),
        'eos_coef': args.eos_coef,
    }
    config.update(loss_config)

    # Add matcher costs
    matcher_config = {
        'set_cost_class': args.set_cost_class,
        'set_cost_bbox': args.set_cost_bbox,
        'set_cost_giou': args.set_cost_giou,
    }
    config.update(matcher_config)

    # Add model config if provided
    if model_config:
        config.update(model_config)

    # Setup project name and tags
    project_name = getattr(args, 'wandb_project', 'wildlifemapper')
    run_name = getattr(args, 'wandb_run_name', None)
    tags = ['wildlife_detection', 'SAM', args.model_type]

    # Add distributed training tag if applicable
    if getattr(args, 'distributed', False):
        tags.append('distributed')

    return WandbLogger(
        project=project_name,
        name=run_name,
        config=config,
        tags=tags,
        notes=f"Wildlife detection training with {args.model_type}",
        dir=args.output_dir if hasattr(args, 'output_dir') else None
    )