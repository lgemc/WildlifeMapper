"""
Enhanced dataloader integrating advanced augmentation strategies and class-aware sampling.
Combines best practices from HerdNet with WildlifeMapper's existing COCO-style dataset.
"""

import torch
import torch.utils.data
import numpy as np
import random
from typing import Dict, Any, Optional, List, Tuple, Union
import hydra
from omegaconf import DictConfig
from pathlib import Path

from .transforms import (
    WildlifeAugmentationPipeline,
    CopyPasteAugmentation,
    MosaicMixupAugmentation
)
from .samplers import (
    ClassAwareBatchSampler,
    WeightedClassSampler,
    MinorityOversamplingBatchSampler
)
from torchvision.datasets import CocoDetection as BaseCocoDetection


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id], dtype=torch.int32)

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        #center-point of the box for Bobby, yo!!
        centre_points = torch.cat((boxes[:, ::2].mean(1, True), boxes[:, 1::2].mean(1, True)), 1)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["center"] = centre_points
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


class EnhancedWildlifeDataset(BaseCocoDetection):
    """
    Enhanced dataset with advanced augmentation and class-aware sampling capabilities.
    """

    def __init__(
        self,
        img_folder: Union[str, Path],
        ann_file: Union[str, Path],
        cfg: DictConfig,
        image_set: str = "train",
        return_masks: bool = False
    ):
        """
        Args:
            img_folder: Path to images
            ann_file: Path to annotation file
            cfg: Hydra configuration containing augmentation settings
            image_set: Dataset split ("train", "val", "test")
            return_masks: Whether to return segmentation masks
        """
        # Initialize parent class without transforms
        super(BaseCocoDetection, self).__init__(img_folder, ann_file)

        self.cfg = cfg
        self.image_set = image_set
        self.return_masks = return_masks
        self.indices = range(self.__len__())

        # Initialize components
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self._init_augmentation_pipeline()
        self._init_object_bank()

        # Image size from config
        aug_cfg = cfg.get('data_augmentation', {}).get('augmentation', {})
        self.img_size = tuple(aug_cfg.get('image_size', [1024, 1024]))

        # Mosaic settings
        self.use_mosaic = (
            image_set == "train" and
            aug_cfg.get('advanced', {}).get('mosaic_prob', 0.0) > 0
        )
        self.mosaic_border = [-self.img_size[0] // 2, -self.img_size[1] // 2]

    def _init_augmentation_pipeline(self):
        """Initialize augmentation pipeline based on configuration."""
        aug_cfg = self.cfg.get('data_augmentation', {}).get('augmentation', {})

        # Main augmentation pipeline
        self.augmentation_pipeline = WildlifeAugmentationPipeline(
            image_size=tuple(aug_cfg.get('image_size', [1024, 1024])),
            strong_aug_prob=aug_cfg.get('strong_aug_prob', 0.6),
            geometric_aug_prob=aug_cfg.get('geometric_aug_prob', 0.7),
            photometric_aug_prob=aug_cfg.get('photometric_aug_prob', 0.6),
            weather_aug_prob=aug_cfg.get('weather_aug_prob', 0.2),
            aerial_specific_prob=aug_cfg.get('aerial_specific_prob', 0.3)
        )

        # Copy-paste augmentation
        copy_paste_cfg = aug_cfg.get('copy_paste', {})
        if copy_paste_cfg.get('enabled', False):
            self.copy_paste = CopyPasteAugmentation(
                paste_prob=copy_paste_cfg.get('paste_prob', 0.3),
                max_paste_objects=copy_paste_cfg.get('max_paste_objects', 5),
                scale_range=tuple(copy_paste_cfg.get('scale_range', [0.5, 1.5])),
                overlap_threshold=copy_paste_cfg.get('overlap_threshold', 0.3)
            )
        else:
            self.copy_paste = None

        # Mosaic and mixup
        advanced_cfg = aug_cfg.get('advanced', {})
        self.mosaic_mixup = MosaicMixupAugmentation(
            mosaic_prob=advanced_cfg.get('mosaic_prob', 0.0),
            mixup_prob=advanced_cfg.get('mixup_prob', 0.0),
            image_size=tuple(aug_cfg.get('image_size', [1024, 1024]))
        )

    def _init_object_bank(self):
        """Initialize object bank for copy-paste augmentation."""
        self.object_bank = []
        self.max_bank_size = 500

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item with enhanced augmentations.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing image and target
        """
        # Handle mosaic augmentation for training
        if self.use_mosaic and random.random() < self.mosaic_mixup.mosaic_prob:
            img, target = self._load_mosaic(idx)
        else:
            # Regular loading
            img, target = super(BaseCocoDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            img, target = self.prepare(img, target)

        # Apply augmentations
        training = self.image_set == "train"

        # Apply copy-paste if enabled and training
        if training and self.copy_paste is not None:
            img, target = self.copy_paste(
                np.array(img) if hasattr(img, 'mode') else img,
                target,
                self.object_bank if len(self.object_bank) > 0 else None
            )

            # Update object bank
            self._update_object_bank(img, target)

        # Apply main augmentation pipeline
        if hasattr(img, 'mode'):  # PIL Image
            img = np.array(img)

        img, target = self.augmentation_pipeline(img, target, training=training)

        return {"image": img, "target": target}

    def _load_mosaic(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load mosaic of 4 images (adapted from original YOLO-style mosaic).

        Args:
            index: Index of primary image

        Returns:
            Mosaic image and combined target
        """
        # Sample 3 additional indices
        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)

        # Load 4 images
        images = []
        targets = []

        for idx in indices:
            img, target = super(BaseCocoDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            img, target = self.prepare(img, target)

            if hasattr(img, 'mode'):  # PIL Image
                img = np.array(img)

            images.append(img)
            targets.append(target)

        # Create mosaic
        mosaic_img, mosaic_target = self.mosaic_mixup.mosaic_augmentation(images, targets)

        return mosaic_img, mosaic_target

    def _update_object_bank(self, image: np.ndarray, target: Dict[str, Any]):
        """Update object bank with new objects for copy-paste."""
        if 'boxes' not in target or len(target['boxes']) == 0:
            return

        boxes = target['boxes']
        labels = target['labels']

        for box, label in zip(boxes, labels):
            if len(self.object_bank) >= self.max_bank_size:
                # Remove oldest object
                self.object_bank.pop(0)

            x1, y1, x2, y2 = map(int, box)
            obj_image = image[y1:y2, x1:x2]

            if obj_image.size > 0:
                obj_mask = np.ones((y2 - y1, x2 - x1), dtype=np.uint8) * 255

                self.object_bank.append({
                    'image': obj_image,
                    'mask': obj_mask,
                    'label': label.item() if torch.is_tensor(label) else label
                })


def create_enhanced_dataloader(
    cfg: DictConfig,
    image_set: str = "train",
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create enhanced dataloader with advanced augmentation and sampling.

    Args:
        cfg: Hydra configuration
        image_set: Dataset split ("train", "val", "test")
        **kwargs: Additional arguments for DataLoader

    Returns:
        Enhanced DataLoader
    """
    # Dataset configuration
    dataset_cfg = cfg.get('dataset', {})
    data_path = Path(dataset_cfg.get('data_path', '.'))

    if image_set == 'train':
        ann_file = data_path / dataset_cfg.get('train_annotation_file', 'annotations/instances_train.json')
        img_folder = data_path / dataset_cfg.get('train_image_folder', 'images/train')
    elif image_set == 'val':
        ann_file = data_path / dataset_cfg.get('val_annotation_file', 'annotations/instances_val.json')
        img_folder = data_path / dataset_cfg.get('val_image_folder', 'images/val')
    else:
        raise ValueError(f"Unknown image_set: {image_set}")

    # Create dataset
    dataset = EnhancedWildlifeDataset(
        img_folder=img_folder,
        ann_file=ann_file,
        cfg=cfg,
        image_set=image_set,
        return_masks=kwargs.get('return_masks', False)
    )

    # Sampling configuration
    aug_cfg = cfg.get('data_augmentation', {}).get('augmentation', {})
    sampling_cfg = aug_cfg.get('sampling', {})
    strategy = sampling_cfg.get('strategy', 'random')
    batch_size = sampling_cfg.get('batch_size', 8)

    # Create sampler based on strategy
    sampler = None
    batch_sampler = None

    if image_set == "train":
        if strategy == "class_aware":
            batch_sampler = ClassAwareBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                samples_per_class=sampling_cfg.get('samples_per_class', None),
                shuffle=sampling_cfg.get('shuffle', True),
                drop_last=True
            )
        elif strategy == "weighted":
            sampler = WeightedClassSampler(
                dataset=dataset,
                num_samples=len(dataset),
                replacement=True
            )
        elif strategy == "minority_oversampling":
            batch_sampler = MinorityOversamplingBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                minority_boost_factor=sampling_cfg.get('minority_boost_factor', 2.0),
                min_class_samples=sampling_cfg.get('min_class_samples', 2),
                shuffle=sampling_cfg.get('shuffle', True)
            )

    # DataLoader configuration
    hardware_cfg = cfg.get('hardware', {})
    dataloader_kwargs = {
        'batch_size': batch_size if batch_sampler is None else 1,
        'shuffle': image_set == "train" and sampler is None and batch_sampler is None,
        'num_workers': hardware_cfg.get('num_workers', 4),
        'pin_memory': hardware_cfg.get('pin_memory', True),
        'drop_last': image_set == "train",
        'collate_fn': collate_fn,
        **kwargs
    }

    # Add sampler if created
    if batch_sampler is not None:
        dataloader_kwargs['batch_sampler'] = batch_sampler
        dataloader_kwargs.pop('batch_size')
        dataloader_kwargs.pop('shuffle')
        dataloader_kwargs.pop('drop_last')
    elif sampler is not None:
        dataloader_kwargs['sampler'] = sampler
        dataloader_kwargs.pop('shuffle')

    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for wildlife detection batches.

    Args:
        batch: List of samples

    Returns:
        Collated batch
    """
    images = []
    targets = []

    for sample in batch:
        images.append(sample["image"])
        targets.append(sample["target"])

    # Stack images
    images = torch.stack(images, 0)

    return {
        "image": images,
        "target": targets
    }


def get_dataloader_from_config(cfg: DictConfig, image_set: str = "train") -> torch.utils.data.DataLoader:
    """
    Convenience function to create dataloader from Hydra config.

    Args:
        cfg: Hydra configuration
        image_set: Dataset split

    Returns:
        DataLoader instance
    """
    return create_enhanced_dataloader(cfg, image_set)


# Example usage with Hydra
@hydra.main(version_base=None, config_path="../../conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    """Example of using enhanced dataloader with Hydra configuration."""

    # Create train dataloader
    train_loader = get_dataloader_from_config(cfg, "train")
    val_loader = get_dataloader_from_config(cfg, "val")

    print(f"Train dataloader created with {len(train_loader)} batches")
    print(f"Val dataloader created with {len(val_loader)} batches")

    # Test first batch
    for batch in train_loader:
        images = batch["image"]
        targets = batch["target"]
        print(f"Batch shape: {images.shape}")
        print(f"Number of targets: {len(targets)}")
        break


if __name__ == "__main__":
    main()