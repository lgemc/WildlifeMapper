"""
Advanced data augmentation and transformation pipeline for wildlife detection.
Inspired by HerdNet's comprehensive augmentation strategies.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import random
import math
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, List, Dict, Optional, Union, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import scipy.ndimage
from copy import deepcopy


class WildlifeAugmentationPipeline:
    """
    Comprehensive augmentation pipeline optimized for aerial wildlife imagery.
    Combines geometric, photometric, and domain-specific augmentations.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (1024, 1024),
        strong_aug_prob: float = 0.8,
        geometric_aug_prob: float = 0.7,
        photometric_aug_prob: float = 0.6,
        weather_aug_prob: float = 0.3,
        aerial_specific_prob: float = 0.4
    ):
        """
        Args:
            image_size: Target image size (height, width)
            strong_aug_prob: Probability of applying strong augmentations
            geometric_aug_prob: Probability of geometric augmentations
            photometric_aug_prob: Probability of photometric augmentations
            weather_aug_prob: Probability of weather-specific augmentations
            aerial_specific_prob: Probability of aerial imagery specific augmentations
        """
        self.image_size = image_size
        self.strong_aug_prob = strong_aug_prob
        self.geometric_aug_prob = geometric_aug_prob
        self.photometric_aug_prob = photometric_aug_prob
        self.weather_aug_prob = weather_aug_prob
        self.aerial_specific_prob = aerial_specific_prob

        self.train_transform = self._build_train_pipeline()
        self.val_transform = self._build_val_pipeline()

    def _build_train_pipeline(self) -> A.Compose:
        """Build comprehensive training augmentation pipeline."""
        return A.Compose([
            # Resize and basic setup
            A.Resize(height=self.image_size[0], width=self.image_size[1], p=1.0),

            # Geometric augmentations
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
            ], p=self.geometric_aug_prob),

            # Advanced geometric transformations
            A.OneOf([
                A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.6),
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-15, 15),
                    shear=(-5, 5),
                    p=0.4
                ),
                A.Perspective(scale=(0.05, 0.1), p=0.3),
            ], p=self.geometric_aug_prob),

            # Photometric augmentations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.7
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=15,
                    p=0.6
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
            ], p=self.photometric_aug_prob),

            # Noise and blur augmentations
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
                A.MotionBlur(blur_limit=5, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.2),
            ], p=0.5),

            # Weather and environmental augmentations
            A.OneOf([
                A.RandomRain(
                    slant_lower=-10, slant_upper=10,
                    drop_length=1, drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=1,
                    brightness_coefficient=0.8,
                    p=0.3
                ),
                A.RandomFog(
                    fog_coef_lower=0.1, fog_coef_upper=0.3,
                    alpha_coef=0.08,
                    p=0.2
                ),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    angle_lower=0, angle_upper=1,
                    num_flare_circles_lower=1, num_flare_circles_upper=3,
                    p=0.15
                ),
            ], p=self.weather_aug_prob),

            # Aerial imagery specific augmentations
            A.OneOf([
                A.CoarseDropout(
                    max_holes=8, max_height=32, max_width=32,
                    min_holes=1, min_height=8, min_width=8,
                    fill_value=0, mask_fill_value=0,
                    p=0.3
                ),
                A.GridDropout(
                    ratio=0.5, unit_size_min=2, unit_size_max=16,
                    holes_number_x=None, holes_number_y=None,
                    shift_x=0, shift_y=0,
                    random_offset=False, fill_value=0, mask_fill_value=0,
                    p=0.2
                ),
            ], p=self.aerial_specific_prob),

            # Strong augmentations for robustness
            A.OneOf([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.2),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            ], p=self.strong_aug_prob * 0.3),

            # Final normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            ),
            ToTensorV2(p=1.0),
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['labels'],
            min_visibility=0.1
        ))

    def _build_val_pipeline(self) -> A.Compose:
        """Build validation augmentation pipeline (minimal augmentations)."""
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1], p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            ),
            ToTensorV2(p=1.0),
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['labels'],
            min_visibility=0.1
        ))

    def __call__(
        self,
        image: Union[np.ndarray, Image.Image],
        target: Dict[str, Any],
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply augmentation pipeline.

        Args:
            image: Input image
            target: Target annotations (COCO format)
            training: Whether in training mode

        Returns:
            Augmented image and target
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Select appropriate pipeline
        transform = self.train_transform if training else self.val_transform

        # Prepare data for albumentations
        bboxes = []
        labels = []

        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            if torch.is_tensor(boxes):
                boxes = boxes.numpy()

            # Convert from [x1, y1, x2, y2] to [x, y, w, h] if needed
            if boxes.shape[1] == 4:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    bboxes.append([x1, y1, w, h])

            if 'labels' in target:
                target_labels = target['labels']
                if torch.is_tensor(target_labels):
                    target_labels = target_labels.numpy()
                labels = target_labels.tolist()
            else:
                labels = [1] * len(bboxes)

        # Apply transformations
        try:
            if bboxes:
                transformed = transform(
                    image=image,
                    bboxes=bboxes,
                    labels=labels
                )
            else:
                transformed = transform(image=image)

            # Update target with transformed bboxes
            if bboxes and 'bboxes' in transformed:
                new_boxes = transformed['bboxes']
                if new_boxes:
                    # Convert back to [x1, y1, x2, y2] format
                    converted_boxes = []
                    for box in new_boxes:
                        x, y, w, h = box
                        converted_boxes.append([x, y, x + w, y + h])

                    target['boxes'] = torch.tensor(converted_boxes, dtype=torch.float32)
                    target['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)
                else:
                    # No boxes after transformation
                    target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                    target['labels'] = torch.zeros(0, dtype=torch.int64)

            return transformed['image'], target

        except Exception as e:
            print(f"Augmentation failed: {e}")
            # Fallback to minimal processing
            fallback_transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            transformed = fallback_transform(image=image)
            return transformed['image'], target


class CopyPasteAugmentation:
    """
    Copy-paste augmentation specifically designed for aerial wildlife imagery.
    Copies animal instances and pastes them into different backgrounds.
    """

    def __init__(
        self,
        paste_prob: float = 0.3,
        max_paste_objects: int = 5,
        scale_range: Tuple[float, float] = (0.5, 1.5),
        overlap_threshold: float = 0.3
    ):
        """
        Args:
            paste_prob: Probability of applying copy-paste
            max_paste_objects: Maximum number of objects to paste
            scale_range: Scale range for pasted objects
            overlap_threshold: Maximum allowed IoU overlap with existing objects
        """
        self.paste_prob = paste_prob
        self.max_paste_objects = max_paste_objects
        self.scale_range = scale_range
        self.overlap_threshold = overlap_threshold

    def __call__(
        self,
        image: np.ndarray,
        target: Dict[str, Any],
        object_bank: Optional[List[Dict]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply copy-paste augmentation.

        Args:
            image: Input image
            target: Target annotations
            object_bank: Bank of objects to paste from

        Returns:
            Augmented image and target
        """
        if random.random() > self.paste_prob or object_bank is None:
            return image, target

        # Extract existing objects from the image
        image_objects = self._extract_objects(image, target)

        # Sample objects to paste
        num_paste = random.randint(1, min(self.max_paste_objects, len(object_bank)))
        paste_objects = random.sample(object_bank, num_paste)

        # Paste objects
        new_image = image.copy()
        new_boxes = target['boxes'].clone() if torch.is_tensor(target['boxes']) else target['boxes'].copy()
        new_labels = target['labels'].clone() if torch.is_tensor(target['labels']) else target['labels'].copy()

        for obj_data in paste_objects:
            obj_image = obj_data['image']
            obj_mask = obj_data['mask']
            obj_label = obj_data['label']

            # Find suitable paste location
            paste_location = self._find_paste_location(new_image, obj_image.shape[:2], new_boxes)

            if paste_location is not None:
                x, y = paste_location
                h, w = obj_image.shape[:2]

                # Apply random scale
                scale = random.uniform(*self.scale_range)
                new_h, new_w = int(h * scale), int(w * scale)

                if new_h > 0 and new_w > 0:
                    # Resize object and mask
                    obj_image_resized = cv2.resize(obj_image, (new_w, new_h))
                    obj_mask_resized = cv2.resize(obj_mask, (new_w, new_h))

                    # Paste object
                    self._paste_object(new_image, obj_image_resized, obj_mask_resized, x, y)

                    # Add new box and label
                    new_box = [x, y, x + new_w, y + new_h]
                    new_boxes = torch.cat([new_boxes, torch.tensor([new_box], dtype=torch.float32)])
                    new_labels = torch.cat([new_labels, torch.tensor([obj_label], dtype=torch.int64)])

        # Update target
        target['boxes'] = new_boxes
        target['labels'] = new_labels

        return new_image, target

    def _extract_objects(self, image: np.ndarray, target: Dict[str, Any]) -> List[Dict]:
        """Extract objects from image for future use in copy-paste."""
        objects = []

        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            labels = target['labels']

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = map(int, box)
                obj_image = image[y1:y2, x1:x2]
                obj_mask = np.ones((y2 - y1, x2 - x1), dtype=np.uint8) * 255

                if obj_image.size > 0:
                    objects.append({
                        'image': obj_image,
                        'mask': obj_mask,
                        'label': label.item() if torch.is_tensor(label) else label
                    })

        return objects

    def _find_paste_location(
        self,
        image: np.ndarray,
        obj_size: Tuple[int, int],
        existing_boxes: torch.Tensor
    ) -> Optional[Tuple[int, int]]:
        """Find suitable location to paste object without significant overlap."""
        h, w = image.shape[:2]
        obj_h, obj_w = obj_size

        max_attempts = 20
        for _ in range(max_attempts):
            x = random.randint(0, max(1, w - obj_w))
            y = random.randint(0, max(1, h - obj_h))

            candidate_box = torch.tensor([x, y, x + obj_w, y + obj_h], dtype=torch.float32)

            # Check overlap with existing boxes
            if len(existing_boxes) > 0:
                ious = self._compute_iou(candidate_box.unsqueeze(0), existing_boxes)
                if torch.max(ious) < self.overlap_threshold:
                    return x, y
            else:
                return x, y

        return None

    def _paste_object(
        self,
        background: np.ndarray,
        obj_image: np.ndarray,
        obj_mask: np.ndarray,
        x: int,
        y: int
    ):
        """Paste object onto background using alpha blending."""
        h, w = obj_image.shape[:2]
        bg_h, bg_w = background.shape[:2]

        # Ensure we don't go out of bounds
        end_x = min(x + w, bg_w)
        end_y = min(y + h, bg_h)
        obj_w = end_x - x
        obj_h = end_y - y

        if obj_w <= 0 or obj_h <= 0:
            return

        # Crop object if necessary
        obj_crop = obj_image[:obj_h, :obj_w]
        mask_crop = obj_mask[:obj_h, :obj_w]

        # Normalize mask
        mask_norm = mask_crop.astype(np.float32) / 255.0

        # Apply blending
        for c in range(background.shape[2]):
            background[y:end_y, x:end_x, c] = (
                background[y:end_y, x:end_x, c] * (1 - mask_norm) +
                obj_crop[:, :, c] * mask_norm
            )

    def _compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between boxes."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)
        intersection = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2 - intersection

        return intersection / union


class MosaicMixupAugmentation:
    """
    Advanced mosaic and mixup augmentation for detection tasks.
    Combines multiple images to create diverse training samples.
    """

    def __init__(
        self,
        mosaic_prob: float = 0.5,
        mixup_prob: float = 0.2,
        image_size: Tuple[int, int] = (1024, 1024)
    ):
        """
        Args:
            mosaic_prob: Probability of applying mosaic augmentation
            mixup_prob: Probability of applying mixup augmentation
            image_size: Target image size
        """
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.image_size = image_size

    def mosaic_augmentation(
        self,
        images: List[np.ndarray],
        targets: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create mosaic from 4 images.

        Args:
            images: List of 4 images
            targets: List of 4 targets

        Returns:
            Mosaic image and combined target
        """
        if len(images) != 4 or len(targets) != 4:
            raise ValueError("Mosaic requires exactly 4 images and targets")

        h, w = self.image_size

        # Create mosaic canvas
        mosaic_image = np.zeros((h, w, 3), dtype=np.uint8)

        # Random center point
        center_x = random.randint(w // 4, 3 * w // 4)
        center_y = random.randint(h // 4, 3 * h // 4)

        combined_boxes = []
        combined_labels = []

        # Place images in mosaic
        positions = [
            (0, 0, center_x, center_y),  # Top-left
            (center_x, 0, w, center_y),  # Top-right
            (0, center_y, center_x, h),  # Bottom-left
            (center_x, center_y, w, h)   # Bottom-right
        ]

        for i, (img, target) in enumerate(zip(images, targets)):
            x1, y1, x2, y2 = positions[i]
            section_w, section_h = x2 - x1, y2 - y1

            # Resize image to fit section
            img_resized = cv2.resize(img, (section_w, section_h))
            mosaic_image[y1:y2, x1:x2] = img_resized

            # Adjust boxes
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                labels = target['labels']

                if torch.is_tensor(boxes):
                    boxes = boxes.numpy()
                if torch.is_tensor(labels):
                    labels = labels.numpy()

                orig_h, orig_w = img.shape[:2]
                scale_x, scale_y = section_w / orig_w, section_h / orig_h

                for box, label in zip(boxes, labels):
                    # Scale and translate box
                    box_scaled = box * np.array([scale_x, scale_y, scale_x, scale_y])
                    box_translated = box_scaled + np.array([x1, y1, x1, y1])

                    # Check if box is valid
                    if (box_translated[2] > box_translated[0] and
                        box_translated[3] > box_translated[1] and
                        box_translated[0] < w and box_translated[1] < h and
                        box_translated[2] > 0 and box_translated[3] > 0):

                        combined_boxes.append(box_translated)
                        combined_labels.append(label)

        # Create combined target
        combined_target = {
            'boxes': torch.tensor(combined_boxes, dtype=torch.float32) if combined_boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(combined_labels, dtype=torch.int64) if combined_labels else torch.zeros(0, dtype=torch.int64)
        }

        return mosaic_image, combined_target

    def mixup_augmentation(
        self,
        image1: np.ndarray,
        target1: Dict[str, Any],
        image2: np.ndarray,
        target2: Dict[str, Any],
        alpha: float = 0.2
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply mixup augmentation to two images.

        Args:
            image1: First image
            target1: First target
            image2: Second image
            target2: Second target
            alpha: Mixup parameter

        Returns:
            Mixed image and combined target
        """
        # Sample lambda from beta distribution
        lam = np.random.beta(alpha, alpha)

        # Resize images to same size
        h, w = self.image_size
        image1_resized = cv2.resize(image1, (w, h))
        image2_resized = cv2.resize(image2, (w, h))

        # Mix images
        mixed_image = (lam * image1_resized + (1 - lam) * image2_resized).astype(np.uint8)

        # Combine targets
        boxes1 = target1.get('boxes', torch.zeros((0, 4)))
        labels1 = target1.get('labels', torch.zeros(0, dtype=torch.int64))
        boxes2 = target2.get('boxes', torch.zeros((0, 4)))
        labels2 = target2.get('labels', torch.zeros(0, dtype=torch.int64))

        # Scale boxes to new image size
        orig_h1, orig_w1 = image1.shape[:2]
        orig_h2, orig_w2 = image2.shape[:2]

        if len(boxes1) > 0:
            scale_x1, scale_y1 = w / orig_w1, h / orig_h1
            boxes1 = boxes1 * torch.tensor([scale_x1, scale_y1, scale_x1, scale_y1])

        if len(boxes2) > 0:
            scale_x2, scale_y2 = w / orig_w2, h / orig_h2
            boxes2 = boxes2 * torch.tensor([scale_x2, scale_y2, scale_x2, scale_y2])

        # Combine targets
        combined_boxes = torch.cat([boxes1, boxes2]) if len(boxes1) > 0 or len(boxes2) > 0 else torch.zeros((0, 4))
        combined_labels = torch.cat([labels1, labels2]) if len(labels1) > 0 or len(labels2) > 0 else torch.zeros(0, dtype=torch.int64)

        combined_target = {
            'boxes': combined_boxes,
            'labels': combined_labels,
            'mixup_lambda': lam  # Store lambda for loss computation if needed
        }

        return mixed_image, combined_target