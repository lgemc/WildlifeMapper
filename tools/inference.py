#!/usr/bin/env python3
"""
Wildlife Detection Inference Tool with Hydra Configuration

This script provides inference capabilities for the WildlifeMapper bbox model,
with support for visualization and comparison against ground truth data.

Usage Examples:
    # Single image inference with default config
    python tools/inference.py image_path=/path/to/image.jpg

    # Multiple images from directory
    python tools/inference.py image_dir=/path/to/images/ output_dir=outputs/

    # With custom model path and threshold
    python tools/inference.py image_path=/path/to/image.jpg model_path=exp/bbox_model/best_checkpoint.pth threshold=0.7

    # With ground truth comparison
    python tools/inference.py image_path=/path/to/image.jpg gt_csv=/path/to/gt.csv

    # Override config values
    python tools/inference.py image_dir=/path/to/images/ model.name=vit_b device=cpu
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import cv2
from glob import glob
from typing import Optional, List, Dict, Any

import hydra
from omegaconf import DictConfig, OmegaConf

# Add wildlifemapper to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'wildlifemapper'))

from segment_anything import sam_model_registry
from segment_anything.network import MedSAM
import segment_anything.utils.misc as utils


class WildlifeInferenceRunner:
    """Main class for running inference with the Wildlife Detection model."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize the inference runner with Hydra config.

        Args:
            cfg (DictConfig): Hydra configuration
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.threshold = cfg.get('threshold', 0.5)

        # Class names for wildlife species
        self.class_names = {
            1: "Shoats",
            2: "Cattle",
            3: "Impala",
            4: "Zebra",
            5: "Wildebeest",
            6: "Buffalo",
            7: "Other"
        }

        # Color dictionary for visualization
        self.colors = {
            1: (255, 105, 180), # Hot Pink - Shoats
            2: (255, 140, 0),   # Dark Orange - Cattle
            3: (0, 191, 255),   # Deep Sky Blue - Impala
            4: (255, 255, 224), # Light Yellow - Zebra
            5: (199, 21, 133),  # Medium Violet Red - Wildebeest
            6: (72, 61, 139),   # Dark Slate Blue - Buffalo
        }

        self._load_model()

    def _load_model(self):
        """Load the trained model using Hydra config."""
        model_path = self.cfg.get('model_path', 'exp/bbox_model/best_checkpoint.pth')
        print(f"Loading model from {model_path}")

        # Convert config to namespace for compatibility with existing code
        class Args:
            def __init__(self, cfg):
                for key, value in cfg.items():
                    setattr(self, key, value)
                # Set default values if not in config
                self.device = self.device if hasattr(self, 'device') else cfg.device
                self.bbox_loss_coef = getattr(self, 'bbox_loss_coef', 5)
                self.giou_loss_coef = getattr(self, 'giou_loss_coef', 2)
                self.eos_coef = getattr(self, 'eos_coef', 0.1)
                self.set_cost_class = getattr(self, 'set_cost_class', 1)
                self.set_cost_bbox = getattr(self, 'set_cost_bbox', 5)
                self.set_cost_giou = getattr(self, 'set_cost_giou', 2)

        # Create args from config
        args = Args(self.cfg)

        # Build the model
        model_name = self.cfg.get('model', {}).get('name', 'vit_l')
        model_checkpoint = self.cfg.get('model', {}).get('checkpoint', None)
        sam_model, criterion, postprocessors = sam_model_registry[model_name](
            checkpoint=model_checkpoint, args=args
        )

        self.model = MedSAM(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
        ).to(self.device)

        self.postprocessors = postprocessors

        # Load trained weights
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            print("Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self.model.eval()

    def preprocess_image(self, image_path: str):
        """
        Preprocess an image for inference.

        Args:
            image_path (str): Path to the image

        Returns:
            tuple: (preprocessed_tensor, original_image, original_size)
        """
        # Load image using PIL as recommended in the codebase
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)

        # Convert to numpy array
        image_np = np.array(image)

        # Resize to model input size
        img_size = self.cfg.get('img_size', 1024)
        image_resized = cv2.resize(image_np, (img_size, img_size))

        # Convert to tensor format (C, H, W)
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()

        # Normalize (SAM normalization)
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        image_tensor = (image_tensor - pixel_mean) / pixel_std

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Create NestedTensor-like structure for compatibility
        class ImageTensors:
            def __init__(self, tensors):
                self.tensors = tensors

            def to(self, device):
                self.tensors = self.tensors.to(device)
                return self

        return ImageTensors(image_tensor), image, original_size

    def run_inference(self, image_path: str) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            image_path (str): Path to the image

        Returns:
            dict: Detection results with boxes, scores, and labels
        """
        # Preprocess image
        image_tensor, original_image, original_size = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        # Create bounding box prompt (whole image)
        b, c, h, w = image_tensor.tensors.shape
        boxes_np = np.array([[0, 0, h, w]])

        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor, boxes_np)

        # Post-process results
        orig_target_sizes = torch.tensor([original_size[::-1]], device=self.device)  # (height, width)
        results = self.postprocessors['bbox'](outputs, orig_target_sizes)

        # Filter by confidence threshold
        result = results[0]
        keep = result['scores'] > self.threshold

        # Apply NMS
        import torchvision
        boxes = result['boxes'][keep]
        scores = result['scores'][keep]
        labels = result['labels'][keep]

        if len(boxes) > 0:
            nms_iou_threshold = self.cfg.get('nms_iou_threshold', 0.4)
            nms_indices = torchvision.ops.nms(boxes, scores, iou_threshold=nms_iou_threshold)
            boxes = boxes[nms_indices]
            scores = scores[nms_indices]
            labels = labels[nms_indices]

        return {
            'boxes': boxes.cpu().numpy(),
            'scores': scores.cpu().numpy(),
            'labels': labels.cpu().numpy(),
            'image_path': image_path,
            'original_size': original_size
        }

    def visualize_results(self, results: Dict[str, Any], output_path: Optional[str] = None,
                         gt_data: Optional[List[Dict]] = None) -> Image.Image:
        """
        Visualize detection results and optionally ground truth.

        Args:
            results (dict): Detection results from run_inference
            output_path (str): Path to save the visualization
            gt_data (list): Ground truth bounding boxes (optional)

        Returns:
            PIL.Image: Visualization image
        """
        # Load original image
        image = Image.open(results['image_path']).convert('RGB')
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.load_default()
        except:
            font = None

        # Draw predictions
        for i, (box, score, label) in enumerate(zip(results['boxes'], results['scores'], results['labels'])):
            # Get color for this class
            color = self.colors.get(int(label), (255, 255, 255))

            # Draw bounding box
            x1, y1, x2, y2 = box
            for thickness in range(3):
                draw.rectangle(
                    [(x1 - thickness, y1 - thickness), (x2 + thickness, y2 + thickness)],
                    outline=color,
                    width=1
                )

            # Draw label and confidence
            class_name = self.class_names.get(int(label), f"Class_{int(label)}")
            text = f"{class_name}: {score:.2f}"

            # Calculate text position
            if font:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            else:
                text_w, text_h = len(text) * 8, 15

            text_x = max(0, x1)
            text_y = max(0, y1 - text_h - 5)

            # Draw text background
            draw.rectangle(
                [(text_x, text_y), (text_x + text_w + 4, text_y + text_h + 4)],
                fill=color
            )

            # Draw text
            draw.text((text_x + 2, text_y + 2), text, fill=(255, 255, 255), font=font)

        # Draw ground truth if provided
        if gt_data:
            gt_color = (0, 255, 0)  # Green for ground truth
            for gt_box in gt_data:
                x1, y1, x2, y2 = gt_box['x_min'], gt_box['y_min'], gt_box['x_max'], gt_box['y_max']

                # Draw dashed line for GT (simulate with multiple small rectangles)
                dash_length = 5
                for x in range(int(x1), int(x2), dash_length * 2):
                    draw.rectangle([(x, y1), (min(x + dash_length, x2), y1 + 2)], fill=gt_color)
                    draw.rectangle([(x, y2), (min(x + dash_length, x2), y2 + 2)], fill=gt_color)
                for y in range(int(y1), int(y2), dash_length * 2):
                    draw.rectangle([(x1, y), (x1 + 2, min(y + dash_length, y2))], fill=gt_color)
                    draw.rectangle([(x2, y), (x2 + 2, min(y + dash_length, y2))], fill=gt_color)

                # GT label
                if 'label' in gt_box:
                    gt_text = f"GT: {gt_box['label']}"
                    gt_text_y = max(0, y1 - 40)
                    draw.text((x1, gt_text_y), gt_text, fill=gt_color, font=font)

        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path, quality=95)
            print(f"Visualization saved to: {output_path}")

        return image

    def load_ground_truth(self, csv_path: str, image_name: str) -> Optional[List[Dict]]:
        """
        Load ground truth data for a specific image.

        Args:
            csv_path (str): Path to CSV file with ground truth
            image_name (str): Name of the image file

        Returns:
            list: List of ground truth bounding boxes
        """
        if not os.path.exists(csv_path):
            return None

        try:
            df = pd.read_csv(csv_path)

            # Handle different column name variations
            image_col = 'images' if 'images' in df.columns else 'Image'

            # Filter for this image
            image_data = df[df[image_col] == image_name]

            gt_boxes = []
            for _, row in image_data.iterrows():
                # Handle different column name variations
                x_min_col = 'x_min' if 'x_min' in row else 'x1'
                y_min_col = 'y_min' if 'y_min' in row else 'y1'
                x_max_col = 'x_max' if 'x_max' in row else 'x2'
                y_max_col = 'y_max' if 'y_max' in row else 'y2'
                label_col = 'labels' if 'labels' in row else ('Label' if 'Label' in row else None)

                gt_box = {
                    'x_min': row[x_min_col],
                    'y_min': row[y_min_col],
                    'x_max': row[x_max_col],
                    'y_max': row[y_max_col],
                }

                if label_col:
                    gt_box['label'] = row[label_col]

                gt_boxes.append(gt_box)

            return gt_boxes

        except Exception as e:
            print(f"Error loading ground truth: {e}")
            return None

    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image with inference and optional visualization.

        Args:
            image_path (str): Path to the image

        Returns:
            dict: Results including detections and paths
        """
        print(f"Processing: {image_path}")

        # Run inference
        results = self.run_inference(image_path)

        # Load ground truth if provided
        gt_data = None
        if self.cfg.get('gt_csv'):
            image_name = os.path.basename(image_path)
            gt_data = self.load_ground_truth(self.cfg.gt_csv, image_name)

        # Save results if output directory specified
        if self.cfg.get('output_dir'):
            os.makedirs(self.cfg.output_dir, exist_ok=True)

            # Save detection results as JSON
            image_name = Path(image_path).stem
            json_path = os.path.join(self.cfg.output_dir, f"{image_name}_detections.json")

            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'image_path': results['image_path'],
                'original_size': results['original_size'],
                'detections': []
            }

            for box, score, label in zip(results['boxes'], results['scores'], results['labels']):
                json_results['detections'].append({
                    'bbox': box.tolist(),
                    'score': float(score),
                    'label': int(label),
                    'class_name': self.class_names.get(int(label), f"Class_{int(label)}")
                })

            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)

            print(f"Results saved to: {json_path}")

            # Create visualization if enabled
            if self.cfg.get('visualize', True):
                vis_path = os.path.join(self.cfg.output_dir, f"{image_name}_visualization.jpg")
                self.visualize_results(results, vis_path, gt_data)

        elif self.cfg.get('visualize', True):
            # Just show the visualization without saving
            self.visualize_results(results, gt_data=gt_data)

        return {
            'results': results,
            'gt_data': gt_data,
            'num_detections': len(results['boxes'])
        }

    def process_image_directory(self) -> List[Dict[str, Any]]:
        """
        Process all images in a directory.

        Returns:
            list: Results for all processed images
        """
        image_dir = self.cfg.image_dir

        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(image_dir, ext)))
            image_files.extend(glob(os.path.join(image_dir, ext.upper())))

        if not image_files:
            print(f"No image files found in {image_dir}")
            return []

        print(f"Found {len(image_files)} images to process")

        all_results = []
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")

            try:
                result = self.process_single_image(image_path)
                all_results.append(result)
                print(f"  Found {result['num_detections']} detections")
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                continue

        # Save summary if output directory specified
        if self.cfg.get('output_dir'):
            summary_path = os.path.join(self.cfg.output_dir, "processing_summary.json")
            summary = {
                'total_images': len(image_files),
                'successfully_processed': len(all_results),
                'total_detections': sum(r['num_detections'] for r in all_results)
            }

            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"Summary saved to: {summary_path}")

        return all_results


@hydra.main(version_base=None, config_path="../conf", config_name="inference_config")
def main(cfg: DictConfig) -> None:
    """Main function with Hydra configuration."""

    print("Inference Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Validate required parameters
    if not cfg.get('image_path') and not cfg.get('image_dir'):
        print("Error: Either 'image_path' or 'image_dir' must be specified")
        print("Usage examples:")
        print("  python tools/inference.py image_path=/path/to/image.jpg")
        print("  python tools/inference.py image_dir=/path/to/images/")
        return

    # Validate model path
    model_path = cfg.get('model_path', 'exp/bbox_model/best_checkpoint.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found: {model_path}")
        return

    # Initialize inference runner
    try:
        runner = WildlifeInferenceRunner(cfg)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # Process input
    try:
        if cfg.get('image_path'):
            # Single image processing
            if not os.path.exists(cfg.image_path):
                print(f"Error: Image not found: {cfg.image_path}")
                return

            result = runner.process_single_image(cfg.image_path)
            print(f"Detection completed. Found {result['num_detections']} objects.")

        elif cfg.get('image_dir'):
            # Directory processing
            if not os.path.exists(cfg.image_dir):
                print(f"Error: Directory not found: {cfg.image_dir}")
                return

            results = runner.process_image_directory()
            total_detections = sum(r['num_detections'] for r in results)
            print(f"Batch processing completed. Processed {len(results)} images with {total_detections} total detections.")

    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    main()