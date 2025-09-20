#!/usr/bin/env python3
"""
Convert bounding box CSV format to COCO JSON format.

Input CSV format expected:
images,labels,base_images,x_min,y_min,x_max,y_max

Output: COCO JSON format with annotations for object detection.
"""

import argparse
import json
import os
import csv
from datetime import datetime
from pathlib import Path
from PIL import Image
from collections import defaultdict


def get_image_info(image_path):
    """Get image dimensions and file info."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"Warning: Could not read image {image_path}: {e}")
        return None, None


def convert_bbox_to_coco(data_configs, output_path):
    """Convert bbox CSV format to COCO JSON format from multiple data sources."""

    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "Wildlife dataset converted from bounding box CSV",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "WildlifeMapper",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Read CSV files from all data configs
    all_annotations_data = []
    for images_path, csv_path in data_configs:
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found at {csv_path}, skipping...")
            continue

        if not os.path.exists(images_path):
            print(f"Warning: Images directory not found at {images_path}, skipping...")
            continue

        print(f"Processing {csv_path} with images from {images_path}...")
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                all_annotations_data.append({
                    'image_name': row['images'],
                    'label': int(row['labels']),
                    'base_image': row['base_images'],
                    'x_min': int(row['x_min']),
                    'y_min': int(row['y_min']),
                    'x_max': int(row['x_max']),
                    'y_max': int(row['y_max']),
                    'images_path': images_path  # Store the path to find the image
                })

    annotations_data = all_annotations_data

    # Get unique categories
    unique_labels = sorted(set(ann['label'] for ann in annotations_data))
    categories = []
    for i, label in enumerate(unique_labels):
        categories.append({
            "id": label,
            "name": f"species_{label}",
            "supercategory": "animal"
        })
    coco_data["categories"] = categories

    # Group annotations by image
    image_annotations = defaultdict(list)
    for ann in annotations_data:
        image_annotations[ann['image_name']].append(ann)

    # Process images and annotations
    image_id = 1
    annotation_id = 1

    for image_name, annotations in image_annotations.items():
        # Use the images_path from the first annotation for this image
        first_ann_images_path = annotations[0]['images_path']
        image_path = os.path.join(first_ann_images_path, image_name)

        # Get image dimensions
        width, height = get_image_info(image_path)
        if width is None or height is None:
            print(f"Skipping image {image_name} - could not read dimensions")
            continue

        # Add image info
        image_info = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_name,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        }
        coco_data["images"].append(image_info)

        # Add annotations for this image
        for ann in annotations:
            x_min, y_min, x_max, y_max = ann['x_min'], ann['y_min'], ann['x_max'], ann['y_max']

            # Calculate COCO bbox format: [x, y, width, height]
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            area = bbox_width * bbox_height

            # Ensure bbox is within image bounds
            x_min = max(0, min(x_min, width))
            y_min = max(0, min(y_min, height))
            bbox_width = min(bbox_width, width - x_min)
            bbox_height = min(bbox_height, height - y_min)

            if bbox_width <= 0 or bbox_height <= 0:
                print(f"Warning: Invalid bbox for {image_name}: {ann}")
                continue

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": ann['label'],
                "segmentation": [],
                "area": area,
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation)
            annotation_id += 1

        image_id += 1

    # Write COCO JSON file
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"Conversion complete!")
    print(f"Images: {len(coco_data['images'])}")
    print(f"Annotations: {len(coco_data['annotations'])}")
    print(f"Categories: {len(coco_data['categories'])}")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert bounding box CSV to COCO JSON format')
    parser.add_argument('--images-paths', nargs='+', required=True,
                        help='Paths to directories containing images')
    parser.add_argument('--csv-paths', nargs='+', required=True,
                        help='Paths to CSV files with bounding box annotations')
    parser.add_argument('--output-path', required=True, help='Output path for COCO JSON file')

    args = parser.parse_args()

    # Validate that we have the same number of images-paths and csv-paths
    if len(args.images_paths) != len(args.csv_paths):
        print("Error: Number of --images-paths must match number of --csv-paths")
        return 1

    # Create data configs (pairs of images_path, csv_path)
    data_configs = list(zip(args.images_paths, args.csv_paths))

    # Validate inputs
    for images_path, csv_path in data_configs:
        if not os.path.isdir(images_path):
            print(f"Error: Images directory does not exist: {images_path}")
            return 1
        if not os.path.isfile(csv_path):
            print(f"Error: CSV file does not exist: {csv_path}")
            return 1

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert
    convert_bbox_to_coco(data_configs, args.output_path)
    return 0


if __name__ == "__main__":
    exit(main())