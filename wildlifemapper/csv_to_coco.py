#!/usr/bin/env python3
"""
CSV to COCO JSON Converter for WildlifeMapper

Converts HerdNet-style CSV annotations to COCO JSON format for training WildlifeMapper.
Supports both point annotations and bounding box annotations.

Usage:
    python csv_to_coco.py --csv_file path/to/annotations.csv --images_dir path/to/images --output_json path/to/output.json
"""

import argparse
import json
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict
from PIL import Image
import sys


def get_image_dimensions(image_path):
    """Get image width and height."""
    try:
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        print(f"Warning: Could not read image {image_path}: {e}")
        return None, None


def point_to_bbox(x, y, bbox_size=10):
    """Convert point annotation to bounding box.

    Args:
        x, y: Point coordinates
        bbox_size: Size of the bounding box around the point

    Returns:
        bbox in COCO format [x_min, y_min, width, height]
    """
    half_size = bbox_size // 2
    x_min = max(0, x - half_size)
    y_min = max(0, y - half_size)
    width = bbox_size
    height = bbox_size
    return [x_min, y_min, width, height]


def validate_csv_headers(df):
    """Validate CSV headers match HerdNet format."""
    # Check for point format: images,x,y,labels
    if set(df.columns) == {'images', 'x', 'y', 'labels'}:
        return 'point'
    # Check for bbox format: images,x_min,y_min,x_max,y_max,labels
    elif set(df.columns) == {'images', 'x_min', 'y_min', 'x_max', 'y_max', 'labels'}:
        return 'bbox'
    else:
        raise ValueError(f"Invalid CSV headers. Expected either:\n"
                        f"  Point format: 'images,x,y,labels'\n"
                        f"  Bbox format: 'images,x_min,y_min,x_max,y_max,labels'\n"
                        f"  Got: {list(df.columns)}")


def convert_csv_to_coco(csv_file, images_dir, output_json, bbox_size=10):
    """Convert CSV annotations to COCO JSON format.

    Args:
        csv_file: Path to CSV file with annotations
        images_dir: Directory containing the images
        output_json: Output path for COCO JSON file
        bbox_size: Size of bounding box for point annotations
    """

    # Read CSV file
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Found {len(df)} annotations")

    # Validate headers and determine format
    annotation_format = validate_csv_headers(df)
    print(f"Detected annotation format: {annotation_format}")

    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "Wildlife dataset converted from CSV",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "CSV to COCO Converter",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown License",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Group annotations by image
    grouped = df.groupby('images')

    # Get unique categories
    unique_labels = sorted(df['labels'].unique())
    print(f"Found {len(unique_labels)} unique categories: {unique_labels}")

    # Create categories
    for idx, label in enumerate(unique_labels):
        coco_data["categories"].append({
            "id": int(label),
            "name": f"class_{label}",
            "supercategory": "animal"
        })

    # Process images and annotations
    image_id = 1
    annotation_id = 1

    for image_name, annotations in grouped:
        image_path = os.path.join(images_dir, image_name)

        # Get image dimensions
        width, height = get_image_dimensions(image_path)
        if width is None or height is None:
            print(f"Skipping {image_name} - could not read image dimensions")
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

        # Process annotations for this image
        for _, row in annotations.iterrows():
            if annotation_format == 'point':
                # Convert point to bounding box
                x, y = float(row['x']), float(row['y'])
                bbox = point_to_bbox(x, y, bbox_size)
                area = bbox_size * bbox_size
            else:  # bbox format
                # Convert from x_min,y_min,x_max,y_max to x_min,y_min,width,height
                x_min, y_min = float(row['x_min']), float(row['y_min'])
                x_max, y_max = float(row['x_max']), float(row['y_max'])
                width_bbox = x_max - x_min
                height_bbox = y_max - y_min
                bbox = [x_min, y_min, width_bbox, height_bbox]
                area = width_bbox * height_bbox

            # Ensure bbox is within image bounds
            bbox[0] = max(0, min(bbox[0], width - 1))
            bbox[1] = max(0, min(bbox[1], height - 1))
            bbox[2] = max(1, min(bbox[2], width - bbox[0]))
            bbox[3] = max(1, min(bbox[3], height - bbox[1]))

            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(row['labels']),
                "bbox": bbox,
                "area": area,
                "segmentation": [],
                "iscrowd": 0
            }

            coco_data["annotations"].append(annotation_info)
            annotation_id += 1

        image_id += 1

    # Save COCO JSON
    print(f"Saving COCO JSON to: {output_json}")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"Conversion complete!")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(coco_data['categories'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HerdNet CSV annotations to COCO JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert point annotations
  python csv_to_coco.py --csv_file train_points.csv --images_dir ./images --output_json ./annotations/train.json

  # Convert bbox annotations
  python csv_to_coco.py --csv_file train_bbox.csv --images_dir ./images --output_json ./annotations/train.json --bbox_size 20

  # Use with HerdNet data
  python csv_to_coco.py --csv_file ../HerdNet/data/groundtruth/csv/train_big_size_A_B_E_K_WH_WB_points.csv --images_dir ../HerdNet/data/train --output_json ./coco_annotations/train.json
        """
    )

    parser.add_argument(
        '--csv_file',
        type=str,
        required=True,
        help='Path to CSV file with annotations (HerdNet format)'
    )

    parser.add_argument(
        '--images_dir',
        type=str,
        required=True,
        help='Directory containing the images'
    )

    parser.add_argument(
        '--output_json',
        type=str,
        required=True,
        help='Output path for COCO JSON file'
    )

    parser.add_argument(
        '--bbox_size',
        type=int,
        default=10,
        help='Size of bounding box for point annotations (default: 10)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        sys.exit(1)

    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        sys.exit(1)

    # Convert
    try:
        convert_csv_to_coco(
            csv_file=args.csv_file,
            images_dir=args.images_dir,
            output_json=args.output_json,
            bbox_size=args.bbox_size
        )
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()