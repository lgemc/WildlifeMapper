#!/usr/bin/env python3
"""
Script to visualize COCO format annotations on images.

Usage:
    # Visualize specific image by ID
    python visualize_coco_annotations.py --json_path data/processed/coco_1024_2014/val.json --image_id 1 --output_dir outputs/

    # Visualize specific image by filename
    python visualize_coco_annotations.py --json_path data/processed/coco_1024_2014/val.json --image_name "005c952ba7a612c40986806cc84a87e1573ef4f2_1.JPG" --output_dir outputs/

    # Visualize multiple images by IDs
    python visualize_coco_annotations.py --json_path data/processed/coco_1024_2014/val.json --image_ids 1,2,5-10 --output_dir outputs/

    # Visualize all images (be careful with large datasets)
    python visualize_coco_annotations.py --json_path data/processed/coco_1024_2014/val.json --output_dir outputs/ --all

    # Specify image directory if images are not in same directory as JSON
    python visualize_coco_annotations.py --json_path data/processed/coco_1024_2014/val.json --image_id 1 --output_dir outputs/ --image_dirs data/images/original/ data/images/processed/
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def load_coco_data(json_path):
    """
    Load COCO format JSON data.

    Args:
        json_path (str): Path to COCO JSON file

    Returns:
        dict: COCO data with images, annotations, and categories
    """
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    print(f"Loaded COCO data with:")
    print(f"  - {len(coco_data['images'])} images")
    print(f"  - {len(coco_data['annotations'])} annotations")
    print(f"  - {len(coco_data['categories'])} categories")

    return coco_data


def create_category_map(categories):
    """
    Create mapping from category ID to category info.

    Args:
        categories (list): List of category dictionaries

    Returns:
        dict: Mapping from category_id to category info
    """
    return {cat['id']: cat for cat in categories}


def create_image_map(images):
    """
    Create mapping from image ID to image info.

    Args:
        images (list): List of image dictionaries

    Returns:
        dict: Mapping from image_id to image info
    """
    return {img['id']: img for img in images}


def group_annotations_by_image(annotations):
    """
    Group annotations by image ID.

    Args:
        annotations (list): List of annotation dictionaries

    Returns:
        dict: Mapping from image_id to list of annotations
    """
    grouped = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in grouped:
            grouped[image_id] = []
        grouped[image_id].append(ann)

    return grouped


def find_image_path(image_filename, base_dir=None, image_dirs=None):
    """
    Find the full path to an image file by searching in multiple directories.

    Args:
        image_filename (str): Name of the image file
        base_dir (str): Base directory containing JSON file
        image_dirs (list): Additional directories to search

    Returns:
        str: Full path to the image file or None if not found
    """
    search_dirs = []

    # Add base directory if provided
    if base_dir:
        search_dirs.append(base_dir)
        search_dirs.append(os.path.join(base_dir, 'images'))  # Common COCO structure

    # Add additional directories if provided
    if image_dirs:
        search_dirs.extend(image_dirs)

    # Try the filename as-is first (in case it's already a full path)
    if os.path.exists(image_filename):
        return image_filename

    # Search in all directories
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        full_path = os.path.join(search_dir, image_filename)
        if os.path.exists(full_path):
            return full_path

    return None


def parse_id_selection(ids_str):
    """
    Parse ID selection string like "1,2,5-10,15" into a list of IDs.

    Args:
        ids_str (str): ID selection string

    Returns:
        list: List of IDs
    """
    if not ids_str:
        return None

    ids = []
    for part in ids_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            ids.extend(range(start, end + 1))
        else:
            ids.append(int(part))

    return sorted(list(set(ids)))  # Remove duplicates and sort


def visualize_coco_image(image_info, annotations, category_map, output_path, base_dir=None, image_dirs=None):
    """
    Visualize COCO annotations on a single image.

    Args:
        image_info (dict): Image information from COCO data
        annotations (list): List of annotations for this image
        category_map (dict): Mapping from category_id to category info
        output_path (str): Path to save the output image
        base_dir (str): Base directory containing JSON file
        image_dirs (list): Additional directories to search for images

    Returns:
        bool: Success status
    """
    # Find the image file
    image_filename = image_info['file_name']
    image_path = find_image_path(image_filename, base_dir, image_dirs)

    if not image_path:
        print(f"Warning: Image not found: {image_filename}")
        if base_dir:
            print(f"  Searched in: {base_dir}")
        if image_dirs:
            print(f"  Also searched in: {image_dirs}")
        return False

    # Load the image
    try:
        image = Image.open(image_path)
        image = image.convert('RGB')  # Ensure RGB mode
    except Exception as e:
        print(f"Warning: Could not load image: {image_path} - {e}")
        return False

    # Create drawing context
    draw = ImageDraw.Draw(image)

    # Color map for different categories
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (255, 192, 203), # Pink
        (165, 42, 42),  # Brown
    ]

    try:
        font = ImageFont.load_default()
    except:
        font = None

    # Draw all annotations
    for ann in annotations:
        # Get category info
        category_id = ann['category_id']
        category_info = category_map.get(category_id, {'name': f'category_{category_id}'})
        category_name = category_info['name']

        # Get color for this category
        color_idx = (category_id - 1) % len(colors)
        color = colors[color_idx]

        # COCO bbox format: [x, y, width, height]
        bbox = ann['bbox']
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[0] + bbox[2])
        y_max = int(bbox[1] + bbox[3])

        # Draw bounding box with thickness of 2
        thickness = 2
        for i in range(thickness):
            draw.rectangle(
                [(x_min - i, y_min - i), (x_max + i, y_max + i)],
                outline=color,
                width=1
            )

        # Add category label
        label_text = f"{category_name} (ID: {category_id})"

        # Get text size
        if font:
            bbox_text = draw.textbbox((0, 0), label_text, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
        else:
            text_w, text_h = len(label_text) * 6, 11

        text_x = max(0, x_min)
        text_y = max(0, y_min - text_h - 5)

        # Add background rectangle for text
        draw.rectangle(
            [(text_x, text_y), (text_x + text_w + 4, text_y + text_h + 4)],
            fill=color
        )

        # Add text
        draw.text((text_x + 2, text_y + 2), label_text, fill=(255, 255, 255), font=font)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the image
    image.save(output_path, quality=95)
    return True


def process_coco_annotations(json_path, output_dir, image_id=None, image_name=None, image_ids=None,
                           visualize_all=False, base_dir=None, image_dirs=None):
    """
    Process COCO annotations and generate visualization images.

    Args:
        json_path (str): Path to COCO JSON file
        output_dir (str): Output directory for generated images
        image_id (int): Single image ID to visualize
        image_name (str): Single image filename to visualize
        image_ids (str): Multiple image IDs to visualize
        visualize_all (bool): Visualize all images
        base_dir (str): Base directory containing JSON file
        image_dirs (list): Additional directories to search for images
    """
    # Load COCO data
    coco_data = load_coco_data(json_path)

    # Create mappings
    category_map = create_category_map(coco_data['categories'])
    image_map = create_image_map(coco_data['images'])
    annotation_groups = group_annotations_by_image(coco_data['annotations'])

    # Determine which images to process
    target_image_ids = []

    if image_id is not None:
        target_image_ids = [image_id]
    elif image_name is not None:
        # Find image by filename
        for img_id, img_info in image_map.items():
            if img_info['file_name'] == image_name:
                target_image_ids = [img_id]
                break
        if not target_image_ids:
            print(f"Error: Image '{image_name}' not found in dataset")
            return
    elif image_ids is not None:
        target_image_ids = parse_id_selection(image_ids)
    elif visualize_all:
        target_image_ids = list(image_map.keys())
        print(f"Warning: Visualizing all {len(target_image_ids)} images. This may take a while.")
    else:
        print("Error: Must specify image_id, image_name, image_ids, or --all")
        return

    # Validate image IDs
    valid_image_ids = []
    for img_id in target_image_ids:
        if img_id in image_map:
            valid_image_ids.append(img_id)
        else:
            print(f"Warning: Image ID {img_id} not found in dataset")

    print(f"Processing {len(valid_image_ids)} images")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each image
    processed_count = 0
    for img_id in valid_image_ids:
        image_info = image_map[img_id]
        annotations = annotation_groups.get(img_id, [])

        # Generate output filename
        image_name = Path(image_info['file_name']).stem
        output_filename = f"{image_name}_coco_ann.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Visualize the image
        success = visualize_coco_image(
            image_info, annotations, category_map, output_path, base_dir, image_dirs
        )

        if success:
            processed_count += 1
            print(f"Generated: {output_path} ({len(annotations)} annotations)")
        else:
            print(f"Failed to process: {image_info['file_name']}")

    print(f"Successfully processed {processed_count}/{len(valid_image_ids)} images")


def main():
    parser = argparse.ArgumentParser(description='Visualize COCO format annotations on images')

    # Required arguments
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to COCO format JSON file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for generated images (default: outputs)')

    # Image selection (mutually exclusive)
    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument('--image_id', type=int,
                                help='Single image ID to visualize')
    selection_group.add_argument('--image_name', type=str,
                                help='Single image filename to visualize')
    selection_group.add_argument('--image_ids', type=str,
                                help='Multiple image IDs to visualize (e.g., "1,2,5-10")')
    selection_group.add_argument('--all', action='store_true',
                                help='Visualize all images in the dataset')

    # Optional arguments
    parser.add_argument('--base_dir', type=str,
                       help='Base directory containing images (defaults to JSON file directory)')
    parser.add_argument('--image_dirs', type=str, nargs='+',
                       help='Additional directories to search for images (space-separated)')

    args = parser.parse_args()

    # Validate JSON file
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found: {args.json_path}")
        return 1

    # Set base directory if not provided
    if args.base_dir is None:
        args.base_dir = os.path.dirname(args.json_path)

    try:
        process_coco_annotations(
            args.json_path,
            args.output_dir,
            args.image_id,
            args.image_name,
            args.image_ids,
            args.all,
            args.base_dir,
            args.image_dirs
        )
    except Exception as e:
        print(f"Error processing COCO annotations: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())