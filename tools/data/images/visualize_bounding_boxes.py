#!/usr/bin/env python3
"""
Script to visualize bounding boxes on images.

Usage:
    # Single image with manual coordinates
    python visualize_bounding_boxes.py --image_path path/to/image.jpg --x_min 100 --x_max 200 --y_min 50 --y_max 150

    # CSV input with row selection
    python visualize_bounding_boxes.py --csv_path data/images/1024-1024/val/gt.csv --rows 1,2,5-10 --output_dir outputs/

    # CSV input with all rows and image directory search
    python visualize_bounding_boxes.py --csv_path data/images/1024-1024/val/gt.csv --output_dir outputs/ --base_image_dir data/images/1024-1024/val/

    # CSV input with multiple image directories
    python visualize_bounding_boxes.py --csv_path data/images/1024-1024/val/gt.csv --output_dir outputs/ --image_dirs data/images/original/ data/images/processed/
"""

import argparse
import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path


def visualize_bounding_box(image_path, x_min, x_max, y_min, y_max, output_path=None, color=(0, 255, 0), thickness=2):
    """
    Visualize a bounding box on an image.

    Args:
        image_path (str): Path to the input image
        x_min (int): Left coordinate of bounding box
        x_max (int): Right coordinate of bounding box
        y_min (int): Top coordinate of bounding box
        y_max (int): Bottom coordinate of bounding box
        output_path (str, optional): Path to save the output image. If None, displays the image
        color (tuple): BGR color for the bounding box (default: green)
        thickness (int): Line thickness for the bounding box (default: 2)
    """
    # Load the image without applying EXIF rotations
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path, cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Validate coordinates
    h, w = image.shape[:2]
    if not (0 <= x_min < x_max <= w and 0 <= y_min < y_max <= h):
        print(f"Warning: Bounding box coordinates may be outside image bounds")
        print(f"Image size: {w}x{h}, Box: ({x_min},{y_min}) to ({x_max},{y_max})")

    # Draw the bounding box
    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)

    # Add coordinate text
    text = f"({x_min},{y_min})-({x_max},{y_max})"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = max(0, x_min)
    text_y = max(text_size[1] + 5, y_min - 5)

    # Add background rectangle for text
    cv2.rectangle(image, (text_x, text_y - text_size[1] - 2),
                  (text_x + text_size[0], text_y + 2), color, -1)
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if output_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(output_path, image)
        print(f"Output saved to: {output_path}")
    else:
        # Display the image
        cv2.imshow('Image with Bounding Box', image)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parse_row_selection(rows_str):
    """
    Parse row selection string like "1,2,5-10,15" into a list of row indices.

    Args:
        rows_str (str): Row selection string

    Returns:
        list: List of row indices (0-based)
    """
    if not rows_str:
        return None

    indices = []
    for part in rows_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.extend(range(start - 1, end))  # Convert to 0-based indexing
        else:
            indices.append(int(part) - 1)  # Convert to 0-based indexing

    return sorted(list(set(indices)))  # Remove duplicates and sort


def get_image_groups(df):
    """
    Group bounding boxes by image to draw multiple boxes on the same image.

    Args:
        df (pd.DataFrame): DataFrame with bounding box data

    Returns:
        dict: Dictionary mapping image paths to list of bounding box data
    """
    groups = {}
    for _, row in df.iterrows():
        # Handle different column name variations
        image_col = 'images' if 'images' in row else 'Image'
        x_min_col = 'x_min' if 'x_min' in row else 'x1'
        y_min_col = 'y_min' if 'y_min' in row else 'y1'
        x_max_col = 'x_max' if 'x_max' in row else 'x2'
        y_max_col = 'y_max' if 'y_max' in row else 'y2'
        label_col = 'labels' if 'labels' in row else ('Label' if 'Label' in row else None)

        image_path = row[image_col]
        if image_path not in groups:
            groups[image_path] = []

        groups[image_path].append({
            'x_min': row[x_min_col],
            'y_min': row[y_min_col],
            'x_max': row[x_max_col],
            'y_max': row[y_max_col],
            'label': row[label_col] if label_col else None
        })

    return groups


def find_image_path(image_name, base_image_dir, image_dirs=None):
    """
    Find the full path to an image file by searching in multiple directories.

    Args:
        image_name (str): Name of the image file
        base_image_dir (str): Primary directory to search
        image_dirs (list): Additional directories to search

    Returns:
        str: Full path to the image file or None if not found
    """
    search_dirs = []

    # Add base directory if provided
    if base_image_dir:
        search_dirs.append(base_image_dir)

    # Add additional directories if provided
    if image_dirs:
        search_dirs.extend(image_dirs)

    # Try the image name as-is first (in case it's already a full path)
    if os.path.exists(image_name):
        return image_name

    # Search in all directories
    for search_dir in search_dirs:
        full_path = os.path.join(search_dir, image_name)
        if os.path.exists(full_path):
            return full_path

    return None


def visualize_multiple_boxes(image_path, boxes_data, output_path, base_image_dir=None, image_dirs=None):
    """
    Visualize multiple bounding boxes on a single image.

    Args:
        image_path (str): Path to the image file
        boxes_data (list): List of dictionaries with bounding box data
        output_path (str): Path to save the output image
        base_image_dir (str): Directory containing the base images
        image_dirs (list): Additional directories to search for images
    """
    # Try to find the image file
    full_image_path = find_image_path(image_path, base_image_dir, image_dirs)

    if not full_image_path:
        print(f"Warning: Image not found: {image_path}")
        if base_image_dir:
            print(f"  Searched in: {base_image_dir}")
        if image_dirs:
            print(f"  Also searched in: {image_dirs}")
        return False

    # Load the image without applying EXIF rotations
    image = cv2.imread(full_image_path, cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        print(f"Warning: Could not load image: {full_image_path}")
        return False

    # Color map for different labels
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]

    # Draw all bounding boxes
    for i, box_data in enumerate(boxes_data):
        # Use different colors for different labels or cycle through colors
        if box_data['label'] is not None:
            color_idx = int(box_data['label']) % len(colors)
        else:
            color_idx = i % len(colors)
        color = colors[color_idx]

        # Draw the bounding box
        cv2.rectangle(image,
                      (int(box_data['x_min']), int(box_data['y_min'])),
                      (int(box_data['x_max']), int(box_data['y_max'])),
                      color, 2)

        # Add label text if available
        if box_data['label'] is not None:
            label_text = f"Label: {box_data['label']}"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = max(0, box_data['x_min'])
            text_y = max(text_size[1] + 5, box_data['y_min'] - 5)

            # Add background rectangle for text
            cv2.rectangle(image, (text_x, text_y - text_size[1] - 2),
                          (text_x + text_size[0], text_y + 2), color, -1)
            cv2.putText(image, label_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the image
    cv2.imwrite(output_path, image)
    return True


def process_csv(csv_path, rows_selection, output_dir, base_image_dir=None, image_dirs=None):
    """
    Process CSV file and generate visualization images.

    Args:
        csv_path (str): Path to the CSV file
        rows_selection (str): Row selection string
        output_dir (str): Output directory for generated images
        base_image_dir (str): Directory containing the base images
        image_dirs (list): Additional directories to search for images
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows")

    # Filter rows if specified
    if rows_selection:
        selected_indices = parse_row_selection(rows_selection)
        if selected_indices:
            # Validate indices
            valid_indices = [i for i in selected_indices if 0 <= i < len(df)]
            if len(valid_indices) != len(selected_indices):
                print(f"Warning: Some row indices are out of range. Using {len(valid_indices)} valid indices.")
            df = df.iloc[valid_indices]
            print(f"Selected {len(df)} rows for processing")

    # Group by image
    image_groups = get_image_groups(df)
    print(f"Processing {len(image_groups)} unique images")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each image group
    processed_count = 0
    for image_path, boxes_data in image_groups.items():
        # Generate output filename
        image_name = Path(image_path).stem
        output_filename = f"{image_name}_bbox.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Visualize the image with all its bounding boxes
        success = visualize_multiple_boxes(image_path, boxes_data, output_path, base_image_dir, image_dirs)
        if success:
            processed_count += 1
            print(f"Generated: {output_path} ({len(boxes_data)} boxes)")
        else:
            print(f"Failed to process: {image_path}")

    print(f"Successfully processed {processed_count}/{len(image_groups)} images")


def main():
    parser = argparse.ArgumentParser(description='Visualize bounding boxes on images')

    # Create mutually exclusive groups for single image vs CSV modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--image_path', type=str,
                           help='Path to the input image (for single image mode)')
    mode_group.add_argument('--csv_path', type=str,
                           help='Path to CSV file with bounding box data')

    # Single image mode arguments
    single_group = parser.add_argument_group('single image mode')
    single_group.add_argument('--x_min', type=int,
                             help='Left coordinate of bounding box')
    single_group.add_argument('--x_max', type=int,
                             help='Right coordinate of bounding box')
    single_group.add_argument('--y_min', type=int,
                             help='Top coordinate of bounding box')
    single_group.add_argument('--y_max', type=int,
                             help='Bottom coordinate of bounding box')
    single_group.add_argument('--output_path', type=str, default=None,
                             help='Path to save the output image (optional)')
    single_group.add_argument('--color', type=str, default='0,255,0',
                             help='BGR color for bounding box as comma-separated values (default: 0,255,0 for green)')
    single_group.add_argument('--thickness', type=int, default=2,
                             help='Line thickness for bounding box (default: 2)')

    # CSV mode arguments
    csv_group = parser.add_argument_group('CSV mode')
    csv_group.add_argument('--rows', type=str,
                          help='Row selection (1-based indexing). Examples: "1,2,5", "1-10", "1,3-5,10"')
    csv_group.add_argument('--image_name', type=str,
                          help='Filter by specific image name (e.g., "L_07_05_16_DSC00604.JPG")')
    csv_group.add_argument('--output_dir', type=str, default='outputs',
                          help='Output directory for generated images (default: outputs)')
    csv_group.add_argument('--base_image_dir', type=str,
                          help='Directory containing the base images (optional)')
    csv_group.add_argument('--image_dirs', type=str, nargs='+',
                          help='Additional directories to search for images (space-separated)')

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.image_path:
        # Single image mode - validate required arguments
        if None in [args.x_min, args.x_max, args.y_min, args.y_max]:
            parser.error("Single image mode requires --x_min, --x_max, --y_min, --y_max")

        # Parse color
        try:
            color = tuple(map(int, args.color.split(',')))
            if len(color) != 3:
                raise ValueError("Color must have 3 values (BGR)")
        except ValueError as e:
            print(f"Error parsing color: {e}")
            print("Using default green color")
            color = (0, 255, 0)

        # Generate default output path if not provided
        if args.output_path is None:
            input_path = Path(args.image_path)
            output_name = f"{input_path.stem}_bbox{input_path.suffix}"
            args.output_path = str(input_path.parent / output_name)

        try:
            visualize_bounding_box(
                args.image_path,
                args.x_min,
                args.x_max,
                args.y_min,
                args.y_max,
                args.output_path,
                color,
                args.thickness
            )
            print(f"Output saved to: {args.output_path}")
        except Exception as e:
            print(f"Error: {e}")
            return 1

    elif args.csv_path:
        # CSV mode
        if not os.path.exists(args.csv_path):
            print(f"Error: CSV file not found: {args.csv_path}")
            return 1

        # Check for conflicting arguments
        if args.rows and args.image_name:
            print("Error: Cannot use both --rows and --image_name options simultaneously.")
            print("Please use either --rows for row selection or --image_name for image filtering.")
            return 1

        try:
            process_csv(args.csv_path, args.rows, args.output_dir, args.base_image_dir, args.image_dirs, args.image_name)
        except Exception as e:
            print(f"Error processing CSV: {e}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())