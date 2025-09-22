#!/usr/bin/env python3
"""
Script to fix COCO dataset class IDs from 1-6 to 0-5
This resolves the mismatch between dataset class IDs and model expectations.
"""

import json
import os
import shutil
from pathlib import Path

def fix_coco_class_ids(input_file, output_file=None, backup=True):
    """
    Fix COCO dataset class IDs by remapping 1-6 to 0-5

    Args:
        input_file: Path to input COCO JSON file
        output_file: Path to output file (defaults to overwriting input)
        backup: Whether to create a backup of the original file
    """
    input_path = Path(input_file)

    if output_file is None:
        output_file = input_file

    # Create backup if requested
    if backup and output_file == input_file:
        backup_path = input_path.with_suffix('.json.backup')
        print(f"Creating backup: {backup_path}")
        shutil.copy2(input_file, backup_path)

    # Load COCO data
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        coco_data = json.load(f)

    # Fix categories: remap IDs from 1-6 to 0-5
    print("Fixing category IDs...")
    for category in coco_data['categories']:
        old_id = category['id']
        new_id = old_id - 1  # Shift from 1-6 to 0-5
        category['id'] = new_id
        print(f"  Category '{category['name']}': {old_id} -> {new_id}")

    # Fix annotations: remap category_id from 1-6 to 0-5
    print("Fixing annotation category IDs...")
    fixed_count = 0
    for annotation in coco_data['annotations']:
        old_id = annotation['category_id']
        new_id = old_id - 1  # Shift from 1-6 to 0-5
        annotation['category_id'] = new_id
        fixed_count += 1

    print(f"Fixed {fixed_count} annotations")

    # Save fixed data
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print("✅ Class ID fix completed!")

    # Verify the fix
    unique_category_ids = set(cat['id'] for cat in coco_data['categories'])
    unique_annotation_ids = set(ann['category_id'] for ann in coco_data['annotations'])

    print(f"Categories now have IDs: {sorted(unique_category_ids)}")
    print(f"Annotations now use category IDs: {sorted(unique_annotation_ids)}")

    return coco_data

def main():
    """Fix all COCO dataset files"""
    dataset_dir = Path("/home/lmanrique/Do/WildlifeMapper/data/processed/coco_1024_2014")

    files_to_fix = [
        dataset_dir / "train.json",
        dataset_dir / "val.json",
        dataset_dir / "test.json"
    ]

    for file_path in files_to_fix:
        if file_path.exists():
            print(f"\n{'='*60}")
            print(f"Processing {file_path}")
            print('='*60)
            fix_coco_class_ids(file_path)
        else:
            print(f"⚠️  File not found: {file_path}")

if __name__ == "__main__":
    main()