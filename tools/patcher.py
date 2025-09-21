__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


import argparse
import os
import PIL
import torchvision
import numpy
import cv2
import pandas

from albumentations import PadIfNeeded

from tqdm import tqdm

from wildlifemapper.data import ImageToPatches, PatchesBuffer, save_batch_images

parser = argparse.ArgumentParser(prog='patcher', description='Cut images into patches')

parser.add_argument('root', type=str,
    help='path to the images directory (str)')
parser.add_argument('height', type=int,
    help='height of the patches, in pixels (int)')
parser.add_argument('width', type=int,
    help='width of the patches, in pixels (int)')
parser.add_argument('overlap', type=int,
    help='overlap between patches, in pixels (int)')
parser.add_argument('dest', type=str,
    help='destination path (str)')
parser.add_argument('-csv', type=str,
    help='path to a csv file containing annotations (str). Defaults to None')
parser.add_argument('-min', type=float, default=0.1,
    help='minimum fraction of area for an annotation to be kept (float). Defautls to 0.1')
parser.add_argument('-all', type=bool, default=False,
    help='set to True to save all patches, not only those containing annotations (bool). Defaults to False')
parser.add_argument('--csv-header-images', type=str, default='images',
    help='name of the column containing image names (str). Defaults to "images"')
parser.add_argument('--csv-header-labels', type=str, default='labels',
    help='name of the column containing labels (str). Defaults to "labels"')
parser.add_argument('--csv-header-x1', type=str, default='x_min',
    help='name of the column containing x1/x_min coordinates (str). Defaults to "x_min"')
parser.add_argument('--csv-header-y1', type=str, default='y_min',
    help='name of the column containing y1/y_min coordinates (str). Defaults to "y_min"')
parser.add_argument('--csv-header-x2', type=str, default='x_max',
    help='name of the column containing x2/x_max coordinates (str). Defaults to "x_max"')
parser.add_argument('--csv-header-y2', type=str, default='y_max',
    help='name of the column containing y2/y_max coordinates (str). Defaults to "y_max"')

args = parser.parse_args()

def preprocess_csv_columns(csv_path, images_col, labels_col, x1_col, y1_col, x2_col, y2_col):
    """Preprocess CSV to rename custom column names to expected format"""
    df = pandas.read_csv(csv_path)

    # Create a mapping of custom column names to expected column names
    column_mapping = {
        images_col: 'images',
        labels_col: 'labels',
        x1_col: 'x_min',
        y1_col: 'y_min',
        x2_col: 'x_max',
        y2_col: 'y_max'
    }

    # Only rename columns that exist and are different from expected names
    rename_dict = {}
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and old_name != new_name:
            rename_dict[old_name] = new_name

    if rename_dict:
        df = df.rename(columns=rename_dict)

    return df

def main():

    images_paths = [os.path.join(args.root, p) for p in os.listdir(args.root) if not p.endswith('.csv')]

    if args.csv is not None:
        # Preprocess CSV to handle custom column names
        processed_df = preprocess_csv_columns(
            args.csv,
            getattr(args, 'csv_header_images'),
            getattr(args, 'csv_header_labels'),
            getattr(args, 'csv_header_x1'),
            getattr(args, 'csv_header_y1'),
            getattr(args, 'csv_header_x2'),
            getattr(args, 'csv_header_y2')
        )

        # Save processed CSV temporarily for PatchesBuffer to use
        temp_csv_path = os.path.join(args.dest, 'temp_processed.csv')
        os.makedirs(args.dest, exist_ok=True)
        processed_df.to_csv(temp_csv_path, index=False)

        patches_buffer = PatchesBuffer(temp_csv_path, args.root, (args.height, args.width), overlap=args.overlap, min_visibility=args.min, images_col='images', labels_col='labels').buffer
        patches_buffer.drop(columns='limits').to_csv(os.path.join(args.dest, 'gt.csv'), index=False)

        # Clean up temporary CSV
        os.remove(temp_csv_path)

        if not args.all:
            images_paths = [os.path.join(args.root, x) for x in processed_df['images'].unique()]

    for img_path in tqdm(images_paths, desc='Exporting patches'):
        pil_img = PIL.Image.open(img_path)
        img_tensor = torchvision.transforms.ToTensor()(pil_img)
        img_name = os.path.basename(img_path)

        if args.csv is not None:
            # save all patches
            if args.all:
                patches = ImageToPatches(img_tensor, (args.height, args.width), overlap=args.overlap).make_patches()
                save_batch_images(patches, img_name, args.dest)

            # or only annotated ones
            else:
                padder = PadIfNeeded(
                    args.height, args.width,
                    position = PadIfNeeded.PositionType.TOP_LEFT,
                    border_mode = cv2.BORDER_CONSTANT,
                    value= 0
                    )
                img_ptch_df = patches_buffer[patches_buffer['base_images']==img_name]
                for row in img_ptch_df[['images','limits']].to_numpy().tolist():
                    ptch_name, limits = row[0], row[1]
                    cropped_img = numpy.array(pil_img.crop(limits.get_tuple))
                    padded_img = PIL.Image.fromarray(padder(image = cropped_img)['image'])
                    padded_img.save(os.path.join(args.dest, ptch_name))

        else:
            patches = ImageToPatches(img_tensor, (args.height, args.width), overlap=args.overlap).make_patches()
            save_batch_images(patches, img_name, args.dest)


if __name__ == '__main__':
    main()