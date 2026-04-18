import os
import glob
import logging

from PIL import Image
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def image_generator(dataset_path):
    png_files = glob.glob(os.path.join(dataset_path, "**", "*.png"), recursive=True)
    
    for img_path in png_files:
        try:
            img = Image.open(img_path)
            img_array = np.asarray(img)

            yield img_path, img_array
        # Handle corrupted files
        except Exception as e:
            # print(f"\nWarning: Could not read {img_path}: {e}")
            yield img_path, None


def count_black_pixels_from_side(img_array, side):
    if side == 'top':
        for i in range(0, img_array.shape[0], 1):
            if not np.all(img_array[i, :] == 0):
                return i

        return img_array.shape[0]
    
    if side == 'bottom':
        for i in range(img_array.shape[0] - 1, -1, -1):
            if not np.all(img_array[i, :] == 0):
                return img_array.shape[0] - 1 - i

        return img_array.shape[0]
    
    if side == 'left':
        for i in range(0, img_array.shape[1], 1):
            if not np.all(img_array[:, i] == 0):
                return i

        return img_array.shape[1]
    
    if side == 'right':
        for i in range(img_array.shape[1] - 1, -1, -1):
            if not np.all(img_array[:, i] == 0):
                return img_array.shape[1] - 1 - i

        return img_array.shape[1]
    
    return 0


def calculate_border(dataset_path):
    png_files = glob.glob(os.path.join(dataset_path, "**", "*.png"), recursive=True)
    total_images = len(png_files)
    
    logger.info(f"Dataset size: {len(png_files)}")
    
    min_top = float('inf')
    min_bottom = float('inf')
    min_left = float('inf')
    min_right = float('inf')
    
    processed_count = 0

    for img_path, img_array in tqdm(image_generator(dataset_path), 
                                     total=total_images, 
                                     desc="Calculating border"):
        if img_array is None:
            logger.warning(f"Corrupt image - {img_path}")
            continue
        
        top = count_black_pixels_from_side(img_array, 'top')
        bottom = count_black_pixels_from_side(img_array, 'bottom')
        left = count_black_pixels_from_side(img_array, 'left')
        right = count_black_pixels_from_side(img_array, 'right')
        
        min_top = min(min_top, top)
        min_bottom = min(min_bottom, bottom)
        min_left = min(min_left, left)
        min_right = min(min_right, right)

        processed_count += 1
    
    logger.info(f"Number of images analyzed to border calculation: {processed_count}")
    logger.info(f"Minimal Border\nTop: {min_top}\nBottom: {min_bottom}\nLeft: {min_left}\nRight: {min_right}")
    
    return min_top, min_bottom, min_left, min_right


if __name__ == "__main__":
    dataset_path = "./alpnach2018"
    
    top, bottom, left, right = calculate_border(dataset_path)
    