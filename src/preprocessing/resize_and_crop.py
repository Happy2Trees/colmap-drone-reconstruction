"""
Preprocess images by resizing and center cropping while adjusting intrinsic parameters.
This script helps reduce memory usage during 3D reconstruction by processing high-resolution images.
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_intrinsic_matrix(k_path: Path) -> np.ndarray:
    """Load intrinsic matrix from file"""
    return np.loadtxt(k_path)


def load_distortion_coeffs(dist_path: Path) -> np.ndarray:
    """Load distortion coefficients from file"""
    return np.loadtxt(dist_path)


def calculate_crop_params(
    original_size: Tuple[int, int],
    target_size: Tuple[int, int]
) -> Tuple[float, Tuple[int, int, int, int]]:
    """
    Calculate scaling factor and crop box for center crop.
    
    Args:
        original_size: (width, height) of original image
        target_size: (width, height) of target image
        
    Returns:
        scale_factor: Scaling factor to apply before cropping
        crop_box: (left, top, right, bottom) for center crop
    """
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    
    # Calculate scale factor to ensure we can crop to target size
    # We scale so the smaller dimension matches the target
    scale_w = target_w / orig_w
    scale_h = target_h / orig_h
    scale_factor = max(scale_w, scale_h)
    
    # Calculate scaled dimensions
    scaled_w = int(orig_w * scale_factor)
    scaled_h = int(orig_h * scale_factor)
    
    # Calculate center crop box
    left = (scaled_w - target_w) // 2
    top = (scaled_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    
    return scale_factor, (left, top, right, bottom)


def adjust_intrinsic_matrix(
    K: np.ndarray,
    scale_factor: float,
    crop_offset: Tuple[int, int]
) -> np.ndarray:
    """
    Adjust intrinsic matrix for scaled and cropped image.
    
    Args:
        K: Original 3x3 intrinsic matrix
        scale_factor: Scaling factor applied to image
        crop_offset: (x_offset, y_offset) from top-left after scaling
        
    Returns:
        Adjusted 3x3 intrinsic matrix
    """
    K_new = K.copy()
    
    # Scale focal lengths and principal point
    K_new[0, 0] *= scale_factor  # fx
    K_new[1, 1] *= scale_factor  # fy
    K_new[0, 2] *= scale_factor  # cx
    K_new[1, 2] *= scale_factor  # cy
    
    # Adjust principal point for cropping
    K_new[0, 2] -= crop_offset[0]
    K_new[1, 2] -= crop_offset[1]
    
    return K_new


def process_image(
    img_path: Path,
    output_path: Path,
    target_size: Tuple[int, int],
    scale_factor: float,
    crop_box: Tuple[int, int, int, int]
) -> None:
    """Process a single image: resize and center crop"""
    img = Image.open(img_path)
    
    # First, scale the image
    scaled_size = (
        int(img.width * scale_factor),
        int(img.height * scale_factor)
    )
    img_scaled = img.resize(scaled_size, Image.Resampling.LANCZOS)
    
    # Then crop
    img_cropped = img_scaled.crop(crop_box)
    
    # Save
    img_cropped.save(output_path, quality=95)


def preprocess_scene(
    scene_dir: Path,
    output_dir: Path,
    target_size: Tuple[int, int],
    force: bool = False
) -> Path:
    """
    Preprocess all images in a scene directory.
    
    Args:
        scene_dir: Input scene directory containing images/, K.txt, dist.txt
        output_dir: Output directory for processed scene
        target_size: Target (width, height) for output images
        force: Overwrite existing output directory
    """
    # Validate input directory
    images_dir = scene_dir / 'images'
    k_path = scene_dir / 'K.txt'
    dist_path = scene_dir / 'dist.txt'
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not k_path.exists():
        raise ValueError(f"Intrinsic matrix file not found: {k_path}")
    if not dist_path.exists():
        raise ValueError(f"Distortion coefficients file not found: {dist_path}")
    
    # Setup output directory
    if output_dir.exists():
        if force:
            logging.warning(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise ValueError(f"Output directory already exists: {output_dir}. Use --force to overwrite.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir = output_dir / 'images'
    output_images_dir.mkdir(exist_ok=True)
    
    # Get list of images
    image_paths = sorted(images_dir.glob('*.jpg'))
    if not image_paths:
        image_paths = sorted(images_dir.glob('*.png'))
    
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    
    logging.info(f"Found {len(image_paths)} images")
    
    # Get original image size from first image
    first_img = Image.open(image_paths[0])
    original_size = (first_img.width, first_img.height)
    first_img.close()
    
    logging.info(f"Original image size: {original_size[0]}x{original_size[1]}")
    logging.info(f"Target image size: {target_size[0]}x{target_size[1]}")
    
    # Calculate preprocessing parameters
    scale_factor, crop_box = calculate_crop_params(original_size, target_size)
    crop_offset = (crop_box[0], crop_box[1])
    
    logging.info(f"Scale factor: {scale_factor:.4f}")
    logging.info(f"Crop box: {crop_box}")
    
    # Load and adjust intrinsic parameters
    K_orig = load_intrinsic_matrix(k_path)
    K_new = adjust_intrinsic_matrix(K_orig, scale_factor, crop_offset)
    
    # Load distortion coefficients (these remain unchanged)
    dist_coeffs = load_distortion_coeffs(dist_path)
    
    # Save adjusted parameters
    np.savetxt(output_dir / 'K.txt', K_new, fmt='%.6f')
    np.savetxt(output_dir / 'dist.txt', dist_coeffs, fmt='%.6f')
    
    logging.info("Saved adjusted intrinsic parameters")
    logging.info(f"Original K:\n{K_orig}")
    logging.info(f"Adjusted K:\n{K_new}")
    
    # Process all images
    logging.info("Processing images...")
    for img_path in tqdm(image_paths, desc="Resizing and cropping"):
        output_path = output_images_dir / img_path.name
        process_image(img_path, output_path, target_size, scale_factor, crop_box)
    
    # Create preprocessing info file
    info = {
        'original_size': {'width': original_size[0], 'height': original_size[1]},
        'target_size': {'width': target_size[0], 'height': target_size[1]},
        'scale_factor': float(scale_factor),
        'crop_box': {
            'left': crop_box[0],
            'top': crop_box[1],
            'right': crop_box[2],
            'bottom': crop_box[3]
        },
        'crop_offset': {'x': crop_offset[0], 'y': crop_offset[1]},
        'num_images': len(image_paths)
    }
    
    with open(output_dir / 'preprocessing_info.yaml', 'w') as f:
        yaml.dump(info, f, default_flow_style=False)
    
    logging.info(f"Preprocessing complete! Output saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess images by resizing and center cropping while adjusting intrinsics'
    )
    parser.add_argument('scene_dir', type=Path, help='Input scene directory')
    parser.add_argument('--output_dir', type=Path, help='Output directory (default: scene_dir_processed)')
    parser.add_argument('--width', type=int, default=1920, help='Target width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080, help='Target height (default: 1080)')
    parser.add_argument('--config', type=Path, help='Config file with target resolution')
    parser.add_argument('--force', action='store_true', help='Overwrite existing output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        target_size = (
            config.get('target_width', args.width),
            config.get('target_height', args.height)
        )
    else:
        target_size = (args.width, args.height)
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = args.scene_dir.parent / f"{args.scene_dir.name}_processed"
    
    # Run preprocessing
    preprocess_scene(
        args.scene_dir,
        args.output_dir,
        target_size,
        args.force
    )


if __name__ == '__main__':
    main()