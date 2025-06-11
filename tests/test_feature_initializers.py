#!/usr/bin/env python3
"""Test script for feature initializers (Grid, SIFT, SuperPoint)"""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.precompute.trackers.feature_initializers import (
    GridInitializer,
    SIFTInitializer,
    SuperPointInitializer
)


def visualize_features(image_path, features_dict, output_path):
    """Visualize features from different methods"""
    from PIL import Image
    
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Create subplots
    fig, axes = plt.subplots(1, len(features_dict), figsize=(5*len(features_dict), 5))
    if len(features_dict) == 1:
        axes = [axes]
    
    for idx, (method_name, (query_points, extra_info)) in enumerate(features_dict.items()):
        ax = axes[idx]
        ax.imshow(img_array)
        
        # Extract x, y coordinates (skip time dimension)
        points = query_points[:, 1:]
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], c='red', s=20, alpha=0.7, 
                  edgecolors='yellow', linewidth=0.5)
        
        ax.set_title(f'{method_name} ({len(points)} points)')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")


def test_initializers():
    """Test all feature initializers"""
    logging.basicConfig(level=logging.INFO)
    
    # Test image path
    script_dir = Path(__file__).parent.parent
    test_images_dir = script_dir / "data/3x_section2/images"
    
    if not test_images_dir.exists():
        print(f"Error: Test images directory not found: {test_images_dir}")
        print("Please ensure test data is available")
        return
    
    # Get first image
    image_paths = sorted(test_images_dir.glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(test_images_dir.glob("*.png"))
    
    if not image_paths:
        print(f"Error: No images found in {test_images_dir}")
        return
    
    test_image = image_paths[0]
    print(f"Using test image: {test_image}")
    
    # Output directory
    output_dir = script_dir / "outputs/feature_initializer_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test each initializer
    results = {}
    
    # 1. Grid initializer
    print("\n1. Testing Grid Initializer...")
    try:
        grid_init = GridInitializer(grid_size=20)
        grid_points, grid_info = grid_init.extract_features(test_image)
        print(f"   Grid: {len(grid_points)} points extracted")
        results['Grid'] = (grid_points, grid_info)
    except Exception as e:
        print(f"   Grid failed: {e}")
    
    # 2. SIFT initializer
    print("\n2. Testing SIFT Initializer...")
    try:
        sift_init = SIFTInitializer(max_features=400, grid_filter=True)
        sift_points, sift_info = sift_init.extract_features(test_image)
        print(f"   SIFT: {len(sift_points)} points extracted")
        if sift_info:
            print(f"   SIFT info keys: {list(sift_info.keys())}")
        results['SIFT'] = (sift_points, sift_info)
    except Exception as e:
        print(f"   SIFT failed: {e}")
    
    # 3. SuperPoint initializer
    print("\n3. Testing SuperPoint Initializer...")
    try:
        superpoint_init = SuperPointInitializer(max_features=400, grid_filter=True)
        sp_points, sp_info = superpoint_init.extract_features(test_image)
        print(f"   SuperPoint: {len(sp_points)} points extracted")
        if sp_info:
            print(f"   SuperPoint info keys: {list(sp_info.keys())}")
            if 'method' in sp_info:
                print(f"   Actually used method: {sp_info['method']}")
        results['SuperPoint'] = (sp_points, sp_info)
    except Exception as e:
        print(f"   SuperPoint failed: {e}")
    
    # Visualize results
    if results:
        print("\nCreating visualization...")
        visualize_features(test_image, results, output_dir / 'feature_comparison.png')
    
    # Save detailed results
    print("\nFeature extraction summary:")
    print("-" * 50)
    for method, (points, info) in results.items():
        print(f"{method}:")
        print(f"  Number of points: {len(points)}")
        print(f"  Points shape: {points.shape}")
        if info:
            for key, value in info.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape {value.shape}")
                elif isinstance(value, list) and len(value) > 0:
                    print(f"  {key}: {len(value)} items")
                else:
                    print(f"  {key}: {value}")
        print()


if __name__ == "__main__":
    test_initializers()