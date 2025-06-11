"""Test preprocessing integration with precompute pipeline"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import preprocess_scene, calculate_crop_params, adjust_intrinsic_matrix
import numpy as np
from PIL import Image


def test_preprocessing_functions():
    """Test individual preprocessing functions"""
    print("Testing preprocessing functions...")
    
    # Test calculate_crop_params
    original_size = (3840, 2160)
    target_size = (1920, 1080)
    
    scale_factor, crop_box = calculate_crop_params(original_size, target_size)
    print(f"Original size: {original_size}")
    print(f"Target size: {target_size}")
    print(f"Scale factor: {scale_factor:.4f}")
    print(f"Crop box: {crop_box}")
    
    # Verify the crop box dimensions
    crop_width = crop_box[2] - crop_box[0]
    crop_height = crop_box[3] - crop_box[1]
    assert crop_width == target_size[0], f"Crop width {crop_width} != target width {target_size[0]}"
    assert crop_height == target_size[1], f"Crop height {crop_height} != target height {target_size[1]}"
    print("✓ Crop parameters calculation correct")
    
    # Test intrinsic adjustment
    K_orig = np.array([
        [9660.362404, 0.000000, 1355.300874],
        [0.000000, 9713.707651, 1632.943811],
        [0.000000, 0.000000, 1.000000]
    ])
    
    crop_offset = (crop_box[0], crop_box[1])
    K_new = adjust_intrinsic_matrix(K_orig, scale_factor, crop_offset)
    
    print("\nOriginal K:")
    print(K_orig)
    print("\nAdjusted K:")
    print(K_new)
    
    # Verify scaling was applied
    assert np.isclose(K_new[0, 0], K_orig[0, 0] * scale_factor), "fx not scaled correctly"
    assert np.isclose(K_new[1, 1], K_orig[1, 1] * scale_factor), "fy not scaled correctly"
    print("✓ Intrinsic matrix adjustment correct")


def test_full_pipeline():
    """Test full preprocessing pipeline"""
    print("\n\nTesting full preprocessing pipeline...")
    
    # Test with the example scene
    scene_dir = Path("/hdd2/0321_block_drone_video/colmap/data/3x_section2")
    if not scene_dir.exists():
        print(f"Test scene not found: {scene_dir}")
        return
    
    # Create a small test by copying just a few images
    test_scene_dir = Path("/tmp/test_scene")
    test_scene_dir.mkdir(exist_ok=True)
    
    # Copy K.txt and dist.txt
    import shutil
    shutil.copy(scene_dir / "K.txt", test_scene_dir / "K.txt")
    shutil.copy(scene_dir / "dist.txt", test_scene_dir / "dist.txt")
    
    # Create test images directory and copy first 3 images
    test_images_dir = test_scene_dir / "images"
    test_images_dir.mkdir(exist_ok=True)
    
    source_images = sorted((scene_dir / "images").glob("*.jpg"))[:3]
    for img_path in source_images:
        shutil.copy(img_path, test_images_dir / img_path.name)
    
    print(f"Created test scene at: {test_scene_dir}")
    print(f"Number of test images: {len(list(test_images_dir.glob('*.jpg')))}")
    
    # Run preprocessing
    output_dir = test_scene_dir.parent / "test_scene_processed"
    try:
        result_dir = preprocess_scene(
            scene_dir=test_scene_dir,
            output_dir=output_dir,
            target_size=(1920, 1080),
            force=True
        )
        print(f"✓ Preprocessing completed successfully")
        print(f"Output directory: {result_dir}")
        
        # Verify output
        assert result_dir.exists()
        assert (result_dir / "K.txt").exists()
        assert (result_dir / "dist.txt").exists()
        assert (result_dir / "images").exists()
        assert (result_dir / "preprocessing_info.yaml").exists()
        
        # Check output image dimensions
        output_images = list((result_dir / "images").glob("*.jpg"))
        if output_images:
            img = Image.open(output_images[0])
            assert img.size == (1920, 1080), f"Output image size {img.size} != (1920, 1080)"
            img.close()
            print(f"✓ Output images have correct dimensions: {img.size}")
        
    finally:
        # Cleanup
        shutil.rmtree(test_scene_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)


def test_config_integration():
    """Test config-based preprocessing"""
    print("\n\nTesting config integration...")
    
    import yaml
    
    # Load a config file
    config_path = Path("/hdd2/0321_block_drone_video/colmap/config/precompute_sift.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Preprocessing config:")
    print(yaml.dump(config['preprocessing'], default_flow_style=False))
    
    assert config['preprocessing']['enabled'] == True
    assert 'target_width' in config['preprocessing']
    assert 'target_height' in config['preprocessing']
    print(f"✓ Config file has preprocessing settings: {config['preprocessing']['target_width']}x{config['preprocessing']['target_height']}")


if __name__ == "__main__":
    test_preprocessing_functions()
    test_full_pipeline()
    test_config_integration()
    print("\n✅ All tests passed!")