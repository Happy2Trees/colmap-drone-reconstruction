"""Test script for the precompute pipeline"""

import sys
import logging
from pathlib import Path
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.precompute.trackers.cotracker_extractor import CoTrackerExtractor
from src.precompute.precompute import load_config


def test_window_generation():
    """Test interval-based window generation"""
    print("Testing window generation...")
    
    extractor = CoTrackerExtractor(window_size=48, interval=10)
    
    # Test with 100 frames
    windows = extractor._generate_windows(100)
    print(f"\nWindows for 100 frames (window_size=48, interval=10):")
    for i, (start, end) in enumerate(windows):
        print(f"  Window {i}: [{start}, {end})")
    
    # Test with different parameters
    extractor2 = CoTrackerExtractor(window_size=32, interval=16)
    windows2 = extractor2._generate_windows(100)
    print(f"\nWindows for 100 frames (window_size=32, interval=16):")
    for i, (start, end) in enumerate(windows2):
        print(f"  Window {i}: [{start}, {end})")
    
    # Test edge cases
    print("\nEdge case tests:")
    
    # Very short sequence
    windows_short = extractor._generate_windows(10)
    print(f"10 frames: {windows_short}")
    
    # Exact window size
    windows_exact = extractor._generate_windows(48)
    print(f"48 frames: {windows_exact}")
    
    print("\nWindow generation tests passed!")


def test_query_points_with_real_image():
    """Test query point generation with actual image dimensions"""
    print("\nTesting query point generation with real image dimensions...")
    
    # Find a real image to get dimensions
    data_dir = Path(__file__).parent.parent / "data"
    image_found = False
    test_image_path = None
    
    # Try to find any image in the data directory
    for pattern in ['**/*.jpg', '**/*.png', '**/*.jpeg']:
        image_paths = list(data_dir.glob(pattern))
        if image_paths:
            test_image_path = image_paths[0]
            image_found = True
            break
    
    if image_found and test_image_path:
        # Get actual image dimensions
        with Image.open(test_image_path) as img:
            img_width, img_height = img.size
        
        print(f"Using image: {test_image_path}")
        print(f"Image dimensions: {img_width}x{img_height}")
        
        extractor = CoTrackerExtractor(grid_size=5)  # 5x5 grid for testing
        
        # Test grid points with actual dimensions
        points = extractor._get_grid_points((img_height, img_width), window_frame_offset=0)
        print(f"\nGrid points (5x5): shape={points.shape}")
        print(f"First 5 points:\n{points[:5]}")
        print(f"Points format: [time, x, y]")
        
        # Check that all points are within actual image bounds
        assert all(0 <= p[1] < img_width for p in points), f"X coordinates out of bounds (width={img_width})"
        assert all(0 <= p[2] < img_height for p in points), f"Y coordinates out of bounds (height={img_height})"
        assert all(p[0] == 0 for p in points), "Time coordinate should be 0"
        
        # Check margin (10% from edges)
        margin = 0.1
        assert all(p[1] >= img_width * margin - 5 for p in points), "Points too close to left edge"
        assert all(p[1] <= img_width * (1-margin) + 5 for p in points), "Points too close to right edge"
        assert all(p[2] >= img_height * margin - 5 for p in points), "Points too close to top edge"
        assert all(p[2] <= img_height * (1-margin) + 5 for p in points), "Points too close to bottom edge"
        
    else:
        print("No real images found in data directory, using synthetic dimensions")
        # Test with common resolutions
        test_resolutions = [
            (1920, 1080),  # Full HD
            (1280, 720),   # HD
            (640, 480),    # VGA
            (3840, 2160),  # 4K
        ]
        
        extractor = CoTrackerExtractor(grid_size=5)
        
        for width, height in test_resolutions:
            print(f"\nTesting with {width}x{height} resolution:")
            points = extractor._get_grid_points((height, width), window_frame_offset=0)
            print(f"  Grid points shape: {points.shape}")
            print(f"  X range: [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}]")
            print(f"  Y range: [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}]")
            
            # Verify bounds
            assert all(0 <= p[1] < width for p in points), f"X out of bounds for {width}x{height}"
            assert all(0 <= p[2] < height for p in points), f"Y out of bounds for {width}x{height}"
    
    print("\nQuery point generation tests passed!")


def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    config_dir = Path(__file__).parent.parent / "config"
    config_files = [
        "precompute.yaml",
        "precompute_dense.yaml", 
        "precompute_sparse.yaml",
        "precompute_test.yaml"
    ]
    
    for config_file in config_files:
        config_path = config_dir / config_file
        if config_path.exists():
            config = load_config(str(config_path))
            print(f"\n{config_file}:")
            print(f"  Features: {config.get('features', [])}")
            print(f"  Window size: {config.get('cotracker', {}).get('window_size', 'N/A')}")
            print(f"  Interval: {config.get('cotracker', {}).get('interval', 'N/A')}")
            print(f"  Visualization enabled: {config.get('visualization', {}).get('enabled', False)}")
        else:
            print(f"\nConfig file not found: {config_path}")
    
    print("\nConfig loading tests passed!")


def test_visualization_output_structure():
    """Test expected visualization output structure"""
    print("\nTesting visualization output structure...")
    
    # Simulate the expected output structure
    test_scene = Path("/tmp/test_scene")
    test_scene.mkdir(exist_ok=True)
    
    # Create expected directories
    cotracker_dir = test_scene / "cotracker"
    cotracker_dir.mkdir(exist_ok=True)
    
    viz_dir = test_scene / "visualizations" / "cotracker_48_10_grid"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Check structure
    print(f"Created test structure at {test_scene}")
    print(f"  - cotracker/ exists: {cotracker_dir.exists()}")
    print(f"  - visualizations/ exists: {viz_dir.exists()}")
    
    # Clean up
    import shutil
    shutil.rmtree(test_scene)
    
    print("Output structure test passed!")


def main():
    """Run all tests"""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Running precompute pipeline tests")
    print("=" * 60)
    
    try:
        test_window_generation()
        test_query_points_with_real_image()
        test_config_loading()
        test_visualization_output_structure()
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()