# Todo: Refactoring Plan for Two-Stage 3D Reconstruction Pipeline

## 🆕 Latest Updates (2025-01-12)

### Image Preprocessing System ✅ COMPLETED
- **Preprocessing Module**: Created `src/preprocessing/resize_and_crop.py` for high-resolution image handling
- **Key Features**:
  - **Resize and Center Crop**: Reduces image resolution while maintaining aspect ratio
  - **Intrinsic Parameter Adjustment**: Automatically adjusts camera matrix (K) for new resolution
  - **Validation System**: Checks existing preprocessed data validity before re-processing
  - **Skip Logic**: Reuses existing preprocessed data when valid
- **Integration with Precompute**:
  - Added `_preprocess_if_needed` method to PrecomputePipeline
  - Added `_is_preprocessing_valid` method for robust validation
  - All config files updated with preprocessing settings
- **Configuration**:
  ```yaml
  preprocessing:
    enabled: true          # Enable/disable preprocessing
    target_width: 1920    # Target width
    target_height: 1080   # Target height  
    force_overwrite: false # Force re-processing
  ```
- **Directory Structure**:
  ```
  data/scene_name/                    # Original scene
  data/scene_name_processed_WxH/      # Preprocessed scene
  ├── images/                         # Resized images
  ├── K.txt                          # Adjusted intrinsics
  ├── dist.txt                       # Original distortion
  └── preprocessing_info.yaml        # Processing metadata
  ```

### Quick Test Commands
```bash
# Test preprocessing + SIFT tracking (1024x576)
python -m src.precompute.precompute data/3x_section2 --config config/precompute_sift.yaml
# → Creates: data/3x_section2_processed_1024x576/

# Direct preprocessing (1920x1080)
python -m src.preprocessing.resize_and_crop data/3x_section2 --width 1920 --height 1080

# Test preprocessing functions
python tests/test_preprocessing_integration.py
```

---

## 🆕 Previous Updates (2025-01-11)

### Feature Initialization System ✅ COMPLETED
- **Modular Architecture**: Created `feature_initializers` module with abstract base class
- **Three Methods Implemented**:
  - **Grid**: 20x20 uniform grid (400 points) with slight randomization
  - **SIFT**: Texture-aware feature detection (up to 400 points)
  - **SuperPoint**: Deep learning features (currently falls back to SIFT)
- **New Config Files**: `precompute_sift.yaml`, `precompute_superpoint.yaml`
- **Test Script**: `tests/test_feature_initializers.py` - compares all methods
- **Full Integration**: CoTrackerExtractor now supports all initialization methods
- **Feature Storage**: Only essential track data is stored (no extra feature info to avoid serialization issues)

### Quick Test Commands
```bash
# Test SIFT-based tracking
python -m src.precompute.precompute data/3x_section2 --config config/precompute_sift.yaml

# Compare all feature methods
python tests/test_feature_initializers.py
```

---

## Overview
Refactor the current monolithic 3D reconstruction system into a two-stage pipeline:
1. **Precompute Stage**: Extract all necessary features (tracks, depth, optical flow)
2. **Reconstruction Stage**: Perform 3D reconstruction using precomputed data

## Current Status

### What We Have
- Working CoTracker integration tests with three initialization methods:
  - Grid-based initialization (20x20 grid = 400 points)
  - SIFT-based initialization  
  - SuperPoint-based initialization
- Overlap-based windowing system that creates NEW tracks for EACH window
- Basic visualization of tracks
- Working code in: `/hdd2/0321_block_drone_video/colmap/tests/cotracker_tests/`

### Current Implementation Details

#### 1. Model Loading
```python
# Current model loading using torch.hub
model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
model = model.to(device)
```

#### 2. Current Window Generation (Overlap-based)
```python
def generate_windows_overlap(total_frames, window_size=24, overlap=12):
    stride = window_size - overlap
    windows = []
    for start in range(0, total_frames, stride):
        end = min(start + window_size, total_frames)
        windows.append((start, end))
        if end >= total_frames:
            break
    return windows

# Example with window_size=48, overlap=24:
# Window 0: [0, 48)
# Window 1: [24, 72)  # 50% overlap
# Window 2: [48, 96)  # 50% overlap
```

#### 3. Query Point Generation
Each window gets NEW query points at its first frame:
```python
def get_grid_points_for_window(image_shape, grid_size=20, window_frame_offset=0):
    h, w = image_shape[:2]
    
    # Create grid with slight randomization
    np.random.seed(None)  # Random seed for variety
    margin = 0.1
    
    y_coords = np.linspace(h * margin, h * (1-margin), grid_size)
    x_coords = np.linspace(w * margin, w * (1-margin), grid_size)
    
    # Add small random perturbation (up to 5 pixels)
    y_coords += np.random.uniform(-5, 5, size=grid_size)
    x_coords += np.random.uniform(-5, 5, size=grid_size)
    
    # Ensure points stay within bounds
    y_coords = np.clip(y_coords, 0, h-1)
    x_coords = np.clip(x_coords, 0, w-1)
    
    # Generate all combinations
    points = []
    for y in y_coords:
        for x in x_coords:
            points.append([window_frame_offset, x, y])  # [time, x, y]
    
    return np.array(points)  # Shape: (400, 3) for 20x20 grid
```

#### 4. Tracking Process
```python
for i, (start, end) in enumerate(windows):
    # Load window frames
    window_frames = load_images_for_window(image_paths, start, end)
    video_tensor = torch.from_numpy(window_frames).permute(0, 3, 1, 2).float()
    video_tensor = video_tensor.unsqueeze(0).to(device)  # [B=1, T, C, H, W]
    
    # Initialize NEW grid for THIS window
    grid_pts = get_grid_points_for_window((img_height, img_width), grid_size=20)
    queries = torch.from_numpy(grid_pts).float().to(device)
    queries = queries.unsqueeze(0)  # [B=1, N=400, 3]
    
    # Run CoTracker
    with torch.no_grad():
        pred_tracks, pred_visibility = model(video_tensor, queries=queries)
        # pred_tracks: [B=1, T, N=400, 2] - pixel coordinates
        # pred_visibility: [B=1, T, N=400] - boolean visibility
    
    # Store results
    all_tracks.append({
        'tracks': pred_tracks.cpu(),
        'visibility': pred_visibility.cpu(),
        'start_frame': start,
        'end_frame': end
    })
```

#### 5. Current Data Storage (In-Memory)
```python
# Each window's tracks stored separately:
track_data = {
    'window_id': int,             # Window identifier
    'start_frame': int,           # Window start frame
    'end_frame': int,             # Window end frame
    'tracks': np.array,           # [T, N, 2] coordinates
    'visibility': np.array,       # [T, N] visibility flags
}
```

### Issues with Current Approach
1. **No Track Continuity**: Each window initializes new query points, losing track continuity
2. **Overlap-based Logic**: Windows generated based on overlap percentage, not flexible
3. **No Precomputation**: All processing happens at runtime
4. **Memory Inefficient**: Loads all data in memory
5. **No Persistence**: Results not saved to disk

## Refactoring Plan

### Phase 1: Modularize CoTracker (Current Priority)

#### 1.1 Create Module Structure
```
src/
├── precompute/
│   ├── __init__.py
│   ├── precompute.py          # Main entry point
│   ├── trackers/
│   │   ├── __init__.py
│   │   ├── base_tracker.py         # Abstract base class
│   │   └── cotracker_extractor.py  # CoTracker implementation
│   ├── depth/
│   │   ├── __init__.py
│   │   └── geometrycrafter_extractor.py
│   └── optical_flow/
│       ├── __init__.py
│       └── raft_extractor.py
```

#### 1.2 NEW Interval-based Window Generation Algorithm

**IMPORTANT CHANGE**: Replace overlap-based with interval-based windowing

```python
def generate_windows_interval(total_frames, window_size=48, interval=10):
    """
    Generate windows starting at regular intervals
    
    Args:
        total_frames: Total number of frames
        window_size: Size of each window
        interval: Frame interval between window starts
        
    Returns:
        List of (start, end) tuples
    """
    windows = []
    window_start = 0
    
    while window_start < total_frames:
        window_end = min(window_start + window_size, total_frames)
        
        # Only add window if it has at least 2 frames
        if window_end - window_start >= 2:
            windows.append((window_start, window_end))
        
        # Move to next window start
        window_start += interval
        
        # Stop if next window would start beyond sequence
        if window_start >= total_frames:
            break
    
    return windows

# Example with window_size=48, interval=10:
# Window 0: [0, 48)
# Window 1: [10, 58)   # starts 10 frames after first window
# Window 2: [20, 68)   # starts 20 frames after first window
# Window 3: [30, 78)   # starts 30 frames after first window
# ...continues until we can't fit another window
```

**Advantages of Interval-based:**
- Flexible overlap (not constrained to percentages)
- Regular, predictable window positions
- Easy to reason about: window N starts at frame `N * interval`
- Can have dense overlap (interval < window_size) or sparse (interval > window_size)

#### 1.3 CoTracker Extractor Implementation

```python
# src/precompute/trackers/cotracker_extractor.py

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
from PIL import Image
import logging

class CoTrackerExtractor:
    """Extract point tracks using CoTracker with interval-based windowing"""
    
    def __init__(self, 
                 window_size: int = 48,
                 interval: int = 10,
                 initialization_method: str = 'grid',
                 grid_size: int = 20,
                 device: str = 'cuda'):
        """
        Args:
            window_size: Number of frames per window
            interval: Frame interval between window starts
            initialization_method: 'grid', 'sift', or 'superpoint'
            grid_size: Grid size for grid initialization (grid_size x grid_size points)
            device: Device to run on
        """
        self.window_size = window_size
        self.interval = interval
        self.initialization_method = initialization_method
        self.grid_size = grid_size
        self.device = device
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load CoTracker model"""
        logging.info("Loading CoTracker3 OFFLINE model...")
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        model = model.to(self.device)
        logging.info("Model loaded successfully")
        return model
    
    def _generate_windows(self, total_frames: int) -> List[Tuple[int, int]]:
        """Generate interval-based windows"""
        windows = []
        window_start = 0
        
        while window_start < total_frames:
            window_end = min(window_start + self.window_size, total_frames)
            
            # Only add window if it has at least 2 frames
            if window_end - window_start >= 2:
                windows.append((window_start, window_end))
            
            # Move to next window start
            window_start += self.interval
            
            # Stop if next window would start beyond sequence
            if window_start >= total_frames:
                break
        
        logging.info(f"Generated {len(windows)} windows for {total_frames} frames")
        for i, (start, end) in enumerate(windows):
            logging.info(f"  Window {i}: frames [{start}, {end})")
        
        return windows
    
    def _get_query_points(self, image_shape: Tuple[int, int], 
                         window_frame_offset: int = 0) -> np.ndarray:
        """Generate query points based on initialization method"""
        
        if self.initialization_method == 'grid':
            return self._get_grid_points(image_shape, window_frame_offset)
        elif self.initialization_method == 'sift':
            return self._get_sift_points(image_shape, window_frame_offset)
        elif self.initialization_method == 'superpoint':
            return self._get_superpoint_points(image_shape, window_frame_offset)
        else:
            raise ValueError(f"Unknown initialization method: {self.initialization_method}")
    
    def _get_grid_points(self, image_shape: Tuple[int, int], 
                        window_frame_offset: int = 0) -> np.ndarray:
        """Generate grid points with slight randomization"""
        h, w = image_shape
        
        # Create grid with slight randomization
        np.random.seed(None)  # Random seed for variety
        margin = 0.1
        
        y_coords = np.linspace(h * margin, h * (1-margin), self.grid_size)
        x_coords = np.linspace(w * margin, w * (1-margin), self.grid_size)
        
        # Add small random perturbation (up to 5 pixels)
        y_coords += np.random.uniform(-5, 5, size=self.grid_size)
        x_coords += np.random.uniform(-5, 5, size=self.grid_size)
        
        # Ensure points stay within bounds
        y_coords = np.clip(y_coords, 0, h-1)
        x_coords = np.clip(x_coords, 0, w-1)
        
        # Generate all combinations
        points = []
        for y in y_coords:
            for x in x_coords:
                points.append([window_frame_offset, x, y])  # [time, x, y]
        
        return np.array(points)
    
    def _load_images_for_window(self, image_paths: List[Path], 
                               start_idx: int, end_idx: int) -> np.ndarray:
        """Load images for a specific window"""
        window_paths = image_paths[start_idx:end_idx]
        images = []
        
        for img_path in window_paths:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            images.append(img_array)
        
        return np.stack(images)
    
    def extract_tracks(self, 
                      image_dir: str,
                      output_path: Optional[str] = None) -> Dict:
        """
        Extract tracks for all windows and save as .npy file
        
        Args:
            image_dir: Directory containing images
            output_path: Output path for .npy file. If None, auto-generated.
            
        Returns:
            Dictionary containing tracks and metadata
        """
        image_dir = Path(image_dir)
        
        # Get image paths
        image_paths = sorted(image_dir.glob("*.jpg"))
        if not image_paths:
            image_paths = sorted(image_dir.glob("*.png"))
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        total_frames = len(image_paths)
        logging.info(f"Found {total_frames} images")
        
        # Get image dimensions
        first_img = Image.open(image_paths[0])
        img_height, img_width = first_img.height, first_img.width
        first_img.close()
        
        # Generate windows
        windows = self._generate_windows(total_frames)
        
        # Process each window
        all_window_tracks = []
        
        for window_idx, (start, end) in enumerate(windows):
            logging.info(f"Processing window {window_idx}/{len(windows)}: frames [{start}, {end})")
            
            # Load window frames
            window_frames = self._load_images_for_window(image_paths, start, end)
            video_tensor = torch.from_numpy(window_frames).permute(0, 3, 1, 2).float()
            video_tensor = video_tensor.unsqueeze(0).to(self.device)
            
            # Get query points for this window
            query_points = self._get_query_points((img_height, img_width), 
                                                 window_frame_offset=0)
            queries = torch.from_numpy(query_points).float().to(self.device)
            queries = queries.unsqueeze(0)  # Add batch dimension
            
            # Run tracking
            with torch.no_grad():
                pred_tracks, pred_visibility = self.model(video_tensor, queries=queries)
            
            # Store window tracks - only essential data
            window_data = {
                'window_id': window_idx,
                'start_frame': start,
                'end_frame': end,
                'tracks': pred_tracks[0].cpu().numpy(),  # Remove batch dim
                'visibility': pred_visibility[0].cpu().numpy(),  # Remove batch dim
            }
            
            all_window_tracks.append(window_data)
            
            # Clear GPU memory
            del window_frames, video_tensor, pred_tracks, pred_visibility
            torch.cuda.empty_cache()
        
        # Prepare output data - minimal structure
        output_data = {
            'tracks': all_window_tracks,
            'metadata': {
                'window_size': self.window_size,
                'interval': self.interval,
                'initialization_method': self.initialization_method,
                'total_frames': total_frames,
            }
        }
        
        # Save to file
        if output_path is None:
            scene_dir = image_dir.parent
            cotracker_dir = scene_dir / 'cotracker'
            cotracker_dir.mkdir(exist_ok=True)
            
            filename = f"{self.window_size}_{self.interval}_{self.initialization_method}.npy"
            output_path = cotracker_dir / filename
        
        np.save(output_path, output_data, allow_pickle=True)
        logging.info(f"Saved tracks to {output_path}")
        
        return output_data
```

#### 1.4 Track Data Format (Saved to Disk)

The `.npy` file contains a dictionary with the following structure:

```python
{
    'tracks': [
        {
            'window_id': 0,
            'start_frame': 0,
            'end_frame': 48,
            'tracks': np.array,         # Shape: (T, N, 2) - pixel coordinates
            'visibility': np.array,     # Shape: (T, N) - boolean visibility
        },
        {
            'window_id': 1,
            'start_frame': 10,
            'end_frame': 58,
            # ... same structure
        },
        # ... more windows
    ],
    'metadata': {
        'window_size': 48,
        'interval': 10,
        'initialization_method': 'grid',
        'total_frames': 300,
    }
}
```

### Phase 2: Main Precompute Script

```python
# src/precompute/precompute.py

import argparse
import logging
from pathlib import Path
from typing import List

from .trackers.cotracker_extractor import CoTrackerExtractor
# Future imports:
# from .depth.geometrycrafter_extractor import GeometryCrafterExtractor
# from .optical_flow.raft_extractor import RAFTExtractor

class PrecomputePipeline:
    """Main pipeline for precomputing all features"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Initialize extractors based on config
        if 'cotracker' in config['features']:
            self.tracker = CoTrackerExtractor(
                window_size=config.get('window_size', 48),
                interval=config.get('interval', 10),
                initialization_method=config.get('initialization_method', 'grid'),
                grid_size=config.get('grid_size', 20),
                device=config.get('device', 'cuda')
            )
        
        # Future: Initialize depth and flow extractors
        
    def run(self, scene_dir: str):
        """Run all configured extractors on the scene"""
        scene_dir = Path(scene_dir)
        image_dir = scene_dir / 'images'
        
        if not image_dir.exists():
            raise ValueError(f"Images directory not found: {image_dir}")
        
        # Extract CoTracker tracks
        if hasattr(self, 'tracker'):
            logging.info("Extracting CoTracker tracks...")
            self.tracker.extract_tracks(image_dir)
        
        # Future: Extract depth and optical flow
        
        logging.info("Precompute pipeline complete!")

def main():
    parser = argparse.ArgumentParser(description='Precompute features for 3D reconstruction')
    parser.add_argument('scene_dir', type=str, help='Path to scene directory')
    parser.add_argument('--features', type=str, nargs='+', 
                       default=['cotracker'], 
                       choices=['cotracker', 'depth', 'flow'],
                       help='Features to extract')
    parser.add_argument('--window_size', type=int, default=48,
                       help='Window size for CoTracker')
    parser.add_argument('--interval', type=int, default=10,
                       help='Interval between window starts')
    parser.add_argument('--initialization_method', type=str, default='grid',
                       choices=['grid', 'sift', 'superpoint'],
                       help='Point initialization method')
    parser.add_argument('--grid_size', type=int, default=20,
                       help='Grid size for grid initialization')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create config
    config = {
        'features': args.features,
        'window_size': args.window_size,
        'interval': args.interval,
        'initialization_method': args.initialization_method,
        'grid_size': args.grid_size,
        'device': args.device
    }
    
    # Run pipeline
    pipeline = PrecomputePipeline(config)
    pipeline.run(args.scene_dir)

if __name__ == '__main__':
    main()
```

### Phase 3: Usage Examples

#### Command Line Usage
```bash
# Basic usage with default settings
python -m src.precompute.precompute /path/to/scene

# Custom window configuration
python -m src.precompute.precompute /path/to/scene \
    --window_size 48 \
    --interval 10 \
    --initialization_method grid \
    --grid_size 20

# Extract multiple features (future)
python -m src.precompute.precompute /path/to/scene \
    --features cotracker depth flow
```

#### Python API Usage
```python
from src.precompute.trackers.cotracker_extractor import CoTrackerExtractor

# Create extractor
extractor = CoTrackerExtractor(
    window_size=48,
    interval=10,
    initialization_method='grid',
    grid_size=20
)

# Extract tracks
tracks_data = extractor.extract_tracks('/path/to/scene/images')

# Load saved tracks
import numpy as np
saved_data = np.load('/path/to/scene/cotracker/48_10_grid.npy', allow_pickle=True).item()
```

### Phase 4: Future Track Merging (Not in Current Scope)

For future optimization, tracks from overlapping windows could be merged:

```python
def find_matching_tracks(window1_data, window2_data, overlap_frames):
    """
    Find tracks that represent the same physical point across windows
    
    Strategy:
    1. For overlapping frames, compute distances between track positions
    2. Use Hungarian algorithm to find optimal matching
    3. Merge tracks with consistent motion patterns
    """
    # Implementation for future phase
    pass
```

## Implementation Checklist

### Week 1: Core CoTracker Refactoring ✅ COMPLETED
- [x] Create directory structure: `src/precompute/trackers/`
- [x] Implement `base_tracker.py` abstract class
- [x] Implement `cotracker_extractor.py` with interval-based windowing
- [x] Add SIFT initialization method ✅
- [x] Add SuperPoint initialization method ✅
- [x] Create unit tests for window generation
- [x] Test with sample data

### Week 2: Integration ✅ COMPLETED
- [x] Implement main `precompute.py` script
- [x] Add command-line interface
- [x] Add configuration file support (YAML)
- [x] Create logging system
- [x] Write documentation and examples
- [x] Integrate with existing codebase

### Additional Features Implemented
- [x] Visualization system with MP4 video output
- [x] Summary plots with window timeline and statistics
- [x] Multiple config files (default, dense, sparse, test, sift, superpoint)
- [x] Visualization settings in config files
- [x] Command-line override for visualization (--no-visualize)
- [x] Updated CLAUDE.md with new usage instructions
- [x] Test script with real image dimension handling
- [x] Image preprocessing module with resize and center crop ✅ NEW (2025-01-12)
- [x] Automatic intrinsic parameter adjustment ✅ NEW
- [x] Preprocessing validation and skip logic ✅ NEW
- [x] Integration of preprocessing into precompute pipeline ✅ NEW
- [x] Test scripts for preprocessing functionality ✅ NEW

### Testing Strategy
1. Unit tests for window generation algorithm
2. Integration tests with small image sequences
3. Performance benchmarks
4. Validation against current implementation

### Expected Output Structure

**Original Scene:**
```
Scene/
├── images/                     # Original high-res images
│   ├── 001.jpg
│   └── ...
├── K.txt                      # Original intrinsics
└── dist.txt                   # Distortion coefficients
```

**After Preprocessing + Precompute:**
```
Scene_processed_1920x1080/
├── images/                     # Preprocessed images
│   ├── 001.jpg
│   └── ...
├── K.txt                      # Adjusted intrinsics
├── dist.txt                   # Same distortion coefficients
├── preprocessing_info.yaml     # Preprocessing metadata
├── cotracker/
│   ├── 48_10_grid.npy         # window_size=48, interval=10, grid method
│   ├── 48_5_grid.npy          # window_size=48, interval=5, grid method
│   └── 32_8_sift.npy          # window_size=32, interval=8, SIFT method
├── visualizations/            # If enabled
│   └── cotracker_48_10_sift/
│       ├── tracking_result.mp4
│       └── tracking_summary.png
├── precompute.log             # Processing log
└── precompute_summary.json    # Results summary
```

## Key Differences from Current Implementation

1. **Window Generation**:
   - OLD: Overlap-based (stride = window_size - overlap)
   - NEW: Interval-based (windows start at 0, interval, 2*interval, ...)

2. **File Output**:
   - OLD: Results stay in memory
   - NEW: Save to .npy files with standardized naming

3. **Modularity**:
   - OLD: Monolithic test scripts
   - NEW: Modular extractors with common interface

4. **Configuration**:
   - OLD: Hard-coded parameters
   - NEW: Command-line and config file support

5. **Scalability**:
   - OLD: Process entire sequence at once
   - NEW: Window-by-window processing with disk storage

## Feature Initialization Implementation (NEW)

### Overview
The CoTracker feature initialization has been fully modularized to support multiple methods:
- **Grid**: Regular grid pattern with slight randomization
- **SIFT**: Scale-Invariant Feature Transform for texture-based points
- **SuperPoint**: Deep learning-based feature detection (with SIFT fallback)

### Architecture

#### 1. Module Structure
```
src/precompute/trackers/feature_initializers/
├── __init__.py
├── base_initializer.py         # Abstract base class
├── grid_initializer.py         # Grid-based initialization
├── sift_initializer.py         # SIFT feature detection
└── superpoint_initializer.py   # SuperPoint features (with fallback)
```

#### 2. Base Class Interface
```python
class BaseFeatureInitializer(ABC):
    def __init__(self, max_features: int = 400):
        self.max_features = max_features
    
    @abstractmethod
    def extract_features(self, image_path: Path, window_frame_offset: int = 0) 
        -> Tuple[np.ndarray, Optional[Dict]]:
        """Extract features from an image
        Returns: (query_points, extra_info)
        Note: extra_info should return None to avoid serialization issues
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of the initialization method"""
        pass
```

#### 3. CoTrackerExtractor Integration
The `CoTrackerExtractor` now uses feature initializers via composition:
```python
# In __init__
self.feature_initializer = self._get_feature_initializer(superpoint_weights)

# In _get_query_points
query_points, extra_info = self.feature_initializer.extract_features(
    image_path, window_frame_offset=0
)
```

### Implementation Details

#### 1. Grid Initializer
- Creates a 20x20 grid by default (400 points)
- Adds small random perturbation (±5 pixels) for robustness
- Maintains 10% margin from image borders

#### 2. SIFT Initializer
```python
class SIFTInitializer(BaseFeatureInitializer):
    def __init__(self, max_features=400, contrast_threshold=0.01, grid_filter=True):
        # SIFT with double extraction then filtering
        self.sift = cv2.SIFT_create(
            nfeatures=max_features * 2,
            contrastThreshold=contrast_threshold
        )
```
- Extracts up to 2x max_features, then filters
- Optional spatial distribution filtering (grid_filter)
- Fallback to grid if no features found
- Returns query points only (extra_info is None to avoid serialization issues)

#### 3. SuperPoint Initializer
```python
class SuperPointInitializer(BaseFeatureInitializer):
    def __init__(self, max_features=400, conf_threshold=0.015, 
                 nms_dist=4, weights_path=None, device='cuda'):
        # Attempts to load SuperPoint, falls back to SIFT if unavailable
```
- Attempts to load SuperPoint from submodules
- Requires superpoint_v1.pth weights file
- Falls back to SIFT if SuperPoint unavailable
- Returns query points only (extra_info is None to avoid serialization issues)

### Configuration Files

#### 1. SIFT Configuration (`config/precompute_sift.yaml`)
```yaml
cotracker:
  window_size: 48
  interval: 10
  initialization_method: sift
  max_features: 400
  device: cuda
```

#### 2. SuperPoint Configuration (`config/precompute_superpoint.yaml`)
```yaml
cotracker:
  window_size: 48
  interval: 10
  initialization_method: superpoint
  max_features: 400
  device: cuda
  # superpoint_weights: /path/to/superpoint_v1.pth  # Optional
```

### Usage Examples

#### Command Line
```bash
# SIFT-based tracking with visualization
python -m src.precompute.precompute /path/to/scene --config config/precompute_sift.yaml

# SuperPoint-based tracking (falls back to SIFT if weights not found)
python -m src.precompute.precompute /path/to/scene --config config/precompute_superpoint.yaml

# Compare different methods
python -m src.precompute.precompute /path/to/scene --config config/precompute.yaml       # Grid
python -m src.precompute.precompute /path/to/scene --config config/precompute_sift.yaml   # SIFT
```

#### Python API
```python
from src.precompute.trackers.cotracker_extractor import CoTrackerExtractor

# Create extractor with SIFT
extractor = CoTrackerExtractor(
    window_size=48,
    interval=10,
    initialization_method='sift',
    max_features=400
)

# Extract tracks
tracks_data = extractor.extract_tracks('/path/to/scene/images')
```

### Output Files
The initialization method is reflected in the output filename:
- Grid: `48_10_grid.npy`
- SIFT: `48_10_sift.npy`
- SuperPoint: `48_10_superpoint.npy`

### Test Scripts

#### 1. Feature Initializer Test (`tests/test_feature_initializers.py`)
Tests all three initialization methods on the same image:
```bash
python tests/test_feature_initializers.py
# Output: outputs/feature_initializer_tests/feature_comparison.png
```

#### 2. Full Pipeline Test
```bash
# Test with 3x_section2 data
python -m src.precompute.precompute data/3x_section2 --config config/precompute_sift.yaml
```

### Performance Comparison
Based on initial tests with `data/3x_section2`:
- **Grid**: 400 points (fixed), uniform distribution
- **SIFT**: ~200-400 points, concentrated on textured regions
- **SuperPoint**: Currently falls back to SIFT (weights needed)

### Future Improvements
1. **SuperPoint weights integration**: Automatic download or bundling
2. **Feature matching across windows**: Use descriptors for track linking
3. **Adaptive feature count**: Adjust based on image texture
4. **Other feature detectors**: ORB, AKAZE, learned features

## Current Implementation Status (Updated)

### ✅ Completed Features
1. **Core Infrastructure**
   - Full directory structure created
   - Base tracker abstract class
   - Main precompute pipeline script
   - All __init__.py files

2. **CoTracker Implementation**
   - Interval-based window generation
   - Modular feature initialization system:
     - Grid-based initialization (20x20 = 400 points)
     - SIFT-based initialization (texture-aware, up to 400 points)
     - SuperPoint initialization (with SIFT fallback)
   - Minimal data storage (no feature info to avoid serialization issues)
   - Track extraction and saving to .npy
   - GPU memory management

3. **Configuration System**
   - YAML config file support
   - Multiple preset configs:
     - `precompute.yaml` (default grid)
     - `precompute_dense.yaml` (dense tracking)
     - `precompute_sparse.yaml` (sparse tracking)
     - `precompute_test.yaml` (quick testing)
     - `precompute_sift.yaml` (SIFT features) ✨ NEW
     - `precompute_superpoint.yaml` (SuperPoint features) ✨ NEW
   - Command-line override option (--no-visualize)
   - Visualization settings in config

4. **Visualization Features**
   - MP4 video generation with tracked points
   - Color-coded tracks per window
   - Trail visualization
   - Summary plots (timeline, statistics)
   - Frame info overlay
   - Optional frame saving

5. **Preprocessing System** ✨ NEW (2025-01-12)
   - Image resize and center crop functionality
   - Automatic intrinsic parameter adjustment
   - Preprocessing validation and skip logic
   - Integration with all config files
   - Separate preprocessing module (`src/preprocessing/`)
   - Support for multiple resolutions per scene

6. **Documentation**
   - Updated CLAUDE.md with usage examples
   - Config file documentation
   - Test scripts:
     - `tests/test_precompute.py` (basic pipeline test)
     - `tests/test_feature_initializers.py` (compare all methods) ✨ NEW
     - `tests/test_preprocessing_integration.py` (preprocessing tests) ✨ NEW
     - `tests/cotracker_tests/test_cotracker_sift.py` (reference)
     - `tests/cotracker_tests/test_cotracker_superpoint.py` (reference)

### 🚧 Pending Features
1. **Other Extractors**
   - GeometryCrafter depth estimation
   - RAFT optical flow
   
2. **Advanced Features**
   - Track merging across windows
   - Track filtering/pruning
   - Multi-GPU support

### 📝 Usage Examples

```bash
# Basic usage with default config (Grid)
python -m src.precompute.precompute /path/to/scene

# SIFT-based feature tracking with preprocessing ✨ NEW
python -m src.precompute.precompute /path/to/scene --config config/precompute_sift.yaml
# Creates: /path/to/scene_processed_1024x576/ (preprocessed)
# Then extracts features from preprocessed images

# Preprocessing only (using resize_and_crop directly) ✨ NEW
python -m src.preprocessing.resize_and_crop /path/to/scene --width 1920 --height 1080

# Use already preprocessed data (preprocessing disabled)
python -m src.precompute.precompute /path/to/scene_processed_1920x1080 --config config/precompute.yaml

# Force re-preprocessing
# Edit config: preprocessing.force_overwrite: true

# Dense tracking with visualization
python -m src.precompute.precompute /path/to/scene --config config/precompute_dense.yaml

# Quick test without visualization
python -m src.precompute.precompute /path/to/scene --config config/precompute_test.yaml --no-visualize

# Test preprocessing integration ✨ NEW
python tests/test_preprocessing_integration.py

# Test feature initializers
python tests/test_feature_initializers.py

# Run basic pipeline tests
python tests/test_precompute.py
```

## Notes for Implementation

1. **Memory Management**: Process one window at a time and clear GPU memory
2. **Error Handling**: Add try-except blocks for model loading and file I/O
3. **Progress Reporting**: Use tqdm or logging for progress updates
4. **Reproducibility**: Set random seeds when needed
5. **Compatibility**: Ensure compatibility with existing COLMAP pipeline

## 🆕 Latest Fixes (2025-01-13)

### Serialization Issue Resolution ✅ COMPLETED
- **Problem**: cv2.KeyPoint objects cannot be pickled when saving to .npy files
- **Solution**: 
  - Modified all feature initializers to return `None` for `extra_info`
  - Simplified CoTracker output to only store essential tracking data
- **Changes**:
  - `SIFTInitializer.extract_features()`: Now returns `(queries, None)`
  - `SuperPointInitializer.extract_features()`: Now returns `(queries, None)`
  - CoTracker window data: Only stores `window_id`, `start_frame`, `end_frame`, `tracks`, `visibility`
  - Metadata: Only stores `window_size`, `interval`, `initialization_method`, `total_frames`
- **Benefits**:
  - Eliminates serialization errors
  - Reduces storage size
  - Simplifies data structure for downstream processing

This document contains all necessary information to implement the CoTracker refactoring without needing to see the original code.