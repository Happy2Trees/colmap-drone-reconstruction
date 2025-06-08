# Co-Tracker + Global Bundle Adjustment Pipeline Implementation Plan

## Overview
Build a pipeline that uses Co-Tracker to extract high-quality feature tracks from video/image sequences using sliding window approach, then performs **global bundle adjustment only** to estimate camera poses for each frame.

## Core Components

### 1. Co-Tracker Integration
- **Library**: Meta's Co-Tracker (already in submodules)
- **Approach**: Use CoTrackerOnlinePredictor for sliding window tracking
- **Key Features**:
  - Overlapping windows for continuous tracking
  - Forward and backward tracking capability
  - Dense or sparse point tracking modes
  - High-quality feature correspondences across frames

### 2. Global Bundle Adjustment Library
**Recommended**: GTSAM (Georgia Tech Smoothing and Mapping)
- **Why GTSAM**: Pure optimization library, NOT a full SfM pipeline
- **Key Point**: "GTSAM simply provides the 'bundle adjustment' optimization"
- Provides projection factors for BA without feature matching/initialization
- Modern, well-documented Python bindings
- Better performance than scipy for large-scale problems
- Can handle thousands of cameras and points efficiently

**Alternatives**:
- pyTheiaSfM: Also supports BA-only, but more SfM-oriented
- scipy.optimize.least_squares: Requires more custom implementation
- g2opy: Good alternative, Python bindings for g2o
- Ceres (via Python bindings): Industry standard but C++ focused

## Implementation Progress & Tasks

### Phase 1: Environment Setup ✅ COMPLETED
- [x] Install Co-Tracker dependencies (already available in submodules)
- [x] Install GTSAM Python package: `pip install gtsam`
- [x] Create project structure under `src/cotracker_ba/`
- [ ] Set up configuration file for pipeline parameters

### Phase 2: Co-Tracker Feature Extraction ✅ COMPLETED
- [x] Create `feature_extractor.py` module
  - [x] Implement sliding window video loader
  - [x] Configure window size and overlap parameters
  - [x] Extract tracks using CoTrackerPredictor (not OnlinePredictor)
  - [x] Handle forward and backward tracking
  - [x] Filter tracks by confidence/visibility scores
- [x] Create `track_manager.py` module
  - [x] Merge overlapping window tracks
  - [x] Handle track continuity across windows
  - [x] Convert tracks to bundle adjustment format

**Implementation Notes**:
- Used `CoTrackerPredictor` from torch.hub instead of `CoTrackerOnlinePredictor`
- Referenced `online_demo.py` and `notebooks/demo.py` for proper API usage
- Implemented sliding window with middle-frame query points for better tracking
- Added visualization capability for tracks

### Phase 3: Camera Model Setup ✅ COMPLETED
- [x] Create `camera_model.py` module
  - [x] Load camera intrinsics from `config/intrinsic/`
  - [x] Support SIMPLE_RADIAL model (matching existing calibration)
  - [x] Handle different magnifications (x3, x7)
  - [x] Added GTSAM calibration support
  - [x] Added COLMAP export functionality

**Camera Parameters Found**:
- x3: fx=9660.36, fy=9713.71, cx=1355.30, cy=1632.94
- x7: fx=19872.64, fy=19873.23, cx=2123.59, cy=1499.44

### Phase 4: Global Bundle Adjustment Implementation ✅ COMPLETED
- [x] Create `bundle_adjustment.py` module using GTSAM
  - [x] Core BA Setup with NonlinearFactorGraph
  - [x] Add projection factors for ALL track observations globally
  - [x] Set up camera pose variables X(i) for each frame
  - [x] Set up 3D point variables L(j) for each track
  - [x] Configure noise models for measurements
  - [x] Implement global optimization (all frames at once)
  - [x] Extract optimized poses and points from result
  - [x] Added robust kernel support (Huber)
  - [x] Added reprojection error computation
  - [x] Added outlier filtering functionality

- [x] Create `initialization.py` module ⚠️ NEEDS REVISION
  - [x] Two-view initialization for first camera pair
  - [x] Incremental pose estimation for remaining cameras
  - [x] Triangulation of 3D points from Co-Tracker tracks
  - [x] RANSAC-based outlier filtering
  - **TODO**: Integrate COLMAP tools for more robust initialization

### Phase 5: Pipeline Integration 🔄 IN PROGRESS
- [ ] Create `pipeline.py` main module
  - [ ] Command-line interface
  - [ ] Video/image sequence input handling
  - [ ] Progress tracking and logging
  - [ ] Output camera poses in COLMAP format
- [ ] Create `utils.py` for helper functions
  - [ ] Track visualization
  - [ ] Pose visualization
  - [ ] Export to COLMAP format
  - [ ] Import from COLMAP format

### Phase 6: Testing and Optimization ⏳ PENDING
- [ ] Create test scripts with sample data
- [ ] Benchmark against existing COLMAP results
- [ ] Optimize sliding window parameters
- [ ] Handle edge cases (few tracks, degenerate motion)

## Key Implementation Decisions & Changes

### 1. Co-Tracker Integration
- Used standard `CoTrackerPredictor` instead of online version
- Model loaded from torch.hub: `torch.hub.load("facebookresearch/co-tracker", "cotracker3")`
- Implemented backward tracking for better results
- Query points sampled from middle frame of each window

### 2. Track Management
- Implemented sophisticated track merging based on overlap distance
- Tracks stored as `MergedTrack` objects with frame-indexed observations
- Minimum track length filtering (default: 10 frames)

### 3. Bundle Adjustment
- Used GTSAM's `GenericProjectionFactorCal3_S2` for projection factors
- Added Huber robust kernel for outlier handling
- Implemented prior on first camera to fix gauge freedom

### 4. Initialization Strategy
- Currently uses traditional two-view initialization with essential matrix
- **Next Step**: Replace with COLMAP's robust initialization tools
  - Option 1: Use pycolmap Python bindings
  - Option 2: Call COLMAP CLI tools for initialization
  - Option 3: Use existing COLMAP sparse models as initialization

## Next Steps for Continuation

1. **Revise initialization.py**:
   ```python
   # Use COLMAP for initialization instead of custom implementation
   # Either through pycolmap or by calling COLMAP tools
   ```

2. **Complete pipeline.py**:
   - Main entry point for the entire pipeline
   - Handle different input formats (video, image sequence)
   - Orchestrate all modules

3. **Implement utils.py**:
   - COLMAP format I/O (cameras.txt, images.txt, points3D.txt)
   - Visualization tools
   - Metric computation

4. **Create configuration system**:
   - YAML configuration for all parameters
   - Support different presets

## Dependencies to Install
```bash
pip install gtsam opencv-python scipy torch torchvision imageio
# Optional but recommended:
pip install pycolmap  # For robust initialization
```

## Important File Locations & References

### Co-Tracker Implementation Reference
- `/hdd2/0321_block_drone_video/colmap/submodules/co-tracker/online_demo.py`
- `/hdd2/0321_block_drone_video/colmap/submodules/co-tracker/notebooks/demo.py`

### Camera Calibration Files
- `/hdd2/0321_block_drone_video/colmap/config/intrinsic/x3/K.txt` - 3x intrinsic matrix
- `/hdd2/0321_block_drone_video/colmap/config/intrinsic/x3/dist.txt` - 3x distortion coefficients
- `/hdd2/0321_block_drone_video/colmap/config/intrinsic/x7/K.txt` - 7x intrinsic matrix
- `/hdd2/0321_block_drone_video/colmap/config/intrinsic/x7/dist.txt` - 7x distortion coefficients

### Existing COLMAP Utils
- `/hdd2/0321_block_drone_video/colmap/src/colmap_utils/read_write_model.py` - COLMAP I/O functions

## Code Structure Summary

### `feature_extractor.py`
```python
class CoTrackerExtractor:
    - load_video_frames()
    - get_initial_queries() # Sample points from middle frame
    - track_window() # Track features in single window
    - extract_tracks() # Process entire video with sliding windows
    - visualize_tracks()
```

### `track_manager.py`
```python
class TrackManager:
    - find_overlapping_tracks()
    - merge_two_tracks()
    - merge_all_tracks() # Main merging algorithm
    - convert_to_bundle_adjustment_format()
```

### `camera_model.py`
```python
class CameraModel:
    - get_camera_params()
    - get_colmap_camera_model()
    - get_gtsam_calibration()
    - save_colmap_cameras_txt()
```

### `bundle_adjustment.py`
```python
class GlobalBundleAdjustment:
    - create_factor_graph()
    - create_initial_values()
    - optimize() # Main BA optimization
    - compute_reprojection_errors()
    - filter_outlier_tracks()
```

## Technical Specifications

### Sliding Window Strategy
```python
window_size = 30  # frames per window
overlap = 15      # overlapping frames between windows
grid_size = 50    # initial grid sampling
confidence_threshold = 0.8  # minimum track confidence
```

### Global Bundle Adjustment with GTSAM
```python
import gtsam
import numpy as np

def global_bundle_adjustment(tracks, initial_poses, calibration):
    """
    Pure bundle adjustment using Co-Tracker tracks
    No feature matching or SfM pipeline - just optimization
    """
    # Create factor graph for global BA
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    
    # Add all observations from Co-Tracker tracks
    for track_id, track in enumerate(tracks):
        for frame_id, (x, y) in track.observations:
            # Create projection factor
            factor = gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(x, y),          # 2D observation
                measurement_noise,            # pixel noise
                gtsam.symbol('x', frame_id), # camera pose
                gtsam.symbol('l', track_id), # 3D point
                calibration                   # camera intrinsics
            )
            graph.add(factor)
    
    # Add initial estimates
    for i, pose in enumerate(initial_poses):
        initial_estimate.insert(gtsam.symbol('x', i), pose)
    
    for j, point in enumerate(initial_3d_points):
        initial_estimate.insert(gtsam.symbol('l', j), point)
    
    # Optimize with Levenberg-Marquardt
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    
    result = optimizer.optimize()
    return result
```

### Bundle Adjustment Parameters
```python
# Noise models
pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)  # 2 pixel std dev
pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])  # rad, meters
)

# Optimization
max_iterations = 100
convergence_threshold = 1e-6
```

### Data Flow
1. Video → Co-Tracker → Feature Tracks
2. Tracks → Track Manager → Filtered & Merged Tracks
3. Merged Tracks → Bundle Adjustment → Camera Poses
4. Camera Poses → Export → COLMAP Format

## Directory Structure
```
src/cotracker_ba/
├── __init__.py
├── feature_extractor.py   # Co-Tracker integration
├── track_manager.py       # Track filtering and merging
├── camera_model.py        # Camera intrinsics handling
├── initialization.py      # Initial pose estimation
├── bundle_adjustment.py   # GTSAM bundle adjustment
├── pipeline.py           # Main pipeline orchestration
├── utils.py              # Visualization and I/O utilities
└── config/
    └── default.yaml      # Pipeline configuration
```

## Usage Example
```bash
python src/cotracker_ba/pipeline.py \
    --input_video /path/to/video.mp4 \
    --output_dir outputs/cotracker_ba/ \
    --camera_model x3 \
    --window_size 30 \
    --overlap 15 \
    --export_colmap
```

## Expected Outputs
- Camera poses for each frame (COLMAP format)
- 3D point cloud from triangulated tracks
- Track visualization videos
- Bundle adjustment statistics

## Advantages Over Traditional Methods
- Better feature tracking in low-texture areas
- Handles motion blur and appearance changes
- No need for feature detection/description
- Direct dense tracking capability
- More robust to challenging conditions

## Critical Next Steps for COLMAP Integration

### 1. Initialization with COLMAP
```python
# Option A: Use pycolmap (if installed)
import pycolmap
reconstruction = pycolmap.incremental_mapping(database_path, image_path, output_path)

# Option B: Call COLMAP CLI
subprocess.run(['colmap', 'mapper', '--database_path', db_path, ...])

# Option C: Use existing COLMAP sparse model
from colmap_utils.read_write_model import read_model
cameras, images, points3D = read_model(path='/path/to/sparse/0', ext='.bin')
```

### 2. Pipeline Integration Flow
```
1. Extract video frames → temporary image directory
2. Run Co-Tracker → get feature tracks
3. Export tracks to COLMAP database format
4. Run COLMAP mapper for initialization only (first N frames)
5. Use COLMAP's initial poses/points as input to GTSAM
6. Run global BA with all tracks
7. Export final results to COLMAP format
```

### 3. COLMAP Database Integration
- Need to create SQLite database with tracks
- Tables: cameras, images, keypoints, descriptors, matches
- Co-Tracker tracks → COLMAP matches format

## Known Issues & Solutions

1. **Co-Tracker Model Loading**
   - Use `torch.hub.load()` instead of local checkpoint
   - Model will be downloaded automatically on first run

2. **Memory Management**
   - Process video in chunks for large files
   - Limit number of tracks per window

3. **Coordinate Systems**
   - Co-Tracker: pixel coordinates (0, 0) at top-left
   - COLMAP: same coordinate system
   - GTSAM: camera looks down +Z axis

## Testing Commands
```bash
# Test with sample video
python src/cotracker_ba/pipeline.py \
    --input_video data/videos/section1_x3.mp4 \
    --output_dir outputs/test_cotracker_ba/ \
    --camera_model x3 \
    --window_size 30 \
    --overlap 15 \
    --max_frames 300  # For quick testing

# Compare with existing COLMAP results
python scripts/compare_poses.py \
    outputs/test_cotracker_ba/sparse/0 \
    outputs/workspaces/section1_3x/sparse/0
```

## References
- Co-Tracker: https://co-tracker.github.io/
- GTSAM: https://gtsam.org/
- pycolmap: https://github.com/colmap/pycolmap
- Bundle Adjustment Theory: https://en.wikipedia.org/wiki/Bundle_adjustment