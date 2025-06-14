# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COLMAP-based 3D reconstruction project for processing drone video footage. Features a two-stage pipeline:
1. **Precompute Stage**: Extract features (tracks, depth, optical flow) from input videos
2. **Reconstruction Stage**: Perform 3D reconstruction using precomputed data

## Key Commands

### Precompute Pipeline (Primary Workflow)
```bash
# Basic feature extraction
python -m src.precompute.precompute /path/to/scene

# With custom config
python -m src.precompute.precompute /path/to/scene --config config/precompute_dense.yaml

# Available configs:
#   precompute.yaml                 # Default balanced settings
#   precompute_geometrycrafter.yaml # GeometryCrafter depth estimation
#   precompute_sift.yaml            # SIFT features + preprocessing
#   precompute_superpoint.yaml      # SuperPoint features
```

### Window-based Bundle Adjustment (GeometryCrafter-style)
```bash
# Run window-based BA with cross-projection
python -m src.window_ba /path/to/scene

# With two-phase optimization (camera + 3D refinement) - FULLY IMPLEMENTED
python -m src.window_ba /path/to/scene --use_refine

# Custom config
python -m src.window_ba /path/to/scene --config config/window_ba.yaml
```

**Key Features:**
- Phase 1: Camera-only optimization with fixed 3D from depth
- Phase 2: Joint optimization of cameras + boundary 3D points
- Automatic visualization generation (PNG files for CLI)
- COLMAP export with optimized boundary points

### Preprocessing Tools (Standalone)
```bash
# Frame sampling
python src/preprocessing/slice_fps.py /path/to/images --target_fps 10 --source_fps 60
python src/preprocessing/slice_fps.py /path/to/images --interval 6

# Image resizing
python -m src.preprocessing.resize_and_crop /path/to/scene --width 1920 --height 1080

# Camera calibration
python src/preprocessing/calibration.py /path/to/calibration/images --output_dir outputs/calibration
```

### COLMAP Reconstruction
```bash
# Run reconstruction scripts
./scripts/run_colmap/run_colmap_3x_0.sh   # Section 1, 3x magnification
./scripts/run_colmap/run_colmap_3x_0w.sh  # Sequential matching (faster)
```

### Visualization
```bash
python src/visualization/visualize_colmap.py /path/to/sparse/0 --output outputs/visualizations/colmap_viz.png
python src/visualization/visualize.py /path/to/sparse/0 --output outputs/visualizations/model.obj
```

## Data Structure

```
Scene/                              # Input scene directory
├── images/                         # Input images (001.jpg, 002.jpg, ...)
├── K.txt                          # Camera intrinsic matrix
├── dist.txt                       # Distortion coefficients
└── [After precompute]
    ├── cotracker/                 # Point tracks
    │   └── {window}_{interval}_{method}.npy
    ├── depth/                     # Depth maps
    │   └── GeometryCrafter/*.npy
    └── precompute_summary.json    # Pipeline results
```

## Preprocessing Pipeline

The precompute system supports automatic preprocessing:

1. **Frame Sampling**: Reduce frame count (e.g., 60fps → 10fps)
2. **Resize & Crop**: Adjust resolution with automatic intrinsic calibration

Configure in YAML:
```yaml
preprocessing:
  enabled: true
  frame_sampling:
    enabled: true
    target_fps: 10
    source_fps: 60
  target_width: 1920
  target_height: 1080
```

Output naming: `scene_fps10_processed_1920x1080/`

## Feature Extractors

### CoTracker
- Tracks points across video sequences
- Window-based processing for long videos
- Supports grid, SIFT, and SuperPoint initialization

### GeometryCrafter
- Monocular depth estimation
- Automatic segmentation for large videos (default: 1000 frames/segment)
- GPU memory aware processing

## Configuration Files

Each config file in `config/` controls:
- Features to extract (cotracker, depth, flow)
- Model parameters (window size, device, etc.)
- Preprocessing settings
- Visualization options

Example workflow:
```bash
# For depth extraction on preprocessed video:
python -m src.precompute.precompute /data/scene --config config/precompute_geometrycrafter.yaml
```

## Important Notes

- GPU required for most operations
- Results stored in scene directory alongside input data
- Preprocessing creates new directories, preserving originals
- Large videos automatically segmented for memory efficiency

## Development References

- Implementation details: `docs/todo_globalba.md` (COMPLETED)
- Window BA modules:
  - `src/window_ba/window_track_loader.py`: Window-based track loading
  - `src/window_ba/window_depth_initializer.py`: Depth-based 3D initialization
  - `src/window_ba/window_bundle_adjuster.py`: Cross-projection BA with Phase 2
  - `src/window_ba/visualization.py`: CLI-compatible visualization
  - `src/window_ba/pipeline.py`: Complete pipeline with COLMAP export
- Submodules: GeometryCrafter, CoTracker, Super-COLMAP