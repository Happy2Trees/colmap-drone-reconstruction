# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a COLMAP-based 3D reconstruction project for processing drone video footage of light emitter blocks. The project is being refactored into a two-stage pipeline:
1. **Precompute Stage**: Extract features (tracks, depth, optical flow) from input videos
2. **Reconstruction Stage**: Perform 3D reconstruction using precomputed data

The project supports both standard COLMAP and Super-COLMAP (with SuperPoint feature detection) for videos captured at different magnifications (x1, x3, x7).

## Project Structure

```
/hdd2/0321_block_drone_video/colmap/
├── src/                          # Python source code
│   ├── preprocessing/            # Video processing and calibration
│   ├── precompute/              # Feature extraction pipeline (NEW)
│   │   ├── precompute.py        # Main precompute entry point
│   │   ├── trackers/            # Track extraction modules
│   │   ├── depth/               # Depth estimation modules
│   │   └── optical_flow/        # Optical flow modules
│   ├── visualization/            # 3D visualization tools
│   ├── colmap_utils/            # COLMAP file I/O utilities
│   ├── cotracker_ba/            # CoTracker bundle adjustment
│   └── utils/                    # General utility functions
├── scripts/                      # Execution scripts
│   └── run_colmap/              # COLMAP pipeline scripts
├── config/                       # Configuration files
│   └── intrinsic/               # Camera calibration parameters
├── outputs/                      # All output files
│   ├── workspaces/              # COLMAP working directories
│   ├── sparse_models/           # Sparse reconstruction results
│   ├── visualizations/          # Visualization outputs
│   └── logs/                    # Execution logs
├── data/                         # Input data (with new structure)
│   └── Scene/                    # Scene directory
│       ├── images/              # Input images
│       ├── K.txt                # Intrinsic parameters
│       ├── dist.txt             # Distortion coefficients
│       ├── cotracker/           # Precomputed tracks
│       ├── depth/               # Precomputed depth maps
│       └── optical_flow/        # Precomputed flow fields
└── submodules/                   # Git submodules
    ├── super-colmap/             # SuperPoint-based COLMAP
    ├── co-tracker/              # CoTracker for point tracking
    └── particle-sfm/            # Particle-based SfM
```

## Key Commands

### Running COLMAP Reconstruction
```bash
# Run COLMAP for different sections and magnifications
./scripts/run_colmap/run_colmap_3x_0.sh   # Section 1, 3x magnification
./scripts/run_colmap/run_colmap_3x_1.sh   # Section 2, 3x magnification
./scripts/run_colmap/run_colmap_3x_2.sh   # Section 3, 3x magnification
./scripts/run_colmap/run_colmap_3x_3.sh   # Section 4, 3x magnification
./scripts/run_colmap/run_colmap_7x_4.sh   # Section 1, 7x magnification

# "w" variants use sequential matching instead of exhaustive
./scripts/run_colmap/run_colmap_3x_0w.sh  # Faster but potentially less accurate
```

### Running Super-COLMAP (SuperPoint features)
```bash
cd submodules/super-colmap
python super_colmap.py \
    --projpath /path/to/project \
    --cameraModel SIMPLE_RADIAL \
    --images_path images \
    --single_camera
```

### Python Utilities
```bash
# Extract frames from video at specific FPS
python src/preprocessing/slice_fps.py /path/to/images --target_fps 10 --source_fps 60

# Perform camera calibration  
python src/preprocessing/calibration.py /path/to/calibration/images --output_dir outputs/calibration

# Preprocess high-resolution images (resize and center crop)
python -m src.preprocessing.resize_and_crop /path/to/scene --width 1920 --height 1080
# Creates: /path/to/scene_processed_1920x1080/

# Precompute features using default config
python -m src.precompute.precompute /path/to/scene

# Precompute with custom config
python -m src.precompute.precompute /path/to/scene --config config/precompute_dense.yaml

# Precompute without visualization (overrides config file setting)
python -m src.precompute.precompute /path/to/scene --no-visualize

# Available config files:
#   config/precompute.yaml           # Default balanced settings (grid)
#   config/precompute_dense.yaml     # Dense tracking (more windows/points)
#   config/precompute_sparse.yaml    # Sparse tracking (fewer windows)
#   config/precompute_test.yaml      # Quick testing configuration
#   config/precompute_sift.yaml      # SIFT feature detection + preprocessing
#   config/precompute_superpoint.yaml # SuperPoint features (falls back to SIFT)

# Visualize COLMAP reconstruction results (as PNG)
python src/visualization/visualize_colmap.py /path/to/sparse/0 --output outputs/visualizations/colmap_viz.png

# Export COLMAP reconstruction to OBJ file
python src/visualization/visualize.py /path/to/sparse/0 --output outputs/visualizations/model.obj
```

### Docker Operations
```bash
# Build and run with Docker Compose
docker-compose up -d

# Direct Docker access
docker exec -it colmap_ct bash
```

### Dependencies Installation
```bash
pip install -r requirements.txt
```

## Architecture

### Input Data Structure

Each scene should be organized as follows:
```
Scene/
├── images/                       # Input images
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── K.txt                        # Camera intrinsic matrix
├── dist.txt                     # Distortion coefficients
├── cotracker/                   # Precomputed tracks (after precompute stage)
│   └── {window_size}_{interval}_{method}.npy
├── depth/                       # Precomputed depth maps
│   └── GeometryCrafter/
│       ├── 001.npy             # Same names as images
│       └── ...
└── optical_flow/               # Precomputed optical flow
    └── raft/
        └── (format TBD)
```

### Data Organization
- **Input Videos**: Located in `data/` directory, organized by magnification and section
- **Working Directories**: `outputs/workspaces/section{1-4}_{3x,7x}{,_seq}/` contain COLMAP databases and sparse reconstructions
  - `section1_3x`: Section 1 at 3x magnification with exhaustive matching
  - `section1_3x_seq`: Section 1 at 3x magnification with sequential matching
  - Similar naming pattern for other sections
- **Intrinsic Calibration**: Camera parameters stored in `config/intrinsic/{x3,x7}/`
- **Scripts**: Shell scripts in `scripts/run_colmap/` handle COLMAP pipeline execution
- **Results**: 
  - Sparse models: `outputs/sparse_models/{x1,x7}_sparse/`
  - Visualizations: `outputs/visualizations/`

### COLMAP Pipeline
1. **Feature Extraction**: Detects keypoints in images using SIFT (standard) or SuperPoint (super-colmap)
2. **Feature Matching**: Matches features between image pairs (exhaustive or sequential)
3. **Sparse Reconstruction**: Estimates camera poses and 3D points
4. **Output**: Binary files (cameras.bin, images.bin, points3D.bin) and text versions

### Super-COLMAP Integration
The `submodules/super-colmap/` directory contains an enhanced version that replaces SIFT with SuperPoint features:
- Uses PyTorch-based SuperPoint model for better feature detection
- Requires downloading the SuperPoint weights: `superpoint_v1.pth`
- Processes images through the same COLMAP pipeline but with superior features

### Two-Stage Pipeline

#### Stage 1: Precompute
Extracts all necessary features before reconstruction:

**1a. Preprocessing (Optional)**
- **Resize and Center Crop**: Reduces high-resolution images to manageable size
  - Automatic intrinsic parameter adjustment
  - Preserves original scene structure
  - Configurable target resolution in config files
  - Skip logic for existing preprocessed data

**1b. Feature Extraction**
- **CoTracker**: Point tracks using interval-based windowing
  - Configurable window size and interval
  - Multiple initialization methods: grid, SIFT, SuperPoint
  - Automatic visualization with MP4 video output
- **GeometryCrafter**: Monocular depth estimation (future)
- **RAFT**: Optical flow between consecutive frames (future)

Output structure after precompute:
```
Scene/
├── cotracker/
│   ├── 48_10_grid.npy          # Tracks file (window_interval_method.npy)
│   │                           # Contains: window_id, start_frame, end_frame, tracks, visibility
│   └── visualizations/         # If visualization enabled
│       ├── tracking_result.mp4 # Video with tracked points
│       ├── tracking_summary.png # Statistics and timeline
│       └── frame_*.png         # Sample frames (optional)
├── precompute.log              # Execution log
└── precompute_summary.json     # Pipeline results summary
```

#### Stage 2: Reconstruction
Uses precomputed features for 3D reconstruction:
- Loads tracks from `.npy` files
- Uses depth priors for initialization
- Incorporates optical flow for motion estimation

### Key Python Scripts
- **src/preprocessing/slice_fps.py**: Reduces video frame rate by sampling frames at specified intervals
- **src/preprocessing/calibration.py**: Performs checkerboard-based camera calibration
- **src/preprocessing/resize_and_crop.py**: Preprocesses high-resolution images with automatic intrinsic adjustment
- **src/precompute/precompute.py**: Main entry point for feature extraction pipeline (with integrated preprocessing)
- **src/precompute/trackers/cotracker_extractor.py**: CoTracker with interval-based windowing
- **src/visualization/visualize_colmap.py**: Creates 3D visualizations of camera poses and point clouds
- **src/visualization/visualize.py**: Exports COLMAP reconstructions to OBJ format with colored points and camera frustums
- **src/colmap_utils/read_write_model.py**: Utilities for reading/writing COLMAP model files

## Important Notes

- The project uses NVIDIA GPUs (specified in docker-compose.yml as devices 4,5,6,7)
- Each run script processes a specific section of drone footage
- The "w" variants of scripts use sequential matching for faster processing
- Results are stored in binary format but can be converted to text using COLMAP tools
- Multiple sparse models (0, 1, 2, etc.) may be generated if the reconstruction produces disconnected components

## Current Development Status

- **Active Refactoring**: Moving from monolithic to two-stage pipeline
- **CoTracker Integration**: ✅ Completed with interval-based windowing and multiple feature methods
- **Preprocessing System**: ✅ Completed with automatic intrinsic adjustment (2025-01-12)
- **Serialization Fix**: ✅ Resolved cv2.KeyPoint pickle issue, simplified output format (2025-01-13)
- **See**: `docs/todo_refactoring.md` for detailed refactoring plan and progress