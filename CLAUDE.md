# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a COLMAP-based 3D reconstruction project for processing drone video footage of light emitter blocks. The project uses both standard COLMAP and Super-COLMAP (with SuperPoint feature detection) to create 3D reconstructions from videos captured at different magnifications (x1, x3, x7).

## Project Structure

```
/hdd2/0321_block_drone_video/colmap/
├── src/                          # Python source code
│   ├── preprocessing/            # Video processing and calibration
│   ├── visualization/            # 3D visualization tools
│   ├── colmap_utils/            # COLMAP file I/O utilities
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
├── data/                         # Input data
└── submodules/                   # Git submodules
    └── super-colmap/             # SuperPoint-based COLMAP
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

### Key Python Scripts
- **src/preprocessing/slice_fps.py**: Reduces video frame rate by sampling frames at specified intervals
- **src/preprocessing/calibration.py**: Performs checkerboard-based camera calibration
- **src/visualization/visualize_colmap.py**: Creates 3D visualizations of camera poses and point clouds
- **src/visualization/visualize.py**: Exports COLMAP reconstructions to OBJ format with colored points and camera frustums
- **src/colmap_utils/read_write_model.py**: Utilities for reading/writing COLMAP model files

## Important Notes

- The project uses NVIDIA GPUs (specified in docker-compose.yml as devices 4,5,6,7)
- Each run script processes a specific section of drone footage
- The "w" variants of scripts use sequential matching for faster processing
- Results are stored in binary format but can be converted to text using COLMAP tools
- Multiple sparse models (0, 1, 2, etc.) may be generated if the reconstruction produces disconnected components