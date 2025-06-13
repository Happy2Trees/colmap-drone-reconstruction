# COLMAP Drone Reconstruction

Advanced 3D reconstruction system for drone footage using a two-stage pipeline with COLMAP integration.

## Project Overview

This project implements a sophisticated two-stage 3D reconstruction pipeline optimized for processing drone video footage. It separates feature extraction (precompute stage) from 3D reconstruction, enabling efficient processing of high-resolution drone videos captured at different magnifications (x1, x3, x7).

## ✨ Key Features

### Two-Stage Pipeline
- **Stage 1: Precompute** - Extract point tracks, depth maps, and optical flow
- **Stage 2: Reconstruction** - Perform 3D reconstruction using precomputed features

### Advanced Point Tracking
- **CoTracker Integration** with interval-based windowing
- **Multiple Feature Detectors**:
  - Grid-based initialization (uniform distribution)
  - SIFT (Scale-Invariant Feature Transform)
  - SuperPoint (deep learning features)

### Smart Preprocessing
- Automatic resize and center crop for high-resolution images
- Intrinsic parameter adjustment
- Validation system to avoid redundant processing

### GeometryCrafter-style Window-based Bundle Adjustment
- **Cross-projection optimization** for global consistency
- **Depth-aware 3D initialization** using monocular depth maps
- **Window-based processing** without track merging
- **COLMAP export** for compatibility with existing tools

### Additional Features
- COLMAP and Super-COLMAP integration
- Comprehensive visualization tools (MP4 videos, 3D models)
- YAML-based configuration system
- Docker environment for easy deployment

## Project Structure

```
├── src/                          # Python source code
│   ├── precompute/              # Feature extraction pipeline
│   ├── preprocessing/           # Image preprocessing tools
│   ├── window_ba/               # Window-based Bundle Adjustment
│   ├── visualization/           # 3D visualization
│   └── colmap_utils/           # COLMAP I/O utilities
├── scripts/                     # Execution scripts
├── config/                      # Configuration files
│   ├── precompute*.yaml        # Precompute configurations
│   └── intrinsic/              # Camera calibration
├── outputs/                     # Output files (excluded from git)
├── data/                        # Input data (excluded from git)
└── submodules/                  # Git submodules
    ├── co-tracker/             # Point tracking
    ├── super-colmap/           # SuperPoint COLMAP
    └── particle-sfm/           # Particle-based SfM
```

## 📁 Scene Data Structure

Each scene should be organized as follows:
```
Scene/
├── images/                      # Input images
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── K.txt                        # Camera intrinsic matrix (3x3)
└── dist.txt                     # Distortion coefficients (k1, k2, p1, p2, k3)
```

After preprocessing and feature extraction:
```
Scene_processed_1920x1080/       # Preprocessed scene
├── images/                      # Resized images
├── K.txt                        # Adjusted intrinsics
├── dist.txt                     # Same distortion
├── cotracker/                   # Extracted tracks
│   └── 48_10_grid.npy          # window_interval_method.npy
├── depth/                       # Depth maps (if computed)
│   └── GeometryCrafter/        # Monocular depth estimation
├── visualizations/              # Optional visualizations
└── window_ba_output/            # Bundle adjustment results
    ├── cameras_final.npz        # Optimized camera poses
    ├── window_tracks_3d.npz     # 3D points per window
    └── colmap/                  # COLMAP export
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU (for COLMAP processing)
- Python 3.8+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Happy2Trees/colmap-drone-reconstruction.git
cd colmap-drone-reconstruction
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Build and run with Docker:
```bash
docker-compose up -d
```

### Usage

#### 1. Precompute Features
```bash
# Basic usage with grid-based tracking
python -m src.precompute.precompute /path/to/scene

# SIFT-based feature tracking with preprocessing
python -m src.precompute.precompute /path/to/scene --config config/precompute_sift.yaml

# Dense tracking for detailed reconstruction
python -m src.precompute.precompute /path/to/scene --config config/precompute_dense.yaml
```

#### 2. Window-based Bundle Adjustment (GeometryCrafter-style)
```bash
# Run window BA with precomputed tracks and depth
python -m src.window_ba /path/to/scene

# With two-phase optimization (camera + 3D refinement)
python -m src.window_ba /path/to/scene --use_refine

# Custom configuration
python -m src.window_ba /path/to/scene --config config/window_ba.yaml
```

#### 3. Run COLMAP Reconstruction
```bash
# Standard COLMAP
./scripts/run_colmap/run_colmap_3x_0.sh  # Section 1, 3x magnification

# Super-COLMAP with SuperPoint features
cd submodules/super-colmap
python super_colmap.py --projpath /path/to/project --cameraModel SIMPLE_RADIAL
```

#### 4. Preprocessing Tools
```bash
# Extract frames from video
python src/preprocessing/slice_fps.py /path/to/video --target_fps 10

# Resize and crop high-resolution images
python -m src.preprocessing.resize_and_crop /path/to/scene --width 1920 --height 1080
```

#### 5. Visualization
```bash
# Visualize COLMAP results as PNG
python src/visualization/visualize_colmap.py /path/to/sparse/0 --output viz.png

# Export to 3D model (OBJ)
python src/visualization/visualize.py /path/to/sparse/0 --output model.obj
```

## ⚙️ Configuration

The project uses YAML configuration files for different scenarios:

- `config/precompute.yaml` - Default balanced settings (20x20 grid)
- `config/precompute_dense.yaml` - Dense tracking (more windows/points)
- `config/precompute_sparse.yaml` - Sparse tracking (faster processing)
- `config/precompute_sift.yaml` - SIFT feature detection
- `config/precompute_superpoint.yaml` - SuperPoint features
- `config/precompute_test.yaml` - Quick testing
- `config/window_ba.yaml` - Window-based Bundle Adjustment settings

Example configuration:
```yaml
cotracker:
  window_size: 48           # Frames per window
  interval: 10              # Frame interval between windows
  initialization_method: grid  # grid, sift, or superpoint
  max_features: 400         # Number of points to track

preprocessing:
  enabled: true
  target_width: 1920
  target_height: 1080

visualization:
  enabled: true
  save_video: true
```

## 🔗 Related Projects

- [COLMAP](https://colmap.github.io/) - Structure-from-Motion and Multi-View Stereo
- [CoTracker](https://co-tracker.github.io/) - Point tracking in video
- [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) - Self-supervised interest point detection
- [GeometryCrafter](https://github.com/GeometryCrafter/GeometryCrafter) - 3D geometry estimation from images

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.