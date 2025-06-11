# Two-Stage 3D Reconstruction Pipeline with Advanced Feature Extraction

## Overview

This PR introduces a major refactoring of the COLMAP-based 3D reconstruction system, implementing a two-stage pipeline optimized for processing drone footage. The new architecture separates feature extraction (precompute stage) from 3D reconstruction, enabling more efficient processing and better resource utilization.

## 🎯 Motivation

Processing high-resolution drone footage for 3D reconstruction faces several challenges:
- Memory constraints when processing long video sequences
- Redundant feature extraction in monolithic pipelines
- Lack of flexibility in feature detection methods
- Difficulty in experimenting with different tracking parameters

This refactoring addresses these issues by:
1. Separating concerns between feature extraction and reconstruction
2. Enabling preprocessing of high-resolution images
3. Supporting multiple feature initialization methods
4. Providing flexible configuration for different use cases

## 🚀 Key Features

### 1. Two-Stage Pipeline Architecture

#### Stage 1: Precompute
- Extract point tracks using CoTracker with interval-based windowing
- Support for depth estimation (GeometryCrafter - future)
- Support for optical flow (RAFT - future)
- Efficient disk-based storage in NumPy format

#### Stage 2: Reconstruction
- Load precomputed features from disk
- Perform 3D reconstruction using COLMAP
- Reduced memory footprint and processing time

### 2. Advanced Point Tracking

**CoTracker Integration** with three initialization methods:
- **Grid**: Uniform 20x20 point distribution (400 points)
- **SIFT**: Scale-Invariant Feature Transform for texture-aware points
- **SuperPoint**: Deep learning-based features (with automatic SIFT fallback)

**Interval-based Windowing**:
```python
# Example: window_size=48, interval=10
Window 0: frames [0, 48)
Window 1: frames [10, 58)   # 79% overlap
Window 2: frames [20, 68)   # 58% overlap with Window 0
```

### 3. Image Preprocessing Module

Handles high-resolution drone footage efficiently:
- **Resize and Center Crop**: Reduces resolution while maintaining aspect ratio
- **Automatic Intrinsic Adjustment**: Updates camera matrix for new resolution
- **Validation System**: Avoids redundant preprocessing
- **Multi-resolution Support**: Different resolutions per scene

### 4. Configuration System

YAML-based configuration with presets:
- `precompute.yaml`: Default balanced settings
- `precompute_dense.yaml`: More windows and points for detailed tracking
- `precompute_sparse.yaml`: Fewer windows for faster processing
- `precompute_sift.yaml`: SIFT-based feature detection
- `precompute_superpoint.yaml`: SuperPoint features
- `precompute_test.yaml`: Quick testing configuration

### 5. Visualization System

- MP4 video generation showing tracked points
- Color-coded tracks per window
- Trail visualization
- Summary plots with statistics

## 📁 Project Structure

```
src/
├── precompute/
│   ├── precompute.py              # Main pipeline entry point
│   ├── trackers/
│   │   ├── base_tracker.py        # Abstract base class
│   │   ├── cotracker_extractor.py # CoTracker implementation
│   │   └── feature_initializers/  # Modular feature detection
│   │       ├── base_initializer.py
│   │       ├── grid_initializer.py
│   │       ├── sift_initializer.py
│   │       └── superpoint_initializer.py
│   ├── depth/                     # Future: GeometryCrafter
│   └── optical_flow/              # Future: RAFT
├── preprocessing/
│   └── resize_and_crop.py         # Image preprocessing
└── tests/
    ├── test_precompute.py
    ├── test_feature_initializers.py
    └── test_preprocessing_integration.py
```

## 📊 Usage Examples

### Basic Usage
```bash
# Precompute features with default settings
python -m src.precompute.precompute /path/to/scene

# SIFT-based tracking with preprocessing
python -m src.precompute.precompute /path/to/scene --config config/precompute_sift.yaml

# Direct preprocessing
python -m src.preprocessing.resize_and_crop /path/to/scene --width 1920 --height 1080
```

### Output Structure
```
Scene_processed_1920x1080/
├── images/                     # Preprocessed images
├── K.txt                      # Adjusted intrinsics
├── dist.txt                   # Distortion coefficients
├── preprocessing_info.yaml     # Metadata
├── cotracker/
│   ├── 48_10_grid.npy        # Tracks: window_size=48, interval=10, grid method
│   └── visualizations/       # If enabled
│       ├── tracking_result.mp4
│       └── tracking_summary.png
└── precompute.log            # Processing log
```

## 🔧 Technical Details

### Memory Efficiency
- Window-by-window processing
- GPU memory cleanup after each window
- Disk-based storage instead of in-memory

### Serialization Solution
- Resolved cv2.KeyPoint pickle issues
- Simplified data structure storing only essential tracking data
- Robust .npy format for cross-platform compatibility

### Performance Optimizations
- Skip logic for existing preprocessed data
- Configurable window overlap via interval parameter
- Future support for multi-GPU processing

## 📋 Testing

Comprehensive test suite included:
- `test_precompute.py`: Pipeline functionality
- `test_feature_initializers.py`: Compare initialization methods
- `test_preprocessing_integration.py`: Preprocessing validation

## 🔄 Migration Guide

For existing users:
1. Camera intrinsics now stored per-scene (not global)
2. Use preprocessing for high-resolution images
3. Configure tracking parameters via YAML files
4. Results stored in scene-specific directories

## 🚦 Future Work

- [ ] GeometryCrafter depth estimation integration
- [ ] RAFT optical flow integration
- [ ] Track merging across windows
- [ ] Multi-GPU support
- [ ] Real-time preview during tracking

## 📝 Documentation

- Updated `CLAUDE.md` with detailed instructions
- `docs/todo_refactoring.md` tracks implementation progress
- Inline documentation in all modules

## ⚠️ Breaking Changes

- Removed global intrinsic parameter files
- Changed from monolithic to modular architecture
- New directory structure for outputs

## 🤝 Contributing

This refactoring sets the foundation for community contributions:
- Modular architecture allows easy addition of new trackers
- Abstract base classes define clear interfaces
- Comprehensive testing ensures stability

---

**Questions or feedback?** Please open an issue or reach out to the maintainers.