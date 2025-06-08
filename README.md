# COLMAP Drone Reconstruction

COLMAP-based 3D reconstruction project for processing drone video footage of light emitter blocks.

## Project Overview

This project uses COLMAP and Super-COLMAP (with SuperPoint feature detection) to create 3D reconstructions from drone videos captured at different magnifications (x1, x3, x7).

## Features

- Video preprocessing and frame extraction
- Camera calibration support
- COLMAP sparse reconstruction pipeline
- Super-COLMAP integration with SuperPoint features
- 3D visualization tools
- Docker environment for easy deployment

## Project Structure

```
├── src/                  # Python source code
├── scripts/             # Execution scripts
├── config/              # Configuration files
├── outputs/             # Output files (excluded from git)
├── data/                # Input data (excluded from git)
└── submodules/          # Git submodules
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

Run COLMAP reconstruction:
```bash
./scripts/run_colmap/run_colmap_3x_0.sh  # For section 1, 3x magnification
```

Extract frames from video:
```bash
python src/preprocessing/slice_fps.py /path/to/video --target_fps 10
```

Visualize results:
```bash
python src/visualization/visualize_colmap.py /path/to/sparse/0
```

## License

This project is licensed under the MIT License.