# Window-based Bundle Adjustment with Depth and Tracks (GeometryCrafter Style)

## 개요 (Overview)

GeometryCrafter의 window-based cross-projection bundle adjustment 방식을 구현한 시스템입니다.

### 핵심 특징
- **Window-based Processing**: Track 병합 없이 window별 독립 처리
- **Cross-Projection BA**: Window 내 모든 프레임 간 상호 projection으로 global consistency 달성
- **Depth-aware**: GeometryCrafter depth를 활용한 scale-aware 3D reconstruction
- **Two-phase Optimization**: 
  - Phase 1: Camera-only (fixed 3D from depth)
  - Phase 2: Camera + boundary 3D points (optional)

## 시스템 아키텍처

```
Input
  ├── GeometryCrafter Depth Maps
  ├── CoTracker Tracks (window-based, bidirectional)
  └── Camera Intrinsics (K.txt)
       ↓
Window Track Loading
  ├── Independent window tracks (no merging)
  ├── Bidirectional query points (start/end frames)
  └── Depth sampling at track locations
       ↓
3D Initialization
  ├── Depth-based unprojection
  ├── Camera-to-world transformation
  └── Confidence scoring
       ↓
Cross-Projection Bundle Adjustment
  ├── Phase 1: Camera-only optimization
  └── Phase 2: Camera + boundary 3D refinement
       ↓
Output
  ├── Optimized camera poses
  ├── Window-based 3D points
  └── COLMAP export
```

## 사용법

### 기본 실행
```bash
# Window BA 실행
python -m src.window_ba /path/to/scene

# Two-phase optimization 사용
python -m src.window_ba /path/to/scene --use_refine

# Custom config
python -m src.window_ba /path/to/scene --config config/window_ba.yaml
```

### 데이터 구조
```
Scene/
├── images/                    # Input images
├── K.txt                     # Camera intrinsic matrix
├── cotracker/                # Window-based tracks
│   └── *_sift_bidirectional.npy  # Bidirectional tracks (preferred)
└── depth/GeometryCrafter/    # Depth maps
    └── *.npz
```

### 출력 파일
```
window_ba_output/
├── cameras_phase1.npz        # Phase 1 결과
├── cameras_final.npz         # 최종 camera parameters
├── window_tracks_3d.npz      # Window별 3D points
├── colmap/                   # COLMAP export
│   ├── cameras.bin/txt
│   ├── images.bin/txt
│   └── points3D.bin/txt
└── visualizations/           # PNG 시각화 파일들
```

## 주요 모듈

### 1. WindowTrackLoader
- Window별 독립적 track 로딩 (병합 없음)
- Bidirectional tracks 자동 감지
- Depth sampling at track locations
- Camera intrinsics 로딩

### 2. WindowDepthInitializer  
- Depth 기반 3D unprojection
- Bilinear interpolation for sub-pixel accuracy
- Confidence scoring based on depth consistency

### 3. WindowBundleAdjuster
- Cross-projection loss 구현
- Camera parametrization: quaternion + translation + FOV
- Two-phase optimization 지원
- Single camera mode (shared FOV)

### 4. Pipeline
- 자동 체크포인트 시스템
- 진행 상황 자동 감지 및 재개
- COLMAP export
- Visualization 생성

## 기술적 특징

### Cross-Projection Loss
```python
# Window 내 모든 프레임 간 상호 projection
for i in range(T):  # Source frame
    for j in range(T):  # Target frame
        if i != j:
            # Project i's 3D to j and compute error
            loss += reprojection_error(project(3D[i], camera[j]), 2D[j])
```

### Camera Model
- Rotation: Quaternion representation (numerically stable)
- Translation: 3-vector
- FOV: Per-frame or single camera mode
- Coordinates: Normalized [-1, 1] to pixel conversion

### Optimization
- Adam optimizer with different learning rates
- Gradient clipping (max_norm=1.0)
- Huber loss for outlier robustness
- Convergence threshold: 1e-6

## 구현 상태 (2025-01-14)

### ✅ 완료된 기능
- Window-based track loading with bidirectional support
- Depth-based 3D initialization
- Cross-projection bundle adjustment
- Two-phase optimization (Phase 2 fully implemented)
- COLMAP export with optimized boundary points
- CLI-compatible visualization
- Auto checkpoint and resume system
- Single camera mode

### 🔧 최근 개선사항
- 자동 체크포인트 시스템 구현
- phase2_history NameError 버그 수정
- 코드 검증 및 디버깅 완료

## Configuration

### window_ba.yaml 주요 설정
```yaml
camera:
  single_camera: true      # 단일 카메라 FOV 공유
  image_width: 1024
  image_height: 576

track_loader:
  track_mode: "sift"       # sift, superpoint, grid
  depth_subdir: "depth/GeometryCrafter"

optimization:
  max_iterations: 10000
  learning_rate_camera: 0.001
  learning_rate_3d: 0.01
  use_robust_loss: true
```

## 참고 자료

- GeometryCrafter SFM: `submodules/GeometryCrafter/sfm/`
- Original implementation inspiration from GeometryCrafter's window-based approach
- COLMAP compatibility for standard 3D reconstruction pipelines