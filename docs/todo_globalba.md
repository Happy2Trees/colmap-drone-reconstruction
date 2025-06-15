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

### 1. WindowTrackLoader (`core/window_track_loader.py`)
- Window별 독립적 track 로딩 (병합 없음)
- Bidirectional tracks 자동 감지
- Depth sampling at track locations
- Camera intrinsics 로딩

### 2. WindowDepthInitializer (`core/window_depth_initializer.py`)
- Depth 기반 3D unprojection
- Bilinear interpolation for sub-pixel accuracy
- Confidence scoring based on depth consistency

### 3. WindowBundleAdjuster (`core/window_bundle_adjuster.py`)
- Cross-projection loss 구현
- Camera parametrization: quaternion + translation + FOV
- Two-phase optimization 지원
- Single camera mode (shared FOV)
- tqdm progress bar로 실시간 진행 상황 표시

### 4. Pipeline (`pipeline.py`)
- 자동 체크포인트 시스템
- 진행 상황 자동 감지 및 재개
- COLMAP export
- Visualization 생성

### 5. CameraModel (`models/camera_model.py`)
- Quaternion 기반 rotation 표현
- Per-frame 또는 single camera FOV 지원
- Device-aware initialization

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

## 구현 상태 (2025-01-15)

### ✅ 완료된 기능
- Window-based track loading with bidirectional support
- Depth-based 3D initialization
- Cross-projection bundle adjustment
- Two-phase optimization (Phase 2 fully implemented)
- COLMAP export with optimized boundary points
- CLI-compatible visualization
- Auto checkpoint and resume system
- Single camera mode
- **진행 상황 가시성 개선** (2025-01-15)
  - tqdm progress bar 추가로 실시간 진행률 표시
  - Phase별 소요 시간 측정 및 표시
  - Window 처리 상황 로깅

### 🔧 최근 개선사항
- 자동 체크포인트 시스템 구현
- phase2_history NameError 버그 수정
- 코드 검증 및 디버깅 완료
- **대규모 코드 리팩토링 완료** (2025-01-14)
- **전체 타입 체킹 오류 수정 완료**
- **Frame count mismatch 버그 수정** (2025-01-15)
  - end_frame을 exclusive로 일관되게 처리하도록 수정
  - 가변 window 크기 지원 (마지막 window가 더 작은 경우 처리)
- **Singular matrix 버그 수정** (2025-01-15)
  - Frame index off-by-one 오류 수정
  - Device 초기화 문제 해결
  - Robust matrix inversion (pseudo-inverse fallback)
- **NaN 문제 해결 및 Loss 계산 수정** (2025-01-15)
  - GeometryCrafter와의 loss 계산 차이점 발견 및 수정
  - Window 간 loss 평균 제거 → 합산으로 변경
  - Window 내 frame pair 평균 제거 (gradient 크기 보존)
  - Adam optimizer epsilon 1e-15 설정 (GeometryCrafter와 동일)
  - Identity pose에서도 안정적인 최적화 가능

## 코드 리팩토링 및 품질 개선 (2025-01-14)

### 주요 리팩토링 내용

#### 1. **모듈 분리 및 구조 개선**
새로운 모듈들이 생성되어 코드가 더 체계적으로 구성됨:
- **`data_models.py`**: Dictionary 대신 구조화된 데이터 클래스 사용
  - `WindowTrackData`, `CameraIntrinsics`, `CameraParameters` 등
- **`camera_model.py`**: Bundle adjuster에서 카메라 모델 분리
- **`config_manager.py`**: 중앙화된 설정 관리
- **`geometry_utils.py`**: 공통 기하학적 연산 유틸리티
- **`checkpoint_manager.py`**: 체크포인트 저장/로드 로직 분리
- **`colmap_exporter.py`**: COLMAP export 기능 독립

#### 2. **코드 품질 향상**
- **Before**: 300줄 이상의 긴 메서드, 혼재된 책임, Dictionary 기반 데이터 전달
- **After**: 50줄 이내의 짧은 메서드, 단일 책임 원칙, Type-safe dataclass 사용

#### 3. **타입 안정성 개선**
- 모든 함수와 클래스에 타입 힌트 추가
- Union 타입을 사용한 np.ndarray와 torch.Tensor 혼용 문제 해결
- 사용하지 않는 import 제거
- Type checking 오류 0개 달성

### 수정된 타입 오류들
1. **base.py**: WindowData 클래스의 타입 안정성 개선
   - np.ndarray와 torch.Tensor를 모두 수용하도록 Union 타입 사용
   - to_torch() 메서드 개선으로 None 타입 처리 강화

2. **camera_model.py**: FOV 파라미터 타입 처리 개선
   - float와 Tensor 타입 모두 적절히 처리
   - load_parameters() 메서드의 타입 안정성 향상

3. **전체 모듈 import 정리**
   - 6개 파일에서 사용하지 않는 import 제거
   - 코드 가독성 및 로딩 성능 향상

### 파일 구조
```
src/window_ba/
├── __init__.py
├── __main__.py
├── run_window_ba.py           # 엔트리 포인트
├── pipeline.py                # 메인 파이프라인
├── core/                      # 핵심 기능
│   ├── __init__.py
│   ├── window_track_loader.py      # Track 로딩
│   ├── window_depth_initializer.py # Depth 초기화  
│   └── window_bundle_adjuster.py   # Bundle adjustment
├── models/                    # 데이터 모델
│   ├── __init__.py
│   ├── base.py               # 추상 베이스 클래스
│   ├── data_models.py        # 데이터 구조
│   └── camera_model.py       # 카메라 파라미터화
└── utils/                    # 유틸리티
    ├── __init__.py
    ├── config_manager.py     # 설정 관리
    ├── checkpoint_manager.py # 체크포인트 처리
    ├── geometry_utils.py     # 기하학적 유틸리티
    ├── colmap_exporter.py    # COLMAP export
    └── visualization.py      # 시각화 유틸리티
```

### 리팩토링 이점
1. **유지보수성**: 명확한 모듈 경계, 기능 찾기 쉬움, 컴포넌트 간 결합도 감소
2. **확장성**: 새로운 optimizer 추가 용이, 데이터 포맷 지원 간편
3. **견고성**: Dataclass를 통한 타입 안정성, 설정 검증, 자동 체크포인트 복구
4. **성능**: 성능 저하 없음, 더 깔끔한 코드 경로, 효율적인 데이터 구조

### 사용법은 동일하게 유지
```bash
# 리팩토링 전후 동일한 명령어
python -m src.window_ba /path/to/scene --use_refine
```

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
  log_interval: 50         # 진행 상황 로그 간격
```

## 알려진 이슈 및 해결

### Singular Matrix 문제 (2025-01-15 해결)
**문제**: "The diagonal element 2 is zero" 오류 발생
- 원인: 
  - Frame index off-by-one 오류 (end_frame에 +1 추가)
  - Device 초기화 문제 (CPU/CUDA 불일치)
  - Identity rotation과 zero translation으로 인한 singular matrix

**해결**:
- `optimize_phase1`: max_frame 계산 시 +1 제거 (end_frame이 이미 exclusive)
- `CameraModel.__init__`: device 파라미터 추가 및 명시적 device 설정
- `_tracks_to_world_coordinates`: pseudo-inverse fallback 추가
- Frame index 범위 검증 추가

### Frame Count Mismatch 문제 (2025-01-15 해결)
**문제**: "Frame count mismatch" 오류 발생
- 원인: CoTracker는 end_frame을 exclusive로 저장하지만, WindowTrackData는 inclusive로 해석
- 예: Window(0, 50) → CoTracker는 0~49 프레임 (50개), WindowTrackData는 0~50 프레임 (51개) 기대

**해결**:
- `WindowTrackData.__post_init__`: frame count validation을 exclusive로 변경
- `WindowTrackData.num_frames`: 계산 로직을 exclusive로 변경  
- `WindowBundleAdjuster`: frame indices 생성 시 end_frame을 exclusive로 처리
- 가변 window 크기 자동 지원 (window_size를 tracks.shape[0]으로 설정)

### NaN Loss 문제 (2025-01-15 해결)
**문제**: Local optima에 빠지고 모든 값이 NaN으로 변환
- 원인:
  - GeometryCrafter와 다른 loss 계산 방식
  - Window 간 loss를 평균내어 gradient가 1/N으로 감소
  - Window 내 frame pair 수로 추가 평균 (1/2450 등으로 감소)
  - 불충분한 gradient로 인한 최적화 실패

**해결**:
- `_compute_phase1_loss`: window 간 loss 평균 제거 (합산 유지)
- `_compute_cross_projection_loss`: frame pair 평균 제거
- Adam optimizer epsilon을 1e-15로 설정 (GeometryCrafter와 동일)
- 결과: Identity pose에서도 depth 차이로 충분한 gradient 생성

## 참고 자료

- GeometryCrafter SFM: `submodules/GeometryCrafter/sfm/`
- Original implementation inspiration from GeometryCrafter's window-based approach
- COLMAP compatibility for standard 3D reconstruction pipelines
- 리팩토링 상세 문서: `docs/window_ba_refactoring_summary.md`