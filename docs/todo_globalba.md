# Global Bundle Adjustment with Depth and Tracks

## 목적 (Purpose)

이 문서는 GeometryCrafter의 depth estimation과 CoTracker의 point tracking 결과를 활용하여 Global Bundle Adjustment (BA)를 수행하는 시스템의 구현 계획을 설명합니다.

### 주요 목표
1. **Depth-aware Bundle Adjustment**: GeometryCrafter로 추출한 monocular depth를 활용한 3D 구조 복원
2. **Long-term Tracking Integration**: CoTracker의 장기간 point tracking 정보를 활용한 정확한 카메라 포즈 추정
3. **Global Optimization**: 전체 시퀀스에 대한 동시 최적화로 drift 최소화

## 현재 구현 분석 (Current Implementation Analysis)

### 1. GeometryCrafter SFM 구조

#### 주요 컴포넌트

**a) SpaTracker (3D-aware Point Tracker)**
- 위치: `submodules/GeometryCrafter/sfm/spatracker/`
- 특징:
  - Tri-plane representation을 사용한 3D-aware tracking
  - Depth 정보를 직접 활용하는 tracking 시스템
  - SuperPoint/SIFT 특징점 기반 초기화

**b) run_track.py (Track Extraction)**
```python
# 주요 기능
- SuperPoint/SIFT 특징점 추출
- SpaTracker를 통한 temporal tracking
- Window-based processing (기본 12 프레임, 6 프레임 간격)
- 각 window의 시작과 끝 프레임에서 query points 추출
- 출력: .npz 파일 (tracks, visibility, query_time)
- Track 병합 없이 window별로 독립적 저장
```

**c) run.py (Camera Pose Optimization)**
```python
# CameraModel 클래스
- 카메라 파라미터 최적화 (rotation as quaternion, translation, FOV)
- Cross-projection loss 기반 bundle adjustment
- Two-stage optimization:
  1. Camera-only optimization (cross-projection)
  2. Camera + 3D points joint optimization (--use_refine)
```

### 2. GeometryCrafter BA의 핵심 메커니즘

#### Cross-Projection Loss
```python
# 각 window에 대해:
for (st_frame, ed_frame, track, track_vis) in tracks:
    # 1. Window 내 모든 프레임의 3D points triangulation
    for i in range(st_frame, ed_frame):
        xyzw_world[i] = triangulate(track[i], camera[i])
    
    # 2. Cross-projection: i번째 프레임의 3D를 j번째 프레임으로 projection
    for j in range(st_frame, ed_frame):
        for i in range(st_frame, ed_frame):
            proj_xy[j,i] = project(xyzw_world[i], camera[j])
            loss += MSE(proj_xy[j,i], track[j])
```

#### 특징
- **Track 병합 없음**: 각 window의 tracks는 독립적으로 유지
- **Global consistency**: Cross-projection으로 달성
- **Window-aware**: 각 window 내에서만 3D-2D correspondence
- **Scalable**: Window 단위 처리로 메모리 효율적

### 3. 현재 BA 구현의 특징

#### 장점
- Depth prior를 활용한 robust한 3D 초기화
- Cross-projection으로 global drift 감소
- Window 단위 처리로 긴 시퀀스 처리 가능
- 메모리 효율적

#### 한계
- Window boundary에서 불연속성 가능
- Long-range correspondence 활용 제한
- Depth uncertainty를 명시적으로 모델링하지 않음

## GeometryCrafter 방식의 Window-based BA 시스템 (GeometryCrafter-style Window-based BA System)

### 1. 시스템 아키텍처

```
Input
  ├── GeometryCrafter Depth Maps (per frame)
  ├── CoTracker Tracks (window-based, no merging)
  └── Camera Intrinsics (K matrix)
       ↓
Window Track Loading
  ├── Load all window tracks independently
  ├── Extract query points (start/end frames)
  └── Prepare depth sampling
       ↓
Initial Camera & 3D Estimation
  ├── Initialize cameras (identity rotation, zero translation)
  ├── Window-wise 3D triangulation with depth
  └── FOV estimation from tracks
       ↓
Cross-Projection Bundle Adjustment
  ├── Window-wise cross-projection loss
  ├── Camera-only optimization (Phase 1)
  └── Optional: Camera + 3D points refinement (Phase 2)
       ↓
Output
  ├── Optimized camera poses
  ├── Per-window 3D points
  └── COLMAP-compatible format
```

### 2. 핵심 구현 모듈

#### a) Window Track Loader
```python
class WindowTrackLoader:
    """Window 단위 track 로딩 및 관리"""
    
    def load_window_tracks(self, track_dir):
        """모든 window tracks를 독립적으로 로드"""
        # Track 병합 없이 window별로 유지
        # 각 window: (start_frame, end_frame, tracks, visibility, query_time)
        
    def prepare_depth_sampling(self, tracks, depths):
        """Track location에서 depth 샘플링"""
        # Grid sampling으로 track 위치의 depth 추출
        
    def load_intrinsics(self, scene_dir):
        """Scene 폴더의 K.txt, dist.txt에서 camera intrinsics 로드"""
        # K.txt: 3x3 intrinsic matrix
        # dist.txt: distortion coefficients (k1, k2, p1, p2, k3)
        # GeometryCrafter는 FOV 기반이지만 초기값으로 K matrix 활용
```

#### b) Window-based 3D Initialization
```python
class WindowDepthInitializer:
    """Window 단위 depth 기반 3D 초기화"""
    
    def triangulate_window_tracks(self, window_tracks, depths, cameras):
        """각 window별로 3D points 초기화"""
        # Window별 독립적 triangulation
        # Depth를 사용한 scale-aware 3D points
        
    def compute_intrinsics_from_tracks(self, tracks):
        """Track에서 camera intrinsics 추정"""
        # GeometryCrafter의 point_map_xy2intrinsic 방식
```

#### c) Cross-Projection Bundle Adjustment
```python
class WindowBundleAdjuster:
    """Window-aware cross-projection 최적화"""
    
    def compute_cross_projection_loss(self, window_tracks):
        """Cross-projection loss 계산"""
        for (st_frame, ed_frame, track, vis) in window_tracks:
            # 1. Window 내 모든 프레임 3D triangulation
            xyzw_world = self.triangulate_all_frames(track, cameras)
            
            # 2. Cross-projection matrix (NxN)
            for i in range(st_frame, ed_frame):
                for j in range(st_frame, ed_frame):
                    # i프레임 3D를 j프레임으로 projection
                    proj_loss[i,j] = project_and_compare()
        
    def optimize_phase1(self):
        """Phase 1: Camera-only optimization"""
        # Camera poses (quaternion + translation)
        # Camera FOVs
        # Fixed 3D points from depth
        
    def optimize_phase2(self):
        """Phase 2: Camera + 3D points refinement"""
        # Window boundary에서만 3D 초기화
        # 3D points도 optimization variables로 추가
```

### 3. 구현 계획 (Implementation Plan)

#### Phase 1: Window Track Loading (수정 필요)
- [ ] GlobalTrackManager → WindowTrackLoader 리팩토링
- [ ] Track 병합 코드 제거
- [ ] Window별 독립적 track 관리 구현

#### Phase 2: Depth-based Initialization (수정 필요)
- [ ] Window 단위 3D triangulation
- [ ] Depth sampling at track locations
- [ ] Camera intrinsics estimation from tracks

#### Phase 3: Cross-Projection BA (새로 구현)
- [ ] Cross-projection loss 구현
- [ ] Window-aware optimization
- [ ] Phase 1: Camera-only optimization
- [ ] Phase 2: Camera + 3D refinement (optional)

#### Phase 4: Integration & Testing
- [ ] Pipeline 전체 플로우 수정
- [ ] COLMAP export 구현
- [ ] GeometryCrafter 결과와 비교

## 기술적 고려사항 (Technical Considerations)

### 1. 메모리 효율성
- Sparse matrix representation for Jacobian
- Batch processing for large sequences
- GPU acceleration for optimization

### 2. Numerical Stability
- Quaternion representation for rotations
- Normalized coordinates
- Adaptive step size control

### 3. Robustness
- RANSAC-based initialization
- M-estimators for outlier handling
- Multi-scale optimization

## 현재 진행 상황 (Current Progress)

### 완료된 작업
- [x] GeometryCrafter SFM 코드 분석 (2025-01-06)
- [x] 현재 BA 구현 이해
- [x] 시스템 아키텍처 설계
- [x] Precompute 데이터 구조 분석 완료 (2025-01-06)
  - CoTracker: 77개 window, 각 50프레임, 407개 tracking points (SIFT)
  - GeometryCrafter: 768개 프레임의 depth maps (576x1024)
  - Camera parameters: K matrix와 distortion coefficients
- [x] GeometryCrafter 방식으로 전체 리팩토링 완료 (2025-01-13)
- [x] WindowTrackLoader 구현 (track 병합 제거)
- [x] WindowDepthInitializer 구현 (window 단위 처리)
- [x] WindowBundleAdjuster 구현 (cross-projection)
- [x] Pipeline 전체 재구성
- [x] Cross-projection loss 구현
- [x] Two-phase optimization 구현 (Phase 2는 부분 구현)
- [x] K.txt, dist.txt에서 intrinsics 로딩 구현

### 남은 작업
- [ ] Phase 2 optimization 완전 구현
- [x] COLMAP export 구현 (2025-01-13)
  - cameras.bin/txt: PINHOLE 모델로 intrinsics 저장
  - images.bin/txt: 최적화된 camera poses와 2D-3D correspondences
  - points3D.bin/txt: Window별 3D points with visibility
- [ ] 실제 데이터로 테스트 및 디버깅
- [ ] 메모리 최적화 (대용량 시퀀스 처리)

## 데이터 분석 결과 (Data Analysis Results) - 2025-01-06

### 분석한 데이터셋: `/hdd2/0321_block_drone_video/colmap/data/3x_section2_fps12_processed_1024x576`

#### 1. 데이터 구조
- **이미지**: 768 프레임 (60fps → 12fps 샘플링)
- **해상도**: 1024×576 (다운샘플링됨)
- **이미지 인터벌**: 5 프레임 간격 (00000.jpg ~ 03395.jpg)

#### 2. CoTracker 데이터
- **파일**: `cotracker/50_10_sift.npy`
- **구조**:
  - 77개의 sliding windows
  - Window size: 50 프레임
  - Interval: 10 프레임 (40프레임 오버랩)
  - 각 window당 407개의 tracking points (SIFT 기반)
  - Track shape: (50, 407, 2) - (frames, points, xy)
  - Visibility mask: (50, 407) - boolean

#### 3. GeometryCrafter Depth
- **파일**: `depth/GeometryCrafter/*.npz` (768개)
- **각 파일 구조**:
  - depth: (576, 1024) float32, 범위 ~0.32 to ~6.68
  - mask: (576, 1024) boolean (모두 True)
  - metadata: model info, window_size=110, overlap=25

#### 4. 카메라 파라미터
- **K.txt**: 
  ```
  fx=2576.10, fy=2590.32
  cx=361.41, cy=435.45
  ```
- **dist.txt**: k1, k2, p1, p2, k3 distortion coefficients

## 구현 완료 내용 (Implementation Completed) - 2025-01-13

### 1. 모듈 구조 (GeometryCrafter 방식으로 재구현)
```
src/window_ba/
├── __init__.py              # 패키지 초기화
├── window_track_loader.py   # Window 단위 track 로딩 (병합 없음)
├── window_depth_initializer.py  # Window 단위 depth 기반 3D 초기화
├── window_bundle_adjuster.py    # Cross-projection BA
├── pipeline.py              # 전체 파이프라인 (COLMAP export 포함)
├── run_window_ba.py         # Main 실행 파일
└── __main__.py              # 모듈 엔트리포인트
```

### 2. 주요 기능 (GeometryCrafter 방식)
- **WindowTrackLoader**:
  - Window별 독립적 track 로딩 (병합 없음)
  - Scene 디렉토리의 K.txt, dist.txt에서 intrinsics 로드
  - Track 위치에서 depth sampling (grid sampling)
  - FOV 계산 from intrinsic matrix
  
- **WindowDepthInitializer**:
  - Window 단위 3D triangulation
  - Depth 기반 unprojection
  - Query points 추출 (window boundaries)
  - Depth consistency 기반 confidence 계산
  
- **WindowBundleAdjuster**:
  - Cross-projection loss 구현
  - Camera model: quaternion + translation + FOV
  - Phase 1: Camera-only optimization
  - Phase 2: Camera + 3D refinement (optional)
  - PyTorch 기반 differentiable optimization
  
- **Pipeline**:
  - 전체 프로세스 통합
  - YAML 기반 configuration
  - 결과 저장 및 COLMAP export (구현 예정)

### 3. 사용법

#### 기본 실행 (GeometryCrafter 방식)
```bash
# 모듈로 실행
python -m src.window_ba /hdd2/0321_block_drone_video/colmap/data/3x_section2_fps12_processed_1024x576

# config 파일 지정
python -m src.window_ba /hdd2/0321_block_drone_video/colmap/data/3x_section2_fps12_processed_1024x576 --config config/window_ba.yaml

# Two-phase optimization (camera + 3D refinement)
python -m src.window_ba /hdd2/0321_block_drone_video/colmap/data/3x_section2_fps12_processed_1024x576 --use_refine

# 직접 스크립트 실행
python src/window_ba/run_window_ba.py /path/to/scene --output_dir outputs/window_ba --verbose
```

#### Configuration 옵션
- `config/window_ba.yaml` 파일로 세부 설정 조정 가능
- Window 처리, depth sampling, optimization 파라미터 설정
- Device (cuda/cpu), learning rates, convergence 등 조정 가능

### 4. 출력 파일
```
window_ba_output/
├── window_ba.log            # 실행 로그
├── cameras_phase1.npz       # Phase 1 카메라 파라미터
├── phase1_history.json      # Phase 1 최적화 히스토리
├── cameras_final.npz        # 최종 카메라 파라미터
├── window_tracks_3d.npz     # Window별 3D tracks
├── pipeline_summary.json    # 전체 결과 요약
├── summary.txt              # 읽기 쉬운 요약
└── colmap/                  # COLMAP export (구현 예정)
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

## GeometryCrafter 방식 알고리즘 상세 (GeometryCrafter-style Algorithm Details) - 2025-01-06

### 1. **Window Track Loading (WindowTrackLoader)**

#### **1.1 Track 로딩 방식**
GeometryCrafter는 track 병합 없이 window별로 독립적으로 처리합니다:
```
Window 0: frames [0, 49]   → 독립적 처리
Window 1: frames [10, 59]  → 독립적 처리 (overlap 있지만 병합하지 않음)
Window 2: frames [20, 69]  → 독립적 처리
```

**핵심 차이점:**

1. **No Track Merging**
   ```python
   # GeometryCrafter 방식
   window_tracks = []
   for track_file in track_files:
       window_tracks.append(load_track(track_file))
   # 병합하지 않고 그대로 사용
   ```

2. **Query Points**
   - 각 window의 시작과 끝 프레임에서만 feature points 추출
   - `query_time` 필드로 구분 (0: 시작, window_size-1: 끝)

#### **1.2 Depth Sampling**
```python
# Track 위치에서 depth 값 추출
for track in window_tracks:
    grid = normalize_coordinates(track[:, :, :2])
    depth_values = F.grid_sample(depth_maps, grid)
    track_with_depth = concat(track, depth_values)
```

### 2. **Depth-based 3D Initialization (DepthInitializer)**

#### **2.1 Sub-pixel Accuracy를 위한 Bilinear Interpolation**
Track 위치가 정수 픽셀이 아닐 때 정확한 depth 값을 얻기 위한 보간:

```python
# 픽셀 (x, y)에서의 depth 계산
depth(x, y) = (1-dx)(1-dy)*d[y0,x0] + dx(1-dy)*d[y0,x1] + 
              (1-dx)dy*d[y1,x0] + dx*dy*d[y1,x1]

여기서:
- (x0, y0) = floor(x, y)
- (x1, y1) = (x0+1, y0+1)  
- (dx, dy) = (x-x0, y-y0)  # fractional parts
```

#### **2.2 3D Point Triangulation**

**Single-view 방식 (현재 구현):**

1. **Median depth 선택**
   - 모든 관찰 중 median depth를 대표값으로 사용
   - Outlier에 robust한 선택

2. **Camera unprojection**
   ```
   X_3D = (x - cx) * depth / fx
   Y_3D = (y - cy) * depth / fy
   Z_3D = depth
   ```
   - (cx, cy): principal point
   - (fx, fy): focal lengths

3. **Confidence score 계산**
   ```python
   confidence = exp(-std(depths) / mean(depths))
   ```
   - Depth 변동이 작을수록 높은 confidence
   - [0, 1] 범위로 정규화

#### **2.3 Multi-view Depth Consistency 평가**
인접 프레임 간 depth 일관성을 통한 신뢰도 평가:

```python
# 상대적 depth 차이 계산
relative_diff = |depth_frame1 - depth_frame2| / max(depth_frame1, depth_frame2)

# Consistency score
consistency = 1.0 - median(relative_diffs)
```

- 동일 3D point의 다른 뷰에서의 depth 비교
- Median 사용으로 outlier에 robust
- Frame 간격이 10 이내인 경우만 비교

### 3. **Cross-Projection Bundle Adjustment (WindowBundleAdjuster)**

#### **3.1 Cross-Projection 메커니즘**

GeometryCrafter의 핵심은 window 내 모든 프레임 간 cross-projection입니다:

```python
def compute_cross_projection_loss(st_frame, ed_frame, track, track_vis):
    # 1. Window 내 모든 프레임에서 3D points triangulation
    for i in range(st_frame, ed_frame):
        z = track[i, :, 2]  # depth
        x = track[i, :, 0] * tanFovX * z  # 3D x
        y = track[i, :, 1] * tanFovY * z  # 3D y
        xyzw_world[i] = to_world_coords(x, y, z, camera[i])
    
    # 2. Cross-projection matrix (T × T)
    for j in range(st_frame, ed_frame):  # target frame
        for i in range(st_frame, ed_frame):  # source frame
            # i번째 프레임의 3D를 j번째 프레임으로 projection
            proj_xy[j,i] = project_to_camera(xyzw_world[i], camera[j])
            
    # 3. Loss computation
    loss = MSE(proj_xy, track_xy_gt)
```

#### **3.2 Camera Parametrization**

**Quaternion + Translation + FOV:**
```python
class CameraModel:
    def __init__(self, K_matrix=None):
        self.rotations = nn.Parameter(quaternions)  # Nx4
        self.translations = nn.Parameter(zeros)     # Nx3
        
        # FOV 초기화: K matrix가 있으면 변환, 없으면 기본값
        if K_matrix is not None:
            # K matrix에서 FOV 계산
            fx, fy = K_matrix[0,0], K_matrix[1,1]
            cx, cy = K_matrix[0,2], K_matrix[1,2]
            width, height = 2*cx, 2*cy
            FovX = 2 * np.arctan(width / (2 * fx))
            FovY = 2 * np.arctan(height / (2 * fy))
        else:
            FovX = FovY = np.deg2rad(60)  # 기본값
            
        self.FovXs = nn.Parameter(torch.full((N,), FovX))
        self.FovYs = nn.Parameter(torch.full((N,), FovY))
```

#### **3.3 Two-Phase Optimization**

**Phase 1: Camera-only**
```python
# Fixed 3D points from depth
# Optimize: camera poses + FOVs
optimizer = Adam([
    {'params': rotations, 'lr': 1e-3},
    {'params': translations, 'lr': 1e-2},
    {'params': [FovXs, FovYs], 'lr': 1e-4}
])
```

**Phase 2: Camera + 3D (--use_refine)**
```python
# Initialize 3D points at window boundaries
tracks_3d = []
for window in windows:
    # Start frame의 query points를 3D로 초기화
    # End frame의 query points를 3D로 초기화
    tracks_3d.append(nn.Parameter(init_3d_points))

# Joint optimization
optimizer = Adam([
    {'params': camera_params, 'lr': ...},
    {'params': tracks_3d, 'lr': 1e-2}
])
```

### 4. **전체 최적화 프로세스**

```
1. Initialization:
   - Cameras: quaternion [1,0,0,0] (identity), translation [0,0,0]
   - 3D points: depth-based initialization

2. Optimization Loop:
   for iteration in range(max_iterations):
       a) Forward pass:
          - Project all 3D points to observing cameras
          - Compute reprojection errors
          
       b) Loss computation:
          - Apply robust loss to residuals
          - (Optional) Add depth consistency term
          
       c) Backward pass:
          - Compute gradients via autograd
          - Clip gradients
          
       d) Parameter update:
          - Adam optimizer step
          - Learning rate scheduling
          
       e) Convergence check:
          - |loss_prev - loss_curr| < 1e-6

3. Post-processing:
   - Extract optimized parameters
   - Convert quaternions back to rotation matrices
```

### 5. **메모리 및 계산 효율성**

#### **5.1 Sparse Operations**
- Visibility matrix: 각 point가 보이는 camera만 저장
- Sparse Jacobian 구조 활용 (미래 구현)

#### **5.2 GPU 가속**
- PyTorch의 CUDA tensors 사용
- Batch projection operations
- Parallel gradient computation

#### **5.3 메모리 관리**
- Depth map caching으로 중복 로딩 방지
- On-demand loading for large sequences
- Track matrix의 효율적인 표현

### 6. **현재 한계 및 향후 개선 방향**

#### **6.1 미구현 기능**

1. **Depth Consistency Loss:**
   ```python
   # 예상 구현
   depth_projected = compute_depth_from_3d_point(X_3d, camera)
   depth_loss = robust_loss(depth_observed - depth_projected)
   total_loss = reproj_loss + λ_depth * depth_loss
   ```

2. **Multi-view Triangulation:**
   - DLT (Direct Linear Transform) 활용
   - Depth prior를 가중치로 사용하는 weighted least squares

3. **COLMAP Export:**
   - Binary format 변환
   - cameras.bin, images.bin, points3D.bin 생성

#### **6.2 알고리즘 개선 방향**

1. **Hierarchical Optimization:**
   - Coarse-to-fine 접근
   - Key frames 먼저 최적화

2. **Loop Closure Detection:**
   - Global consistency 향상
   - Drift 누적 방지

3. **Dynamic Depth Weight:**
   - Iteration에 따라 depth weight 조정
   - 초기: depth 중시, 후기: reprojection 중시

이 구현은 전통적인 SfM의 검증된 최적화 기법과 최신 딥러닝 기반 depth/tracking의 강점을 효과적으로 결합한 하이브리드 시스템입니다.

## COLMAP Export 구현 (COLMAP Export Implementation) - 2025-01-13

### Export 기능 개요
Window BA 결과를 COLMAP 포맷으로 변환하여 저장하는 기능 구현:

#### 1. **Cameras Export**
```python
def _create_colmap_cameras(self, intrinsics, width, height):
    # PINHOLE model 사용 (fx, fy, cx, cy)
    # K.txt의 intrinsic matrix를 COLMAP 파라미터로 변환
    cameras = {
        1: Camera(model='PINHOLE', params=[fx, fy, cx, cy])
    }
```

#### 2. **Images Export**
```python
def _create_colmap_images(self, camera_model, window_tracks):
    # 최적화된 camera poses (quaternion + translation)
    # Window tracks에서 2D-3D correspondences 추출
    # Frame별로 visible points 수집
    images[frame_id] = Image(
        qvec=normalized_quaternion,
        tvec=translation,
        xys=2d_points,
        point3D_ids=corresponding_3d_ids
    )
```

#### 3. **Points3D Export**
```python
def _create_colmap_points3D(self, window_tracks):
    # Window별 3D points 수집
    # Unique ID: window_idx * 10000 + point_idx
    # Median 3D position across observations
    # Track visibility across frames
    points3D[id] = Point3D(
        xyz=median_position,
        image_ids=observing_frames,
        error=std_deviation
    )
```

### Export 파일 구조
```
colmap/
├── cameras.bin/txt     # Camera intrinsics
├── images.bin/txt      # Camera poses + 2D points
└── points3D.bin/txt    # 3D points + visibility
```

### 사용법
```python
# Pipeline에서 자동 export (config에서 설정)
output:
  save_colmap: true
  colmap_format: binary  # or 'text'
```

## 요약: GeometryCrafter 방식으로의 전환 (Summary: Transition to GeometryCrafter Style)

### 주요 변경사항

1. **Track 처리 방식 변경**
   - Before: Track 병합 → Global tracks
   - After: Window별 독립적 tracks 유지

2. **최적화 방식 변경**
   - Before: Global Bundle Adjustment
   - After: Window-based Cross-projection BA

3. **구현 모듈 변경**
   - `GlobalTrackManager` → `WindowTrackLoader`
   - `DepthInitializer` → `WindowDepthInitializer`
   - `GlobalBundleAdjuster` → `WindowBundleAdjuster`

4. **COLMAP Export 추가**
   - `colmap_utils/read_write_model.py` 활용
   - Binary/Text 포맷 지원
   - Window tracks → COLMAP sparse reconstruction

### GeometryCrafter 방식의 장점
- Window 단위 처리로 메모리 효율적
- Cross-projection으로 global consistency 유지
- Two-phase optimization으로 정확도 향상
- 긴 시퀀스에 대한 확장성

### 구현 로드맵
1. 현재 구현 백업
2. 모듈별 리팩토링
3. Cross-projection loss 구현
4. Two-phase optimization 구현
5. 테스트 및 검증

이 문서는 GeometryCrafter의 검증된 방식을 따라 더 robust하고 확장 가능한 BA 시스템을 구축하는 가이드를 제공합니다.

## 참고 자료 (References)

### 코드베이스
- GeometryCrafter SFM: `submodules/GeometryCrafter/sfm/`
- CoTracker integration: `src/feature_extractors/cotracker_extractor.py`
- COLMAP utilities: `src/colmap/`

### 관련 논문
- Bundle Adjustment in the Large (Agarwal et al.)
- Structure from Motion Revisited (Schönberger et al.)
- Depth-aware Multi-view Stereo (relevant papers)

## 다음 단계 (Next Steps)

1. **즉시 시작 가능한 작업**:
   - Track association 모듈 구현 시작
   - 기존 run.py의 CameraModel 클래스 확장

2. **필요한 리소스**:
   - GPU 메모리 요구사항 분석
   - 테스트 데이터셋 준비

3. **협업 필요 사항**:
   - COLMAP 인터페이스 정의
   - Evaluation metric 합의