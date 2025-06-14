# Window-based Bundle Adjustment 코드 상세 분석

## 개요

이 문서는 GeometryCrafter 방식의 window-based cross-projection bundle adjustment 구현을 상세히 분석합니다. 이 구현은 전통적인 Structure-from-Motion의 feature matching과 bundle adjustment를 window 기반의 deep learning 접근법(CoTracker + GeometryCrafter)과 결합한 하이브리드 방식입니다.

## 1. 전체 파이프라인 구조

### 1.1 입력 데이터 구조
```
Scene Directory/
├── images/                      # 입력 이미지들 (001.jpg, 002.jpg, ...)
├── K.txt                       # Camera intrinsic matrix (3x3)
├── dist.txt                    # Distortion coefficients (k1, k2, p1, p2, k3) [Optional]
├── cotracker/                  # Window별 tracking 결과
│   ├── 50_10_sift.npy         # Unidirectional tracks (legacy)
│   └── 50_10_sift_bidirectional.npy  # Bidirectional tracks (preferred)
└── depth/GeometryCrafter/      # Depth maps
    └── 00000.npz              # 각 프레임의 depth (keys: 'depth', 'confidence' [optional])
```

**데이터 포맷 상세:**
- **tracks.npy**: Dictionary with keys:
  - `tracks`: List of window dictionaries
  - `metadata`: {'window_size': 50, 'interval': 10, ...}
- **depth.npz**: {'depth': (H, W) array in meters}
- **K.txt**: 3x3 matrix, row-major order
- **dist.txt**: 5 values [k1, k2, p1, p2, k3] (OpenCV convention)

### 1.2 처리 흐름
```
Window Track Loading (WindowTrackLoader)
    ↓ Track 병합 없이 window별 독립 유지
    ↓ Depth sampling at track locations
3D Initialization (WindowDepthInitializer)  
    ↓ Depth 기반 unprojection
Cross-Projection BA (WindowBundleAdjuster)
    ↓ Phase 1: Camera-only optimization
    ↓ Phase 2: Camera + 3D refinement (optional)
COLMAP Export
    ↓ Binary/Text format
    ↓ cameras.bin, images.bin, points3D.bin
```

## 2. WindowTrackLoader 상세 분석

### 2.1 핵심 특징: Track 병합 없음

GeometryCrafter의 핵심 철학은 window별 track을 병합하지 않고 독립적으로 유지하는 것입니다:

```python
# 전통적 방식 (병합)
Window 0: frames [0, 49]   ─┐
Window 1: frames [10, 59]  ─┼─→ Global merged tracks
Window 2: frames [20, 69]  ─┘

# GeometryCrafter 방식 (독립)
Window 0: frames [0, 49]   → 독립적 처리
Window 1: frames [10, 59]  → 독립적 처리 (overlap 있지만 병합 안함)
Window 2: frames [20, 69]  → 독립적 처리
```

### 2.2 Track 데이터 구조

```python
window_tracks = [
    {
        'window_idx': 0,
        'start_frame': 0,
        'end_frame': 49,
        'tracks': np.array,        # (50, 407, 2) - 50프레임, 407개 points, xy좌표
        'visibility': np.array,    # (50, 407) boolean - 각 point의 visibility
        'query_time': np.array,    # (407,) - window 내 상대 시간 (0=첫프레임, T-1=마지막프레임)
        'window_size': 50,         # 실제 window 크기
        'interval': 10,            # Window 간격
        'tracks_3d': np.array,     # (50, 407, 3) - depth sampling 후 [x, y, depth]
        'depth_sampled': True,     # Depth sampling 완료 여부
    },
    ...
]
```

**query_time 상세 설명:**
- `query_time`은 각 포인트가 **해당 window 내에서** 언제 추출되었는지를 나타냄
- 값 0: 해당 window의 첫 프레임에서 추출 (예: Window 1이면 frame 10)
- 값 T-1: 해당 window의 마지막 프레임에서 추출

**현재 구현 vs GeometryCrafter:**
- GeometryCrafter: 일부 points는 query_time=0 (forward tracking), 일부는 query_time=T-1 (backward tracking)
- 현재 구현: 모든 points가 query_time=0 (forward tracking만)

### 2.3 Depth Sampling 메커니즘

Track 위치에서 정확한 depth 값을 얻기 위한 bilinear interpolation:

```python
def prepare_depth_sampling(tracks, depths):
    # 1. Pixel 좌표를 [-1, 1]로 normalize
    x_norm = 2.0 * x / (W - 1) - 1.0
    y_norm = 2.0 * y / (H - 1) - 1.0
    
    # 2. PyTorch grid_sample로 sub-pixel accuracy 달성
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).unsqueeze(0)
    depth_values = F.grid_sample(depth_map, grid, mode='bilinear', align_corners=True)
    
    # 3. tracks_3d 생성: [x, y, depth]
    tracks_3d = torch.cat([tracks, depth_values], dim=-1)  # (T, N, 3)
```

**Frame Interval Handling:**
```python
# Depth 파일명과 frame index 매핑 처리
for local_idx in range(end_frame - start_frame):
    frame_idx = start_frame + local_idx
    actual_frame_num = frame_idx * frame_interval  # e.g., frame_interval=5
    depth_file = f"{actual_frame_num:05d}.npz"    # 00000.npz, 00005.npz, ...
```

## 3. WindowDepthInitializer 상세 분석

### 3.1 Depth 기반 3D 초기화

Single-view unprojection 방식:

```python
def unproject_to_3d(points_2d, depths):
    # Camera intrinsics
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    
    # 2D pixel + depth → 3D camera space
    X = (x_2d - cx) * depth / fx
    Y = (y_2d - cy) * depth / fy  
    Z = depth
    
    return torch.stack([X, Y, Z], dim=-1)
```

### 3.2 Window별 3D Points 구조

```python
# 각 window, 각 프레임별로 독립적 3D points
xyzw_world[t, n, :] = [X, Y, Z, 1]  # (T, N, 4) homogeneous coords

# Query points (GeometryCrafter의 boundary optimization을 위한 points)
query_mask_start = (query_time == 0)                      # Window 시작에서 추출된 points
query_mask_end = (query_time == window_size - 1)          # Window 끝에서 추출된 points

query_3d_start = xyzw_world[0, query_mask_start, :3]      # Start frame의 3D positions
query_3d_end = xyzw_world[-1, query_mask_end, :3]         # End frame의 3D positions
```

**Boundary Optimization의 핵심 아이디어:**
- GeometryCrafter는 모든 3D points가 아닌 window boundaries의 points만 최적화
- 이는 메모리 효율적이며 window 간 consistency를 보장
- 현재는 모든 points가 각 window의 첫 프레임에서 추출되므로 `query_3d_end`는 비어있음

### 3.3 Depth Consistency in GeometryCrafter

GeometryCrafter는 explicit confidence score 대신 optimization 과정에서 depth consistency를 직접 활용:

- **Visibility mask**: 각 프레임에서 포인트의 가시성 여부를 binary로 추적
- **Depth consistency loss**: Cross-projection 시 depth 비율이 1에 가까워지도록 최적화
- **Direct optimization**: 별도의 confidence 계산 없이 loss function에서 직접 처리

## 4. WindowBundleAdjuster: Cross-Projection 메커니즘

### 4.1 Camera Model Parametrization

```python
class CameraModel(nn.Module):
    def __init__(self, num_frames, init_fov_x, init_fov_y):
        # Quaternion rotation (더 안정적)
        quaternions = torch.zeros(num_frames, 4)
        quaternions[:, 0] = 1.0  # w=1, x=y=z=0 (identity)
        self.quaternions = nn.Parameter(quaternions)     # (N, 4) [w, x, y, z]
        
        # Translation
        self.translations = nn.Parameter(torch.zeros(num_frames, 3))  # (N, 3)
        
        # Field of View (GeometryCrafter style) - per-frame FOV
        self.tan_fov_x = nn.Parameter(torch.full((num_frames,), init_fov_x))  # (N,)
        self.tan_fov_y = nn.Parameter(torch.full((num_frames,), init_fov_y))  # (N,)
```

**Quaternion Normalization:**
```python
def normalize_quaternions(self):
    """Unit quaternion constraint 강제"""
    with torch.no_grad():
        self.quaternions.data = F.normalize(self.quaternions.data, p=2, dim=1)
```

### 4.2 Cross-Projection Loss 상세

GeometryCrafter의 핵심 아이디어:

```
Window 내 모든 프레임 쌍에 대해:
- Frame i의 3D points를 Frame j로 projection
- Ground truth 2D positions와 비교
- T×T cross-projection matrix 생성
```

```python
def compute_cross_projection_loss(window_tracks, camera_model):
    for window in windows:
        # 1. 각 프레임에서 3D points 계산
        for i in range(T):  # window 내 각 프레임
            # Normalized coords → 3D camera space (FOV 사용)
            x_cam = x_norm * tan_fov_x[i] * depth
            y_cam = y_norm * tan_fov_y[i] * depth
            
            # Camera space → World space
            xyzw_world[i] = transform_to_world([x_cam, y_cam, depth], camera[i])
        
        # 2. Cross-projection matrix (T × T)
        for j in range(T):  # target frame
            for i in range(T):  # source frame
                if i == j: continue
                
                # i프레임의 3D points를 j프레임으로 projection
                xyzw_j = xyzw_world[i] @ camera[j].projection_matrix
                
                # Perspective division
                x_proj = xyzw_j.x / xyzw_j.z
                y_proj = xyzw_j.y / xyzw_j.z
                
                # Pixel coordinates conversion
                x_pix = (x_proj / tan_fov_x[j] + 1.0) * width/2
                y_pix = (y_proj / tan_fov_y[j] + 1.0) * height/2
                
                # Reprojection error (XY)
                error_xy = [x_pix, y_pix] - ground_truth_2d[j]
                loss_xy = robust_loss(error_xy)
                
                # Depth consistency loss (optional)
                if depth_loss_weight > 0:
                    # GeometryCrafter의 predicted depth at frame j
                    predicted_depth_j = tracks_3d[j].depth
                    
                    # Depth ratio should be close to 1
                    depth_ratio = xyzw_j.z / predicted_depth_j
                    loss_depth = MSE(depth_ratio, 1.0)
                    
                    loss += proj_loss_weight * loss_xy + depth_loss_weight * loss_depth
                else:
                    loss += proj_loss_weight * loss_xy
```

### 4.3 Robust Loss Function

Outlier 처리를 위한 Huber loss:

```python
def robust_loss(error, sigma=1.0):
    error_norm = torch.norm(error, dim=-1)
    
    # Huber loss: quadratic for small errors, linear for large
    loss = torch.where(
        error_norm < sigma,
        0.5 * error_norm**2,              # Small errors: quadratic
        sigma * error_norm - 0.5 * sigma**2  # Large errors: linear
    )
    return loss.mean()
```

## 5. 최적화 프로세스

### 5.1 Phase 1: Camera-only Optimization (완전 구현됨)

```python
# 고정: Depth에서 초기화한 3D points (모든 프레임의 모든 points)
# 최적화: Camera poses + FOV

optimizer = Adam([
    {'params': quaternions, 'lr': 1e-3},      # Rotation
    {'params': translations, 'lr': 1e-2},     # Translation (higher LR)
    {'params': [tan_fov_x, tan_fov_y], 'lr': 1e-4}  # FOV (lower LR)
])

for iteration in range(max_iterations):
    # 1. Normalize quaternions (unit constraint)
    camera_model.normalize_quaternions()
    
    # 2. Compute loss
    loss = compute_cross_projection_loss(windows, camera_model)
    
    # 3. Backprop with gradient clipping
    loss.backward()
    clip_grad_norm_(parameters, max_norm=1.0)
    
    # 4. Update
    optimizer.step()
    
    # 5. Convergence check
    if |loss_prev - loss| < 1e-6: break
```

### 5.2 Phase 2: Joint Camera + 3D Refinement (구현됨, 부분적으로 활성화됨)

**현재 상태: 기본 프레임워크 구현 완료, 양방향 tracking 필요**

GeometryCrafter의 원래 디자인:
```python
# Query points만 최적화 (window boundaries)
tracks_3d_params = []
for window in windows:
    # Start/end frame의 query points를 parameters로
    tracks_3d_params.append(nn.Parameter(query_3d_start))
    tracks_3d_params.append(nn.Parameter(query_3d_end))

optimizer = Adam([
    {'params': camera_params, 'lr': 1e-4},    # Lower LR for cameras
    {'params': tracks_3d_params, 'lr': 1e-2}  # 3D points
])
```

**Phase 2의 목적:**
- Window boundaries의 3D points를 최적화하여 window 간 consistency 향상
- Sparse 3D reconstruction: 모든 points가 아닌 경계 points만 조정
- 메모리 효율적이며 대규모 시퀀스에 적합

**구현 시 고려사항:**
- 현재 CoTracker는 각 window의 첫 프레임에서만 points 추출
- 완전한 구현을 위해서는 window 끝 프레임에서도 points 추출 필요
- `--use_refine` 플래그는 있지만 실제 구현은 Phase 1 결과를 반환

## 6. COLMAP Export

### 6.1 데이터 변환 전략

```python
# 1. Cameras: PINHOLE model
cameras = {
    1: Camera(
        model='PINHOLE',
        params=[fx, fy, cx, cy],  # From K matrix
        width=1024, height=576
    )
}

# 2. Images: Optimized poses
images[frame_id] = Image(
    qvec=normalized_quaternion,   # [w, x, y, z]
    tvec=translation,            # [tx, ty, tz]
    camera_id=1,                 # Single camera
    xys=2d_points,              # Observed 2D points
    point3D_ids=corresponding_3d # 3D point IDs
)

# 3. Points3D: Window-based numbering
# Point ID = window_idx * 10000 + point_idx
points3D[id] = Point3D(
    xyz=median_3d_position,      # Median across observations
    rgb=[128, 128, 128],        # Default gray
    error=std_deviation,         # Uncertainty
    image_ids=observing_frames,  # Visibility
    point2D_idxs=indices        # Correspondence
)
```

### 6.2 Window Points 병합 전략

각 window의 points는 독립적으로 유지되며, unique ID로 구분:

```
Window 0: Point IDs [0, 9999]
Window 1: Point IDs [10000, 19999]  
Window 2: Point IDs [20000, 29999]
...
```

**양방향 tracking 구현 시 변화:**
- 각 window에서 더 많은 3D points (시작 + 끝 프레임 features)
- Window overlap region에서 더 조밀한 3D reconstruction
- Phase 2 최적화로 window 간 연결부 개선

## 7. 핵심 특징과 장점

### 7.1 메모리 효율성
- Window 단위 처리로 대규모 시퀀스 가능
- Track 병합 없어 메모리 사용량 예측 가능
- Sparse matrix operations 활용 가능

### 7.2 Robustness
- Depth prior로 안정적인 3D 초기화
- Cross-projection으로 global consistency
- Huber loss로 outlier 처리

### 7.3 Scalability
- Window 크기/overlap 조정 가능
- 병렬 처리 가능한 구조
- Incremental 처리 가능

## 8. 실행 방법

### 8.1 기본 실행
```bash
# Phase 1만 실행 (camera-only)
python -m src.window_ba /path/to/scene

# Output structure
window_ba_output/
├── window_ba.log            # 실행 로그
├── cameras_phase1.npz       # Phase 1 결과
├── cameras_final.npz        # 최종 카메라
├── window_tracks_3d.npz     # 3D tracks
├── pipeline_summary.json    # 결과 요약
└── colmap/                  # COLMAP export
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

### 8.2 고급 옵션
```bash
# Two-phase optimization
python -m src.window_ba /path/to/scene --use_refine

# Custom configuration
python -m src.window_ba /path/to/scene --config config/window_ba.yaml

# Verbose output
python -m src.window_ba /path/to/scene --verbose
```

## 9. Configuration 옵션

```yaml
# config/window_ba.yaml
device: cuda

track_loader:
  track_mode: sift  # Options: sift, superpoint, grid
  depth_subdir: depth/GeometryCrafter

optimization:
  max_iterations: 10000
  learning_rate_camera: 1e-3
  learning_rate_translation: 1e-2
  learning_rate_fov: 1e-4
  learning_rate_3d: 1e-2  # For Phase 2 (미구현)
  convergence_threshold: 1e-6
  use_robust_loss: true
  robust_loss_sigma: 1.0
  proj_loss_weight: 1.0  # XY reprojection loss weight
  depth_loss_weight: 0.0  # Depth consistency loss (disabled by default, following GeometryCrafter)

camera:
  image_width: 1024
  image_height: 576
  default_fov_degrees: 60

output:
  save_intermediate: true
  save_colmap: true
  colmap_format: binary  # or 'text'
```

## 10. 알고리즘의 이론적 배경 및 수학적 기초

### 10.1 Cross-Projection의 장점
1. **Temporal Consistency**: 모든 프레임 간 일관성 확보
2. **No Track Merging**: ID association 문제 회피
3. **Global Optimization**: Window 내에서 global 최적화

### 10.2 Depth Prior의 역할
1. **Scale Awareness**: Monocular SfM의 scale ambiguity 해결
2. **Initialization**: 안정적인 3D 초기값 제공
3. **Constraint**: 최적화 과정에서 depth consistency 유지

**중요**: Depth consistency는 실제 ground truth depth와의 비교가 아니라, GeometryCrafter가 각 프레임에서 예측한 depth 값들 간의 일관성을 의미합니다. 같은 3D 포인트가 다른 프레임으로 투영됐을 때도 비슷한 depth를 가져야 한다는 제약입니다.

### 10.5 수학적 기초

#### Quaternion Rotation
```
q = [w, x, y, z] where ||q|| = 1
R = quaternion_to_rotation_matrix(q)
```

#### Projection 변환
```
1. World → Camera: X_cam = R^T(X_world - t)
2. Camera → Normalized: x_norm = X_cam.x / (tan_fov_x * X_cam.z)
3. Normalized → Pixel: x_pix = (x_norm + 1) * width/2
```

#### Cross-projection 수식
```
Frame i의 3D point P_i를 Frame j로 projection:
P_j = K_j * [R_j | t_j] * [R_i | t_i]^(-1) * P_i
```

### 10.3 Window-based 처리의 의미
1. **Memory Bound**: O(window_size) 메모리 복잡도
2. **Parallelizable**: Window별 독립 처리 가능
3. **Incremental**: 새로운 window 추가 용이

### 10.4 양방향 Tracking의 이론적 장점
1. **Occlusion Handling**: Forward/backward tracking으로 occlusion 극복
2. **Boundary Completeness**: Window 양끝에 확실한 3D points 확보
3. **Temporal Symmetry**: 시간 방향 대칭성으로 더 안정적인 reconstruction
4. **Overlap Consistency**: Window 간 overlap region에서 더 많은 correspondence

## 11. 현재 구현의 한계와 개선 방향

### 11.1 미구현 기능
- [ ] Phase 2 완전 구현 (boundary 3D points 최적화)
- [ ] 양방향 tracking 구현 (window 시작 + 끝에서 features 추출)
- [ ] CoTracker에서 window 끝 프레임 points 추출
- [x] Depth consistency loss term (구현 완료, 기본값 비활성화)
- [ ] Multi-scale optimization
- [ ] Loop closure detection

### 11.2 최적화 가능 영역
- [ ] Sparse Jacobian 활용
- [ ] GPU 병렬화 강화
- [ ] Adaptive learning rate scheduling
- [ ] Coarse-to-fine 전략

### 11.3 확장 가능성
- [ ] Different camera models (RADIAL, OPENCV)
- [ ] Rolling shutter correction
- [ ] Dynamic scene handling
- [ ] IMU integration

이 구현은 전통적 multi-view geometry와 최신 deep learning의 강점을 결합한 state-of-the-art 접근법입니다. GeometryCrafter의 핵심 아이디어인 window-based cross-projection은 대규모 비디오 시퀀스에서 효율적이고 확장 가능한 3D reconstruction을 가능하게 합니다.

## 12. 현재 구현 상태 요약

### 12.1 구현 완료
- ✅ Window-based track loading (병합 없이)
- ✅ Depth-based 3D initialization
- ✅ Phase 1: Camera-only optimization
- ✅ Cross-projection loss computation
- ✅ Depth consistency loss (GeometryCrafter style)
- ✅ COLMAP export (binary/text formats)
- ✅ Configurable image dimensions
- ✅ Visualization pipeline (PNG output for CLI)
- ✅ Phase 2 framework (boundary point optimization)
- ✅ Robust loss (Huber) implementation
- ✅ FOV optimization per frame
- ✅ Gradient clipping for stable optimization

### 12.2 부분 구현
- ⚠️ Query time tracking (bidirectional support exists but requires CoTracker modification)
- ⚠️ Boundary points extraction (framework complete, needs bidirectional tracks)
- ⚠️ Phase 2 optimization (code complete, effectiveness limited by unidirectional tracks)

### 12.3 미구현
- ❌ CoTracker의 양방향 tracking integration
- ❌ Window 끝 프레임에서의 point extraction in CoTracker
- ❌ Complete boundary optimization (needs bidirectional data)

### 12.4 GeometryCrafter의 양방향 Tracking 메커니즘

GeometryCrafter의 원본 구현 분석 결과, 다음과 같은 핵심 기능이 발견됨:

**양방향 Feature Extraction:**
```python
# Window 시작 프레임에서 SIFT + SuperPoint features 추출
queries = get_query_points(superpoint, sift, frame[0], mask)

# Window 끝 프레임에서도 동일하게 추출
queries2 = get_query_points(superpoint, sift, frame[-1], mask)

# 양방향 tracking 수행
query_time = [0, 0, ..., T-1, T-1, ...]  # 각 point의 시작 시간
tracks = SpaTracker(video, concat([queries, queries2]))
```

**현재 구현과의 차이:**
- GeometryCrafter: Window 양끝에서 features 추출 → 양방향 tracking
- 현재 구현: Window 시작에서만 추출 → 단방향 tracking만 가능

**이것이 중요한 이유:**
1. Boundary optimization을 위해서는 window 양끝에 3D points 필요
2. Occlusion 처리: 시작/끝에서만 보이는 features 포착
3. Window overlap region에서 더 robust한 correspondence

### 12.5 향후 작업
GeometryCrafter의 완전한 구현을 위해서는:
1. **CoTracker에 양방향 tracking 구현**:
   - Window 시작과 끝에서 SIFT/SuperPoint features 추출
   - Forward tracking (start→end) + Backward tracking (end→start)
   - query_time 정보를 올바르게 저장
2. **Phase 2 optimization 구현**:
   - Boundary 3D points (query_3d_start, query_3d_end) 최적화
   - Window 간 consistency constraints
3. **Window overlap region 처리 개선**

## 13. 코드 상세 분석: 각 모듈의 핵심 함수

### 13.1 WindowTrackLoader - 핵심 메서드 분석

#### load_window_tracks() 상세 동작 (lines 67-162)
```python
def load_window_tracks(self, track_dir: Path, track_pattern: str = "*.npy") -> List[Dict]:
    # 1. Track 파일 검색 및 정렬
    track_files = sorted(track_dir.glob(track_pattern))
    
    # 2. 각 파일에서 window tracks 추출
    for track_file in track_files:
        track_data = np.load(track_file, allow_pickle=True).item()
        
        # 3. Bidirectional tracking 감지
        if 'query_times' in window_data:
            query_time = window_data['query_times'].astype(np.int32)
        else:
            # Legacy unidirectional data
            query_time = np.zeros(num_points, dtype=np.int32)
            
        # 4. Bidirectional 확인
        is_bidirectional = window_data.get('bidirectional', False)
        if is_bidirectional or np.any(query_time > 0):
            num_start_queries = np.sum(query_time == 0)
            num_end_queries = np.sum(query_time > 0)
```

**핵심 포인트:**
- Bidirectional tracks를 우선적으로 찾음 (lines 119-141)
- query_time이 없는 legacy data 처리 (lines 120-127)
- Window metadata (size, interval) 보존 (lines 100-101)

#### create_tracks_with_depth() 분석 (lines 205-266)
```python
def create_tracks_with_depth(self, window_tracks: List[Dict], depth_dir: Path) -> List[Dict]:
    # 1. Frame interval 감지 (lines 222-238)
    # Depth 파일명 패턴에서 interval 추론
    frame_numbers = []
    for f in depth_files[:10]:
        if f.stem.isdigit():
            frame_numbers.append(int(f.stem))
    frame_interval = frame_numbers[1] - frame_numbers[0] if len(frame_numbers) > 1 else 1
    
    # 2. Depth map 로드 및 sampling (lines 240-258)
    for local_idx in range(end_frame - start_frame):
        frame_idx = start_frame + local_idx
        actual_frame_num = frame_idx * frame_interval  # 실제 depth 파일 번호
        depth_file = depth_dir / f"{actual_frame_num:05d}.npz"
```

**핵심 포인트:**
- Frame interval 자동 감지 기능
- Missing depth 처리 (dummy depth 사용)
- GPU memory efficient depth loading

### 13.2 WindowDepthInitializer - 3D Unprojection 상세

#### triangulate_window_tracks() 분석 (lines 50-127)
```python
def triangulate_window_tracks(self, window_tracks: List[Dict], cameras: Optional[Dict] = None):
    for window in window_tracks:
        # 1. Frame별 3D 초기화 (lines 76-94)
        for t in range(T):
            points_2d = tracks_3d[t, :, :2]  # (N, 2)
            depths = tracks_3d[t, :, 2]      # (N,)
            points_3d = self.unproject_to_3d(points_2d, depths)
            xyzw_world[t, :, :3] = points_3d
            xyzw_world[t, :, 3] = 1.0
            
        # 2. Boundary points 추출 (lines 99-125)
        query_mask_start = (query_time == 0)
        query_mask_end = (query_time == window['window_size'] - 1)
        
        if query_mask_start.any():
            query_3d_start = xyzw_world[0, query_mask_start, :3]
            window['query_3d_start'] = query_3d_start.cpu().numpy()
```

**핵심 포인트:**
- 각 프레임에서 독립적 3D 초기화
- Boundary points만 별도로 저장 (Phase 2를 위해)
- Camera space에서 world space 변환 준비 (lines 91-93)

### 13.3 WindowBundleAdjuster - Cross-Projection 엔진

#### compute_cross_projection_loss() 상세 분석 (lines 130-271)
```python
def compute_cross_projection_loss(self, window_tracks, camera_model):
    for window in window_tracks:
        # 1. 3D 포인트 생성 (lines 161-179)
        for i in range(T):
            # Normalized coordinates → Camera space (FOV 사용)
            x_cam = x_norm * tan_fov_x[i] * z
            y_cam = y_norm * tan_fov_y[i] * z
            xyzw_cam = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=-1)
            xyzw_world[i] = xyzw_cam @ proj_mats[i].T
            
        # 2. T×T Cross-projection (lines 184-258)
        for j in range(T):  # Target frame
            for i in range(T):  # Source frame
                if i == j: continue
                
                # Visibility check
                valid_mask = visibility[i] & visibility[j]
                
                # Project i→j
                xyzw_j = xyzw_world[i, valid_mask] @ proj_mats[j]
                
                # Perspective division & pixel conversion
                x_proj_pix = (x_proj_norm + 1.0) * (self.image_width / 2)
                y_proj_pix = (y_proj_norm + 1.0) * (self.image_height / 2)
```

**Cross-projection 핵심 아이디어:**
1. 각 window 내에서 모든 frame pair (i,j)에 대해 projection
2. i 프레임의 3D points를 j 프레임으로 투영
3. Ground truth 2D와 비교하여 reprojection error 계산
4. T×T matrix로 모든 가능한 projection 고려

#### Depth Consistency Loss (lines 235-249)
```python
if self.config.depth_loss_weight > 0:
    # GeometryCrafter's predicted depth at target frame j
    predicted_depth_j = tracks_3d[j, valid_mask, 2][valid_depth]
    
    # Depth ratio should be close to 1 for consistent depths
    depth_ratio = z_proj[valid_depth] / (predicted_depth_j + 1e-6)
    z_loss = F.mse_loss(depth_ratio, torch.ones_like(depth_ratio))
```

**중요:** 이것은 ground truth depth와의 비교가 아니라, GeometryCrafter가 각 프레임에서 예측한 depth 간의 consistency를 체크하는 것임.

### 13.4 Phase 2 Optimization 상세 (lines 592-674)

#### setup_init_track() - Boundary Points 초기화 (lines 332-422)
```python
def setup_init_track(self, window_tracks, camera_model):
    for window in window_tracks:
        # 1. First frame boundary points (lines 358-383)
        mask_first = (query_times == 0)
        if mask_first.any():
            # Depth + 2D → 3D world coordinates
            x_cam = x_norm * tan_fov_x_first * z
            y_cam = y_norm * tan_fov_y_first * z
            xyzw_cam = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=-1)
            xyzw_world = xyzw_cam @ torch.linalg.inv(proj_mat_first.float()).T
            
        # 2. Last frame boundary points (lines 385-414)
        mask_last = (query_times == window_size - 1)
        if mask_last.any():
            # Similar process for end frame
```

**Phase 2 핵심 개념:**
- 오직 window boundaries의 3D points만 최적화
- 모든 3D points가 아닌 sparse subset만 사용
- Window 간 consistency 보장을 위한 설계

### 13.5 COLMAP Export 상세 (pipeline.py lines 302-534)

#### _create_colmap_points3D() 분석 (lines 410-534)
```python
def _create_colmap_points3D(self, window_tracks):
    # 1. Phase 2 optimized boundary points 우선 (lines 419-455)
    for window in window_tracks:
        if 'boundary_3d_optimized' in window:
            # Optimized points from Phase 2
            boundary_3d = window['boundary_3d_optimized']
            
    # 2. Regular 3D points from depth (lines 458-531)
    for window in window_tracks:
        # Skip if already added as optimized boundary point
        if 'boundary_3d_optimized' in window and is_boundary_point:
            continue
            
        # Median 3D position across observations
        xyz_median = np.median(xyz_observations, axis=0)
```

**COLMAP Export 전략:**
1. Phase 2로 최적화된 boundary points 우선 export
2. 일반 3D points는 median position 사용
3. Point ID = window_idx * 10000 + point_idx로 unique ID 보장

## 14. 성능 최적화 및 메모리 관리

### 14.1 GPU 메모리 최적화
- Window 단위 처리로 메모리 사용량 예측 가능
- Depth maps는 필요한 경우에만 로드
- Cross-projection에서 visibility mask로 불필요한 계산 회피

### 14.2 계산 효율성
- Quaternion normalization은 gradient 계산 전에 수행
- Robust loss (Huber)로 outlier 영향 최소화
- Convergence check로 불필요한 iteration 회피

### 14.3 병렬 처리 가능 영역
- Window들은 독립적이므로 병렬 처리 가능
- Cross-projection matrix 계산 병렬화 가능
- COLMAP export는 window별로 독립적 처리 가능

## 15. 디버깅 및 트러블슈팅

### 15.1 주요 로그 포인트
- Window loading: 각 window의 frame 범위와 point 수
- Depth sampling: Missing depth files 경고
- Optimization: 매 100 iteration마다 loss 기록
- COLMAP export: 총 point 수와 image 수

### 15.2 일반적인 문제와 해결법
1. **"No track files found"**: track_pattern이 실제 파일명과 일치하는지 확인
2. **"Depth file not found"**: Frame interval 설정 확인
3. **Convergence 실패**: Learning rate 조정 또는 robust_loss_sigma 증가
4. **GPU OOM**: Window size 감소 또는 더 적은 points 사용

### 15.3 시각화를 통한 검증
- camera_trajectory.png: 카메라 경로가 매끄러운지 확인
- reprojection_errors.png: 평균 error가 합리적인지 확인
- summary.png: Loss curve가 수렴하는지 확인

## 16. 예제 실행 및 결과 분석

### 16.1 기본 실행 예제
```bash
# 1. Precompute 단계 (필수)
python -m src.precompute.precompute /data/scene --config config/precompute_geometrycrafter.yaml

# 2. Window BA 실행
python -m src.window_ba /data/scene --output_dir outputs/window_ba_results

# 3. Phase 2 포함 실행
python -m src.window_ba /data/scene --output_dir outputs/window_ba_phase2 --use_refine
```

### 16.2 예상 출력 구조
```
outputs/window_ba_results/
├── window_ba.log               # 상세 실행 로그
├── cameras_phase1.npz          # Phase 1 카메라 파라미터
├── cameras_final.npz           # 최종 카메라 파라미터
├── window_tracks_3d.npz        # 3D tracks 데이터
├── phase1_history.json         # 최적화 히스토리
├── pipeline_summary.json       # 실행 요약
├── summary.txt                 # 인간이 읽을 수 있는 요약
├── colmap/                     # COLMAP export
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── visualizations/             # 시각화 결과
    ├── camera_trajectory.png
    ├── 3d_points.png
    ├── reprojection_errors.png
    └── summary.png
```

### 16.3 특정 시나리오별 설정

#### 드론 비디오 (60fps → 10fps)
```yaml
# config/window_ba_drone.yaml
track_loader:
  track_mode: "sift"
  depth_subdir: "depth/GeometryCrafter"

optimization:
  max_iterations: 5000  # 드론은 비교적 매끄러운 경로
  robust_loss_sigma: 2.0  # 빠른 움직임에 대한 허용치 증가

camera:
  image_width: 1920
  image_height: 1080
```

#### 정적 카메라
```yaml
# config/window_ba_static.yaml
optimization:
  learning_rate_translation: 0.001  # 작은 translation 변화
  learning_rate_fov: 0.00001  # FOV 거의 고정
  depth_loss_weight: 0.1  # Depth consistency 강화
```

### 16.4 성능 벤치마크

| 메트릭 | 예상값 | 설명 |
|---------|---------|------|
| Phase 1 수렴 시간 | 2-5분 | 100 frames, 50 window size |
| 평균 reprojection error | < 2 pixels | 양호한 depth 품질 가정 |
| GPU 메모리 사용량 | 4-8GB | Window size에 비례 |
| COLMAP points 수 | ~10K-50K | Window 수와 track 밀도에 따라 |

### 16.5 결과 해석

#### pipeline_summary.json 예시
```json
{
  "scene_dir": "/data/scene",
  "num_windows": 15,
  "total_frames": 200,
  "phase1": {
    "final_loss": 0.000234,
    "iterations": 3421,
    "converged": true
  },
  "success": true
}
```

#### 로그 해석
```
[INFO] Loaded 15 window tracks
[INFO] Window 0: Bidirectional tracks with 250 start queries and 250 end queries
[INFO] Phase 1 Iteration 100: loss=0.0234, projections=12450
[INFO] Converged at iteration 3421
[INFO] Created 15234 3D points from window tracks
```

### 16.6 COLMAP과의 통합
```bash
# COLMAP GUI에서 결과 확인
colmap gui --database_path outputs/window_ba_results/colmap/database.db \
           --image_path /data/scene/images

# 직접 MVS 실행
colmap image_undistorter --image_path /data/scene/images \
                        --input_path outputs/window_ba_results/colmap \
                        --output_path outputs/dense
```

## 17. 결론 및 향후 발전 방향

### 17.1 현재 구현의 강점
1. **확장성**: Window 단위 처리로 대규모 비디오 처리 가능
2. **안정성**: Depth prior로 초기화 실패 가능성 감소
3. **호환성**: COLMAP과 완벽한 호환
4. **유연성**: 다양한 tracking 방법 지원 (SIFT, SuperPoint, Grid)

### 17.2 개선 가능한 부분
1. **양방향 Tracking**: CoTracker에 bidirectional tracking 완전 통합
2. **Multi-scale 처리**: Coarse-to-fine 최적화
3. **Dynamic Scene**: 움직이는 객체 처리
4. **Real-time 처리**: Incremental window 처리

### 17.3 연구 및 확장 방향
1. **Loop Closure**: 긴 비디오에서 loop detection 및 closure
2. **Multi-camera**: 다중 카메라 시스템 지원
3. **Semantic Integration**: Semantic segmentation과 통합
4. **Neural Rendering**: NeRF/3DGS와의 통합

이 구현은 GeometryCrafter의 핵심 아이디어를 충실히 따르면서도, 실용적인 COLMAP 통합과 확장 가능한 구조를 제공합니다.