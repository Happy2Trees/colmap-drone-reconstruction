# GeometryCrafter Depth Integration Plan

## 목표 (Objectives) - REVISED (2025-01-21)

GeometryCrafter 코드를 포팅하여 precompute 파이프라인에 통합합니다. Submodule의 원본 파일은 수정하지 않고, 필요한 코드를 복사하여 로컬에서 관리합니다.

### 주요 목표 (포팅 방식)
1. **Code Porting**: Submodule에서 필요한 코드를 복사하여 로컬에서 관리
2. **Minimal Modification**: 원본 코드를 최대한 그대로 유지하여 수정 용이성 확보
3. **MoGe 통합**: MoGe depth prior 모델을 pip 패키지로 활용
4. **Independent Module**: 외부 의존성 최소화로 독립적인 모듈 구성
5. **표준화된 인터페이스**: precompute 파이프라인과 일관된 인터페이스 제공

### 포팅 방식 채택 이유
- **독립성**: Submodule 업데이트에 영향받지 않는 안정적인 환경
- **커스터마이징**: 필요시 코드 수정이 자유로움
- **디버깅 용이성**: 전체 코드가 로컬에 있어 디버깅이 쉬움
- **의존성 관리**: 프로젝트에 필요한 부분만 선택적으로 포팅

## 기술 아키텍처 (Technical Architecture)

### 1. 모듈 구조 (Ported Architecture)
```
src/precompute/depth/
├── __init__.py
├── base_depth_estimator.py          # 추상 기반 클래스 (✅ 완료)
└── geometrycrafter/
    ├── __init__.py
    ├── extractor.py                # GeometryCrafter 메인 인터페이스
    ├── utils.py                    # 헬퍼 함수들
    ├── models/                     # 포팅된 모델 코드
    │   ├── __init__.py
    │   ├── unet.py                # UNet 모델 (from submodule)
    │   ├── pmap_vae.py            # Point map VAE (from submodule)
    │   ├── determ_ppl.py          # Deterministic pipeline (from submodule)
    │   └── diff_ppl.py            # Diffusion pipeline (from submodule)
    └── configs/                    # 모델 설정
        └── default_configs.py      # 기본 설정값들

submodules/GeometryCrafter/          # 원본 참조용 (읽기 전용)
├── geometrycrafter/                 # 포팅할 소스 코드
├── third_party/                     # 외부 의존성
└── run.py                          # 참조용 실행 스크립트
```

### 2. MoGe Integration

MoGe는 GeometryCrafter의 **필수 컴포넌트**로, 각 프레임에 대한 depth prior를 제공합니다:

- **역할**: Monocular depth estimation prior
- **설치**: `pip install git+https://github.com/microsoft/MoGe.git`
- **출력**:
  - Point map: 3D 좌표 (x right, y down, z forward)
  - Depth map: Scale-invariant depth
  - Valid mask: 유효한 픽셀 마스크
  - Camera intrinsics: 정규화된 카메라 파라미터

### 3. 구현 전략 (Implementation Strategy)

#### 3.1 포팅 접근 방식 (Porting Approach)
```python
# Step 1: 필요한 모듈 식별 및 복사
# - geometrycrafter/unet.py
# - geometrycrafter/pmap_vae.py
# - geometrycrafter/determ_ppl.py
# - geometrycrafter/diff_ppl.py
# - 필요한 유틸리티 함수들

# Step 2: 로컬 모듈 구조 생성
# src/precompute/depth/geometrycrafter/models/ 디렉토리에 포팅

# Step 3: Import 경로 수정
# 원본: from geometrycrafter import ...
# 포팅: from .models import ...
```

#### 3.2 Minimal Modification Strategy
```python
# geometrycrafter/models/unet.py (포팅된 파일)
# 원본 코드를 그대로 복사하되, import 경로만 수정

# 원본
# from geometrycrafter.utils import ...

# 수정
# from ..utils import ...

class GeometryCrafterExtractor(BaseDepthEstimator):
    """포팅된 GeometryCrafter 구현체"""
    
    def __init__(self, config):
        super().__init__(config)
        self.is_video_model = True
        
        # 포팅된 로컬 모델 사용
        from .models.unet import UNetSpatioTemporalConditionModelVid2vid
        from .models.pmap_vae import PMapAutoencoderKLTemporalDecoder
        
        # Initialize models
        self.unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(...)
        self.point_map_vae = PMapAutoencoderKLTemporalDecoder.from_pretrained(...)
        self.prior_model = MoGe(cache_dir=self.cache_dir)  # pip 패키지 사용
        
        # Create pipeline using ported implementation
        if self.model_type == 'diff':
            from .models.diff_ppl import GeometryCrafterDiffPipeline
            self.pipe = GeometryCrafterDiffPipeline(...)
        else:
            from .models.determ_ppl import GeometryCrafterDetermPipeline
            self.pipe = GeometryCrafterDetermPipeline(...)
```

#### 3.3 Processing Flow
```python
def extract_depth(self, image_dir, output_path=None):
    """Process video sequence for depth extraction"""
    # 1. Load images as video frames
    frames = self._load_images_as_video(image_paths)
    
    # 2. Use original pipeline with sliding window
    with torch.inference_mode():
        rec_point_map, rec_valid_mask = self.pipe(
            frames_tensor,
            self.point_map_vae,
            self.prior_model,
            window_size=self.window_size,
            overlap=self.overlap,
            # ... other parameters
        )
    
    # 3. Extract depth from point maps
    depth_maps = rec_point_map[..., 2]  # Z coordinate is depth
    
    # 4. Save results in precompute format
    for i, (depth, mask) in enumerate(zip(depth_maps, rec_valid_mask)):
        self.save_depth_map(depth, output_file, mask=mask)
```

### 4. GeometryCrafter Extractor 구현
```python
# geometrycrafter/extractor.py
class GeometryCrafterExtractor(BaseDepthEstimator):
    def __init__(self, config):
        self.device = config.get('device', 'cuda')
        self.batch_size = config.get('batch_size', 4)
        self.model_type = config.get('model_type', 'determ')  # 'determ' or 'diff'
        
        # MoGe 초기화 (필수)
        from moge import MoGe
        self.moge_model = MoGe(cache_dir='./weights/moge').to(self.device)
        
        # GeometryCrafter 파이프라인 초기화
        self.pipeline = self._load_pipeline()
    
    def extract_depth(self, image_dir: Path, output_path: Optional[Path] = None):
        """이미지 디렉토리에서 depth map 추출"""
        # 1. 이미지 로드 및 전처리
        # 2. 배치 단위로 처리
        # 3. Point map → Depth 변환
        # 4. 개별 파일로 저장
    
    def process_batch(self, image_batch):
        """배치 단위 depth 추출 (MoGe prior 활용)"""
        with torch.no_grad():
            # MoGe depth prior + GeometryCrafter refinement
            point_maps, masks = self.pipeline(image_batch)
            
            # Point map에서 depth 추출
            depth_maps = self.extract_depth_from_point_maps(point_maps)
            
        return depth_maps, masks
```

### 5. 포팅 구현 세부사항

#### 5.1 주요 특징
- **독립적 모듈**: Submodule 의존성 없이 독립적으로 작동
- **최소한의 수정**: Import 경로와 필수 변경사항만 수정
- **완전한 호환성**: 원본 모델 가중치 그대로 사용
- **쉬운 유지보수**: 원본 코드 구조를 그대로 유지하여 업데이트 용이

#### 5.2 메모리 관리
```python
def extract_depth(self, image_dir, output_path=None):
    """Memory-efficient processing with original sliding window"""
    # GeometryCrafter의 native sliding window 활용
    # - window_size: 110 frames (default)
    # - overlap: 25 frames
    # - decode_chunk_size: 8 frames
    
    # low_memory_usage 옵션 활용
    if self.low_memory_usage:
        # CPU-GPU 간 데이터 이동 최적화
        # 중간 결과 CPU로 오프로드
```

#### 5.3 에러 처리 및 로깅
```python
import logging
logger = logging.getLogger(__name__)

try:
    # Model loading with detailed logging
    logger.info("Loading GeometryCrafter models...")
    self.unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(...)
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise
```

## 구현 단계 (Implementation Steps)

### Phase 1: 기본 구조 구축
- [x] GeometryCrafter 코드 분석
- [x] MoGe 통합 방식 분석
- [x] ~~구현 전략 변경 (포팅 → direct import)~~ → 포팅 방식으로 재변경 (2025-01-21)
- [x] base_depth_estimator.py 작성 (video support 포함)
- [x] 디렉토리 구조 생성

### Phase 2: 코드 포팅
- [x] GeometryCrafter 핵심 모듈 포팅
  - [x] unet.py 포팅 및 import 경로 수정
  - [x] pmap_vae.py 포팅 및 import 경로 수정
  - [x] determ_ppl.py 포팅 및 import 경로 수정
  - [x] diff_ppl.py 포팅 및 import 경로 수정
  - [x] 필요한 유틸리티 함수 포팅 (수정 불필요)
- [x] 포팅된 코드 테스트
  - [x] 모델 로딩 테스트
  - [x] 추론 파이프라인 테스트
  - [x] 원본과 동일한 결과 확인

### Phase 3: Extractor 구현
- [x] GeometryCrafterExtractor 클래스 구현
  - [x] 포팅된 모델 초기화
  - [x] 이미지 로딩 및 전처리
  - [x] Pipeline 실행 (포팅된 코드 사용)
  - [x] 결과 저장 메서드
- [x] 헬퍼 유틸리티 구현
  - [x] 이미지 시퀀스 로더
  - [x] Point map → Depth 변환
  - [x] 시각화 함수

### Phase 4: Integration
- [x] precompute.py에 depth 모듈 통합
- [x] 설정 파일에 depth 옵션 추가
  - [x] precompute_geometrycrafter.yaml 재작성 (포팅 버전용)
  - [x] 기본 파라미터 설정
- [x] 배치 처리 최적화
- [x] 로깅 및 진행상황 표시

### Phase 5: 검증 및 최적화
- [ ] 단위 테스트 작성
- [x] 메모리 사용량 프로파일링
  - 3837 프레임 처리 시 OOM(Out of Memory)로 인한 프로세스 종료 확인
  - 메모리 부족으로 인한 "Killed" 에러 발생
- [x] 다양한 비디오 길이 테스트
  - 20 프레임: 성공 ✅
  - 3837 프레임: 실패 (OOM) ❌
- [x] 캐시 시스템 구현 (2025-06-12)
  - `produce_priors` 함수에 캐싱 기능 추가
  - 진행 상황 표시 (tqdm progress bar)
  - GPU 메모리 사용량 실시간 모니터링
  - 효율적인 해시 생성 (전체 데이터 대신 샘플 사용)
  - 2단계 캐시: raw 결과와 processed 결과 분리
  - 코드 중복 제거 (`_process_priors` 메서드)
- [ ] 에러 케이스 처리
  - 메모리 부족 감지 및 graceful degradation 필요
- [ ] 속도 최적화
  - 배치 처리 구현 필요 (현재 전체 시퀀스를 한 번에 로드)

### Phase 6: 문서화
- [ ] API 문서 작성
- [ ] 사용 가이드 작성
- [ ] CLAUDE.md 업데이트

## 포팅 실행 계획 (Porting Execution Plan)

### Step 1: 디렉토리 구조 생성
```bash
mkdir -p src/precompute/depth/geometrycrafter/models
```

### Step 2: 파일 복사
```bash
# Core model files
cp submodules/GeometryCrafter/geometrycrafter/__init__.py src/precompute/depth/geometrycrafter/models/
cp submodules/GeometryCrafter/geometrycrafter/unet.py src/precompute/depth/geometrycrafter/models/
cp submodules/GeometryCrafter/geometrycrafter/pmap_vae.py src/precompute/depth/geometrycrafter/models/
cp submodules/GeometryCrafter/geometrycrafter/determ_ppl.py src/precompute/depth/geometrycrafter/models/
cp submodules/GeometryCrafter/geometrycrafter/diff_ppl.py src/precompute/depth/geometrycrafter/models/
```

### Step 3: Import 경로 수정
- 대부분의 파일은 외부 라이브러리만 import하므로 수정 불필요
- 내부 import가 있는 경우만 상대 경로로 변경

### Step 4: MoGe Wrapper 구현
```python
# src/precompute/depth/geometrycrafter/moge_wrapper.py
import torch
import torch.nn as nn
from moge import MoGeModel

class MoGe(nn.Module):
    def __init__(self, cache_dir):
        super().__init__()
        self.model = MoGeModel.from_pretrained(
            'Ruicheng/moge-vitl', cache_dir=cache_dir).eval()
    
    @torch.no_grad()
    def forward_image(self, image: torch.Tensor, **kwargs):
        output = self.model.infer(image, resolution_level=9, apply_mask=False, **kwargs)
        points = output['points']
        masks = output['mask']
        return points, masks
```

### Step 5: Extractor 구현
- GeometryCrafterExtractor 클래스 구현
- 포팅된 모델들을 사용하여 depth extraction 수행

## 현재 상태 (Current Status)

**마지막 업데이트**: 2025-01-21

- ✅ GeometryCrafter 분석 완료
- ✅ MoGe 통합 방식 확인 완료
- ✅ ~~포팅 계획 수립 완료~~ → Direct import 방식으로 변경 → **포팅 방식으로 재변경** (2025-01-21)
- ✅ 긴 비디오 시퀀스 처리 분석 완료
  - GeometryCrafter는 native sliding window 지원 (window_size=110, overlap=25)
  - 메모리 최적화를 위한 chunk-based processing 내장
- ✅ 기본 구조 구축 완료
  - base_depth_estimator.py (video support 포함)
  - 디렉토리 구조 생성
- ✅ **포팅 구현 완료** (2025-06-12): 
  - **구현 내용**: 
    - GeometryCrafter 핵심 모듈 포팅 완료 (unet, pmap_vae, determ_ppl, diff_ppl)
    - GeometryCrafterExtractor 클래스 구현 완료
    - MoGe를 pip 패키지로 통합 (forward_image wrapper 구현)
    - precompute 파이프라인 통합 완료
  - **테스트 결과**: 
    - 20개 이미지로 테스트 성공
    - 모델 로딩 및 추론 정상 작동
    - Depth map 및 visualization 생성 확인
    - 메타데이터 저장 정상 작동
  - **출력 형식**: 
    - `.npz` 파일 (depth, mask, metadata 포함)
    - `_vis.png` 시각화 파일
- 🚧 **메모리 이슈 및 최적화** (2025-06-12):
  - **문제점**: 
    - 3837 프레임 처리 시 OOM으로 프로세스 종료
    - 전체 시퀀스를 메모리에 로드하는 구조적 문제
  - **개선사항**:
    - `produce_priors` 캐싱 시스템 구현
    - 진행 상황 표시 및 메모리 모니터링 추가
    - 효율적인 캐시 키 생성 (샘플 기반 해싱)
    - 2단계 캐시 구조 (raw/processed)
  - **진행 중**:
    - 배치 처리 방식 구현 검토
    - 메모리 사용량 디버깅 도구 추가
- ✅ **세그먼트 기반 처리 구현 완료** (2025-01-21):
  - **구현 내용**:
    - GeometryCrafterExtractor에 `frame_start`, `frame_end` 파라미터 추가
    - precompute.py에서 자동 세그먼트 분할 처리 구현
    - 통일된 출력 디렉토리 구조 (모든 depth map이 하나의 디렉토리에 저장)
    - 세그먼트별 metadata + 통합 metadata 생성
  - **주요 개선사항**:
    - `segment_size: 1000` 설정 옵션 추가 (config 파일)
    - cache 관련 코드 제거 (produce_priors에서 불필요한 cache 로직 삭제)
    - 24GB GPU에서 3800 프레임 안전하게 처리 가능
  - **출력 구조**:
    ```
    depth/
    ├── GeometryCrafter/          # 모든 depth map (통합)
    │   ├── 001.npy
    │   ├── 002.npy
    │   └── ... (3800개)
    ├── depth_metadata.json       # 통합 metadata
    └── depth_metadata_segment_*.json  # 각 세그먼트 metadata
    ```

### 📊 실용적 가이드라인 (1024x576 기준)
- **RTX 3090 (24GB)**: ~3400 frames 처리 가능
- **RTX 4090 (24GB)**: ~3400 frames 처리 가능 
- **A100 (40GB)**: ~5600 frames 처리 가능
- **권장사항**: 
  - 안전한 처리를 위해 GPU 메모리의 70% 정도만 사용
  - 24GB GPU에서는 ~2000 frames까지 안정적
  - 더 긴 비디오는 segment 단위로 분할

## 주요 기술적 세부사항

### 1. Two-Stage Depth Estimation
```
Stage 1: MoGe Prior
- Input: RGB image
- Output: Initial depth map, point map, intrinsics
- Purpose: Monocular depth prior

Stage 2: GeometryCrafter Refinement  
- Input: RGB image + MoGe prior
- Output: Refined point map with temporal consistency
- Purpose: High-quality depth with temporal smoothing
```

### 2. MoGe Prior의 중요성
- **필수 컴포넌트**: GeometryCrafter는 MoGe prior 없이 작동 불가
- **역할**: 
  - 초기 depth 추정값 제공
  - Scale-invariant geometry 제공
  - Camera intrinsics 추정
- **통합 방식**: Prior conditioning을 통해 UNet에 입력

### 3. 포팅 계획 (Porting Plan)

#### 3.1 포팅할 파일 목록
```
submodules/GeometryCrafter/geometrycrafter/
├── __init__.py                     # 모듈 export 정의
├── unet.py                         # UNet 모델 정의 (diffusers 기반)
├── pmap_vae.py                     # Point map VAE (diffusers 기반)
├── determ_ppl.py                   # Deterministic pipeline
└── diff_ppl.py                     # Diffusion pipeline

외부 의존성 (pip 설치):
- diffusers >= 0.31.0
- MoGe (git+https://github.com/microsoft/MoGe.git)
- torch, torchvision, numpy 등 표준 라이브러리
```

#### 3.2 포팅 전략
1. **최소 수정 원칙**:
   - 원본 코드를 그대로 복사
   - Import 경로만 수정
   - 절대 필요한 경우만 코드 수정

2. **의존성 처리**:
   - diffusers 라이브러리는 그대로 사용
   - MoGe는 pip 패키지로 사용
   - 내부 import만 상대 경로로 변경

3. **파일별 수정 사항**:
   ```python
   # __init__.py 수정 없음 (상대 import 사용)
   from .pmap_vae import PMapAutoencoderKLTemporalDecoder
   from .unet import UNetSpatioTemporalConditionModelVid2vid
   from .diff_ppl import GeometryCrafterDiffPipeline
   from .determ_ppl import GeometryCrafterDetermPipeline
   
   # unet.py, pmap_vae.py - 외부 import만 사용하므로 수정 불필요
   # determ_ppl.py, diff_ppl.py - 내부 import 확인 필요
   ```

4. **MoGe 통합 방식**:
   ```python
   # Direct pip package 사용
   from moge import MoGeModel
   # 또는 wrapper 구현
   class MoGe(nn.Module):
       def __init__(self, cache_dir):
           self.model = MoGeModel.from_pretrained('Ruicheng/moge-vitl', cache_dir=cache_dir)
   ```

### 4. 메모리 최적화 전략

#### 4.1 현재 상태 (2025-01-21)
- ✅ **세그먼트 기반 처리 구현 완료**: 큰 비디오를 자동으로 분할하여 처리
- **메모리 사용량 (1024x576 기준)**: 
  - 1024x576 @ 500 frames: ~3.5GB GPU 메모리
  - 1024x576 @ 1000 frames: ~7GB GPU 메모리
  - 1024x576 @ 2000 frames: ~14GB GPU 메모리
  - 1024x576 @ 3800 frames: ~26.6GB GPU 메모리 (세그먼트 분할로 처리 가능)
  - 1024x576 @ 4800 frames: ~34GB GPU 메모리 (세그먼트 분할로 처리 가능)

#### 4.2 구현된 접근 방식
1. ✅ **비디오 분할 처리 (구현 완료)**:
   ```yaml
   # config/precompute_geometrycrafter.yaml
   depth:
     segment_size: 1000  # 자동으로 1000 프레임씩 분할 처리
   ```
   - 3800 프레임 → 4개 세그먼트로 자동 분할
   - 각 세그먼트 독립적으로 처리
   - 최종 출력은 하나의 디렉토리에 통합

2. **해상도 감소 (옵션)**:
   ```yaml
   depth:
     downsample_ratio: 2.0  # 1/2 해상도로 처리
   ```

3. **Window size 조정 (옵션)**:
   ```yaml
   depth:
     window_size: 50  # 기본값: 110
     overlap: 10      # 기본값: 25
   ```

#### 4.3 최적화 개선사항
- ✅ **Cache 코드 제거**: produce_priors의 불필요한 cache 로직 제거로 메모리 절약
- ✅ **통합 출력 구조**: 세그먼트별 처리해도 하나의 디렉토리에 통합 저장
- ✅ **Metadata 관리**: 세그먼트별 + 통합 metadata로 완전한 추적 가능

## 설정 파라미터 (Configuration)

```yaml
depth:
  model: geometrycrafter
  device: cuda
  
  # 세그먼트 처리 설정 (대용량 비디오용)
  segment_size: 1000      # 프레임 단위 세그먼트 크기
  
  # 모델 설정
  model_type: diff        # 'diff' (고품질) 또는 'determ' (빠름)
  cache_dir: workspace/cache  # 모델 가중치 캐시 디렉토리
  
  # 비디오 처리 설정
  window_size: 110        # 시간적 윈도우 크기 (기본값)
  overlap: 25             # 윈도우 간 오버랩
  decode_chunk_size: 8    # VAE 디코딩 청크 크기
  
  # 추론 설정
  num_inference_steps: 5  # 디노이징 스텝 수 (diff 모델용)
  guidance_scale: 1.0     # 가이던스 스케일 (diff 모델용)
  downsample_ratio: 1.0   # 입력 다운샘플링 비율
  
  # 모델 옵션
  force_projection: true  # 원근 투영 강제
  force_fixed_focal: true # 고정 초점 거리 사용
  use_extract_interp: false # 추출시 보간 사용
  low_memory_usage: false # 저메모리 모드 (느림)
  
  # 출력 설정
  save_visualization: true  # Depth map 시각화 저장
  output_format: npy       # 출력 형식: 'npy', 'png', 'pfm'
  save_moge_prior: false   # MoGe prior 별도 저장
  save_ply: false          # 3D 포인트 클라우드 PLY 저장
  
  # 랜덤 시드
  seed: 42
```

## 현재 이슈 및 해결 방안 (Current Issues & Solutions)

### 1. OOM (Out of Memory) 문제
**증상**: 
- 3837 프레임 처리 시 "Killed" 메시지와 함께 프로세스 종료
- 이미지 로딩 단계에서 발생 (pipeline 실행 전)

**원인**:
- 전체 비디오 시퀀스를 numpy array로 메모리에 로드
- 3837 frames × 1024×576 × 3 channels × float32 = ~25GB RAM 필요

**해결 방안**:
1. **즉시 적용 가능**:
   - `low_memory_usage: true` 설정 사용
   - 비디오를 여러 세그먼트로 분할 처리
   - `downsample_ratio` 증가 (품질 저하)
   
2. **구현 필요**:
   - 배치 단위 이미지 로딩 및 처리
   - Streaming 방식 구현
   - 디스크 캐싱 활용

### 2. 캐시 시스템 개선점
**현재 구현**:
- `produce_priors` 함수에 캐싱 추가
- 진행 상황 표시 (tqdm)
- GPU 메모리 모니터링

**개선 사항**:
- 효율적인 해시 생성 (첫/마지막 프레임 샘플 사용)
- 2단계 캐시: raw 결과와 processed 결과 분리
- 코드 중복 제거 (`_process_priors` 메서드)

### 3. 디버깅 방안
**추가된 로깅**:
```python
# 메모리 상태 로깅
logger.info(f"System memory: {memory_info.percent:.1f}% used")
logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# 캐시 상태 로깅
logger.info(f"Loading cached priors from {cache_path}")
logger.info(f"Saved processed priors to cache: {cache_processed_path}")
```

**디버깅 명령어**:
```bash
# 시스템 메모리 모니터링
watch -n 1 free -h

# GPU 메모리 모니터링  
watch -n 1 nvidia-smi

# 프로세스별 메모리 사용량
htop
```

## 의존성 요구사항 (Dependencies)

### 필수 의존성 (requirements.txt 추가)
```
# Core dependencies
torch>=2.3.1
torchvision>=0.18.1
einops>=0.8.0
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0

# Model specific
diffusers>=0.31.0  # GeometryCrafter 컴포넌트
transformers>=4.39.1  # Tokenizer 및 유틸리티
safetensors>=0.4.0  # 가중치 로딩
accelerate>=0.24.0  # 모델 로딩 최적화

# MoGe (Microsoft Monocular Geometry Estimation)
git+https://github.com/microsoft/MoGe.git  # Depth prior 모델

# Optional optimizations
xformers>=0.0.22  # 메모리 효율적인 attention (선택사항)
torch-compile>=2.0  # 속도 최적화 (선택사항)
```

### MoGe 의존성 (자동 설치됨)
- timm (Vision Transformer)
- huggingface-hub (모델 다운로드)
- 기타 MoGe 요구사항

## 사용법 (Usage)

```bash
# GeometryCrafter depth 추출
python -m src.precompute.precompute /path/to/scene --config config/precompute_geometrycrafter.yaml

# Visualization 없이 실행
python -m src.precompute.precompute /path/to/scene --config config/precompute_geometrycrafter.yaml --no-visualize

# 예시 (1024x576 preprocessing + depth extraction)
python -m src.precompute.precompute /data/scene_001 --config config/precompute_geometrycrafter.yaml
```

### Configuration Options
- `model_type`: 'determ' (faster, recommended) or 'diff' (higher quality)
- `window_size`: 110 frames (default) - 메모리 부족시 감소 권장
- `overlap`: 25 frames (default) - 메모리 부족시 감소 권장
- `downsample_ratio`: 1.0 (default) - 메모리 부족시 2.0 이상 권장
- `preprocessing`: 1024x576 (GeometryCrafter 최적 해상도)
- `decode_chunk_size`: 8 (default) - 메모리 부족시 4-6으로 감소
- `low_memory_usage`: false (default) - true로 설정시 느리지만 메모리 절약
  - true: 텐서를 CPU에 유지하고 필요시에만 GPU로 이동
  - false: 모든 텐서를 GPU에 유지 (빠르지만 메모리 사용량 높음)

## 예상 출력 구조

```
Scene/
├── depth/
│   ├── GeometryCrafter/          # GeometryCrafter 최종 출력
│   │   ├── 001.npy              # Refined depth map
│   │   ├── 002.npy
│   │   └── ...
│   └── MoGe/                     # MoGe prior (선택사항)
│       ├── 001_depth.npy         # Initial depth
│       ├── 001_points.npy        # 3D points
│       └── ...
└── depth_metadata.json           # 처리 정보 및 파라미터
```

## 리스크 및 대응 방안

### 1. MoGe 의존성
- **리스크**: MoGe 패키지 설치 실패 또는 버전 충돌
- **대응**: 
  - 특정 커밋 해시로 고정
  - 로컬 캐시 및 fallback 옵션
  - Docker 이미지에 사전 설치

### 2. 메모리 사용량
- **리스크**: MoGe + GeometryCrafter 동시 실행시 OOM
- **대응**: 
  - Two-stage 분리 실행
  - Prior 캐싱 전략
  - 동적 배치 크기 조정

### 3. 처리 속도
- **리스크**: Two-stage로 인한 속도 저하
- **대응**: 
  - MoGe prior 캐싱
  - 병렬 처리 최적화
  - Deterministic 모드 기본 사용

## 테스트 계획

### 1. 단위 테스트
```python
# tests/test_geometrycrafter_depth.py
def test_moge_integration():
    """MoGe 모델 로딩 및 추론 테스트"""

def test_prior_conditioning():
    """MoGe prior를 사용한 conditioning 테스트"""

def test_two_stage_pipeline():
    """전체 two-stage pipeline 테스트"""
```

### 2. 통합 테스트
- MoGe + GeometryCrafter 전체 플로우
- 다양한 이미지 크기 및 종횡비
- 메모리 사용량 모니터링

### 3. 품질 검증
- MoGe prior vs GeometryCrafter 최종 결과 비교
- 시간적 일관성 검증
- COLMAP reconstruction 개선도 측정

## 참고 자료

- [GeometryCrafter GitHub](https://github.com/VAST-AI-Research/GeometryCrafter)
- [MoGe GitHub](https://github.com/microsoft/MoGe)
- [GeometryCrafter Paper](https://arxiv.org/abs/2412.07068)
- [MoGe Paper](https://arxiv.org/abs/2410.05737)