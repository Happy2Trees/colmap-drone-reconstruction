# 3D Registration 파이프라인 개요 (src/3d_registration)

키포인트 검출 JSON들로부터 프레임별 키포인트를 추적하여 인스턴스(트랙)로 묶고, COLMAP의 카메라 파라미터를 이용해 RANSAC 기반 다중뷰 삼각측량으로 3D 점을 생성합니다. 현재 버전은 번들 어저스트먼트(BA)를 제거하고 삼각측량만 수행합니다.

## 구성 파일
- `keypoint_tracking.py`: 프레임별 키포인트를 근접-매칭으로 연결하여 트랙(인스턴스) 생성, `tracks.json`/`tracks.csv` 저장.
- `triangulate_ransac_ba.py`: COLMAP `cameras.bin`/`images.bin`과 `tracks.csv`를 받아 RANSAC 삼각측량 수행. 출력으로 `*_points.csv`, `*_points.ply` 저장. (BA 코드는 제거됨)
- `triangulate_ransac_ba.sh`: 삼각측량 실행 예시 스크립트.
- `npz_to_ply.py`: NPZ(points_xyz, points_ids)를 PLY로 변환하는 유틸(BA 결과 등 외부 NPZ에도 사용 가능). 현재 파이프라인 기본 출력은 CSV/PLY이므로 선택 사항.
- `results/`, `tracks_sec2_x7.csv`: 예시 결과/데이터.
 - `main.py`: 전체 파이프라인 오케스트레이션(트래킹 → 삼각측량 → 측정→PLY → 정합/평가)
 - `lib3d/evaluation/ply_from_measure.py`: 측정 XYZ + 후보 인덱스 → PLY 저장 함수
 - `lib3d/evaluation/registration_o3d.py`: Open3D 기반 정합(RANSAC+ICP) 및 거리 통계

## 입력 데이터 요구사항
1) 검출 JSON 디렉토리(Tracking 입력)
- 파일 패턴: 기본 `*.json`.
- 필드 요구:
  - `image_name` 또는 `image` 경로(파일명에 프레임 번호 숫자 포함 필요. 예: `img_000123.jpg` → 123 프레임으로 사용).
  - `detections`: 배열. 각 항목에 `conf`(bbox score), `keypoints_xy`([[x,y], ...]), 선택적으로 `keypoints_conf`.
- 좌표는 원본 이미지 픽셀 좌표 기준. COLMAP 모델(`images.bin`)의 이미지 이름과 일치해야 합니다.

2) COLMAP 모델 디렉토리(삼각측량 입력)
- 필수 파일: `cameras.bin`, `images.bin`.
- `images.bin`의 이미지 이름이 JSON의 `image_name`(또는 `image`의 basename)과 정확히 매칭되어야 합니다.

## 출력
- Tracking 출력(`keypoint_tracking.py`):
  - `tracks.json`: 인스턴스별 프레임 히스토리.
  - `tracks.csv`: 다음 컬럼을 갖는 CSV
    - `instance_id, frame, image_name, x, y`
- Triangulation 출력(`triangulate_ransac_ba.py`):
  - `<out_prefix>_points.csv`: 포인트별 정보
    - `instance_id, X, Y, Z, num_obs, inliers, mean_reproj, max_reproj, fallback_no_ransac`
  - `<out_prefix>_points.ply`: 컬러(인스턴스ID 기반)로 색칠된 포인트 클라우드(ASCII)

## 사용 방법
### A) main.py로 일괄 실행(권장)
예시:
```bash
python -m src.3d_registration.main \
  --json_dir /path/to/detect/json \
  --model_dir /path/to/colmap_model/sparse/0 \
  --exp_dir experiments/my_exp \
  --xyz_npy /hdd2/0321_block_drone_video/colmap/data/measurement_xyz.npy \
  --candidate_txt /hdd2/0321_block_drone_video/colmap/data/candidate_list.txt
```

주요 옵션(스텝 제어):
- `--skip_tracking`, `--reuse_tracks_csv path`
- `--skip_triangulation`, `--reuse_triangulated_ply path`
- `--skip_measure_ply`
- `--skip_register`

파라미터(발췌):
- Tracking: `--max_match_dist 80.0`, `--max_missed 2`, `--conf_min 0.8`
- Triangulation: `--ransac_thresh 3.0`, `--ransac_iters 400`, `--min_inliers 3`, `--pos_depth_ratio 0.5`, `--min_views 2`
- Registration: `--voxel_size 0.05`, `--ransac_n 4`, `--distance_thresh_ratio 1.5`, `--icp_max_iter 50`

출력(기본):
- `tracks.json`, `tracks.csv`
- `<out_prefix>_points.csv`, `<out_prefix>_points.ply`
- `output/measurement_candidates.ply`
- `output/register_debug/` + `register_eval.json`

참고: `--exp_dir`를 사용하면 위 모든 출력이 지정한 실험 디렉토리 아래(예: `experiments/my_exp`)에 저장됩니다. 이 경우 내부적으로 `out_prefix = <exp_dir>/triangulated`, `measurement_candidates.ply = <exp_dir>/measurement_candidates.ply`, `register_debug = <exp_dir>/register_debug`로 설정됩니다.

---
### 1) 키포인트 트래킹 실행
`keypoint_tracking.py`는 예시용 `__main__` 블록이 있으며, 내부 상수 `JSON_DIR`을 편집해 실행하거나, 함수 호출 방식으로 사용할 수 있습니다.

- 스크립트 실행(간단):
```bash
python src/3d_registration/keypoint_tracking.py
```
- 주요 파라미터(함수 인자):
  - `load_frames_keypoints_from_det_jsons(...)`
    - `conf_min`: 이 값 이하의 detection은 무시(기본 0.8)
    - `kp_conf_min`: 키포인트 신뢰도 임계(선택)
    - `take`: `all`(모든 keypoint) 또는 `first`(첫 키포인트만)
    - `fill_missing`: 누락 프레임을 빈 리스트로 채워 미스 반영(기본 True)
  - `link_keypoints_by_instance(...)`
    - `max_match_dist`: 매칭 허용 거리(px). 해상도에 맞춰 조정 권장(4K 기준 60~80)
    - `max_missed`: 미스 허용 프레임 수(기본 2)
    - `prefer_hungarian`: SciPy가 있으면 헝가리안, 없으면 그리디 사용

실행 후 현재 디렉토리에 `tracks.json`, `tracks.csv`가 생성됩니다.

### 2) 삼각측량 실행
`triangulate_ransac_ba.py`는 BA 없이 삼각측량만 수행합니다.

예시:
```bash
python src/3d_registration/triangulate_ransac_ba.py \
  --model_dir /path/to/colmap_model/sparse/0 \
  --csv tracks.csv \
  --out_prefix output/triangulated \
  --ransac_thresh 3.0 \
  --ransac_iters 400 \
  --min_inliers 3 \
  --pos_depth_ratio 0.5
```

주요 옵션:
- `--model_dir`: `cameras.bin`, `images.bin`가 있는 디렉토리.
- `--csv`: 트래킹 출력 CSV(컬럼: `instance_id, frame, image_name, x, y`).
- `--out_prefix`: 출력 접두사(예: `output/triangulated`).
- `--min_views`: 인스턴스 최소 관측 수(기본 2).
- RANSAC 관련:
  - `--ransac_thresh`: reprojection 임계(px). 값이 작을수록 엄격.
  - `--ransac_iters`: 샘플링 페어 수(반복 횟수 제한).
  - `--min_inliers`: 최소 인라이어 수.
  - `--pos_depth_ratio`: 양의 깊이(cheirality) 만족 비율.

출력으로 `<out_prefix>_points.csv`, `<out_prefix>_points.ply`가 생성됩니다.

## 내부 동작 개요
- Tracking:
  1. JSON에서 프레임별 키포인트 수집(`conf_min` 등으로 필터링).
  2. 프레임 번호는 파일명/`image_name`의 숫자에서 추출.
  3. 최근 속도를 고려한 예측-게이팅 후(행별 gate) 할당(헝가리안 또는 그리디).
  4. 트랙 히스토리를 `tracks.json/csv`로 저장.
- Triangulation:
  1. COLMAP `images.bin` 포즈(qvec,tvec)와 `cameras.bin` 내재파라미터로 투영행렬 P 생성.
  2. 인스턴스별 관측을 기반으로 페어 샘플링 → SVD 다중뷰 삼각측량.
  3. 체이럴리티(양의 깊이)와 reprojection 오류로 인라이어 선별.
  4. 선택 인라이어로 가우스-뉴턴(점만) 미세화.
  5. CSV/PLY 저장.

## 환경/의존성
- Python 3.8+
- `numpy`, `pandas`
- (선택) `scipy` — 있으면 Tracking에 헝가리안 사용. Triangulation은 필수 아님.

## 주의사항/팁
- 이름 매칭: JSON의 `image_name`(또는 `image` basename)과 COLMAP `images.bin`의 이미지 이름이 정확히 같아야 합니다.
- 해상도 일치: 키포인트 좌표가 COLMAP 내재 파라미터가 추정된 이미지 해상도와 동일 기준이어야 합니다.
- 프레임 인덱스: 파일명에 숫자가 없으면 스킵됩니다(정렬/동기 어려움). 규칙적인 이름을 권장.
- 성능: `--ransac_iters`, `--ransac_thresh`가 속도/정확도에 영향. 너무 빡빡하면 인라이어 부족으로 포인트가 걸러질 수 있습니다.
- `fallback_no_ransac`: RANSAC에서 적절한 샘플을 찾지 못하면 모든 뷰로 직접 삼각화 후 점만 미세화하는 경로를 사용했다는 표식입니다.

---
문의나 개선이 필요하면 파라미터 기본값을 알려주세요. 사용 환경/데이터 특성에 맞춰 튜닝해 드릴 수 있습니다.
