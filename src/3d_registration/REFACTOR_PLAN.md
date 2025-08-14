# Refactor Plan: function-first + main.py orchestration

본 문서는 “기존 스크립트를 함수로 쪼개고, `main.py`에서 전체 파이프라인을 실행”하는 계획만을 명확히 정리합니다. 지금은 계획 수립 단계이며, 아래 구조·함수 시그니처·입출력 규약만 합의합니다.

## 목표(요약)
- 기존 스크립트 내부 코드를 함수화(run_* 진입점 제공)하여 모듈처럼 사용 가능하게 함
- `src/3d_registration/main.py` 하나로 전체 파이프라인 실행(기본 전체 실행, 단계 스킵/재사용 옵션 제공)
- 추가 기능(측정 XYZ→PLY, PLY 정합/거리 평가)은 보조 모듈로 분리하되 3d_registration 디렉토리 내부에 위치

## 디렉터리(최소 변경)
```
src/3d_registration/
  main.py                 # 전체 오케스트레이션(기본 전체 실행)
  lib3d/
    __init__.py
    keypoint_tracking.py     # run_tracking(...)
    triangulate_ransac_ba.py # run_triangulation(..., BA 제거 상태 유지)
    evaluation/
      __init__.py
      ply_from_measure.py # measure_to_ply(...)
      registration_o3d.py # register_and_eval(...)
  README.md
  REFACTOR_PLAN.md
```

## main.py 설계(기본 전체 실행)
- 기본 실행: tracking → triangulation → measurement→PLY → registration/eval 순차 실행
- 공통 인자: 로그/출력 폴더, 덮어쓰기 여부, 단계 스킵/재사용 옵션 제공
- 예시 실행(단일 커맨드):
  - `python main.py \
      --json_dir <det_json_dir> \
      --model_dir <colmap_model_dir> \
      --out_prefix output/triangulated \
      --xyz_npy /hdd2/0321_block_drone_video/colmap/data/measurement_xyz.npy \
      --candidate_txt /hdd2/0321_block_drone_video/colmap/data/candidate_list.txt`
- 단계 제어 옵션(예): `--skip_tracking`, `--reuse_tracks_csv path`, `--skip_triangulation`, `--reuse_triangulated_ply path`, `--skip_measure_ply`, `--skip_register`

## 함수화 항목(정확한 시그니처)
1) keypoint_tracking.py
- 유지: `parse_frame_idx_from_name`, `load_frames_keypoints_from_det_jsons`, `link_keypoints_by_instance`, `save_tracks_json_csv`
- 추가: `def run_tracking(json_dir: str, pattern: str = "*.json", conf_min: float = 0.8, kp_conf_min: Optional[float] = None, take: str = "all", fill_missing: bool = True, max_match_dist: float = 80.0, max_missed: int = 2, prefer_hungarian: bool = True, out_json: str = "tracks.json", out_csv: str = "tracks.csv") -> Tuple[str, str]:`
  - 동작: JSON 로드 → 링크 → 저장 → `(out_json, out_csv)` 경로 반환

2) triangulate_ransac_ba.py
- 유지: COLMAP bin 로더/기하/삼각측량 유틸(`read_*`, `camera_K`, `projection_matrix`, `triangulate_nviews`, `refine_point_gauss_newton`, `ransac_triangulate`, `save_ply`)
- 추가: `def run_triangulation(model_dir: str, tracks_csv: str, out_prefix: str, min_views: int = 2, ransac_thresh: float = 2.0, ransac_iters: int = 200, min_inliers: int = 2, pos_depth_ratio: float = 0.5) -> Tuple[str, str]:`
  - 동작: tracks.csv 로드 → 인스턴스별 RANSAC 삼각측량 → `<out_prefix>_points.csv/ply` 저장 → 경로 반환

3) lib3d/evaluation/ply_from_measure.py
- 신설: `def measure_to_ply(xyz_npy: str, candidate_idx_txt: str, out_ply: str, color: str = "id") -> int`
  - 동작: NPY 로드 → 후보 인덱스만 추출 → 색상 부여 → PLY 저장 → 개수 반환
  - 기본 입력 경로: `/hdd2/0321_block_drone_video/colmap/data/measurement_xyz.npy`, `/hdd2/0321_block_drone_video/colmap/data/candidate_list.txt`

4) lib3d/evaluation/registration_o3d.py
- 신설: `def register_and_eval(source_ply: str, target_ply: str, voxel_size: float = 0.05, ransac_n: int = 4, distance_thresh_ratio: float = 1.5, icp_max_iter: int = 50, save_debug_dir: Optional[str] = None) -> Dict[str, Any]`
  - 동작: FPFH + RANSAC global reg → ICP refine → C2C 거리 통계(평균/중앙값/표준편차/최대) → 결과 dict 반환 및 파일 저장 옵션

## End-to-End 흐름/입출력 규약
1) Tracking: `tracks.json`, `tracks.csv`
2) Triangulation: `<out_prefix>_points.csv`, `<out_prefix>_points.ply`
3) Measure→PLY: `measurement_candidates.ply`
4) Registration/Eval: `register_eval.json`(+옵션 `register_eval.csv`, `transform.txt`, 정합된 PLY)

## 파라미터 기본값(권장)
- Tracking: `max_match_dist=80.0`, `max_missed=2`, `conf_min=0.8`
- Triangulation: `ransac_thresh=3.0`, `ransac_iters=400`, `min_inliers=3`, `pos_depth_ratio=0.5`, `min_views=2`
- Registration/Eval(O3D): `voxel_size=0.05`, `ransac_n=4`, `distance_thresh_ratio=1.5`, `icp_max_iter=50`

## 구현 순서(체크리스트)
1) keypoint_tracking.py에 `run_tracking(...)` 추가, `__main__`는 예시만 유지
2) triangulate_ransac_ba.py에 `run_triangulation(...)` 추가(BA 제거 상태 유지 확인), CLI는 예시만 유지
3) lib3d/evaluation에 `measure_to_ply(...)`, `register_and_eval(...)` 스켈레톤 추가
4) main.py 작성: 인자 파싱 → 단계별 실행/스킵/재사용 제어 → 경로 관리 → 로그 출력
5) 문서 갱신: README.md 실행 예시(main.py 기준) 업데이트
6) 최소 스모크 테스트: 작은 샘플로 각 단계 실행해 산출물 생성 여부 확인

## 유의사항
- 이름/해상도 매칭: JSON의 `image_name`과 COLMAP의 이름 일치 필요, 좌표 해상도도 일치해야 함
- Open3D 의존성: 설치 실패 시 등록/평가는 스킵될 수 있으며, 그 경우 경고와 함께 종료
- 성능: 대용량일 때 다운샘플/임계 조정 필수. 파라미터는 main.py에서 외부 주입 가능해야 함

본 계획에 맞춰 구현을 진행하고, 필요 시 세부 시그니처/옵션을 추가 조정하겠습니다.

---

## 진행상황(업데이트)
- [x] 1) `keypoint_tracking.py`에 `run_tracking(...)` 추가 (기존 함수/클래스 무변경, 디렉토리 생성만 안전 추가)
- [x] 2) `triangulate_ransac_ba.py`에 `run_triangulation(...)` 추가 (BA 제거 상태 유지, 기존 CLI는 새 함수 호출로 위임)
- [x] 3) `lib3d/evaluation` 추가: `measure_to_ply(...)`, `register_and_eval(...)`
- [x] 4) `main.py` 오케스트레이션 구현(단계 스킵/재사용/파라미터 전달)
- [x] 5) `README.md` 실행 예시(main.py 기준) 업데이트
- [x] 변경: `keypoint_tracking.py`, `triangulate_ransac_ba.py`를 `lib3d/`로 이동하고 `main.py` 임포트를 해당 경로로 수정
- [ ] 6) 최소 스모크 테스트: 샘플 데이터로 각 단계 산출물 생성 확인 (데이터 수급/검증 대기)

## 실행 방법(요약)
### A) 전체 파이프라인 실행(권장)
```bash
python -m src.3d_registration.main \
  --json_dir /path/to/detect/json \
  --model_dir /path/to/colmap_model/sparse/0 \
  --out_prefix output/triangulated \
  --xyz_npy /hdd2/0321_block_drone_video/colmap/data/measurement_xyz.npy \
  --candidate_txt /hdd2/0321_block_drone_video/colmap/data/candidate_list.txt
```

주요 단계 제어 옵션:
- `--skip_tracking`, `--reuse_tracks_csv path`
- `--skip_triangulation`, `--reuse_triangulated_ply path`
- `--skip_measure_ply`, `--skip_register`

핵심 파라미터(기본값):
- Tracking: `max_match_dist=80.0`, `max_missed=2`, `conf_min=0.8`
- Triangulation: `ransac_thresh=3.0`, `ransac_iters=400`, `min_inliers=3`, `pos_depth_ratio=0.5`, `min_views=2`
- Registration: `voxel_size=0.05`, `ransac_n=4`, `distance_thresh_ratio=1.5`, `icp_max_iter=50`

출력(기본 경로 기준):
- `tracks.json`, `tracks.csv`
- `<out_prefix>_points.csv`, `<out_prefix>_points.ply`
- `output/measurement_candidates.ply`
- `output/register_debug/` + `register_eval.json`

### B) 단계별 실행/재사용 예시
- Tracking만 수행 후 재사용:
  ```bash
  python -m src.3d_registration.main \
    --json_dir /path/to/detect/json \
    --model_dir /path/to/colmap_model/sparse/0 \
    --out_prefix output/triangulated
  # 이후
  python -m src.3d_registration.main \
    --model_dir /path/to/colmap_model/sparse/0 \
    --reuse_tracks_csv output/tracks.csv \
    --out_prefix output/triangulated \
    --skip_measure_ply --skip_register
  ```

- 이미 삼각측량된 PLY를 재사용하여 정합만:
  ```bash
  python -m src.3d_registration.main \
    --reuse_triangulated_ply output/triangulated_points.ply \
    --skip_tracking --skip_triangulation
  ```

## 참고/주의
- JSON의 `image_name`과 COLMAP `images.bin`의 이름은 정확히 일치해야 하며, 키포인트 좌표 기준 해상도도 일치해야 합니다.
- Open3D 미설치 시 `register_and_eval(...)`은 건너뛰고 `{"skipped": true, "reason": ...}` 형태로 보고합니다.
- 모듈 실행은 패키지 방식(`python -m src.3d_registration.main`)을 권장합니다.
