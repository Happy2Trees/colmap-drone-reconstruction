#!/usr/bin/env python3
"""
CoTracker save_all_tracks (*.npy)  ➜  COLMAP database (.db)

(기반 코드: COLMAPDatabase helper – ETH & UNC LICENSE)
(수정 2: 좌표 유효성 검사 및 전역 인덱스 맵을 사용한 견고한 매칭)


Usage
-----
python conv_sift.py \
    --npy /home/sanggyun/hdseo/colmap-drone-reconstruction/data/sec2_1_processed_1024x576/cotracker/80_40_sift.npy \
    --image_dir /home/sanggyun/hdseo/colmap-drone-reconstruction/data/sec2_1_processed_1024x576/images \
    --K /home/sanggyun/hdseo/colmap-drone-reconstruction/data/sec2_1_processed_1024x576/K.txt \
    --width 1024 --height 576 \
    --out_db /home/sanggyun/hdseo/colmap-drone-reconstruction/data/sec2_1_processed_1024x576/outputs/x3_section2_cotracker_sift.db

"""

# python conv_sift.py \
#     --npy /home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2_processed_1024x576_80_40/cotracker/80_40_sift.npy \
#     --image_dir /home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2_processed_1024x576_80_40/images \
#     --K /home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2_processed_1024x576_80_40/K.txt \
#     --width 1024 --height 576 \
#     --out_db /home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2_processed_1024x576_80_40/outputs/x3_section2_cotracker_sift.db


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# 이 라인을 파일의 시작 부분으로 옮겼습니다.
from __future__ import annotations 
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

import argparse, importlib, sys
from pathlib import Path
from collections import defaultdict
import os
import numpy as np
from tqdm import tqdm

# ───────────────────── 1. COLMAP helper 클래스 가져오기 ──────────────────────
COLMAP_DB_MODULE = "colmap_database"
try:
    COLMAP = importlib.import_module(COLMAP_DB_MODULE)
    COLMAPDatabase = COLMAP.COLMAPDatabase
    image_ids_to_pair_id = COLMAP.image_ids_to_pair_id
    pair_id_to_image_ids = COLMAP.pair_id_to_image_ids
except ImportError:
    print(f"Error: Could not import '{COLMAP_DB_MODULE}.py'. Make sure it's in the same directory.")
    sys.exit(1)

# ───────────────────── 2. 변환 루틴 ─────────────────────────────────────────
def load_tracks(npy_path: Path):
    """NumPy 2.x 호환 로더 + list[dict] 반환"""
    if np.__version__.startswith('2.'):
        np.core.numerictypes.ScalarType = (np.core.numerictypes.ScalarType, type(np.dtype('int64').type()))
    try:
        data = np.load(npy_path, allow_pickle=True)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            print("Handling NumPy 2.x compatibility issue...")
            sys.modules['numpy._core'] = importlib.import_module('numpy.core')
            data = np.load(npy_path, allow_pickle=True)
        else:
            raise
    if data.ndim == 0:
        data = data.item()
    return data.tolist() if isinstance(data, np.ndarray) else data


def main(args):
    # 0) DB 파일이 이미 있으면 삭제
    if os.path.exists(args.out_db):
        print(f"Warning: Deleting existing database '{args.out_db}'")
        os.remove(args.out_db)

    db = COLMAPDatabase.connect(str(args.out_db))
    db.create_tables()

    # 1) 카메라 및 이미지 정보 등록
    K = np.loadtxt(args.K)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    camera_params = np.array([fx, fy, cx, cy], np.float64)
    cam_id = db.add_camera(model=1, width=args.width, height=args.height, params=camera_params, prior_focal_length=True)

    image_paths = sorted([p for p in os.listdir(args.image_dir) if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    frame_to_imageid = {i: db.add_image(name, cam_id) for i, name in enumerate(image_paths)}
    raw_npy = load_tracks(args.npy)
    all_segs = raw_npy['tracks']
    # ==========================================================================
    # 수정된 로직: 3단계 접근 (전역 인덱스 맵 사용)
    # ==========================================================================

    # ⭐️ 1단계: 각 이미지에 어떤 세그먼트의 (모든) 키포인트들이 속하는지 수집
    image_keypoints_per_seg = defaultdict(list)
    for seg_idx, seg in enumerate(tqdm(all_segs, desc="Phase 1/3: Collecting all keypoints")):
        tracks = seg["tracks"][0]
        start_frame, end_frame = seg["start_frame"], seg["end_frame"]
        for t_local, frame_abs in enumerate(range(start_frame, end_frame)):
            if frame_abs in frame_to_imageid:
                image_id = frame_to_imageid[frame_abs]
                image_keypoints_per_seg[image_id].append((seg_idx, tracks[t_local]))

    # ⭐️ 2단계: 유효한 키포인트만 DB에 저장하고, (seg, track_idx) -> global_idx 맵 생성
    keypoint_map = defaultdict(lambda: defaultdict(dict)) # keypoint_map[image_id][seg_idx][track_idx] = global_idx
    for image_id, seg_kps_list in tqdm(image_keypoints_per_seg.items(), desc="Phase 2/3: Storing valid keypoints & building map"):
        seg_kps_list.sort(key=lambda x: x[0]) # 세그먼트 순서로 정렬하여 일관성 유지
        
        valid_kps_for_image = []
        global_idx = 0
        for seg_idx, kps in seg_kps_list:
            for track_idx_local, (x, y) in enumerate(kps):
                # ★★★ 키포인트 유효성 검사 (좌표가 이미지 범위 안에 있는지) ★★★
                if 0 <= x < args.width and 0 <= y < args.height:
                    valid_kps_for_image.append((x, y))
                    keypoint_map[image_id][seg_idx][track_idx_local] = global_idx
                    global_idx += 1
        
        if valid_kps_for_image:
            final_keypoints_np = np.array(valid_kps_for_image, dtype=np.float32)
            db.add_keypoints(image_id, final_keypoints_np)

    # ⭐️ 3단계: 전역 인덱스 맵을 사용하여 정확한 매칭 정보 생성
    matches_dict = defaultdict(list)
    for seg_idx, seg in enumerate(tqdm(all_segs, desc="Phase 3/3: Generating matches with map")):
        vis = seg["visibility"][0]
        start_frame, end_frame = seg["start_frame"], seg["end_frame"]
        T, N = vis.shape

        for t1 in range(T):
            for t2 in range(t1 + 1, T):
                frame1_abs, frame2_abs = start_frame + t1, start_frame + t2
                if frame1_abs not in frame_to_imageid or frame2_abs not in frame_to_imageid:
                    continue

                image_id1, image_id2 = frame_to_imageid[frame1_abs], frame_to_imageid[frame2_abs]
                
                # 공통으로 보이는 트랙 찾기
                common_visible_indices = np.where(vis[t1] & vis[t2])[0]
                
                new_matches = []
                for track_idx_local in common_visible_indices:
                    # ★★★ 맵을 조회하여 두 이미지 모두에서 유효한 키포인트인지 확인 ★★★
                    global_idx1 = keypoint_map[image_id1][seg_idx].get(track_idx_local)
                    global_idx2 = keypoint_map[image_id2][seg_idx].get(track_idx_local)
                    
                    if global_idx1 is not None and global_idx2 is not None:
                        new_matches.append((global_idx1, global_idx2))
                
                if new_matches:
                    pair_id = image_ids_to_pair_id(image_id1, image_id2)
                    matches_dict[pair_id].extend(new_matches)

    # 최종적으로 수집된 매칭 정보를 DB에 기록
    print("Committing matches to database...")
    for pair_id, matches_list in tqdm(matches_dict.items(), desc="Committing matches"):
        if not matches_list: continue
        
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        
        # 중복 매칭 제거 (서로 다른 세그먼트에서 동일한 매칭이 생성될 수 있음)
        unique_matches = np.unique(np.array(matches_list, dtype=np.uint32), axis=0)
        
        db.add_matches(image_id1, image_id2, unique_matches)
        db.add_two_view_geometry(image_id1, image_id2, unique_matches)
    
    db.commit()
    db.close()
    print(f"✅ Finished! Database saved to ➜ {args.out_db}")

# ───────────────────── 4. CLI ───────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--npy", type=Path, required=True, help="CoTracker save_all_tracks.npy")
    ap.add_argument("--image_dir", type=Path, required=True, help="Frame directory (e.g., 00123.jpg)")
    ap.add_argument("--K", type=Path, required=True, help="3x3 intrinsic matrix file (*.txt)")
    ap.add_argument("--width", type=int, required=True, help="Image width")
    ap.add_argument("--height", type=int, required=True, help="Image height")
    ap.add_argument("--out_db", type=Path, default="x3_section2_cotracker_grid.db", help="Output COLMAP database path")
    main(ap.parse_args())