from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import json
import csv
import os
import glob
import re

try:
    # 선택: 있으면 헝가리안 사용, 없으면 자동 그리디로 대체
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -----------------------------
# 유틸: image_name에서 숫자 프레임 인덱스 추출
# 예) "img_05853.jpg" -> 5853
# -----------------------------
def parse_frame_idx_from_name(name: str) -> int:
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else -1


# -----------------------------
# JSON 디렉토리에서 프레임별 키포인트 로드
# conf <= conf_min 인 detection은 무시 (기본: 0.8)
# kp_conf_min 설정 시, keypoints_conf < kp_conf_min 인 키포인트는 무시
# fill_missing=True면 최소~최대 프레임 사이의 누락 프레임은 빈 리스트로 채움
# -----------------------------
def load_frames_keypoints_from_det_jsons(
    json_dir: str,
    pattern: str = "*.json",
    conf_min: float = 0.8,
    kp_conf_min: Optional[float] = None,
    take: str = "all",         # "all"이면 detection의 모든 keypoint 사용, "first"면 첫 keypoint만
    fill_missing: bool = True
) -> Tuple[Dict[int, List[Tuple[float, float]]], Dict[int, str]]:
    paths = sorted(glob.glob(os.path.join(json_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No JSON files found under: {json_dir}")

    # 먼저 (frame_idx, image_name, points) 수집
    tmp: Dict[int, Tuple[List[Tuple[float, float]], str]] = {}
    min_idx, max_idx = +10**18, -10**18

    for jp in paths:
        with open(jp, "r") as f:
            D = json.load(f)

        img_name = D.get("image_name") or os.path.basename(D.get("image", os.path.basename(jp)))
        fidx = parse_frame_idx_from_name(img_name)
        if fidx == -1:
            # 파일명에서 숫자를 못 찾으면 순서를 보장하기 어려움 → 스킵하거나 0부터 enumerate해도 됨
            # 여기서는 스킵 대신 안전하게 enumerate 대체를 권하면 좋지만, 단순히 continue 처리
            continue

        min_idx = min(min_idx, fidx)
        max_idx = max(max_idx, fidx)

        pts: List[Tuple[float, float]] = []
        for det in D.get("detections", []):
            # bbox confidence 기준 필터: conf <= conf_min 면 무시
            if float(det.get("conf", 0.0)) <= conf_min:
                continue

            kps = det.get("keypoints_xy", []) or []
            kpc = det.get("keypoints_conf", []) or []

            if take == "first":
                if len(kps) >= 1:
                    if (kp_conf_min is None) or (len(kpc) == 0) or (kpc[0] >= kp_conf_min):
                        x, y = float(kps[0][0]), float(kps[0][1])
                        pts.append((x, y))
            else:  # take == "all"
                for j, kp in enumerate(kps):
                    if not (isinstance(kp, (list, tuple)) and len(kp) >= 2):
                        continue
                    if (kp_conf_min is not None) and (j < len(kpc)) and (kpc[j] < kp_conf_min):
                        continue
                    x, y = float(kp[0]), float(kp[1])
                    pts.append((x, y))

        tmp[fidx] = (pts, img_name)

    frames_keypoints: Dict[int, List[Tuple[float, float]]] = {}
    frame_name_map: Dict[int, str] = {}

    if not tmp:
        raise RuntimeError("No usable detections after filtering. "
                           "Check conf_min/kp_conf_min and JSON contents.")

    if fill_missing:
        for f in range(min_idx, max_idx + 1):
            if f in tmp:
                pts, name = tmp[f]
                frames_keypoints[f] = pts
                frame_name_map[f] = name
            else:
                # 누락 프레임: 빈 검출 (미스 증가에 사용)
                frames_keypoints[f] = []
                # 이름은 비워둠
    else:
        for f in sorted(tmp.keys()):
            pts, name = tmp[f]
            frames_keypoints[f] = pts
            frame_name_map[f] = name

    return frames_keypoints, frame_name_map


@dataclass
class Track:
    id: int
    last_pt: np.ndarray               # shape (2,)
    last_frame: int
    misses: int = 0
    history: List[dict] = field(default_factory=list)
    v: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))  # <-- 속도 벡터
    hits: int = 1  # 누적 관측 수
class KeypointTracker:
    def __init__(self,
                 max_match_dist: float = 80.0,   # 기본 임계 상향(4K/20fps 기준 60~80 권장)
                 max_missed: int = 2,
                 prefer_hungarian: bool = True,
                 vel_alpha: float = 0.8,         # 속도 업데이트 가중치(최근 관측 반영 비율)
                 gate_scale_with_dt: bool = True # dt 스케일 게이팅 적용
                 ):
        self.max_match_dist = float(max_match_dist)
        self.max_missed = int(max_missed)
        self.prefer_hungarian = bool(prefer_hungarian)
        self.vel_alpha = float(vel_alpha)
        self.gate_scale_with_dt = bool(gate_scale_with_dt)
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}

    @staticmethod
    def _pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if len(A) == 0 or len(B) == 0:
            return np.empty((len(A), len(B)), dtype=np.float64)
        diff = A[:, None, :] - B[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def _predict_pos(self, tr: Track, frame_idx: int) -> Tuple[np.ndarray, int, float]:
        """현재 트랙의 frame_idx에서의 예측 위치와 dt, 게이트 반경 반환"""
        dt = max(1, int(frame_idx - tr.last_frame))
        pred = tr.last_pt + tr.v * dt
        # dt-스케일 게이팅: 기본 임계 * dt (or dt=1이면 기본값)
        gate = self.max_match_dist * (dt if self.gate_scale_with_dt else 1.0)
        return pred, dt, gate

    def _assign(self, tracks_idx: List[int], dets: np.ndarray, frame_idx: int
               ) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
        if len(tracks_idx) == 0:
            return [], [], list(range(len(dets)))
        if len(dets) == 0:
            return [], tracks_idx, []

        # 트랙별 예측위치/게이트 계산
        preds = []
        gates = []
        for tid in tracks_idx:
            tr = self.tracks[tid]
            pred, dt, gate = self._predict_pos(tr, frame_idx)
            preds.append(pred)
            gates.append(gate)
        preds = np.asarray(preds, dtype=np.float64)      # (N,2)
        gates = np.asarray(gates, dtype=np.float64)      # (N,)

        # 예측 위치와의 거리로 비용행렬 구성
        D = self._pairwise_dist(preds, dets)             # (N,M)

        # 행별(트랙별) 게이팅
        gated = D.copy()
        if gated.size > 0:
            gated[gated > gates[:, None]] = 1e9

        matches = []
        used_tracks = set()
        used_dets = set()

        if self.prefer_hungarian and _HAS_SCIPY and gated.size > 0:
            row_ind, col_ind = linear_sum_assignment(gated)
            for r, c in zip(row_ind, col_ind):
                if gated[r, c] < 1e9:
                    matches.append((tracks_idx[r], c))
                    used_tracks.add(tracks_idx[r])
                    used_dets.add(c)
        else:
            # 그리디 할당
            N, M = gated.shape
            triplets = [(gated[r, c], r, c) for r in range(N) for c in range(M)]
            triplets.sort(key=lambda x: x[0])
            for cost, r, c in triplets:
                if cost >= 1e9: break
                tid = tracks_idx[r]
                if (tid not in used_tracks) and (c not in used_dets):
                    matches.append((tid, c))
                    used_tracks.add(tid)
                    used_dets.add(c)

        unmatched_tracks = [tid for tid in tracks_idx if tid not in used_tracks]
        unmatched_dets = [j for j in range(len(dets)) if j not in used_dets]
        return matches, unmatched_tracks, unmatched_dets

    def update(self,
               detections: List[Tuple[float, float]],
               frame_idx: int,
               frame_name: Optional[str] = None) -> Dict[int, Tuple[float, float]]:

        dets = np.array(detections, dtype=np.float64).reshape(-1, 2) if len(detections) else np.zeros((0,2), np.float64)
        active_ids = [tid for tid, tr in self.tracks.items() if tr.misses <= self.max_missed]

        matches, unmatched_tracks, unmatched_dets = self._assign(active_ids, dets, frame_idx)

        frame_assignments: Dict[int, Tuple[float, float]] = {}

        # 1) 매칭된 트랙 업데이트 (+ 속도 업데이트)
        for tid, j in matches:
            pt = dets[j]
            tr = self.tracks[tid]
            dt = max(1, int(frame_idx - tr.last_frame))
            # 측정기반 속도 추정
            v_meas = (pt - tr.last_pt) / float(dt)
            tr.v = (1.0 - self.vel_alpha) * tr.v + self.vel_alpha * v_meas  # 지수평활
            tr.last_pt = pt
            tr.last_frame = frame_idx
            tr.misses = 0
            tr.hits += 1
            tr.history.append({"frame": frame_idx, "x": float(pt[0]), "y": float(pt[1]), "name": frame_name})
            frame_assignments[tid] = (float(pt[0]), float(pt[1]))

        # 2) 매칭 안 된 트랙: 미스 증가
        for tid in unmatched_tracks:
            self.tracks[tid].misses += 1

        # 3) 매칭 안 된 검출: 새 트랙 생성
        for j in unmatched_dets:
            pt = dets[j]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(
                id=tid,
                last_pt=pt,
                last_frame=frame_idx,
                misses=0,
                history=[{"frame": frame_idx, "x": float(pt[0]), "y": float(pt[1]), "name": frame_name}],
                v=np.zeros(2, dtype=np.float64),
                hits=1
            )
            frame_assignments[tid] = (float(pt[0]), float(pt[1]))

        return frame_assignments

    def get_tracks(self, drop_dead: bool = True) -> Dict[int, List[dict]]:
        out = {}
        for tid, tr in self.tracks.items():
            if drop_dead and len(tr.history) <= 0:
                continue
            out[tid] = tr.history
        return out


def link_keypoints_by_instance(frames_keypoints: Dict[int, List[Tuple[float,float]]],
                               frame_name_map: Optional[Dict[int, str]] = None,
                               max_match_dist: float = 50.0,
                               max_missed: int = 2,
                               prefer_hungarian: bool = True) -> Dict[int, List[dict]]:
    """
    프레임별 키포인트들을 인스턴스별로 묶어서 반환.
    """
    tracker = KeypointTracker(max_match_dist=max_match_dist,
                              max_missed=max_missed,
                              prefer_hungarian=prefer_hungarian)
    for f in sorted(frames_keypoints.keys()):
        detections = frames_keypoints[f]
        name = frame_name_map.get(f) if frame_name_map else None
        tracker.update(detections, frame_idx=f, frame_name=name)
    return tracker.get_tracks(drop_dead=True)


def save_tracks_json_csv(tracks: Dict[int, List[dict]],
                         json_path: Optional[str] = "tracks.json",
                         csv_path: Optional[str] = "tracks.csv") -> None:
    if json_path:
        with open(json_path, "w") as f:
            json.dump(tracks, f, indent=2)
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "frame", "image_name", "x", "y"])
            for tid, hist in tracks.items():
                for rec in hist:
                    writer.writerow([tid, rec.get("frame"), rec.get("name"), rec.get("x"), rec.get("y")])


def run_tracking(
    json_dir: str,
    pattern: str = "*.json",
    conf_min: float = 0.8,
    kp_conf_min: Optional[float] = None,
    take: str = "all",
    fill_missing: bool = True,
    max_match_dist: float = 80.0,
    max_missed: int = 2,
    prefer_hungarian: bool = True,
    out_json: str = "tracks.json",
    out_csv: str = "tracks.csv",
) -> Tuple[str, str]:
    """
    JSON 디렉토리에서 키포인트를 로드하고 인스턴스별 링크 후 저장.
    반환: (out_json_path, out_csv_path)
    """
    frames_keypoints, frame_name_map = load_frames_keypoints_from_det_jsons(
        json_dir,
        pattern=pattern,
        conf_min=conf_min,
        kp_conf_min=kp_conf_min,
        take=take,
        fill_missing=fill_missing,
    )

    tracks = link_keypoints_by_instance(
        frames_keypoints,
        frame_name_map=frame_name_map,
        max_match_dist=max_match_dist,
        max_missed=max_missed,
        prefer_hungarian=prefer_hungarian,
    )

    # 경로 디렉토리 생성
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    save_tracks_json_csv(tracks, out_json, out_csv)
    return out_json, out_csv


# -----------------------------
# 사용 예시
# -----------------------------
if __name__ == "__main__":
    # 1) JSON 폴더에서 프레임별 키포인트 읽기
    JSON_DIR = "/hdd2/0321_block_drone_video/detect/json"  # <- 네 경로로 바꿔줘
    frames_keypoints, frame_name_map = load_frames_keypoints_from_det_jsons(
        JSON_DIR,
        pattern="*.json",
        conf_min=0.8,          # <= 0.8 인 detection 무시
        kp_conf_min=None,      # 필요하면 예: 0.5
        take="all",            # detection 안의 모든 keypoint 사용 (보통 1개)
        fill_missing=True      # 누락 프레임을 빈 리스트로 채워 미스 카운트 반영
    )

    # 2) 인스턴스별로 링크 (근접-매칭)
    tracks = link_keypoints_by_instance(
        frames_keypoints,
        frame_name_map=frame_name_map,
        max_match_dist=80.0,   # 3840x2160 해상도면 40~60px 정도가 무난. 더 타이트하게 하고 싶으면 낮춰도 OK
        max_missed=2,          # 미스 2프레임까지 허용
        prefer_hungarian=True
    )

    # 3) 저장
    save_tracks_json_csv(tracks, "tracks.json", "tracks.csv")
    print(json.dumps(tracks, indent=2)[:2000])  # 너무 길면 앞부분만 프린트
