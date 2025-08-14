from typing import Optional
import os
import re
import numpy as np


def _color_from_id(idx: int):
    h = (idx * 47) % 360
    s = 0.9
    v = 0.95
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


def _save_ply(points_xyz: np.ndarray, colors_rgb: np.ndarray, path: str) -> None:
    N = int(points_xyz.shape[0])
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points_xyz, colors_rgb):
            f.write(f"{float(x)} {float(y)} {float(z)} {int(r)} {int(g)} {int(b)}\n")


def measure_to_ply(
    xyz_npy: str,
    candidate_idx_txt: str,
    out_ply: str,
    color: str = "id",
    deduplicate: bool = True,
) -> int:
    """
    measurement_xyz.npy와 후보 인덱스 텍스트를 읽어 PLY 저장.
    반환값은 저장된 포인트 수.
    """
    os.makedirs(os.path.dirname(out_ply) or ".", exist_ok=True)

    pts = np.load(xyz_npy)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("xyz_npy should be of shape (N,3[+]).")
    pts = pts[:, :3].astype(float)

    # 후보 인덱스 읽기 (쉼표/공백 구분 모두 지원)
    idxs = []
    with open(candidate_idx_txt, "r") as f:
        for line in f:
            # 라인 주석(# ...) 제거 후 트림
            line = re.split(r"#", line, 1)[0].strip()
            if not line:
                continue
            # 쉼표 또는 공백으로 토큰 분리
            tokens = re.split(r"[\s,]+", line)
            for tok in tokens:
                if tok == "":
                    continue
                try:
                    idxs.append(int(tok))
                except ValueError:
                    # 토큰에 숫자가 아닌게 섞여 있으면 무시
                    continue

    if len(idxs) == 0:
        raise ValueError("No candidate indices parsed from text file.")
    idxs = np.array(idxs, dtype=int)
    idxs = idxs[(idxs >= 0) & (idxs < len(pts))]
    if deduplicate and len(idxs) > 0:
        # 고유 인덱스만 유지 (첫 등장 순서 유지)
        _, first_idx = np.unique(idxs, return_index=True)
        idxs = idxs[np.sort(first_idx)]

    chosen = pts[idxs]
    if color == "id":
        cols = np.array([_color_from_id(int(i)) for i in range(len(chosen))], dtype=np.uint8)
    else:
        cols = np.full((len(chosen), 3), 200, dtype=np.uint8)

    _save_ply(chosen, cols, out_ply)
    return int(len(chosen))


def xyz_to_ply_sampled(
    xyz_npy: str,
    out_ply: str,
    sample_count: Optional[int] = None,
    seed: int = 42,
    color: str = "gray",
) -> int:
    """
    Load a measurement xyz .npy and save a (possibly sampled) PLY.

    - sample_count <= 0 or None: save all points.
    - sample_count > 0: random sample (without replacement) up to that many points.
    Returns number of saved points.
    """
    os.makedirs(os.path.dirname(out_ply) or ".", exist_ok=True)

    pts = np.load(xyz_npy)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("xyz_npy should be of shape (N,3[+]).")
    pts = pts[:, :3].astype(float)

    N = pts.shape[0]
    k = None if (sample_count is None) else int(sample_count)
    if (k is None) or (k <= 0) or (k >= N):
        chosen = pts
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=k, replace=False)
        chosen = pts[idx]

    if color == "id":
        cols = np.array([_color_from_id(int(i)) for i in range(len(chosen))], dtype=np.uint8)
    else:
        cols = np.full((len(chosen), 3), 180, dtype=np.uint8)

    _save_ply(chosen, cols, out_ply)
    return int(len(chosen))
