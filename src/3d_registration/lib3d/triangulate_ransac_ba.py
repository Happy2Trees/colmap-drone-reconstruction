
import os
import struct
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import itertools
import random

# ============================
# COLMAP binary readers
# ============================

CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

def _read_uint64(f): return struct.unpack('<Q', f.read(8))[0]
def _read_uint32(f): return struct.unpack('<I', f.read(4))[0]
def _read_int32(f):  return struct.unpack('<i', f.read(4))[0]
def _read_int64(f):  return struct.unpack('<q', f.read(8))[0]
def _read_double(f): return struct.unpack('<d', f.read(8))[0]

def _read_string(f):
    chars = []
    while True:
        c = f.read(1)
        if c == b'\x00' or c == b'':
            break
        chars.append(c)
    return b''.join(chars).decode('utf-8', errors='ignore')

def read_cameras_bin(path: str):
    cams = {}
    with open(path, 'rb') as f:
        num_cams = _read_uint64(f)
        for _ in range(num_cams):
            cam_id = _read_uint32(f)
            model_id = _read_int32(f)
            width = _read_uint64(f)
            height = _read_uint64(f)
            model_name, num_params = CAMERA_MODELS[model_id]
            params = [ _read_double(f) for _ in range(num_params) ]
            cams[cam_id] = {'model_id': model_id,'model': model_name,'width': int(width),'height': int(height),'params': np.array(params, dtype=float)}
    return cams

def read_images_bin(path: str):
    imgs = {}
    with open(path, 'rb') as f:
        num_images = _read_uint64(f)
        for _ in range(num_images):
            img_id = _read_uint32(f)
            qvec = np.array([_read_double(f) for _ in range(4)], dtype=float)
            tvec = np.array([_read_double(f) for _ in range(3)], dtype=float)
            cam_id = _read_uint32(f)
            name = _read_string(f)
            num_points2D = _read_uint64(f)
            for __ in range(num_points2D):
                _ = _read_double(f); _ = _read_double(f); _ = _read_int64(f)
            imgs[img_id] = {'qvec': qvec,'tvec': tvec,'camera_id': cam_id,'name': name}
    return imgs

# ============================
# Geometry helpers
# ============================

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w    ],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w    ],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ], dtype=float)

def camera_K(cam: Dict[str, Any]) -> np.ndarray:
    model = cam['model']; p = cam['params'].astype(float)
    if model in ('SIMPLE_PINHOLE','SIMPLE_RADIAL','SIMPLE_RADIAL_FISHEYE','RADIAL','RADIAL_FISHEYE','FOV'):
        f, cx, cy = p[0], p[1], p[2]; fx = fy = f
    elif model in ('PINHOLE','OPENCV','OPENCV_FISHEYE','FULL_OPENCV','THIN_PRISM_FISHEYE'):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        fx = fy = p[0]; cx = p[1]; cy = p[2]
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)

def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.hstack([R, t.reshape(3,1)])

def reproj_residuals(X, P, uv):
    x = P @ np.hstack([X, 1.0])
    u = x[0]/x[2]; v = x[1]/x[2]
    return np.array([u - uv[0], v - uv[1]])

def triangulate_nviews(P_list, uv_list):
    A = []
    for P, uv in zip(P_list, uv_list):
        u, v = float(uv[0]), float(uv[1])
        A.append(u * P[2,:] - P[0,:])
        A.append(v * P[2,:] - P[1,:])
    A = np.stack(A, axis=0)
    _, _, VT = np.linalg.svd(A)
    X_h = VT[-1]; X_h /= (X_h[-1] + 1e-12)
    return X_h[:3]

def positive_depth_count(X, Ps):
    cnt = 0
    for P in Ps:
        Xc = P @ np.hstack([X,1.0])
        if Xc[2] > 0: cnt += 1
    return cnt

def refine_point_gauss_newton(X0, P_list, uv_list, iters=15):
    X = X0.copy()
    for _ in range(iters):
        J = []; r = []
        for P, uv in zip(P_list, uv_list):
            x = P @ np.hstack([X, 1.0])
            Xc, Yc, Zc = x[0], x[1], x[2] + 1e-12
            u = Xc / Zc; v = Yc / Zc
            r.extend([u - uv[0], v - uv[1]])
            P3 = P[2,:3]; P1 = P[0,:3]; P2 = P[1,:3]
            du_dX = (P1*Zc - P3*Xc) / (Zc*Zc)
            dv_dX = (P2*Zc - P3*Yc) / (Zc*Zc)
            J.append(du_dX); J.append(dv_dX)
        if len(J) < 2: break
        J = np.vstack(J); r = np.array(r)
        H = J.T @ J + 1e-6 * np.eye(3); g = J.T @ r
        try:
            dx = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        X = X + dx
        if np.linalg.norm(dx) < 1e-6: break
    return X

# ============================
# RANSAC triangulation
# ============================

def ransac_triangulate(P_list, uv_list, reproj_thresh_px=2.0, max_iters=200, min_inliers=2, require_pos_depth_ratio=0.5, seed=42):
    assert len(P_list) == len(uv_list) and len(P_list) >= 2
    N = len(P_list)
    rnd = random.Random(seed)
    pairs = list(itertools.combinations(range(N), 2))
    if len(pairs) > max_iters:
        pairs_sampled = rnd.sample(pairs, max_iters)
    else:
        pairs_sampled = pairs

    best_inliers = []
    best_X = None
    best_err = float('inf')

    for (i, j) in pairs_sampled:
        P2 = [P_list[i], P_list[j]]
        uv2 = [uv_list[i], uv_list[j]]
        try:
            Xc = triangulate_nviews(P2, uv2)
        except Exception:
            continue
        if positive_depth_count(Xc, P2) < 2:  # cheirality
            continue

        errs = []
        inliers = []
        pos_depth_all = positive_depth_count(Xc, P_list)
        for k in range(N):
            e = np.linalg.norm(reproj_residuals(Xc, P_list[k], uv_list[k]))
            errs.append(e); inliers.append(e <= reproj_thresh_px)
        inliers = np.array(inliers, dtype=bool)
        num_inl = int(inliers.sum())
        mean_err = float(np.mean([errs[k] for k in range(N) if inliers[k]])) if num_inl>0 else float('inf')

        if pos_depth_all < max(2, int(require_pos_depth_ratio * N)):
            continue

        better = (num_inl > len(best_inliers)) or (num_inl == len(best_inliers) and mean_err < best_err)
        if better:
            best_inliers = inliers; best_X = Xc; best_err = mean_err

    if best_X is None:
        X0 = triangulate_nviews(P_list, uv_list)
        inliers = np.array([True]*N, dtype=bool)
        Xr = refine_point_gauss_newton(X0, P_list, uv_list, iters=15)
        errs = [np.linalg.norm(reproj_residuals(Xr, P_list[k], uv_list[k])) for k in range(N)]
        return Xr, inliers, {'num_inliers': N, 'mean_err': float(np.mean(errs)), 'max_err': float(np.max(errs)), 'fallback': True}

    in_idx = np.where(best_inliers)[0].tolist()
    P_in = [P_list[k] for k in in_idx]; uv_in = [uv_list[k] for k in in_idx]
    X0 = triangulate_nviews(P_in, uv_in)
    Xr = refine_point_gauss_newton(X0, P_in, uv_in, iters=15)
    errs = [np.linalg.norm(reproj_residuals(Xr, P_in[k], uv_in[k])) for k in range(len(P_in))]
    info = {'num_inliers': int(len(in_idx)), 'mean_err': float(np.mean(errs)) if errs else None, 'max_err': float(np.max(errs)) if errs else None, 'fallback': False}
    return Xr, best_inliers, info


def _residuals_all(X: np.ndarray, P_list: List[np.ndarray], uv_list: List[np.ndarray]) -> np.ndarray:
    return np.array([np.linalg.norm(reproj_residuals(X, P_list[k], uv_list[k])) for k in range(len(P_list))], dtype=float)


def _msac_cost(residuals: np.ndarray, thresh: float) -> float:
    t2 = float(thresh) * float(thresh)
    r2 = residuals * residuals
    return float(np.sum(np.minimum(r2, t2)))


def magsac_triangulate(P_list: List[np.ndarray], uv_list: List[np.ndarray], reproj_thresh_px: float = 2.0,
                       max_iters: int = 200, min_inliers: int = 2, require_pos_depth_ratio: float = 0.5,
                       seed: int = 42):
    """
    Approximate MAGSAC++ style: select hypothesis by MSAC cost and refine on inliers.
    This is not a full MAGSAC++ implementation but follows the robust scoring spirit.
    """
    assert len(P_list) == len(uv_list) and len(P_list) >= 2
    N = len(P_list)
    rnd = random.Random(seed)
    pairs = list(itertools.combinations(range(N), 2))
    if len(pairs) > max_iters:
        pairs_sampled = rnd.sample(pairs, max_iters)
    else:
        pairs_sampled = pairs

    best_cost = float('inf')
    best_X = None
    best_inliers = None

    for (i, j) in pairs_sampled:
        P2 = [P_list[i], P_list[j]]
        uv2 = [uv_list[i], uv_list[j]]
        try:
            Xc = triangulate_nviews(P2, uv2)
        except Exception:
            continue
        if positive_depth_count(Xc, P2) < 2:
            continue

        res = _residuals_all(Xc, P_list, uv_list)
        if positive_depth_count(Xc, P_list) < max(2, int(require_pos_depth_ratio * N)):
            continue
        cost = _msac_cost(res, reproj_thresh_px)
        if cost < best_cost:
            best_cost = cost
            best_X = Xc
            best_inliers = res <= reproj_thresh_px

    if best_X is None:
        # fallback to direct LS + GN
        X0 = triangulate_nviews(P_list, uv_list)
        inliers = np.ones(len(P_list), dtype=bool)
        Xr = refine_point_gauss_newton(X0, P_list, uv_list, iters=15)
        res = _residuals_all(Xr, P_list, uv_list)
        return Xr, inliers, {'num_inliers': int(inliers.sum()), 'mean_err': float(np.mean(res)) if len(res)>0 else None, 'max_err': float(np.max(res)) if len(res)>0 else None, 'fallback': True}

    # Local optimization on inliers
    in_idx = np.where(best_inliers)[0].tolist()
    P_in = [P_list[k] for k in in_idx]; uv_in = [uv_list[k] for k in in_idx]
    X0 = triangulate_nviews(P_in, uv_in)
    Xr = refine_point_gauss_newton(X0, P_in, uv_in, iters=15)
    res_in = _residuals_all(Xr, P_in, uv_in)
    info = {'num_inliers': int(len(in_idx)), 'mean_err': float(np.mean(res_in)) if len(res_in)>0 else None,
            'max_err': float(np.max(res_in)) if len(res_in)>0 else None, 'fallback': False}
    return Xr, np.array(best_inliers, dtype=bool), info


def superransac_triangulate(P_list: List[np.ndarray], uv_list: List[np.ndarray], reproj_thresh_px: float = 2.0,
                            max_iters: int = 500, min_inliers: int = 2, require_pos_depth_ratio: float = 0.5,
                            seed: int = 42, lo_max_iters: int = 3, anneal: bool = True):
    """
    Approximate SuperRANSAC: LO-RANSAC with optional threshold annealing.
    - Hypothesize from 2-view minimal samples
    - Score by inlier count
    - Local optimization: re-estimate from inliers + GN, reselect inliers (repeat up to lo_max_iters)
    - Optional annealing: progressively tighten threshold as better models emerge
    """
    assert len(P_list) == len(uv_list) and len(P_list) >= 2
    N = len(P_list)
    rnd = random.Random(seed)
    pairs = list(itertools.combinations(range(N), 2))
    if len(pairs) > max_iters:
        pairs_sampled = rnd.sample(pairs, max_iters)
    else:
        pairs_sampled = pairs

    best_inliers = np.zeros(N, dtype=bool)
    best_X = None
    best_score = -1
    cur_thresh = float(reproj_thresh_px)

    def score_and_lo(X_init: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray, float]:
        # Initial inliers
        res = _residuals_all(X_init, P_list, uv_list)
        inliers = res <= thresh
        if int(inliers.sum()) < 2:
            return X_init, inliers, float('-inf')
        # LO iterations
        X_lo = X_init.copy()
        for _ in range(lo_max_iters):
            idx = np.where(inliers)[0]
            if len(idx) < 2:
                break
            P_in = [P_list[k] for k in idx]; uv_in = [uv_list[k] for k in idx]
            X0 = triangulate_nviews(P_in, uv_in)
            X_lo = refine_point_gauss_newton(X0, P_in, uv_in, iters=15)
            res = _residuals_all(X_lo, P_list, uv_list)
            new_inliers = res <= thresh
            if np.array_equal(new_inliers, inliers):
                break
            inliers = new_inliers
        score = int(inliers.sum())
        return X_lo, inliers, float(score)

    for (i, j) in pairs_sampled:
        P2 = [P_list[i], P_list[j]]
        uv2 = [uv_list[i], uv_list[j]]
        try:
            Xc = triangulate_nviews(P2, uv2)
        except Exception:
            continue
        if positive_depth_count(Xc, P2) < 2:
            continue
        if positive_depth_count(Xc, P_list) < max(2, int(require_pos_depth_ratio * N)):
            continue

        X_lo, inl, score = score_and_lo(Xc, cur_thresh)
        if score > best_score:
            best_score = score
            best_inliers = inl
            best_X = X_lo
            # Anneal threshold to be tighter as model improves
            if anneal and best_score >= 3 and cur_thresh > 0.5 * reproj_thresh_px:
                cur_thresh = max(0.5 * reproj_thresh_px, cur_thresh * 0.9)

    if best_X is None:
        X0 = triangulate_nviews(P_list, uv_list)
        inliers = np.ones(N, dtype=bool)
        Xr = refine_point_gauss_newton(X0, P_list, uv_list, iters=15)
        res = _residuals_all(Xr, P_list, uv_list)
        return Xr, inliers, {'num_inliers': int(inliers.sum()), 'mean_err': float(np.mean(res)) if len(res)>0 else None, 'max_err': float(np.max(res)) if len(res)>0 else None, 'fallback': True}

    # Final stats on inliers
    idx = np.where(best_inliers)[0]
    P_in = [P_list[k] for k in idx]; uv_in = [uv_list[k] for k in idx]
    res_in = _residuals_all(best_X, P_in, uv_in)
    info = {'num_inliers': int(len(idx)), 'mean_err': float(np.mean(res_in)) if len(res_in)>0 else None,
            'max_err': float(np.max(res_in)) if len(res_in)>0 else None, 'fallback': False}
    return best_X, np.array(best_inliers, dtype=bool), info

# (Bundle Adjustment code removed to keep only triangulation.)

# ============================
# I/O helpers
# ============================

def color_from_id(idx: int):
    h = (idx * 47) % 360; s = 0.9; v = 0.95
    c = v * s; x = c * (1 - abs((h/60)%2 - 1)); m = v - c
    if   0 <= h < 60:   r,g,b = c,x,0
    elif 60 <= h < 120: r,g,b = x,c,0
    elif 120<= h <180:  r,g,b = 0,c,x
    elif 180<= h <240:  r,g,b = 0,x,c
    elif 240<= h <300:  r,g,b = x,0,c
    else:               r,g,b = c,0,x
    return (int((r+m)*255), int((g+m)*255), int((b+m)*255))

def save_ply(points_xyz: np.ndarray, colors_rgb: np.ndarray, path: str):
    N = points_xyz.shape[0]
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {N}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for (x,y,z), (r,g,b) in zip(points_xyz, colors_rgb):
            f.write(f'{x} {y} {z} {int(r)} {int(g)} {int(b)}\n')

# ============================
# Main
# ============================

def _resolve_model_bin_paths(model_dir: Path) -> Tuple[Path, Path, Path]:
    """Resolve paths to cameras.bin and images.bin.

    Accepts a COLMAP model directory which may be either:
    - a directory that directly contains cameras.bin/images.bin
    - a workspace directory that contains a 'sparse' subdir with numbered runs
      (e.g., sparse/0, sparse/1). In this case, choose the highest-numbered
      subfolder that contains both binaries.
    Returns (cameras_bin_path, images_bin_path, chosen_dir).
    Raises AssertionError if not found.
    """
    md = Path(model_dir)

    # 1) Directly under model_dir
    cam_direct = md / 'cameras.bin'
    img_direct = md / 'images.bin'
    if cam_direct.exists() and img_direct.exists():
        return cam_direct, img_direct, md

    # 2) If model_dir looks like .../sparse/<n>
    if md.name.isdigit() and md.parent.name == 'sparse':
        cam = md / 'cameras.bin'
        img = md / 'images.bin'
        if cam.exists() and img.exists():
            return cam, img, md

    # 3) If there's a 'sparse' directory inside model_dir, scan it
    sp = md / 'sparse'
    candidates: List[Path] = []
    if sp.exists() and sp.is_dir():
        # Case: binaries directly under 'sparse'
        if (sp / 'cameras.bin').exists() and (sp / 'images.bin').exists():
            candidates.append(sp)
        # Case: numbered subfolders under 'sparse'
        for sub in sp.iterdir():
            if sub.is_dir() and (sub / 'cameras.bin').exists() and (sub / 'images.bin').exists():
                candidates.append(sub)
        if candidates:
            numeric = [(int(p.name), p) for p in candidates if p.name.isdigit()]
            if numeric:
                numeric.sort(key=lambda x: x[0])
                chosen = numeric[-1][1]
            else:
                # Fallback: choose the most recently modified candidate
                chosen = max(candidates, key=lambda p: p.stat().st_mtime)
            return chosen / 'cameras.bin', chosen / 'images.bin', chosen

    # 4) Fallback: recursive search under model_dir for sibling binaries
    found_dirs: List[Path] = []
    try:
        for img in md.rglob('images.bin'):
            cam = img.parent / 'cameras.bin'
            if cam.exists():
                found_dirs.append(img.parent)
    except Exception:
        pass
    if found_dirs:
        numeric = [(int(p.name), p) for p in found_dirs if p.name.isdigit()]
        if numeric:
            numeric.sort(key=lambda x: x[0])
            chosen = numeric[-1][1]
        else:
            # Prefer a dir under a 'sparse' ancestor if available
            sparse_dirs = [p for p in found_dirs if 'sparse' in [a.name for a in p.parents]]
            if sparse_dirs:
                chosen = max(sparse_dirs, key=lambda p: p.stat().st_mtime)
            else:
                chosen = max(found_dirs, key=lambda p: p.stat().st_mtime)
        return chosen / 'cameras.bin', chosen / 'images.bin', chosen

    raise AssertionError('cameras.bin / images.bin not found under given model_dir (including sparse/*).')


def run_triangulation(
    model_dir: str,
    tracks_csv: str,
    out_prefix: str,
    min_views: int = 2,
    ransac_method: str = 'ransac',
    ransac_thresh: float = 2.0,
    ransac_iters: int = 200,
    min_inliers: int = 2,
    pos_depth_ratio: float = 0.5,
) -> Tuple[str, str]:
    """
    tracks.csv와 COLMAP 모델에서 RANSAC 삼각측량을 수행하고
    `<out_prefix>_points.csv` 및 `<out_prefix>_points.ply`를 저장하여 경로를 반환합니다.
    """
    # 출력 경로 디렉토리 확보
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cam_path, img_path, chosen_dir = _resolve_model_bin_paths(Path(model_dir))
    print(f'[*] Using COLMAP model dir: {chosen_dir}')

    cams = read_cameras_bin(str(cam_path))
    imgs = read_images_bin(str(img_path))

    # Intrinsics per camera_id; Extrinsics per image_id
    K_by_cam = {cid: camera_K(cam) for cid, cam in cams.items()}
    name_to_image_id = {d['name']: img_id for img_id, d in imgs.items()}
    image_id_to_pose = {img_id: {'R': qvec2rotmat(d['qvec']), 't': d['tvec'], 'camera_id': d['camera_id']}
                        for img_id, d in imgs.items()}

    df = pd.read_csv(tracks_csv)
    req = {'instance_id','frame','image_name','x','y'}
    if not req.issubset(df.columns):
        raise ValueError(f'CSV must contain columns: {sorted(list(req))}')
    df = df.sort_values(['instance_id','frame']).reset_index(drop=True)

    # Build per-instance view lists
    instances: Dict[int, Tuple[List[np.ndarray], List[np.ndarray], List[int]]] = {}
    for iid, g in df.groupby('instance_id'):
        P_list: List[np.ndarray] = []
        uv_list: List[np.ndarray] = []
        used_image_ids: List[int] = []
        for _, row in g.iterrows():
            name = str(row['image_name'])
            if name not in name_to_image_id:
                continue
            img_id = name_to_image_id[name]
            pose = image_id_to_pose[img_id]
            K = K_by_cam[pose['camera_id']]
            P = projection_matrix(K, pose['R'], pose['t'])
            P_list.append(P)
            uv_list.append(np.array([row['x'], row['y']], dtype=float))
            used_image_ids.append(img_id)
        if len(P_list) >= max(2, min_views):
            instances[int(iid)] = (P_list, uv_list, used_image_ids)

    # Robust triangulation per instance
    X_map: Dict[int, np.ndarray] = {}
    stats: List[Dict[str, Any]] = []
    for iid in sorted(instances.keys()):
        P_list, uv_list, used_img_ids = instances[iid]
        if ransac_method.lower() == 'magsac':
            Xr, inliers, info = magsac_triangulate(
                P_list, uv_list,
                reproj_thresh_px=ransac_thresh,
                max_iters=ransac_iters,
                min_inliers=min_inliers,
                require_pos_depth_ratio=pos_depth_ratio,
            )
        elif ransac_method.lower() == 'superransac':
            Xr, inliers, info = superransac_triangulate(
                P_list, uv_list,
                reproj_thresh_px=ransac_thresh,
                max_iters=max(ransac_iters, 300),
                min_inliers=min_inliers,
                require_pos_depth_ratio=pos_depth_ratio,
            )
        else:
            Xr, inliers, info = ransac_triangulate(
                P_list, uv_list,
                reproj_thresh_px=ransac_thresh,
                max_iters=ransac_iters,
                min_inliers=min_inliers,
                require_pos_depth_ratio=pos_depth_ratio,
            )
        if int(inliers.sum()) < min_inliers:
            continue
        X_map[iid] = Xr
        stats.append({'instance_id': iid,
                      'num_obs': int(len(P_list)),
                      'inliers': int(inliers.sum()),
                      'mean_reproj': info['mean_err'],
                      'max_reproj': info['max_err'],
                      'fallback_no_ransac': bool(info['fallback'])})

    # Save triangulated CSV + PLY
    out_csv = f'{out_prefix}_points.csv'
    recs: List[Dict[str, Any]] = []
    for s in stats:
        X = X_map[s['instance_id']]
        recs.append({'instance_id': s['instance_id'],
                     'X': X[0], 'Y': X[1], 'Z': X[2],
                     'num_obs': s['num_obs'],
                     'inliers': s['inliers'],
                     'mean_reproj': s['mean_reproj'],
                     'max_reproj': s['max_reproj'],
                     'fallback_no_ransac': s['fallback_no_ransac']})
    pd.DataFrame.from_records(recs).to_csv(out_csv, index=False)
    print(f'[*] Saved: {out_csv} ({len(recs)} points)')

    out_ply = f'{out_prefix}_points.ply'
    if len(recs) > 0:
        pts = np.vstack([X_map[s['instance_id']] for s in stats])
        cols = np.array([color_from_id(s['instance_id']) for s in stats], dtype=np.uint8)
        save_ply(pts, cols, out_ply)
        print(f'[*] Saved: {out_ply}')

    print('[*] Done.')
    return out_csv, out_ply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True, help='Dir with cameras.bin, images.bin')
    ap.add_argument('--csv', required=True, help='CSV: instance_id, frame, image_name, x, y')
    ap.add_argument('--out_prefix', default='triangulated', help='Output prefix')
    ap.add_argument('--min_views', type=int, default=2, help='Min views per instance')
    # RANSAC
    ap.add_argument('--ransac_method', default='ransac', choices=['ransac', 'magsac', 'superransac'], help='Robust estimator for triangulation')
    ap.add_argument('--ransac_thresh', type=float, default=2.0, help='RANSAC reprojection threshold (px)')
    ap.add_argument('--ransac_iters', type=int, default=200, help='RANSAC max iterations (sampled pairs)')
    ap.add_argument('--min_inliers', type=int, default=2, help='RANSAC: minimum inliers to accept point')
    ap.add_argument('--pos_depth_ratio', type=float, default=0.5, help='Require positive depth on this fraction of views')
    args = ap.parse_args()

    run_triangulation(
        model_dir=args.model_dir,
        tracks_csv=args.csv,
        out_prefix=args.out_prefix,
        min_views=args.min_views,
        ransac_method=args.ransac_method,
        ransac_thresh=args.ransac_thresh,
        ransac_iters=args.ransac_iters,
        min_inliers=args.min_inliers,
        pos_depth_ratio=args.pos_depth_ratio,
    )

if __name__ == '__main__':
    main()
