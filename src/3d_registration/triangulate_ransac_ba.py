
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

# ============================
# Bundle Adjustment (per-image extrinsics + points; intrinsics fixed)
# ============================

def rodrigues_to_R(r: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(r)
    if theta < 1e-12: return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[ -k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
    return R

def R_to_rodrigues(R: np.ndarray) -> np.ndarray:
    theta = np.arccos(max(min((np.trace(R)-1)/2, 1.0), -1.0))
    if theta < 1e-12: return np.zeros(3)
    rx = (R[2,1]-R[1,2])/(2*np.sin(theta))
    ry = (R[0,2]-R[2,0])/(2*np.sin(theta))
    rz = (R[1,0]-R[0,1])/(2*np.sin(theta))
    return theta*np.array([rx,ry,rz])

def pack_params(cam_params: Dict[int, Dict[str,Any]], X_map: Dict[int, np.ndarray], cam_ids: List[int], point_ids: List[int]):
    theta = []; trans = []
    for cid in cam_ids:
        R = cam_params[cid]['R']; t = cam_params[cid]['t']
        theta.append(R_to_rodrigues(R)); trans.append(t)
    theta = np.concatenate(theta) if len(theta)>0 else np.zeros(0)
    trans = np.concatenate(trans) if len(trans)>0 else np.zeros(0)
    Xs = np.concatenate([X_map[pid] for pid in point_ids]) if len(point_ids)>0 else np.zeros(0)
    return np.concatenate([theta, trans, Xs])

def unpack_params(vec: np.ndarray, cam_params, cam_ids, point_ids):
    idx = 0
    for cid in cam_ids:
        r = vec[idx:idx+3]; idx+=3
        t = vec[idx:idx+3]; idx+=3
        cam_params[cid]['R'] = rodrigues_to_R(r)
        cam_params[cid]['t'] = t
    X_map = {}
    for pid in point_ids:
        X_map[pid] = vec[idx:idx+3]; idx+=3
    return cam_params, X_map

def ba_residuals(param_vec, obs: List[Tuple[int,int,np.ndarray]], K_map, cam_params, cam_ids, point_ids):
    cam_params_upd = {k: {'R': v['R'].copy(), 't': v['t'].copy()} for k,v in cam_params.items()}
    cam_params_upd, X_map = unpack_params(param_vec, cam_params_upd, cam_ids, point_ids)
    res = []
    for (cid, pid, uv) in obs:
        K = K_map[cid]; R = cam_params_upd[cid]['R']; t = cam_params_upd[cid]['t']
        P = projection_matrix(K, R, t)
        X = X_map[pid]
        x = P @ np.hstack([X,1.0]); u = x[0]/x[2]; v = x[1]/x[2]
        res.extend([u - uv[0], v - uv[1]])
    return np.array(res, dtype=float)

def run_global_ba(obs, K_map, cam_params, X_init):
    try:
        from scipy.optimize import least_squares
    except Exception as e:
        print('[BA] SciPy not available; skipping BA. Reason:', e)
        return cam_params, X_init, None
    cam_ids = sorted(list({cid for cid,_,_ in obs}))
    point_ids = sorted(list(X_init.keys()))
    if len(cam_ids) == 0 or len(point_ids) == 0:
        print('[BA] Not enough variables; skipping.')
        return cam_params, X_init, None
    p0 = pack_params(cam_params, X_init, cam_ids, point_ids)
    fun = lambda p: ba_residuals(p, obs, K_map, cam_params, cam_ids, point_ids)
    res = least_squares(fun, p0, verbose=1, xtol=1e-8, ftol=1e-8, max_nfev=100)
    cam_params_opt = {cid: {'R': cam_params[cid]['R'].copy(), 't': cam_params[cid]['t'].copy()} for cid in cam_ids}
    cam_params_opt, X_opt = unpack_params(res.x, cam_params_opt, cam_ids, point_ids)
    return cam_params_opt, X_opt, res

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True, help='Dir with cameras.bin, images.bin')
    ap.add_argument('--csv', required=True, help='CSV: instance_id, frame, image_name, x, y')
    ap.add_argument('--out_prefix', default='triangulated', help='Output prefix')
    ap.add_argument('--min_views', type=int, default=2, help='Min views per instance')
    # RANSAC
    ap.add_argument('--ransac_thresh', type=float, default=2.0, help='RANSAC reprojection threshold (px)')
    ap.add_argument('--ransac_iters', type=int, default=200, help='RANSAC max iterations (sampled pairs)')
    ap.add_argument('--min_inliers', type=int, default=2, help='RANSAC: minimum inliers to accept point')
    ap.add_argument('--pos_depth_ratio', type=float, default=0.5, help='Require positive depth on this fraction of views')
    # BA
    ap.add_argument('--do_ba', action='store_true', help='Run global BA (SciPy required)')
    args = ap.parse_args()
    os.makedirs(args.out_prefix.split('/')[0], exist_ok=True)

    cam_path = Path(args.model_dir) / 'cameras.bin'
    img_path = Path(args.model_dir) / 'images.bin'
    assert cam_path.exists() and img_path.exists(), 'cameras.bin / images.bin not found.'

    cams = read_cameras_bin(str(cam_path))
    imgs = read_images_bin(str(img_path))

    # Intrinsics per camera_id; Extrinsics per image_id
    K_by_cam = {cid: camera_K(cam) for cid, cam in cams.items()}
    name_to_image_id = {d['name']: img_id for img_id, d in imgs.items()}
    image_id_to_pose = {img_id: {'R': qvec2rotmat(d['qvec']), 't': d['tvec'], 'camera_id': d['camera_id']}
                        for img_id, d in imgs.items()}

    df = pd.read_csv(args.csv)
    req = {'instance_id','frame','image_name','x','y'}
    if not req.issubset(df.columns):
        raise ValueError(f'CSV must contain columns: {sorted(list(req))}')
    df = df.sort_values(['instance_id','frame']).reset_index(drop=True)

    # Build per-instance view lists
    instances = {}
    for iid, g in df.groupby('instance_id'):
        P_list = []; uv_list = []; used_image_ids = []
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
        if len(P_list) >= max(2, args.min_views):
            instances[int(iid)] = (P_list, uv_list, used_image_ids)

    # RANSAC triangulation per instance
    X_map = {}
    stats = []
    for iid in sorted(instances.keys()):
        P_list, uv_list, used_img_ids = instances[iid]
        Xr, inliers, info = ransac_triangulate(
            P_list, uv_list,
            reproj_thresh_px=args.ransac_thresh,
            max_iters=args.ransac_iters,
            min_inliers=args.min_inliers,
            require_pos_depth_ratio=args.pos_depth_ratio
        )
        if int(inliers.sum()) < args.min_inliers:
            continue
        X_map[iid] = Xr
        stats.append({'instance_id': iid,
                      'num_obs': int(len(P_list)),
                      'inliers': int(inliers.sum()),
                      'mean_reproj': info['mean_err'],
                      'max_reproj': info['max_err'],
                      'fallback_no_ransac': bool(info['fallback'])})

    # Save triangulated CSV + PLY
    out_csv = f'{args.out_prefix}_points.csv'
    recs = []
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

    if len(recs) > 0:
        pts = np.vstack([X_map[s['instance_id']] for s in stats])
        cols = np.array([color_from_id(s['instance_id']) for s in stats], dtype=np.uint8)
        out_ply = f'{args.out_prefix}_points.ply'
        save_ply(pts, cols, out_ply)
        print(f'[*] Saved: {out_ply}')

    # Build BA problem (observations only for triangulated points)
    if args.do_ba and len(X_map) > 0:
        try:
            from scipy.optimize import least_squares  # check availability
        except Exception as e:
            print('[BA] SciPy not available; skipping BA. Reason:', e)
        else:
            # Per-image extrinsics as variables; intrinsics fixed
            # Only include images that observe at least one triangulated point
            obs = []
            involved_images = set()
            for _, row in df.iterrows():
                iid = int(row['instance_id'])
                if iid not in X_map: 
                    continue
                name = str(row['image_name'])
                if name not in name_to_image_id: 
                    continue
                img_id = name_to_image_id[name]
                uv = np.array([row['x'], row['y']], dtype=float)
                obs.append((img_id, iid, uv))
                involved_images.add(img_id)

            involved_images = sorted(list(involved_images))
            if len(involved_images) == 0:
                print('[BA] No involved images; skipping.')
            else:
                cam_params = {img_id: {'R': image_id_to_pose[img_id]['R'].copy(),
                                       't': image_id_to_pose[img_id]['t'].copy()}
                              for img_id in involved_images}
                K_map = {img_id: K_by_cam[image_id_to_pose[img_id]['camera_id']] for img_id in involved_images}
                obs_sel = [o for o in obs if o[0] in cam_params]

                print('[*] Running global BA on involved images and triangulated points...')
                cam_params_opt, X_opt, res = run_global_ba(obs_sel, K_map, cam_params, X_map)
                if res is not None:
                    print(f'[*] BA done. Final cost: {res.cost:.6f}, nfev={res.nfev}')

                    # Save BA outputs
                    out_npz = f'{args.out_prefix}_ba_results.npz'
                    np.savez(out_npz,
                             points_ids=np.array(sorted(list(X_opt.keys())), dtype=int),
                             points_xyz=np.vstack([X_opt[i] for i in sorted(list(X_opt.keys()))]) if len(X_opt)>0 else np.zeros((0,3)),
                             image_ids=np.array(involved_images, dtype=int),
                             Rs=np.stack([cam_params_opt[i]['R'] for i in involved_images]) if len(involved_images)>0 else np.zeros((0,3,3)),
                             ts=np.stack([cam_params_opt[i]['t'] for i in involved_images]) if len(involved_images)>0 else np.zeros((0,3)))
                    print(f'[*] Saved: {out_npz}')

                    # Also save BA-refined points as CSV for convenience
                    out_csv_ba = f'{args.out_prefix}_points_ba.csv'
                    recs_ba = []
                    for pid in sorted(list(X_opt.keys())):
                        Xb = X_opt[pid]
                        recs_ba.append({'instance_id': pid, 'X': Xb[0], 'Y': Xb[1], 'Z': Xb[2]})
                    pd.DataFrame.from_records(recs_ba).to_csv(out_csv_ba, index=False)
                    print(f'[*] Saved: {out_csv_ba}')

    print('[*] Done.')

if __name__ == '__main__':
    main()
