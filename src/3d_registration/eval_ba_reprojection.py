#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate reprojection error using BA results (poses+points).
Inputs:
  - --model_dir: path containing cameras.bin, images.bin
  - --csv:       observation csv (instance_id, frame, image_name, x, y)
  - --ba_npz:    BA results npz (saved by the previous script)
Outputs:
  - {out_prefix}_ba_reproj_per_obs.csv
  - {out_prefix}_ba_reproj_per_point.csv
  - {out_prefix}_ba_reproj_per_image.csv
Prints overall summary to stdout.
"""

import os
import struct
import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

# ============================
# COLMAP binary readers (minimal)
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
            cams[cam_id] = {
                'model_id': model_id,
                'model': model_name,
                'width': int(width),
                'height': int(height),
                'params': np.array(params, dtype=float)
            }
    return cams

def read_images_bin(path: str):
    imgs = {}
    with open(path, 'rb') as f:
        num_images = _read_uint64(f)
        for _ in range(num_images):
            img_id = _read_uint32(f)
            # qvec, tvec, cam_id, name, points2D (skip)
            _ = [_read_double(f) for __ in range(4)]  # qvec
            _ = [_read_double(f) for __ in range(3)]  # tvec
            cam_id = _read_uint32(f)
            name = _read_string(f)
            num_points2D = _read_uint64(f)
            for __ in range(num_points2D):
                _ = _read_double(f); _ = _read_double(f); _ = _read_int64(f)
            imgs[img_id] = {'camera_id': cam_id, 'name': name}
    return imgs

# ============================
# Geometry
# ============================

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

# ============================
# Evaluation
# ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True, help='Dir with cameras.bin, images.bin')
    ap.add_argument('--csv', required=True, help='CSV: instance_id, frame, image_name, x, y')
    ap.add_argument('--ba_npz', required=True, help='BA results npz (points_ids, points_xyz, image_ids, Rs, ts)')
    ap.add_argument('--out_prefix', default='ba_eval', help='Output prefix for CSVs')
    ap.add_argument('--report_topk', type=int, default=20, help='Print top-k largest errors')
    args = ap.parse_args()

    cam_path = Path(args.model_dir) / 'cameras.bin'
    img_path = Path(args.model_dir) / 'images.bin'
    assert cam_path.exists() and img_path.exists(), 'cameras.bin / images.bin not found.'

    # Load intrinsics & image->camera mapping
    cams = read_cameras_bin(str(cam_path))
    imgs = read_images_bin(str(img_path))
    image_id_to_name = {iid: d['name'] for iid, d in imgs.items()}
    image_id_to_cam = {iid: d['camera_id'] for iid, d in imgs.items()}
    K_by_cam = {cid: camera_K(cams[cid]) for cid in cams.keys()}

    # Load BA results
    data = np.load(args.ba_npz, allow_pickle=False)
    points_ids = data['points_ids']           # shape [P]
    points_xyz = data['points_xyz']           # shape [P,3]
    image_ids  = data['image_ids']            # shape [M]
    Rs         = data['Rs']                   # shape [M,3,3]
    ts         = data['ts']                   # shape [M,3]
    # Build lookups
    idx_of_image = {int(img_id): i for i, img_id in enumerate(image_ids.tolist())}
    idx_of_point = {int(pid): i for i, pid in enumerate(points_ids.tolist())}

    # Load observations
    df = pd.read_csv(args.csv)
    required_cols = {'instance_id','image_name','x','y'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f'CSV must contain columns: {sorted(list(required_cols))}')
    # Map image_name -> image_id
    name_to_image_id = {name: iid for iid, name in image_id_to_name.items()}

    # Filter to (1) points present in BA, (2) images present in BA
    rows = []
    missing_images = 0
    for _, row in df.iterrows():
        pid = int(row['instance_id'])
        if pid not in idx_of_point:
            continue
        name = str(row['image_name'])
        if name not in name_to_image_id:
            missing_images += 1
            continue
        img_id = int(name_to_image_id[name])
        if img_id not in idx_of_image:
            # image exists in model, but BA did not include it
            continue

        im_idx = idx_of_image[img_id]
        pt_idx = idx_of_point[pid]
        R = Rs[im_idx]; t = ts[im_idx]
        cam_id = image_id_to_cam[img_id]
        if cam_id not in K_by_cam:
            # Should not happen; skip if camera not found
            continue
        K = K_by_cam[cam_id]
        P = projection_matrix(K, R, t)

        X = points_xyz[pt_idx]
        x = P @ np.hstack([X, 1.0])
        u_proj = x[0] / x[2]
        v_proj = x[1] / x[2]
        u_obs = float(row['x'])
        v_obs = float(row['y'])
        err = float(np.sqrt((u_proj - u_obs)**2 + (v_proj - v_obs)**2))

        rows.append({
            'image_id': img_id,
            'image_name': name,
            'camera_id': cam_id,
            'instance_id': pid,
            'u_obs': u_obs, 'v_obs': v_obs,
            'u_proj': float(u_proj), 'v_proj': float(v_proj),
            'reproj_err': err
        })

    if len(rows) == 0:
        print('[!] No overlapping observations between BA results and CSV.')
        if missing_images > 0:
            print(f'    Note: {missing_images} rows referenced image names not found in images.bin')
        return

    per_obs = pd.DataFrame(rows)
    out_obs = f'{args.out_prefix}_ba_reproj_per_obs.csv'
    per_obs.to_csv(out_obs, index=False)
    print(f'[*] Saved: {out_obs} ({len(per_obs)} observations)')

    # Per-point stats
    grp_pt = per_obs.groupby('instance_id')['reproj_err']
    per_point = pd.DataFrame({
        'instance_id': grp_pt.mean().index.astype(int),
        'num_obs': grp_pt.size().values.astype(int),
        'mean_err': grp_pt.mean().values,
        'median_err': grp_pt.median().values,
        'max_err': grp_pt.max().values,
        'min_err': grp_pt.min().values,
        'std_err': grp_pt.std(ddof=0).values
    }).sort_values('mean_err').reset_index(drop=True)
    out_pt = f'{args.out_prefix}_ba_reproj_per_point.csv'
    per_point.to_csv(out_pt, index=False)
    print(f'[*] Saved: {out_pt} ({len(per_point)} points)')

    # Per-image stats
    grp_im = per_obs.groupby(['image_id','image_name'])['reproj_err']
    per_image = pd.DataFrame({
        'image_id': [i for (i,_) in grp_im.mean().index],
        'image_name': [n for (_,n) in grp_im.mean().index],
        'num_obs': grp_im.size().values.astype(int),
        'mean_err': grp_im.mean().values,
        'median_err': grp_im.median().values,
        'max_err': grp_im.max().values,
        'min_err': grp_im.min().values,
        'std_err': grp_im.std(ddof=0).values
    }).sort_values('mean_err').reset_index(drop=True)
    out_im = f'{args.out_prefix}_ba_reproj_per_image.csv'
    per_image.to_csv(out_im, index=False)
    print(f'[*] Saved: {out_im} ({len(per_image)} images)')

    # Overall summary
    errs = per_obs['reproj_err'].values
    print('\n=== Overall Reprojection Error (pixels) ===')
    print(f'Count : {len(errs)}')
    print(f'Mean  : {errs.mean():.6f}')
    print(f'Median: {np.median(errs):.6f}')
    print(f'Std   : {errs.std():.6f}')
    print(f'Max   : {errs.max():.6f}')
    # Top-K largest errors
    if args.report_topk > 0:
        topk = per_obs.nlargest(args.report_topk, 'reproj_err')[[
            'reproj_err','image_id','image_name','instance_id','u_obs','v_obs','u_proj','v_proj'
        ]]
        print(f'\nTop-{min(args.report_topk, len(topk))} largest errors:')
        with pd.option_context('display.max_rows', None, 'display.width', 200):
            print(topk.to_string(index=False))

if __name__ == '__main__':
    main()
