#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPZ(points_ids, points_xyz[*,3]) -> PLY(ASCII) 변환 스크립트

예시:
  python npz_to_ply.py --npz triangulated_ba_results.npz --out points.ply --color id
  python npz_to_ply.py --npz triangulated_ba_results.npz --out points_white.ply --color white
  python npz_to_ply.py --npz triangulated_ba_results.npz --out points_rand.ply --color random --seed 42
  python npz_to_ply.py --npz triangulated_ba_results.npz --out points_zgray.ply --color z
"""

import argparse
import numpy as np
from typing import Tuple

def color_from_id(idx: int) -> Tuple[int,int,int]:
    """안 겹치게 대략적인 고정색 생성 (HSV 기반)"""
    h = (idx * 47) % 360
    s = 0.9; v = 0.95
    c = v * s
    x = c * (1 - abs((h/60) % 2 - 1))
    m = v - c
    if   0 <= h < 60:   r,g,b = c,x,0
    elif 60 <= h < 120: r,g,b = x,c,0
    elif 120<= h <180:  r,g,b = 0,c,x
    elif 180<= h <240:  r,g,b = 0,x,c
    elif 240<= h <300:  r,g,b = x,0,c
    else:               r,g,b = c,0,x
    return (int((r+m)*255), int((g+m)*255), int((b+m)*255))

def save_ply(points_xyz: np.ndarray, colors_rgb: np.ndarray, path: str):
    assert points_xyz.shape[0] == colors_rgb.shape[0]
    N = points_xyz.shape[0]
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {N}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for (x,y,z), (r,g,b) in zip(points_xyz, colors_rgb.astype(np.uint8)):
            f.write(f'{float(x)} {float(y)} {float(z)} {int(r)} {int(g)} {int(b)}\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='BA 결과 NPZ: points_ids, points_xyz[, image_ids, Rs, ts]')
    ap.add_argument('--out', required=True, help='출력 PLY 경로')
    ap.add_argument('--color', choices=['id','white','random','z'], default='id',
                    help='색상 모드: id=포인트ID기반 고정색, white=흰색, random=랜덤, z=깊이 그레이스케일')
    ap.add_argument('--seed', type=int, default=123, help='random 색상 시드')
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    # points_xyz는 [N,3], points_ids는 [N] (없으면 0..N-1로 대체)
    if 'points_xyz' not in data:
        raise ValueError('NPZ에 points_xyz가 없습니다.')
    points_xyz = data['points_xyz']
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError('points_xyz shape이 (N,3)이 아닙니다.')
    N = points_xyz.shape[0]
    if N == 0:
        raise ValueError('포인트가 비어 있습니다.')

    if 'points_ids' in data:
        points_ids = data['points_ids'].astype(int)
        if points_ids.shape[0] != N:
            raise ValueError('points_ids 길이와 points_xyz 개수가 일치하지 않습니다.')
    else:
        points_ids = np.arange(N, dtype=int)

    # 색상 결정
    if args.color == 'white':
        colors = np.full((N,3), 255, dtype=np.uint8)
    elif args.color == 'random':
        rng = np.random.default_rng(args.seed)
        colors = rng.integers(low=0, high=256, size=(N,3), dtype=np.uint8)
    elif args.color == 'z':
        z = points_xyz[:,2]
        zmin, zmax = float(np.min(z)), float(np.max(z))
        if zmax - zmin < 1e-12:
            gray = np.full(N, 200, dtype=np.uint8)
        else:
            norm = (z - zmin) / (zmax - zmin)
            gray = np.clip((norm * 255.0), 0, 255).astype(np.uint8)
        colors = np.stack([gray, gray, gray], axis=1)
    else:  # 'id'
        colors = np.zeros((N,3), dtype=np.uint8)
        for i, pid in enumerate(points_ids.tolist()):
            colors[i] = color_from_id(int(pid))

    save_ply(points_xyz.astype(float), colors.astype(np.uint8), args.out)
    print(f'[*] Saved PLY: {args.out}  (N={N}, color="{args.color}")')

if __name__ == '__main__':
    main()
