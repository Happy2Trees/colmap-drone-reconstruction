from typing import Optional, Dict, Any, List, Tuple
import os
import json
import numpy as np
from dataclasses import dataclass


# ===============================
# Point-set alignment (Umeyama + IRLS)
# ===============================

@dataclass
class AlignResult:
    scale: float
    R: np.ndarray
    t: np.ndarray
    X: np.ndarray  # aligned A
    loss: str
    with_scale: bool
    iterations: int


def weighted_umeyama(
    A: np.ndarray,
    B: np.ndarray,
    w: np.ndarray,
    with_scale: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    w = np.asarray(w, float)
    w = np.clip(w, 1e-12, None)
    W = w / w.sum()
    muA = (W[:, None] * A).sum(axis=0)
    muB = (W[:, None] * B).sum(axis=0)
    A0, B0 = A - muA, B - muB
    Sigma = B0.T @ (W[:, None] * A0)
    U, D, Vt = np.linalg.svd(Sigma)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    if with_scale:
        varA = (W * (A0 ** 2).sum(axis=1)).sum()
        s = D.sum() / (varA + 1e-15)
    else:
        s = 1.0
    t = muB - s * (muA @ R.T)
    return float(s), R, t


def transform_points(A: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return s * (A @ R.T) + t


def l2_align(A: np.ndarray, B: np.ndarray, with_scale: bool = True) -> AlignResult:
    w = np.ones(len(A), dtype=float)
    s, R, t = weighted_umeyama(A, B, w, with_scale)
    X = transform_points(A, s, R, t)
    return AlignResult(scale=float(s), R=R, t=t, X=X, loss="L2", with_scale=with_scale, iterations=1)


def _auto_delta_from_residuals(r: np.ndarray) -> float:
    med = np.median(r)
    mad = np.median(np.abs(r - med)) * 1.4826
    if mad <= 1e-12:
        mad = np.std(r) + 1e-9
    return 2.0 * mad


def irls_align(
    A: np.ndarray,
    B: np.ndarray,
    with_scale: bool = True,
    loss: str = "huber",
    delta: Optional[float] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> AlignResult:
    init = l2_align(A, B, with_scale)
    if delta is None:
        r0 = np.linalg.norm(init.X - B, axis=1)
        delta = _auto_delta_from_residuals(r0)
    w = np.ones(len(A), dtype=float)
    prev_obj = np.inf
    s, R, t = init.scale, init.R, init.t
    iters = 0
    for it in range(int(max_iter)):
        s, R, t = weighted_umeyama(A, B, w, with_scale)
        X = transform_points(A, s, R, t)
        r = np.linalg.norm(X - B, axis=1)
        if loss == "huber":
            obj = (np.where(r <= delta, 0.5 * r ** 2, delta * (r - 0.5 * delta))).sum()
            w = np.where(r <= delta, 1.0, delta / np.clip(r, 1e-12, None))
        elif loss == "cauchy":
            obj = (0.5 * delta ** 2 * np.log1p((r / delta) ** 2)).sum()
            w = 1.0 / (1.0 + (r / delta) ** 2)
        else:
            raise ValueError("loss must be 'huber' or 'cauchy'")
        iters = it + 1
        if abs(prev_obj - obj) < tol:
            break
        prev_obj = obj
    X = transform_points(A, s, R, t)
    return AlignResult(scale=float(s), R=R, t=t, X=X, loss=loss, with_scale=with_scale, iterations=iters)


def l1_metrics(X: np.ndarray, B: np.ndarray):
    E = X - B
    d1 = np.abs(E).sum(axis=1)
    mae1 = float(d1.mean())
    med1 = float(np.median(d1))
    max1 = float(d1.max())
    return d1, mae1, med1, max1


def sim3_to_matrix(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def _load_ply_points(path: str) -> np.ndarray:
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError(f"open3d import failed: {e}")
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=float)


def _nn_pairs(A: np.ndarray, B: np.ndarray, bidir: bool = True, max_dist: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError(f"open3d import failed: {e}")
    pA = o3d.geometry.PointCloud(); pA.points = o3d.utility.Vector3dVector(A)
    pB = o3d.geometry.PointCloud(); pB.points = o3d.utility.Vector3dVector(B)
    kdtA = o3d.geometry.KDTreeFlann(pA)
    kdtB = o3d.geometry.KDTreeFlann(pB)
    fwd_j = np.full(A.shape[0], -1, dtype=int)
    fwd_d = np.full(A.shape[0], np.inf, dtype=float)
    for i, p in enumerate(A):
        _, idx, d2 = kdtB.search_knn_vector_3d(p, 1)
        if len(idx) > 0:
            fwd_j[i] = int(idx[0])
            fwd_d[i] = float(np.sqrt(d2[0]))
    if bidir:
        rev_i = np.full(B.shape[0], -1, dtype=int)
        for j, q in enumerate(B):
            _, idx, _ = kdtA.search_knn_vector_3d(q, 1)
            if len(idx) > 0:
                rev_i[j] = int(idx[0])
        ii, jj = [], []
        for i, j in enumerate(fwd_j):
            if j < 0:
                continue
            if max_dist is not None and fwd_d[i] > max_dist:
                continue
            if rev_i[j] == i:
                ii.append(i); jj.append(j)
        return np.array(ii, dtype=int), np.array(jj, dtype=int)
    else:
        # one-way NN; dedup by keeping best per target
        best_i = {}
        for i, j in enumerate(fwd_j):
            if j < 0:
                continue
            if max_dist is not None and fwd_d[i] > max_dist:
                continue
            if (j not in best_i) or (fwd_d[i] < fwd_d[best_i[j]]):
                best_i[j] = i
        jj = sorted(best_i.keys())
        ii = [best_i[j] for j in jj]
        return np.array(ii, dtype=int), np.array(jj, dtype=int)


def register_and_eval(
    source_ply: str,
    target_ply: str,
    voxel_size: float = 0.0,
    ransac_n: int = 0,
    distance_thresh_ratio: float = 0.0,
    icp_max_iter: int = 0,
    save_debug_dir: Optional[str] = None,
    aligned_ply_out: Optional[str] = None,
    with_scaling: bool = True,
    # New options for PLY-based Umeyama/IRLS evaluation
    source_indices: Optional[List[int]] = None,
    target_indices: Optional[List[int]] = None,
    methods: Optional[List[str]] = None,  # e.g., ["L2","huber","cauchy"]
    with_scale_options: Optional[List[bool]] = None,  # [True, False]
    match_mode: Optional[str] = None,  # 'explicit' or 'nn' (auto if indices provided)
    nn_bidir: bool = True,
    nn_max_dist: Optional[float] = None,
) -> Dict[str, Any]:
    """PLY-based evaluation using Umeyama/IRLS alignment.

    Notes:
      - If source_indices and target_indices are provided with equal length, they
        define explicit correspondences (match_mode='explicit').
      - Otherwise, nearest-neighbor correspondences are computed between points
        (match_mode='nn'), optionally filtered by mutual check and max distance.
      - voxel_size>0 will downsample both clouds for NN pairing only; alignment is
        estimated from those pairs but applied to the full source for saving.
    """
    try:
        import open3d as o3d  # for PLY I/O and KDTree
    except Exception as e:
        return {"skipped": True, "reason": f"open3d import failed: {e}"}

    if save_debug_dir:
        os.makedirs(save_debug_dir, exist_ok=True)

    # Load full-resolution points
    src = o3d.io.read_point_cloud(source_ply)
    tgt = o3d.io.read_point_cloud(target_ply)
    A_full = np.asarray(src.points, dtype=float)
    B_full = np.asarray(tgt.points, dtype=float)

    if methods is None:
        methods = ["L2", "huber", "cauchy"]
    if with_scale_options is None:
        with_scale_options = [True, False]

    # Determine correspondences and the points used for estimation
    used_src_idx: np.ndarray
    used_tgt_idx: np.ndarray
    A_est: np.ndarray
    B_est: np.ndarray

    source_indices = [0,1,2,3,4,5,6,7,8]
    target_indices = [0,1,2,3,4,5,6,7,8]
    if source_indices is not None and target_indices is not None:
        if len(source_indices) != len(target_indices):
            raise ValueError("source_indices and target_indices must have the same length.")
        used_src_idx = np.array(source_indices, dtype=int)
        used_tgt_idx = np.array(target_indices, dtype=int)
        if used_src_idx.size == 0:
            raise ValueError("No correspondences provided in indices.")
        A_est = A_full[used_src_idx]
        B_est = B_full[used_tgt_idx]
        mmode = 'explicit'
    else:
        # NN correspondences, optionally with voxel downsampling for pairing speed
        if voxel_size and voxel_size > 0:
            vs = float(voxel_size)
            src_d = src.voxel_down_sample(vs)
            tgt_d = tgt.voxel_down_sample(vs)
            A_use = np.asarray(src_d.points, dtype=float)
            B_use = np.asarray(tgt_d.points, dtype=float)
        else:
            A_use = A_full
            B_use = B_full
        ii, jj = _nn_pairs(A_use, B_use, bidir=bool(nn_bidir), max_dist=nn_max_dist)
        if ii.size == 0:
            return {"skipped": True, "reason": "No NN correspondences found."}
        # Use the matched sets directly for estimation
        A_est = A_use[ii]
        B_est = B_use[jj]
        # Keep track of which points used (note: indices refer to downsampled clouds if used)
        used_src_idx = ii
        used_tgt_idx = jj
        mmode = 'nn'

    # Evaluate across methods and scale options
    summary_rows: List[List] = []
    results_by_mode: Dict[str, Any] = {}
    best_key = None
    best_mae = np.inf

    for ws in with_scale_options:
        for m in methods:
            m_low = m.lower()
            mode_name = f"{m} ({'Sim3' if ws else 'SE3'})"
            if m_low == 'l2':
                res = l2_align(A_est, B_est, with_scale=ws)
            elif m_low in ('huber','cauchy'):
                res = irls_align(A_est, B_est, with_scale=ws, loss=m_low)
            else:
                raise ValueError("Unknown method: %s" % m)
            d1, mae1, med1, max1 = l1_metrics(res.X, B_est)
            T = sim3_to_matrix(res.scale, res.R, res.t)

            summary_rows.append([mode_name, ws, res.loss, float(res.scale), int(res.iterations), mae1, med1, max1])
            results_by_mode[mode_name] = {
                "scale": float(res.scale),
                "R": res.R.tolist(),
                "t": res.t.tolist(),
                "T": T.tolist(),
                "iterations": int(res.iterations),
                "loss": res.loss,
                "with_scale": bool(ws),
                "mae_l1": float(mae1),
                "median_l1": float(med1),
                "max_l1": float(max1),
                "num_pairs": int(A_est.shape[0]),
                "match_mode": mmode,
            }
            if mae1 < best_mae:
                best_mae = mae1
                best_key = mode_name

    # Save aligned source PLY for the best mode if requested
    if aligned_ply_out and best_key is not None:
        bk = results_by_mode[best_key]
        T = np.array(bk["T"], dtype=float)
        # Apply to full-resolution source
        A_aligned = (A_full @ (T[:3,:3].T)) + T[:3,3]
        try:
            import open3d as o3d  # type: ignore
            p_aligned = o3d.geometry.PointCloud()
            p_aligned.points = o3d.utility.Vector3dVector(A_aligned)
            # Preserve colors if any
            try:
                if len(src.colors) > 0:
                    p_aligned.colors = src.colors
            except Exception:
                pass
            o3d.io.write_point_cloud(aligned_ply_out, p_aligned)
        except Exception as e:
            results_by_mode[best_key]["aligned_ply_error"] = str(e)

    stats = {
        "summary": [
            {
                "Mode": row[0],
                "with_scale": row[1],
                "loss": row[2],
                "Scale s": row[3],
                "Iterations": row[4],
                "MAE_L1": row[5],
                "Median_L1": row[6],
                "Max_L1": row[7],
            }
            for row in summary_rows
        ],
        "results_by_mode": results_by_mode,
        "best_mode": best_key,
        "num_points_source": int(A_full.shape[0]),
        "num_points_target": int(B_full.shape[0]),
        "match_mode": mmode,
    }

    if save_debug_dir:
        try:
            with open(os.path.join(save_debug_dir, "register_eval.json"), "w") as f:
                json.dump(stats, f, indent=2)
            np.save(os.path.join(save_debug_dir, "used_src_idx.npy"), used_src_idx)
            np.save(os.path.join(save_debug_dir, "used_tgt_idx.npy"), used_tgt_idx)
        except Exception:
            pass

    return stats
