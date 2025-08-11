"""
Similarity ICP utilities to align two point clouds with scale.

Provides:
- Umeyama closed-form similarity transform estimation (s, R, t)
- Trimmed ICP with scale using NumPy + SciPy KDTree
- Optional Open3D pipeline (global FPFH + ICP with scaling) if Open3D is installed

Main entry: align_point_clouds(source, target, ...)

All point arrays are expected as float32/float64 NumPy arrays of shape (N, 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

try:
    # SciPy is in requirements; prefer cKDTree for speed
    from scipy.spatial import cKDTree as KDTree
except Exception:  # pragma: no cover
    KDTree = None  # type: ignore


def _as_numpy_xyz(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"Expected array of shape (N,3), got {x.shape}")
    return x.astype(np.float64, copy=False)


@dataclass
class Similarity:
    s: float
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    def matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.s * self.R
        T[:3, 3] = self.t
        return T


def apply_similarity(points: np.ndarray, sim: Similarity) -> np.ndarray:
    P = _as_numpy_xyz(points)
    return sim.s * (P @ sim.R.T) + sim.t


def umeyama_similarity(
    source: np.ndarray,
    target: np.ndarray,
    with_scaling: bool = True,
    eps: float = 1e-12,
) -> Similarity:
    """
    Estimate similarity transform that maps source -> target using Umeyama (1991).

    Args:
        source: (N,3) source points
        target: (N,3) target points (corresponding 1:1)
        with_scaling: if False, constrain scale to 1
        eps: numerical stability epsilon

    Returns:
        Similarity(s, R, t)
    """
    X = _as_numpy_xyz(source)
    Y = _as_numpy_xyz(target)
    if X.shape[0] != Y.shape[0] or X.shape[0] < 3:
        raise ValueError("Need same number of >=3 correspondences for Umeyama")

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    # Covariance
    Sigma = (Yc.T @ Xc) / X.shape[0]

    U, D, Vt = np.linalg.svd(Sigma)
    # Ensure a proper rotation with det = +1
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    if with_scaling:
        var_x = (Xc ** 2).sum() / X.shape[0]
        # trace(D @ S) == sum(D * diag(S))
        s = (D * np.diag(S)).sum() / max(var_x, eps)
    else:
        s = 1.0

    t = mu_y - s * (R @ mu_x)

    return Similarity(float(s), R, t)


def _estimate_initial_scale(source: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    """Robust initial scale estimate using RMS radius to centroid."""
    X = _as_numpy_xyz(source)
    Y = _as_numpy_xyz(target)
    rx = np.sqrt(((X - X.mean(0)) ** 2).sum(axis=1) + eps)
    ry = np.sqrt(((Y - Y.mean(0)) ** 2).sum(axis=1) + eps)
    # Use median to reduce outlier impact
    med_rx = float(np.median(rx))
    med_ry = float(np.median(ry))
    if med_rx < eps:
        return 1.0
    return med_ry / med_rx


def initial_guess_from_pca(source: np.ndarray, target: np.ndarray) -> Similarity:
    """
    Build a crude initial guess using centroid alignment, RMS scale, and PCA axes.
    Useful when Open3D is unavailable and rotation differs moderately.
    """
    X = _as_numpy_xyz(source)
    Y = _as_numpy_xyz(target)
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    # PCA axes
    Cx = (Xc.T @ Xc) / max(Xc.shape[0], 1)
    Cy = (Yc.T @ Yc) / max(Yc.shape[0], 1)
    _, Ux = np.linalg.eigh(Cx)  # columns eigenvectors (ascending)
    _, Uy = np.linalg.eigh(Cy)
    Ux = Ux[:, ::-1]
    Uy = Uy[:, ::-1]
    R0 = Uy @ Ux.T
    if np.linalg.det(R0) < 0:
        Uy[:, -1] *= -1
        R0 = Uy @ Ux.T

    s0 = _estimate_initial_scale(X, Y)
    t0 = mu_y - s0 * (R0 @ mu_x)
    return Similarity(float(s0), R0, t0)


def _build_kdtree(points: np.ndarray) -> Any:
    if KDTree is None:
        raise RuntimeError("SciPy KDTree not available. Install scipy.")
    return KDTree(_as_numpy_xyz(points))


def compose_similarity(delta: Similarity, base: Similarity) -> Similarity:
    """Compose two similarity transforms: result = delta ∘ base."""
    s = float(delta.s * base.s)
    R = delta.R @ base.R
    t = delta.s * (delta.R @ base.t) + delta.t
    return Similarity(s, R, t)


def _trim_by_percentile(distances: np.ndarray, keep_ratio: float) -> np.ndarray:
    n = distances.shape[0]
    if keep_ratio >= 1.0:
        return np.ones(n, dtype=bool)
    k = max(3, int(np.ceil(n * keep_ratio)))
    if k >= n:
        return np.ones(n, dtype=bool)
    thresh = np.partition(distances, k - 1)[k - 1]
    return distances <= thresh


def icp_similarity(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    inlier_ratio: float = 0.8,
    init: Optional[Similarity] = None,
    max_pairs: Optional[int] = 50000,
    verbose: bool = False,
) -> Tuple[Similarity, Dict[str, Any]]:
    """
    Trimmed ICP with scaling using NumPy + SciPy KDTree.

    Args:
        source: (Ns,3) source points
        target: (Nt,3) target points
        max_iterations: maximum ICP iterations
        tolerance: convergence threshold on parameter update
        inlier_ratio: keep this fraction of closest pairs each iteration (0<r<=1)
        init: optional initial similarity (s,R,t). If None, s is initialized via RMS radii, R=I, t=0.
        max_pairs: randomly sample up to this many source points per iteration for speed
        verbose: print per-iteration diagnostics

    Returns:
        (Similarity, metrics)
    """
    X_full = _as_numpy_xyz(source)
    Y_full = _as_numpy_xyz(target)

    if X_full.shape[0] < 3 or Y_full.shape[0] < 3:
        raise ValueError("Both point clouds need at least 3 points")

    if init is None:
        s0 = _estimate_initial_scale(X_full, Y_full)
        R0 = np.eye(3)
        # Initialize translation from centroids given s0,R0
        t0 = Y_full.mean(axis=0) - s0 * (R0 @ X_full.mean(axis=0))
        sim = Similarity(float(s0), R0, t0)
    else:
        sim = init

    # Pre-build KDTree on target
    kdtree = _build_kdtree(Y_full)

    # Sampling indices (fixed across iterations for stability)
    if max_pairs is not None and X_full.shape[0] > max_pairs:
        rng = np.random.default_rng(42)
        sample_idx = np.asarray(rng.choice(X_full.shape[0], size=max_pairs, replace=False))
        X = X_full[sample_idx]
    else:
        sample_idx = None
        X = X_full

    history = {
        "rmse": [],
        "inliers": [],
        "s": [],
    }

    for it in range(max_iterations):
        X_trans = apply_similarity(X, sim)
        dists, nn_idx = kdtree.query(X_trans, k=1, workers=-1)
        dists = dists.astype(np.float64)
        nn_idx = nn_idx.astype(np.int64)

        # Trim correspondences
        keep_mask = _trim_by_percentile(dists, keep_ratio=float(inlier_ratio))
        X_corr = X[keep_mask]
        Y_corr = Y_full[nn_idx[keep_mask]]
        X_trans_corr = X_trans[keep_mask]

        if X_corr.shape[0] < 3:
            raise RuntimeError("Too few correspondences after trimming; try higher inlier_ratio or better init.")

        # Estimate delta transform on current aligned points, then compose
        delta = umeyama_similarity(X_trans_corr, Y_corr, with_scaling=True)

        # Convergence check
        dR = delta.R
        angle = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.arccos(angle))
        ds = abs(delta.s - 1.0)
        dt = float(np.linalg.norm(delta.t))

        sim = compose_similarity(delta, sim)

        rmse = float(np.sqrt(np.mean(dists[keep_mask] ** 2)))
        history["rmse"].append(rmse)
        history["inliers"].append(int(keep_mask.sum()))
        history["s"].append(sim.s)

        if verbose:
            print(f"[ICP {it:02d}] rmse={rmse:.6f}, inliers={keep_mask.sum()}, ds={ds:.3e}, dtheta={angle:.3e}, dt={dt:.3e}")

        # Stop when all small
        if max(ds, angle, dt) < tolerance:
            break

    # Compute fitness metrics using final transform on sampled points
    X_final = apply_similarity(X, sim)
    d_final, _ = kdtree.query(X_final, k=1, workers=-1)
    rmse_final = float(np.sqrt(np.mean(d_final ** 2)))

    metrics = {
        "rmse": rmse_final,
        "iterations": it + 1,
        "used_open3d": False,
        "history": history,
    }
    return sim, metrics


def _o3d_available() -> bool:
    try:
        import open3d as o3d  # noqa: F401
        return True
    except Exception:
        return False


def _to_o3d_pcd(points: np.ndarray):  # type: ignore
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_as_numpy_xyz(points))
    return pcd


def open3d_icp_similarity(
    source: np.ndarray,
    target: np.ndarray,
    max_correspondence_distance: float,
    max_iterations: int = 50,
    init: Optional[np.ndarray] = None,
    with_scaling: bool = True,
    estimate_normals: bool = False,
    voxel_downsample: Optional[float] = None,
):
    """
    Run Open3D ICP that estimates scale as well as R,t.

    Returns: (Similarity, metrics)
    """
    import open3d as o3d

    src_pcd = _to_o3d_pcd(source)
    tgt_pcd = _to_o3d_pcd(target)

    if voxel_downsample and voxel_downsample > 0:
        src_pcd = src_pcd.voxel_down_sample(voxel_downsample)
        tgt_pcd = tgt_pcd.voxel_down_sample(voxel_downsample)

    if estimate_normals:
        src_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max_correspondence_distance * 2, max_nn=30))
        tgt_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max_correspondence_distance * 2, max_nn=30))

    if init is None:
        init = np.eye(4)

    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=with_scaling)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)

    reg = o3d.pipelines.registration.registration_icp(
        src_pcd,
        tgt_pcd,
        max_correspondence_distance=max_correspondence_distance,
        init=init,
        estimation_method=estimation,
        criteria=criteria,
    )

    T = np.asarray(reg.transformation)
    # T[:3,:3] = s*R in Open3D when with_scaling=True
    Rs = T[:3, :3]
    s = float(np.cbrt(np.linalg.det(Rs)))  # extract uniform scale from Rs
    if s == 0:
        s = 1.0
    R = Rs / s
    t = T[:3, 3]

    sim = Similarity(s, R, t)
    metrics = {
        "rmse": float(reg.inlier_rmse),
        "fitness": float(reg.fitness),
        "iterations": max_iterations,  # Open3D does not expose actual count directly
        "used_open3d": True,
    }
    return sim, metrics


def open3d_global_and_icp(
    source: np.ndarray,
    target: np.ndarray,
    voxel_size: float = 0.05,
    ransac_n: int = 4,
    icp_distance: Optional[float] = None,
    max_icp_iterations: int = 50,
) -> Tuple[Similarity, Dict[str, Any]]:
    """
    If Open3D is available: do FPFH global registration (RANSAC) to get an initial
    transform, then refine with scaled ICP.
    """
    import open3d as o3d

    def preprocess(pcd: "o3d.geometry.PointCloud", voxel: float):
        pcd_down = pcd.voxel_down_sample(voxel)
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
        )
        return pcd_down, fpfh

    src = _to_o3d_pcd(source)
    tgt = _to_o3d_pcd(target)
    src_down, src_fpfh = preprocess(src, voxel_size)
    tgt_down, tgt_fpfh = preprocess(tgt, voxel_size)

    estimation_scaled = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)

    distance_threshold = icp_distance if icp_distance is not None else voxel_size * 1.5

    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold * 2.5,
        estimation_method=estimation_scaled,
        ransac_n=ransac_n,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold * 2.5),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500),
    )

    init_T = np.asarray(ransac.transformation)

    sim_icp, metrics_icp = open3d_icp_similarity(
        source,
        target,
        max_correspondence_distance=distance_threshold,
        max_iterations=max_icp_iterations,
        init=init_T,
        with_scaling=True,
        estimate_normals=False,
        voxel_downsample=None,
    )

    metrics = {
        "rmse": metrics_icp.get("rmse", np.nan),
        "fitness": metrics_icp.get("fitness", np.nan),
        "iterations": metrics_icp.get("iterations", max_icp_iterations),
        "used_open3d": True,
        "init_ransac_fitness": float(getattr(ransac, "fitness", 0.0)),
        "init_ransac_rmse": float(getattr(ransac, "inlier_rmse", 0.0)),
    }
    return sim_icp, metrics


def align_point_clouds(
    source: np.ndarray,
    target: np.ndarray,
    method: str = "auto",
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    inlier_ratio: float = 0.8,
    max_pairs: Optional[int] = 50000,
    use_global_init: bool = True,
    voxel_size: float = 0.05,
) -> Tuple[Similarity, Dict[str, Any]]:
    """
    Estimate similarity transform (s, R, t) aligning source -> target.

    method:
        - "open3d": use Open3D global+ICP with scaling (requires open3d)
        - "numpy": use pure NumPy trimmed ICP with scaling
        - "auto": prefer Open3D if available, else NumPy ICP
    """
    method = method.lower()
    if method == "auto":
        method = "open3d" if _o3d_available() else "numpy"

    if method == "open3d":
        if not _o3d_available():
            raise RuntimeError("Open3D not available. Install open3d or use method='numpy'.")
        if use_global_init:
            return open3d_global_and_icp(source, target, voxel_size=voxel_size)
        # else: directly ICP with scaling
        distance = voxel_size * 2.0
        return open3d_icp_similarity(source, target, max_correspondence_distance=distance, max_iterations=max_iterations)

    elif method == "numpy":
        return icp_similarity(
            source,
            target,
            max_iterations=max_iterations,
            tolerance=tolerance,
            inlier_ratio=inlier_ratio,
            init=None,
            max_pairs=max_pairs,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def _load_points_from_file(path: str, max_points: Optional[int] = None) -> np.ndarray:
    """Load points from a .ply/.pcd using Open3D or Trimesh if available."""
    pts = None
    # Try Open3D first
    if _o3d_available():
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise ValueError(f"Failed to read points from {path}")
        pts = np.asarray(pcd.points)
    else:
        try:
            import trimesh

            m = trimesh.load(path)
            if hasattr(m, "vertices") and len(m.vertices) > 0:
                pts = np.asarray(m.vertices)
            elif hasattr(m, "points"):
                pts = np.asarray(m.points)
            else:
                raise ValueError("Unsupported mesh/point object from trimesh")
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"Unable to load point cloud from '{path}'. Install open3d or trimesh. Error: {e}"
            )

    if max_points is not None and pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return _as_numpy_xyz(pts)


def _print_transform(sim: Similarity) -> None:
    R = sim.R
    t = sim.t
    s = sim.s
    print("Estimated scale s:", s)
    print("Estimated rotation R (row-major):\n", R)
    print("Estimated translation t:", t)
    print("4x4 matrix (sR | t):\n", sim.matrix())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Align two point clouds with scale and estimate s,R,t.")
    parser.add_argument("source", help="Path to source point cloud (.ply/.pcd) or npy with (N,3)")
    parser.add_argument("target", help="Path to target point cloud (.ply/.pcd) or npy with (M,3)")
    parser.add_argument("--method", default="auto", choices=["auto", "open3d", "numpy"], help="Alignment backend")
    parser.add_argument("--max_points", type=int, default=200000, help="Randomly sample up to this many points from each cloud")
    parser.add_argument("--iters", type=int, default=50, help="Max ICP iterations")
    parser.add_argument("--voxel", type=float, default=0.05, help="Voxel size (Open3D) / scale of distances")
    parser.add_argument("--inlier_ratio", type=float, default=0.8, help="Trimmed ICP inlier ratio (NumPy)")
    args = parser.parse_args()

    def load_any(path: str, max_points: int) -> np.ndarray:
        if path.lower().endswith(".npy"):
            arr = np.load(path)
            return _as_numpy_xyz(arr[:max_points])
        return _load_points_from_file(path, max_points=max_points)

    src = load_any(args.source, args.max_points)
    tgt = load_any(args.target, args.max_points)

    sim, metrics = align_point_clouds(
        src,
        tgt,
        method=args.method,
        max_iterations=args.iters,
        inlier_ratio=args.inlier_ratio,
        voxel_size=args.voxel,
    )

    _print_transform(sim)
    print("Metrics:", {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in metrics.items() if k != "history"})

# -----------------------------------------------------------------------------
# CLI 실행 예제 (Examples)
#
# 1) 기본 실행: Open3D가 설치되어 있으면 자동으로 Open3D(Global+ICP) 경로 사용
#    python src/preprocessing/icp.py source.ply target.ply --method auto --voxel 0.05
#
# 2) Open3D 경로 강제 사용
#    python src/preprocessing/icp.py source.ply target.ply --method open3d --voxel 0.05
#
# 3) NumPy ICP(대응점 없이 Trimmed ICP, 스케일 포함) 강제 사용
#    python src/preprocessing/icp.py source.ply target.ply --method numpy --iters 60 --inlier_ratio 0.8
#
# 4) Numpy 배열(.npy, shape=(N,3)) 입력 사용
#    python src/preprocessing/icp.py source.npy target.npy --method auto
#
# 팁:
#  - --voxel: 장면 스케일의 1~5%로 시작, 필요 시 조정
#  - 포인트가 많으면 --max_points로 다운샘플, Open3D가 더 안정적
#  - Open3D 미설치 시 --method auto는 자동으로 NumPy ICP로 폴백
# -----------------------------------------------------------------------------
