from typing import Optional, Dict, Any
import os
import json
import numpy as np


def register_and_eval(
    source_ply: str,
    target_ply: str,
    voxel_size: float = 0.05,
    ransac_n: int = 4,
    distance_thresh_ratio: float = 1.5,
    icp_max_iter: int = 50,
    save_debug_dir: Optional[str] = None,
    aligned_ply_out: Optional[str] = None,
    with_scaling: bool = True,
) -> Dict[str, Any]:
    try:
        import open3d as o3d
    except Exception as e:
        return {"skipped": True, "reason": f"open3d import failed: {e}"}

    if save_debug_dir:
        os.makedirs(save_debug_dir, exist_ok=True)

    src = o3d.io.read_point_cloud(source_ply)
    tgt = o3d.io.read_point_cloud(target_ply)

    # Downsample
    vs = float(voxel_size)
    src_d = src.voxel_down_sample(vs)
    tgt_d = tgt.voxel_down_sample(vs)

    # Estimate normals
    radius_normal = vs * 2.0
    src_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    tgt_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # FPFH features
    radius_feature = vs * 5.0
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_d, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_d, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    # RANSAC
    dist_thresh = float(distance_thresh_ratio) * vs
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, src_fpfh, tgt_fpfh, True,
        dist_thresh,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling),
        ransac_n,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000),
    )

    T_init = ransac_result.transformation

    # ICP refine
    if with_scaling:
        icp = o3d.pipelines.registration.registration_icp(
            src_d, tgt_d, dist_thresh,
            T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(icp_max_iter)),
        )
    else:
        icp = o3d.pipelines.registration.registration_icp(
            src_d, tgt_d, dist_thresh,
            T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(icp_max_iter)),
        )

    T = icp.transformation

    # C2C distance stats (source -> target)
    src_aligned = src.transform(T.copy()) if hasattr(src, "transform") else src
    # After transform, src is already transformed by open3d (in-place); rebuild for distances
    src_aligned = src

    tgt_tree = o3d.geometry.KDTreeFlann(tgt)
    dists = []
    for p in np.asarray(src_aligned.points):
        _, idx, d = tgt_tree.search_knn_vector_3d(p, 1)
        if len(d) > 0:
            dists.append(np.sqrt(d[0]))
    dists = np.array(dists, dtype=float) if len(dists) > 0 else np.zeros((0,), dtype=float)

    # Estimate uniform scale from the linear part (s * R)
    A = T[:3, :3]
    s_est = float(np.mean(np.linalg.norm(A, axis=0))) if A.size else 1.0

    stats = {
        "voxel_size": vs,
        "with_scaling": bool(with_scaling),
        "ransac_fitness": float(ransac_result.fitness),
        "ransac_inlier_rmse": float(ransac_result.inlier_rmse),
        "icp_fitness": float(icp.fitness),
        "icp_inlier_rmse": float(icp.inlier_rmse),
        "transform": T.tolist(),
        "scale": s_est,
        "c2c_mean": float(np.mean(dists)) if dists.size else None,
        "c2c_median": float(np.median(dists)) if dists.size else None,
        "c2c_std": float(np.std(dists)) if dists.size else None,
        "c2c_max": float(np.max(dists)) if dists.size else None,
        "num_points_source": int(np.asarray(src.points).shape[0]),
        "num_points_target": int(np.asarray(tgt.points).shape[0]),
    }

    # Save aligned source to a specified output path, if provided
    if aligned_ply_out:
        try:
            import open3d as o3d  # type: ignore
            o3d.io.write_point_cloud(aligned_ply_out, src_aligned)
            stats["aligned_ply_out"] = aligned_ply_out
        except Exception as e:
            stats["aligned_ply_error"] = str(e)

    if save_debug_dir:
        # Save transform
        np.savetxt(os.path.join(save_debug_dir, "transform.txt"), T)
        # Save aligned source
        o3d.io.write_point_cloud(os.path.join(save_debug_dir, "aligned_source.ply"), src_aligned)
        # Save summary json
        with open(os.path.join(save_debug_dir, "register_eval.json"), "w") as f:
            json.dump(stats, f, indent=2)

    return stats
