"""Global Bundle Adjustment Module using GTSAM

This module performs global bundle adjustment on Co-Tracker tracks using GTSAM.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

try:
    import gtsam
    from gtsam import symbol
except ImportError:
    raise ImportError("GTSAM is required. Install with: pip install gtsam")

from .track_manager import MergedTrack
from .camera_model import CameraModel

logger = logging.getLogger(__name__)


@dataclass
class BundleAdjustmentResult:
    """Results from bundle adjustment optimization"""
    camera_poses: Dict[int, gtsam.Pose3]  # frame_id -> camera pose
    points_3d: Dict[int, gtsam.Point3]  # track_id -> 3D point
    initial_error: float
    final_error: float
    iterations: int
    converged: bool


class GlobalBundleAdjustment:
    """Performs global bundle adjustment using GTSAM"""
    
    def __init__(self,
                 camera_model: CameraModel,
                 camera_name: str,
                 max_iterations: int = 100,
                 convergence_threshold: float = 1e-6,
                 pixel_noise_sigma: float = 2.0,
                 robust_kernel: bool = True,
                 huber_delta: float = 1.0):
        """
        Initialize Global Bundle Adjustment
        
        Args:
            camera_model: CameraModel instance with intrinsics
            camera_name: Name of camera configuration to use
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold for optimization
            pixel_noise_sigma: Standard deviation of pixel measurements in pixels
            robust_kernel: Whether to use robust kernel for outliers
            huber_delta: Delta for Huber robust kernel
        """
        self.camera_model = camera_model
        self.camera_name = camera_name
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.pixel_noise_sigma = pixel_noise_sigma
        self.robust_kernel = robust_kernel
        self.huber_delta = huber_delta
        
        # Get camera calibration
        self.calibration = camera_model.get_gtsam_calibration(camera_name)
        
        # Create noise model for pixel measurements
        self.measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, pixel_noise_sigma)
        
        if robust_kernel:
            # Add robust kernel for outlier rejection
            huber = gtsam.noiseModel.mEstimator.Huber(huber_delta)
            self.measurement_noise = gtsam.noiseModel.Robust(huber, self.measurement_noise)
        
        logger.info(f"Initialized GlobalBundleAdjustment with camera '{camera_name}', "
                   f"pixel_noise={pixel_noise_sigma}, robust={robust_kernel}")
    
    def create_factor_graph(self, 
                           tracks: List[MergedTrack],
                           initial_poses: Dict[int, gtsam.Pose3],
                           initial_points: Dict[int, gtsam.Point3]) -> gtsam.NonlinearFactorGraph:
        """Create factor graph for bundle adjustment
        
        Args:
            tracks: List of merged tracks with observations
            initial_poses: Initial camera poses (frame_id -> Pose3)
            initial_points: Initial 3D points (track_id -> Point3)
            
        Returns:
            GTSAM NonlinearFactorGraph with projection factors
        """
        graph = gtsam.NonlinearFactorGraph()
        
        # Add projection factors for each observation
        n_factors = 0
        for track in tracks:
            track_key = symbol('l', track.track_id)  # 3D point
            
            for frame_id, observation in track.observations.items():
                if track.visibility[frame_id] < 0.5:  # Skip invisible points
                    continue
                
                if frame_id not in initial_poses:
                    continue  # Skip if no initial pose
                
                camera_key = symbol('x', frame_id)  # Camera pose
                
                # Create projection factor
                factor = gtsam.GenericProjectionFactorCal3_S2(
                    gtsam.Point2(observation[0], observation[1]),  # 2D measurement
                    self.measurement_noise,                         # Measurement noise
                    camera_key,                                     # Camera pose key
                    track_key,                                      # 3D point key
                    self.calibration                                # Camera calibration
                )
                
                graph.add(factor)
                n_factors += 1
        
        # Add prior on first camera to fix gauge freedom
        if 0 in initial_poses:
            pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])  # rot (rad), trans (m)
            )
            graph.addPriorPose3(symbol('x', 0), initial_poses[0], pose_prior_noise)
        
        logger.info(f"Created factor graph with {n_factors} projection factors")
        return graph
    
    def create_initial_values(self,
                             initial_poses: Dict[int, gtsam.Pose3],
                             initial_points: Dict[int, gtsam.Point3]) -> gtsam.Values:
        """Create initial values for optimization
        
        Args:
            initial_poses: Initial camera poses (frame_id -> Pose3)
            initial_points: Initial 3D points (track_id -> Point3)
            
        Returns:
            GTSAM Values with initial estimates
        """
        initial_estimate = gtsam.Values()
        
        # Add camera poses
        for frame_id, pose in initial_poses.items():
            initial_estimate.insert(symbol('x', frame_id), pose)
        
        # Add 3D points
        for track_id, point in initial_points.items():
            initial_estimate.insert(symbol('l', track_id), point)
        
        logger.info(f"Created initial values with {len(initial_poses)} poses "
                   f"and {len(initial_points)} points")
        return initial_estimate
    
    def optimize(self,
                tracks: List[MergedTrack],
                initial_poses: Dict[int, gtsam.Pose3],
                initial_points: Dict[int, gtsam.Point3]) -> BundleAdjustmentResult:
        """Run global bundle adjustment optimization
        
        Args:
            tracks: List of merged tracks with observations
            initial_poses: Initial camera poses (frame_id -> Pose3)
            initial_points: Initial 3D points (track_id -> Point3)
            
        Returns:
            BundleAdjustmentResult with optimized poses and points
        """
        # Create factor graph
        graph = self.create_factor_graph(tracks, initial_poses, initial_points)
        
        # Create initial values
        initial_estimate = self.create_initial_values(initial_poses, initial_points)
        
        # Compute initial error
        initial_error = graph.error(initial_estimate)
        logger.info(f"Initial error: {initial_error:.2f}")
        
        # Set up optimizer
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.max_iterations)
        params.setRelativeErrorTol(self.convergence_threshold)
        params.setAbsoluteErrorTol(self.convergence_threshold)
        params.setVerbosityLM("SUMMARY")
        
        # Run optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()
        
        # Extract results
        final_error = graph.error(result)
        iterations = optimizer.iterations()
        
        logger.info(f"Optimization complete: iterations={iterations}, "
                   f"final_error={final_error:.2f}, "
                   f"error_reduction={initial_error - final_error:.2f}")
        
        # Extract optimized poses and points
        optimized_poses = {}
        for frame_id in initial_poses.keys():
            key = symbol('x', frame_id)
            if result.exists(key):
                optimized_poses[frame_id] = result.atPose3(key)
        
        optimized_points = {}
        for track_id in initial_points.keys():
            key = symbol('l', track_id)
            if result.exists(key):
                optimized_points[track_id] = result.atPoint3(key)
        
        return BundleAdjustmentResult(
            camera_poses=optimized_poses,
            points_3d=optimized_points,
            initial_error=initial_error,
            final_error=final_error,
            iterations=iterations,
            converged=(iterations < self.max_iterations)
        )
    
    def compute_reprojection_errors(self,
                                   tracks: List[MergedTrack],
                                   poses: Dict[int, gtsam.Pose3],
                                   points: Dict[int, gtsam.Point3]) -> Dict[str, float]:
        """Compute reprojection error statistics
        
        Args:
            tracks: List of merged tracks
            poses: Camera poses
            points: 3D points
            
        Returns:
            Dictionary with error statistics
        """
        errors = []
        
        for track in tracks:
            if track.track_id not in points:
                continue
            
            point_3d = points[track.track_id]
            
            for frame_id, observation in track.observations.items():
                if frame_id not in poses or track.visibility[frame_id] < 0.5:
                    continue
                
                # Project point
                pose = poses[frame_id]
                camera = gtsam.PinholeCameraCal3_S2(pose, self.calibration)
                
                try:
                    projected = camera.project(point_3d)
                    error = np.linalg.norm(projected - gtsam.Point2(observation[0], observation[1]))
                    errors.append(error)
                except:
                    # Point behind camera or other projection error
                    continue
        
        if not errors:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'max': 0.0,
                'count': 0
            }
        
        errors = np.array(errors)
        return {
            'mean': np.mean(errors),
            'median': np.median(errors),
            'std': np.std(errors),
            'max': np.max(errors),
            'count': len(errors)
        }
    
    def filter_outlier_tracks(self,
                             tracks: List[MergedTrack],
                             poses: Dict[int, gtsam.Pose3],
                             points: Dict[int, gtsam.Point3],
                             max_error: float = 10.0) -> List[MergedTrack]:
        """Filter tracks with high reprojection errors
        
        Args:
            tracks: List of merged tracks
            poses: Camera poses
            points: 3D points
            max_error: Maximum allowed reprojection error in pixels
            
        Returns:
            Filtered list of tracks
        """
        filtered_tracks = []
        
        for track in tracks:
            if track.track_id not in points:
                continue
            
            point_3d = points[track.track_id]
            errors = []
            
            for frame_id, observation in track.observations.items():
                if frame_id not in poses or track.visibility[frame_id] < 0.5:
                    continue
                
                # Project point
                pose = poses[frame_id]
                camera = gtsam.PinholeCameraCal3_S2(pose, self.calibration)
                
                try:
                    projected = camera.project(point_3d)
                    error = np.linalg.norm(projected - gtsam.Point2(observation[0], observation[1]))
                    errors.append(error)
                except:
                    errors.append(float('inf'))
            
            # Keep track if median error is below threshold
            if errors and np.median(errors) < max_error:
                filtered_tracks.append(track)
        
        logger.info(f"Filtered {len(tracks) - len(filtered_tracks)} outlier tracks "
                   f"(kept {len(filtered_tracks)} / {len(tracks)})")
        
        return filtered_tracks