"""Initialization Module for Bundle Adjustment

This module handles initial pose estimation and triangulation using GTSAM's built-in functions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

try:
    import gtsam
    from gtsam import (
        EssentialMatrixConstraint, EssentialMatrix,
        CameraSetCal3_S2, PinholeCamera, Cal3_S2,
        Rot3, Point3, Pose3, Unit3
    )
except ImportError:
    raise ImportError("GTSAM is required. Install with: pip install gtsam")

# Try to import pycolmap for more robust initialization
try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False
    logger.warning("pycolmap not found. Using GTSAM-only initialization.")

from .track_manager import MergedTrack
from .camera_model import CameraModel

logger = logging.getLogger(__name__)


class PoseInitializer:
    """Initializes camera poses and 3D points for bundle adjustment"""
    
    def __init__(self,
                 camera_model: CameraModel,
                 camera_name: str,
                 min_parallax_deg: float = 1.0,
                 min_inliers: int = 30,
                 ransac_threshold: float = 4.0):
        """
        Initialize Pose Initializer
        
        Args:
            camera_model: CameraModel instance
            camera_name: Name of camera configuration
            min_parallax_deg: Minimum parallax angle for initialization (degrees)
            min_inliers: Minimum number of inliers for valid initialization
            ransac_threshold: RANSAC threshold in pixels
        """
        self.camera_model = camera_model
        self.camera_name = camera_name
        self.min_parallax_deg = min_parallax_deg
        self.min_inliers = min_inliers
        self.ransac_threshold = ransac_threshold
        
        # Get camera parameters
        self.camera_params = camera_model.get_camera_params(camera_name)
        self.K = self.camera_params['K']
        self.dist = self.camera_params['dist']
        
        logger.info(f"Initialized PoseInitializer with camera '{camera_name}'")
    
    def find_best_initial_pair(self, 
                              tracks: List[MergedTrack],
                              max_frames_to_check: int = 30) -> Tuple[int, int]:
        """Find best frame pair for initialization
        
        Args:
            tracks: List of merged tracks
            max_frames_to_check: Maximum number of frames to check
            
        Returns:
            Tuple of (frame1_id, frame2_id) for best initialization pair
        """
        # Get all frame IDs
        all_frames = set()
        for track in tracks:
            all_frames.update(track.observations.keys())
        all_frames = sorted(list(all_frames))[:max_frames_to_check]
        
        best_pair = (0, 1)
        best_score = 0
        
        # Try different frame pairs
        for i in range(len(all_frames)):
            for j in range(i + 1, min(i + 20, len(all_frames))):  # Check up to 20 frames ahead
                frame1, frame2 = all_frames[i], all_frames[j]
                
                # Count common tracks
                common_tracks = 0
                for track in tracks:
                    if (frame1 in track.observations and 
                        frame2 in track.observations and
                        track.visibility[frame1] > 0.5 and
                        track.visibility[frame2] > 0.5):
                        common_tracks += 1
                
                # Score based on number of tracks and frame distance
                frame_distance = j - i
                score = common_tracks * np.sqrt(frame_distance)
                
                if score > best_score and common_tracks >= self.min_inliers:
                    best_score = score
                    best_pair = (frame1, frame2)
        
        logger.info(f"Selected initial pair: frames {best_pair[0]} and {best_pair[1]} "
                   f"with score {best_score:.2f}")
        return best_pair
    
    def get_corresponding_points(self,
                                tracks: List[MergedTrack],
                                frame1: int,
                                frame2: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Get corresponding points between two frames
        
        Args:
            tracks: List of merged tracks
            frame1: First frame ID
            frame2: Second frame ID
            
        Returns:
            Tuple of (points1, points2, track_ids)
        """
        points1 = []
        points2 = []
        track_ids = []
        
        for track in tracks:
            if (frame1 in track.observations and 
                frame2 in track.observations and
                track.visibility[frame1] > 0.5 and
                track.visibility[frame2] > 0.5):
                
                points1.append(track.observations[frame1])
                points2.append(track.observations[frame2])
                track_ids.append(track.track_id)
        
        return np.array(points1), np.array(points2), track_ids
    
    def estimate_relative_pose(self,
                              points1: np.ndarray,
                              points2: np.ndarray) -> Tuple[Optional[np.ndarray], 
                                                           Optional[np.ndarray], 
                                                           np.ndarray]:
        """Estimate relative pose between two views
        
        Args:
            points1: Points in first frame (N, 2)
            points2: Points in second frame (N, 2)
            
        Returns:
            Tuple of (R, t, inlier_mask) or (None, None, empty_mask) if failed
        """
        if len(points1) < 8:
            return None, None, np.array([])
        
        # Compute essential matrix
        E, mask = cv2.findEssentialMat(
            points1, points2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self.ransac_threshold
        )
        
        if E is None:
            return None, None, np.array([])
        
        # Recover pose
        _, R, t, mask_pose = cv2.recoverPose(E, points1, points2, self.K, mask=mask)
        
        # Combine masks
        inlier_mask = (mask.ravel() == 1) & (mask_pose.ravel() > 0)
        
        n_inliers = np.sum(inlier_mask)
        if n_inliers < self.min_inliers:
            logger.warning(f"Not enough inliers: {n_inliers} < {self.min_inliers}")
            return None, None, inlier_mask
        
        logger.info(f"Estimated relative pose with {n_inliers} inliers")
        return R, t.ravel(), inlier_mask
    
    def triangulate_points(self,
                          points1: np.ndarray,
                          points2: np.ndarray,
                          R: np.ndarray,
                          t: np.ndarray) -> np.ndarray:
        """Triangulate 3D points from two views
        
        Args:
            points1: Points in first frame (N, 2)
            points2: Points in second frame (N, 2)
            R: Rotation matrix from frame 1 to frame 2
            t: Translation vector from frame 1 to frame 2
            
        Returns:
            3D points in frame 1 coordinates (N, 3)
        """
        # Create projection matrices
        P1 = self.K @ np.eye(3, 4)
        P2 = self.K @ np.hstack([R, t.reshape(-1, 1)])
        
        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T
    
    def check_parallax(self,
                      points_3d: np.ndarray,
                      baseline: np.ndarray) -> float:
        """Compute median parallax angle
        
        Args:
            points_3d: 3D points (N, 3)
            baseline: Camera baseline vector
            
        Returns:
            Median parallax angle in degrees
        """
        # Compute viewing rays
        rays1 = points_3d / np.linalg.norm(points_3d, axis=1, keepdims=True)
        rays2 = (points_3d - baseline) / np.linalg.norm(points_3d - baseline, axis=1, keepdims=True)
        
        # Compute angles
        cos_angles = np.sum(rays1 * rays2, axis=1)
        cos_angles = np.clip(cos_angles, -1, 1)
        angles_deg = np.degrees(np.arccos(cos_angles))
        
        return np.median(angles_deg)
    
    def initialize_two_view(self,
                           tracks: List[MergedTrack]) -> Tuple[Dict[int, gtsam.Pose3],
                                                               Dict[int, gtsam.Point3]]:
        """Initialize poses and points from two views
        
        Args:
            tracks: List of merged tracks
            
        Returns:
            Tuple of (initial_poses, initial_points)
        """
        # Find best initial pair
        frame1, frame2 = self.find_best_initial_pair(tracks)
        
        # Get corresponding points
        points1, points2, track_ids = self.get_corresponding_points(tracks, frame1, frame2)
        
        if len(points1) < self.min_inliers:
            raise ValueError(f"Not enough correspondences: {len(points1)} < {self.min_inliers}")
        
        # Estimate relative pose
        R, t, inlier_mask = self.estimate_relative_pose(points1, points2)
        
        if R is None:
            raise ValueError("Failed to estimate relative pose")
        
        # Triangulate points
        points1_inliers = points1[inlier_mask]
        points2_inliers = points2[inlier_mask]
        points_3d = self.triangulate_points(points1_inliers, points2_inliers, R, t)
        
        # Check parallax
        parallax = self.check_parallax(points_3d, t)
        logger.info(f"Median parallax angle: {parallax:.2f} degrees")
        
        if parallax < self.min_parallax_deg:
            logger.warning(f"Low parallax: {parallax:.2f} < {self.min_parallax_deg} degrees")
        
        # Create initial poses
        initial_poses = {
            frame1: gtsam.Pose3(),  # First camera at origin
            frame2: gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t))
        }
        
        # Create initial 3D points
        initial_points = {}
        inlier_track_ids = np.array(track_ids)[inlier_mask]
        for i, track_id in enumerate(inlier_track_ids):
            initial_points[track_id] = gtsam.Point3(points_3d[i])
        
        logger.info(f"Initialized with {len(initial_poses)} poses and {len(initial_points)} points")
        return initial_poses, initial_points
    
    def add_next_view(self,
                     tracks: List[MergedTrack],
                     current_poses: Dict[int, gtsam.Pose3],
                     current_points: Dict[int, gtsam.Point3],
                     frame_id: int) -> Optional[gtsam.Pose3]:
        """Add next view using PnP
        
        Args:
            tracks: List of merged tracks
            current_poses: Current camera poses
            current_points: Current 3D points
            frame_id: Frame ID to add
            
        Returns:
            Estimated pose for the new frame, or None if failed
        """
        # Collect 2D-3D correspondences
        points_2d = []
        points_3d = []
        
        for track in tracks:
            if (track.track_id in current_points and
                frame_id in track.observations and
                track.visibility[frame_id] > 0.5):
                
                points_2d.append(track.observations[frame_id])
                point_3d = current_points[track.track_id]
                points_3d.append([point_3d.x(), point_3d.y(), point_3d.z()])
        
        points_2d = np.array(points_2d)
        points_3d = np.array(points_3d)
        
        if len(points_2d) < 10:
            logger.warning(f"Not enough correspondences for frame {frame_id}: {len(points_2d)}")
            return None
        
        # Solve PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, self.K, self.dist,
            confidence=0.999,
            reprojectionError=self.ransac_threshold
        )
        
        if not success or len(inliers) < self.min_inliers:
            logger.warning(f"PnP failed for frame {frame_id}")
            return None
        
        # Convert to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.ravel()
        
        # Create pose (note: OpenCV returns world-to-camera transform)
        pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t))
        
        logger.info(f"Added frame {frame_id} with {len(inliers)} inliers")
        return pose
    
    def initialize_all_poses(self,
                            tracks: List[MergedTrack],
                            n_frames: int) -> Tuple[Dict[int, gtsam.Pose3],
                                                   Dict[int, gtsam.Point3]]:
        """Initialize all camera poses incrementally
        
        Args:
            tracks: List of merged tracks
            n_frames: Total number of frames
            
        Returns:
            Tuple of (initial_poses, initial_points)
        """
        # Initialize with two views
        initial_poses, initial_points = self.initialize_two_view(tracks)
        
        # Get all frame IDs with observations
        all_frames = set()
        for track in tracks:
            all_frames.update(track.observations.keys())
        all_frames = sorted(list(all_frames))
        
        # Add remaining frames incrementally
        for frame_id in all_frames:
            if frame_id in initial_poses:
                continue
            
            pose = self.add_next_view(tracks, initial_poses, initial_points, frame_id)
            if pose is not None:
                initial_poses[frame_id] = pose
                
                # Triangulate new points visible in this frame
                self.triangulate_new_points(tracks, initial_poses, initial_points, frame_id)
        
        logger.info(f"Initialized {len(initial_poses)} / {len(all_frames)} poses")
        return initial_poses, initial_points
    
    def triangulate_new_points(self,
                              tracks: List[MergedTrack],
                              poses: Dict[int, gtsam.Pose3],
                              points: Dict[int, gtsam.Point3],
                              new_frame_id: int):
        """Triangulate new points visible in the new frame
        
        Args:
            tracks: List of merged tracks
            poses: Current camera poses
            points: Current 3D points (will be updated)
            new_frame_id: ID of newly added frame
        """
        for track in tracks:
            # Skip if already triangulated
            if track.track_id in points:
                continue
            
            # Check if visible in new frame
            if (new_frame_id not in track.observations or
                track.visibility[new_frame_id] < 0.5):
                continue
            
            # Find another frame where this track is visible
            for other_frame_id in track.observations.keys():
                if (other_frame_id != new_frame_id and
                    other_frame_id in poses and
                    track.visibility[other_frame_id] > 0.5):
                    
                    # Triangulate
                    pose1 = poses[new_frame_id]
                    pose2 = poses[other_frame_id]
                    
                    # Convert to OpenCV format
                    R1 = pose1.rotation().matrix()
                    t1 = pose1.translation()
                    R2 = pose2.rotation().matrix()
                    t2 = pose2.translation()
                    
                    P1 = self.K @ np.hstack([R1, t1.reshape(-1, 1)])
                    P2 = self.K @ np.hstack([R2, t2.reshape(-1, 1)])
                    
                    points1 = track.observations[new_frame_id].reshape(1, 2)
                    points2 = track.observations[other_frame_id].reshape(1, 2)
                    
                    points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
                    point_3d = points_4d[:3, 0] / points_4d[3, 0]
                    
                    # Add to points
                    points[track.track_id] = gtsam.Point3(point_3d)
                    break