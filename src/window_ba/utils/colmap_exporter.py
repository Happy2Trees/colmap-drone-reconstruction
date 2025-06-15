"""
COLMAP export utilities for Window-based Bundle Adjustment.

This module handles exporting optimized camera poses and 3D points
to COLMAP format for compatibility with standard 3D reconstruction pipelines.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import sys

# Add parent directory to path for colmap_utils
sys.path.append(str(Path(__file__).parent.parent.parent))
from colmap_utils import read_write_model

from ..models.data_models import WindowTrackData, CameraIntrinsics
from ..models.camera_model import CameraModel

logger = logging.getLogger(__name__)


class COLMAPExporter:
    """Export Window BA results to COLMAP format."""
    
    def __init__(self, output_dir: Path, binary_format: bool = True):
        """
        Initialize COLMAP exporter.
        
        Args:
            output_dir: Directory to save COLMAP model
            binary_format: If True, save as binary; otherwise as text
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ext = '.bin' if binary_format else '.txt'
        
    def export(self,
              camera_model: CameraModel,
              window_tracks: List[WindowTrackData],
              intrinsics: CameraIntrinsics):
        """
        Export complete COLMAP model.
        
        Args:
            camera_model: Optimized camera model
            window_tracks: List of window tracks with 3D points
            intrinsics: Camera intrinsic parameters
        """
        logger.info("Starting COLMAP export...")
        
        # 1. Create cameras dictionary
        cameras = self._create_cameras(intrinsics)
        
        # 2. Create images dictionary
        images = self._create_images(camera_model, window_tracks)
        
        # 3. Create points3D dictionary
        points3D = self._create_points3D(window_tracks)
        
        # 4. Write model
        read_write_model.write_model(
            cameras, images, points3D, 
            str(self.output_dir), self.ext
        )
        
        # Log statistics
        logger.info(f"Exported COLMAP model to {self.output_dir}:")
        logger.info(f"  - Format: {self.ext}")
        logger.info(f"  - Cameras: {len(cameras)}")
        logger.info(f"  - Images: {len(images)}")
        logger.info(f"  - Points3D: {len(points3D)}")
        
        # Save metadata
        self._save_metadata(camera_model, window_tracks, intrinsics)
    
    def _create_cameras(self, intrinsics: CameraIntrinsics) -> Dict:
        """
        Create COLMAP cameras dictionary.
        
        For simplicity, we use a single camera model for all frames.
        COLMAP uses PINHOLE model: fx, fy, cx, cy
        """
        cameras = {
            1: read_write_model.Camera(
                id=1,
                model='PINHOLE',
                width=intrinsics.width,
                height=intrinsics.height,
                params=np.array([
                    intrinsics.fx,
                    intrinsics.fy,
                    intrinsics.cx,
                    intrinsics.cy
                ])
            )
        }
        
        return cameras
    
    def _create_images(self, 
                      camera_model: CameraModel,
                      window_tracks: List[WindowTrackData]) -> Dict:
        """
        Create COLMAP images dictionary from optimized camera poses.
        
        Each image contains:
        - Camera pose (quaternion + translation)
        - 2D-3D correspondences for visible points
        """
        images = {}
        
        with torch.no_grad():
            # Get total number of frames
            max_frame = max(track.end_frame for track in window_tracks) + 1
            
            # Process each frame
            for frame_idx in range(max_frame):
                # Get camera pose
                qvec = camera_model.quaternions[frame_idx].cpu().numpy()
                tvec = camera_model.translations[frame_idx].cpu().numpy()
                
                # Normalize quaternion
                qvec = qvec / np.linalg.norm(qvec)
                
                # Collect 2D-3D correspondences for this frame
                xys, point3D_ids = self._get_frame_observations(
                    frame_idx, window_tracks
                )
                
                # Create image entry
                image_name = f"{frame_idx:05d}.jpg"  # Assuming jpg format
                
                images[frame_idx + 1] = read_write_model.Image(
                    id=frame_idx + 1,  # COLMAP uses 1-based indexing
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=1,  # Single camera model
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids
                )
        
        return images
    
    def _get_frame_observations(self, 
                               frame_idx: int,
                               window_tracks: List[WindowTrackData]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 2D observations and corresponding 3D point IDs for a specific frame.
        
        Args:
            frame_idx: Global frame index
            window_tracks: List of window tracks
            
        Returns:
            xys: (N, 2) array of 2D points
            point3D_ids: (N,) array of corresponding 3D point IDs
        """
        xys_list = []
        point3D_ids_list = []
        
        for window in window_tracks:
            if window.start_frame <= frame_idx <= window.end_frame:
                # Frame index within window
                window_frame_idx = frame_idx - window.start_frame
                
                # Get visibility for this frame
                visibility = window.visibility[window_frame_idx]
                
                # Get 2D points
                tracks_2d = window.tracks[window_frame_idx]  # (N, 2)
                
                # Add visible points
                for point_idx in range(len(visibility)):
                    if visibility[point_idx]:
                        # Create unique point3D ID
                        # Using window_idx * 10000 + point_idx to ensure uniqueness
                        point3D_id = self._get_point3d_id(window.window_idx, point_idx)
                        
                        xys_list.append(tracks_2d[point_idx])
                        point3D_ids_list.append(point3D_id)
        
        if xys_list:
            xys = np.array(xys_list)
            point3D_ids = np.array(point3D_ids_list)
        else:
            xys = np.zeros((0, 2))
            point3D_ids = np.zeros(0, dtype=np.int32)
        
        return xys, point3D_ids
    
    def _create_points3D(self, window_tracks: List[WindowTrackData]) -> Dict:
        """
        Create COLMAP points3D dictionary from window tracks.
        
        If Phase 2 optimization was used, prioritize the optimized boundary points.
        """
        points3D = {}
        
        # First pass: Add Phase 2 optimized boundary points if available
        self._add_optimized_boundary_points(points3D, window_tracks)
        
        # Second pass: Add regular 3D points from depth initialization
        self._add_regular_3d_points(points3D, window_tracks)
        
        logger.info(f"Created {len(points3D)} 3D points for COLMAP export")
        
        return points3D
    
    def _add_optimized_boundary_points(self,
                                     points3D: Dict,
                                     window_tracks: List[WindowTrackData]):
        """Add Phase 2 optimized boundary points to points3D dictionary."""
        point_id = 1
        
        for window in window_tracks:
            if window.boundary_3d_optimized is None:
                continue
                
            # These are the boundary points optimized in Phase 2
            boundary_3d = window.boundary_3d_optimized  # (M, 3)
            
            # Separate start and end boundary points
            mask_first = window.boundary_mask_start
            mask_last = window.boundary_mask_end
            
            # Process start boundary points
            if mask_first.any():
                n_first = mask_first.sum()
                for local_idx in range(n_first):
                    # Find global point index
                    point_indices = np.where(mask_first)[0]
                    if local_idx >= len(point_indices):
                        continue
                    global_point_idx = point_indices[local_idx]
                    
                    # Get 3D position
                    xyz = boundary_3d[local_idx]
                    
                    # Find observations
                    image_ids, point2D_idxs = self._find_point_observations(
                        window, global_point_idx
                    )
                    
                    if len(image_ids) > 0:
                        points3D[point_id] = read_write_model.Point3D(
                            id=point_id,
                            xyz=xyz,
                            rgb=np.array([255, 128, 128]),  # Red for start boundary
                            error=0.1,  # Default error
                            image_ids=image_ids,
                            point2D_idxs=point2D_idxs
                        )
                        point_id += 1
            
            # Process end boundary points (similar logic)
            if mask_last.any():
                n_last = mask_last.sum()
                n_first = mask_first.sum() if mask_first.any() else 0
                
                for local_idx in range(n_last):
                    # Find global point index
                    point_indices = np.where(mask_last)[0]
                    if local_idx >= len(point_indices):
                        continue
                    global_point_idx = point_indices[local_idx]
                    
                    # Get 3D position (offset by n_first)
                    xyz = boundary_3d[n_first + local_idx]
                    
                    # Find observations
                    image_ids, point2D_idxs = self._find_point_observations(
                        window, global_point_idx
                    )
                    
                    if len(image_ids) > 0:
                        points3D[point_id] = read_write_model.Point3D(
                            id=point_id,
                            xyz=xyz,
                            rgb=np.array([128, 255, 128]),  # Green for end boundary
                            error=0.1,
                            image_ids=image_ids,
                            point2D_idxs=point2D_idxs
                        )
                        point_id += 1
        
        if point_id > 1:
            logger.info(f"Added {point_id - 1} optimized boundary points")
    
    def _add_regular_3d_points(self,
                              points3D: Dict,
                              window_tracks: List[WindowTrackData]):
        """Add regular 3D points from depth initialization."""
        # Start from next available ID
        point_id = max(points3D.keys()) + 1 if points3D else 1
        
        for window in window_tracks:
            if window.xyzw_world is None:
                continue
            
            xyzw_world = window.xyzw_world  # (T, N, 4)
            visibility = window.visibility  # (T, N)
            
            T, N, _ = xyzw_world.shape
            
            # For each point in the window
            for point_idx in range(N):
                # Skip if this is a boundary point already added from Phase 2
                if window.boundary_3d_optimized is not None:
                    # Check if this point is a boundary point
                    is_boundary = (window.boundary_mask_start[point_idx] or 
                                 window.boundary_mask_end[point_idx])
                    if is_boundary:
                        continue  # Skip, already added
                
                # Find frames where this point is visible
                visible_frames = []
                xyz_observations = []
                
                for t in range(T):
                    if visibility[t, point_idx]:
                        global_frame_idx = window.start_frame + t
                        visible_frames.append(global_frame_idx)
                        xyz_observations.append(xyzw_world[t, point_idx, :3])
                
                if len(visible_frames) == 0:
                    continue
                
                # Use median 3D position across all observations
                xyz_median = np.median(xyz_observations, axis=0)
                
                # Error estimation
                if len(xyz_observations) > 1:
                    error = float(np.std(xyz_observations))
                else:
                    error = 1.0
                
                # Get observations
                image_ids, point2D_idxs = self._find_point_observations(
                    window, point_idx
                )
                
                if len(image_ids) > 0:
                    points3D[point_id] = read_write_model.Point3D(
                        id=point_id,
                        xyz=xyz_median,
                        rgb=np.array([128, 128, 128]),  # Gray for regular points
                        error=error,
                        image_ids=image_ids,
                        point2D_idxs=point2D_idxs
                    )
                    point_id += 1
    
    def _find_point_observations(self,
                                window: WindowTrackData,
                                point_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find all image observations of a specific point.
        
        Returns:
            image_ids: Array of image IDs (1-based)
            point2D_idxs: Array of point indices in each image
        """
        image_ids = []
        point2D_idxs = []
        
        for t in range(window.num_frames):
            if window.visibility[t, point_idx]:
                global_frame_idx = window.start_frame + t
                image_ids.append(global_frame_idx + 1)  # 1-based
                
                # For now, using a simple index
                # In practice, this should match the order in the image's xys array
                point2D_idxs.append(len(point2D_idxs))
        
        return np.array(image_ids, dtype=np.int32), np.array(point2D_idxs, dtype=np.int32)
    
    def _get_point3d_id(self, window_idx: int, point_idx: int) -> int:
        """Generate unique point3D ID."""
        return window_idx * 10000 + point_idx + 1  # 1-based
    
    def _save_metadata(self,
                      camera_model: CameraModel,
                      window_tracks: List[WindowTrackData],
                      intrinsics: CameraIntrinsics):
        """Save additional metadata for debugging and analysis."""
        metadata = {
            'num_frames': camera_model.num_frames,
            'num_windows': len(window_tracks),
            'single_camera': camera_model.single_camera,
            'image_width': intrinsics.width,
            'image_height': intrinsics.height,
            'intrinsics': {
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'cx': intrinsics.cx,
                'cy': intrinsics.cy
            },
            'window_info': []
        }
        
        # Add window information
        for window in window_tracks:
            window_info = {
                'window_idx': window.window_idx,
                'start_frame': window.start_frame,
                'end_frame': window.end_frame,
                'num_points': window.num_points,
                'has_optimized_boundary': window.boundary_3d_optimized is not None
            }
            metadata['window_info'].append(window_info)
        
        # Save as JSON
        import json
        with open(self.output_dir / 'export_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Saved export metadata")