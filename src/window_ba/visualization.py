"""
Visualization utilities for Window BA results.
Following GeometryCrafter's visualization approach.
"""

import numpy as np
import matplotlib
# Use non-interactive backend for CLI environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

class WindowBAVisualizer:
    """Visualize Window BA results including camera trajectory and 3D points."""
    
    def __init__(self, output_dir: Path, image_width: int = 1024, image_height: int = 576):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_width = image_width
        self.image_height = image_height
        
    def visualize_camera_trajectory(self, 
                                   camera_model: torch.nn.Module,
                                   window_tracks: List[Dict],
                                   save_path: Optional[Path] = None) -> None:
        """
        Visualize camera trajectory in 3D.
        
        Args:
            camera_model: Optimized camera model
            window_tracks: Window tracks with frame information
            save_path: Where to save the visualization
        """
        if save_path is None:
            save_path = self.output_dir / 'camera_trajectory.png'
            
        # Extract camera positions
        with torch.no_grad():
            positions = camera_model.translations.cpu().numpy()  # (N, 3)
            quaternions = camera_model.quaternions.cpu().numpy()  # (N, 4)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot camera trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2, label='Camera path')
        
        # Plot camera positions
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='red', s=20, label='Camera positions')
        
        # Mark window boundaries
        window_boundaries = set()
        for window in window_tracks:
            window_boundaries.add(window['start_frame'])
            window_boundaries.add(window['end_frame'])
        
        boundary_positions = positions[list(window_boundaries)]
        ax.scatter(boundary_positions[:, 0], boundary_positions[:, 1], boundary_positions[:, 2],
                  c='green', s=100, marker='^', label='Window boundaries')
        
        # Add camera orientations (show every 10th camera)
        for i in range(0, len(positions), 10):
            # Convert quaternion to rotation matrix
            R = self._quaternion_to_rotation_matrix(quaternions[i])
            
            # Camera forward direction (negative Z in camera space)
            forward = R @ np.array([0, 0, -1])
            
            # Draw arrow showing camera direction
            ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                     forward[0], forward[1], forward[2],
                     length=0.5, color='orange', alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Trajectory')
        ax.legend()
        
        # Equal aspect ratio
        self._set_axes_equal(ax)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved camera trajectory visualization to {save_path}")
    
    def visualize_3d_points(self,
                           window_tracks: List[Dict],
                           camera_model: Optional[torch.nn.Module] = None,
                           max_points: int = 10000,
                           save_path: Optional[Path] = None) -> None:
        """
        Visualize 3D point cloud from window tracks.
        
        Args:
            window_tracks: Window tracks with 3D points
            camera_model: Optional camera model to show cameras
            max_points: Maximum number of points to visualize
            save_path: Where to save the visualization
        """
        if save_path is None:
            save_path = self.output_dir / '3d_points.png'
            
        # Collect all 3D points
        all_points = []
        boundary_points = []
        
        for window in window_tracks:
            if 'xyzw_world' in window:
                xyzw = window['xyzw_world']  # (T, N, 4)
                
                # Sample points to avoid too many
                T, N, _ = xyzw.shape
                for t in range(0, T, 5):  # Sample every 5th frame
                    for n in range(0, N, 10):  # Sample every 10th point
                        if window['visibility'][t, n]:
                            all_points.append(xyzw[t, n, :3])
            
            # Add optimized boundary points if available
            if 'boundary_3d_optimized' in window:
                boundary_3d = window['boundary_3d_optimized']
                for point in boundary_3d:
                    boundary_points.append(point)
        
        # Convert to arrays
        if all_points:
            all_points = np.array(all_points[:max_points])
        else:
            all_points = np.zeros((0, 3))
            
        if boundary_points:
            boundary_points = np.array(boundary_points)
        
        # Create plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot regular 3D points
        if all_points.shape[0] > 0:
            ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],
                      c='blue', s=1, alpha=0.3, label=f'3D points ({all_points.shape[0]})')
        
        # Plot boundary points in different color
        if boundary_points.shape[0] > 0:
            ax.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2],
                      c='red', s=10, alpha=0.8, label=f'Boundary points ({boundary_points.shape[0]})')
        
        # Add cameras if provided
        if camera_model is not None:
            with torch.no_grad():
                positions = camera_model.translations.cpu().numpy()
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                   'g-', linewidth=2, label='Camera path')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Point Cloud')
        ax.legend()
        
        # Equal aspect ratio
        self._set_axes_equal(ax)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved 3D points visualization to {save_path}")
        
    def visualize_reprojection_errors(self,
                                     window_tracks: List[Dict],
                                     camera_model: torch.nn.Module,
                                     sample_frames: int = 5,
                                     save_path: Optional[Path] = None) -> None:
        """
        Visualize reprojection errors for sample frames.
        
        Args:
            window_tracks: Window tracks with 2D/3D correspondences
            camera_model: Optimized camera model
            sample_frames: Number of frames to visualize
            save_path: Where to save the visualization
        """
        if save_path is None:
            save_path = self.output_dir / 'reprojection_errors.png'
            
        # Select sample frames evenly distributed
        num_frames = max(w['end_frame'] for w in window_tracks) + 1
        frame_indices = np.linspace(0, num_frames-1, sample_frames, dtype=int)
        
        fig, axes = plt.subplots(1, sample_frames, figsize=(4*sample_frames, 4))
        if sample_frames == 1:
            axes = [axes]
            
        for idx, frame_idx in enumerate(frame_indices):
            ax = axes[idx]
            
            # Collect reprojection errors for this frame
            errors = []
            
            for window in window_tracks:
                if window['start_frame'] <= frame_idx <= window['end_frame']:
                    window_frame_idx = frame_idx - window['start_frame']
                    
                    # Get 2D tracks and visibility
                    tracks_2d = window['tracks'][window_frame_idx]  # (N, 2)
                    visibility = window['visibility'][window_frame_idx]  # (N,)
                    
                    if 'xyzw_world' in window:
                        xyzw = window['xyzw_world'][window_frame_idx]  # (N, 4)
                        
                        # Project 3D points to 2D
                        with torch.no_grad():
                            # Get camera parameters
                            proj_mat = camera_model.get_projection_matrices(
                                torch.tensor([frame_idx])
                            )[0].cpu().numpy()
                            
                            tan_fov_x = camera_model.tan_fov_x[frame_idx].cpu().numpy()
                            tan_fov_y = camera_model.tan_fov_y[frame_idx].cpu().numpy()
                            
                        # Project each visible point
                        for n in range(len(visibility)):
                            if visibility[n]:
                                # Project 3D to 2D
                                xyz_homo = xyzw[n]
                                xyz_cam = proj_mat @ xyz_homo
                                
                                if xyz_cam[2] > 0.1:  # Valid depth
                                    x_proj = xyz_cam[0] / xyz_cam[2]
                                    y_proj = xyz_cam[1] / xyz_cam[2]
                                    
                                    # Convert to pixel coordinates
                                    x_norm = x_proj / (tan_fov_x * xyz_cam[2])
                                    y_norm = y_proj / (tan_fov_y * xyz_cam[2])
                                    
                                    x_pix = (x_norm + 1.0) * (self.image_width / 2)
                                    y_pix = (y_norm + 1.0) * (self.image_height / 2)
                                    
                                    # Compute error
                                    error = np.linalg.norm([x_pix - tracks_2d[n, 0], 
                                                          y_pix - tracks_2d[n, 1]])
                                    errors.append(error)
            
            # Plot histogram of errors
            if errors:
                ax.hist(errors, bins=30, alpha=0.7, color='blue')
                ax.axvline(np.mean(errors), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(errors):.2f}px')
                ax.set_xlabel('Reprojection Error (pixels)')
                ax.set_ylabel('Count')
                ax.set_title(f'Frame {frame_idx}')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'Frame {frame_idx}')
        
        plt.suptitle('Reprojection Error Distribution')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved reprojection error visualization to {save_path}")
    
    def create_summary_visualization(self,
                                   camera_model: torch.nn.Module,
                                   window_tracks: List[Dict],
                                   phase1_history: Dict,
                                   phase2_history: Optional[Dict] = None,
                                   save_path: Optional[Path] = None) -> None:
        """
        Create a summary visualization with multiple plots.
        
        Args:
            camera_model: Optimized camera model
            window_tracks: Window tracks
            phase1_history: Phase 1 optimization history
            phase2_history: Optional Phase 2 optimization history
            save_path: Where to save the visualization
        """
        if save_path is None:
            save_path = self.output_dir / 'summary.png'
            
        # Create figure with subplots
        if phase2_history and 'losses' in phase2_history:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 2)
        else:
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(2, 2)
            
        # 1. Optimization loss curves
        ax1 = fig.add_subplot(gs[0, 0])
        if 'losses' in phase1_history:
            ax1.plot(phase1_history['iterations'], phase1_history['losses'], 
                    'b-', label='Phase 1', linewidth=2)
        if phase2_history and 'losses' in phase2_history:
            ax1.plot(phase2_history['iterations'], phase2_history['losses'], 
                    'r-', label='Phase 2', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Optimization Progress')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Camera trajectory (top view)
        ax2 = fig.add_subplot(gs[0, 1])
        with torch.no_grad():
            positions = camera_model.translations.cpu().numpy()
        ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
        ax2.scatter(positions[::10, 0], positions[::10, 1], c='red', s=20)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Camera Trajectory (Top View)')
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)
        
        # 3. Window coverage
        ax3 = fig.add_subplot(gs[1, 0])
        window_coverage = np.zeros(max(w['end_frame'] for w in window_tracks) + 1)
        for window in window_tracks:
            window_coverage[window['start_frame']:window['end_frame']+1] += 1
        ax3.bar(range(len(window_coverage)), window_coverage, color='blue', alpha=0.7)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Window Count')
        ax3.set_title('Window Coverage')
        ax3.grid(True, alpha=0.3)
        
        # 4. 3D points per window
        ax4 = fig.add_subplot(gs[1, 1])
        window_ids = []
        num_points = []
        num_boundary = []
        
        for window in window_tracks:
            window_ids.append(window['window_idx'])
            if 'tracks' in window:
                num_points.append(window['tracks'].shape[1])
            else:
                num_points.append(0)
                
            if 'boundary_3d_optimized' in window:
                num_boundary.append(len(window['boundary_3d_optimized']))
            else:
                num_boundary.append(0)
        
        x = np.arange(len(window_ids))
        width = 0.35
        ax4.bar(x - width/2, num_points, width, label='Total points', alpha=0.7)
        if any(num_boundary):
            ax4.bar(x + width/2, num_boundary, width, label='Boundary points', alpha=0.7)
        ax4.set_xlabel('Window ID')
        ax4.set_ylabel('Number of Points')
        ax4.set_title('Points per Window')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. If Phase 2 was used, add boundary points visualization
        if phase2_history and 'losses' in phase2_history:
            ax5 = fig.add_subplot(gs[2, :], projection='3d')
            
            # Collect boundary points
            boundary_points_start = []
            boundary_points_end = []
            
            for window in window_tracks:
                if 'boundary_3d_optimized' in window:
                    boundary_3d = window['boundary_3d_optimized']
                    query_time = window.get('query_time', None)
                    
                    if query_time is not None:
                        mask_first = (query_time == 0)
                        mask_last = (query_time == window.get('window_size', 50) - 1)
                        
                        for i, is_first in enumerate(mask_first):
                            if is_first and i < len(boundary_3d):
                                boundary_points_start.append(boundary_3d[i])
                        
                        for i, is_last in enumerate(mask_last):
                            if is_last and i < len(boundary_3d):
                                boundary_points_end.append(boundary_3d[i])
            
            if boundary_points_start:
                points_start = np.array(boundary_points_start)
                ax5.scatter(points_start[:, 0], points_start[:, 1], points_start[:, 2],
                          c='blue', s=20, alpha=0.6, label='Start boundary')
            
            if boundary_points_end:
                points_end = np.array(boundary_points_end)
                ax5.scatter(points_end[:, 0], points_end[:, 1], points_end[:, 2],
                          c='red', s=20, alpha=0.6, label='End boundary')
            
            # Add camera path
            ax5.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                    'g-', linewidth=2, alpha=0.5, label='Camera path')
            
            ax5.set_xlabel('X')
            ax5.set_ylabel('Y')
            ax5.set_zlabel('Z')
            ax5.set_title('Optimized Boundary Points')
            ax5.legend()
            self._set_axes_equal(ax5)
        
        plt.suptitle('Window Bundle Adjustment Summary', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved summary visualization to {save_path}")
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        q = q / np.linalg.norm(q)  # Normalize
        w, x, y, z = q
        
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        
        return R
    
    def _set_axes_equal(self, ax: Axes3D) -> None:
        """Set equal aspect ratio for 3D axes."""
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        centers = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        
        ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
        ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
        ax.set_zlim3d([centers[2] - radius, centers[2] + radius])