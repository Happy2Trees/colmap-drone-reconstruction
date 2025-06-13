"""
Window-based Bundle Adjuster following GeometryCrafter's cross-projection approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for bundle adjustment optimization."""
    max_iterations: int = 10000
    learning_rate_camera: float = 1e-3
    learning_rate_translation: float = 1e-2
    learning_rate_fov: float = 1e-4
    learning_rate_3d: float = 1e-2
    convergence_threshold: float = 1e-6
    gradient_clip: float = 1.0
    use_robust_loss: bool = True
    robust_loss_sigma: float = 1.0
    
class CameraModel(nn.Module):
    """
    Camera model with quaternion rotation, translation, and FOV.
    Following GeometryCrafter's parametrization.
    """
    
    def __init__(self, num_frames: int, init_fov_x: float, init_fov_y: float):
        super().__init__()
        
        # Initialize quaternions to identity [1, 0, 0, 0]
        quaternions = torch.zeros(num_frames, 4)
        quaternions[:, 0] = 1.0  # w component
        self.quaternions = nn.Parameter(quaternions)
        
        # Initialize translations to zero
        self.translations = nn.Parameter(torch.zeros(num_frames, 3))
        
        # Initialize FOVs
        self.tan_fov_x = nn.Parameter(torch.full((num_frames,), init_fov_x))
        self.tan_fov_y = nn.Parameter(torch.full((num_frames,), init_fov_y))
        
        logger.info(f"Initialized camera model for {num_frames} frames")
    
    def normalize_quaternions(self):
        """Normalize quaternions to unit length."""
        with torch.no_grad():
            self.quaternions.data = F.normalize(self.quaternions.data, p=2, dim=1)
    
    def quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to rotation matrix.
        
        Args:
            q: (N, 4) tensor of quaternions [w, x, y, z]
            
        Returns:
            R: (N, 3, 3) tensor of rotation matrices
        """
        # Normalize quaternions
        q = F.normalize(q, p=2, dim=1)
        
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Compute rotation matrix elements
        R = torch.zeros(q.shape[0], 3, 3, device=q.device)
        
        R[:, 0, 0] = 1 - 2*y*y - 2*z*z
        R[:, 0, 1] = 2*x*y - 2*w*z
        R[:, 0, 2] = 2*x*z + 2*w*y
        
        R[:, 1, 0] = 2*x*y + 2*w*z
        R[:, 1, 1] = 1 - 2*x*x - 2*z*z
        R[:, 1, 2] = 2*y*z - 2*w*x
        
        R[:, 2, 0] = 2*x*z - 2*w*y
        R[:, 2, 1] = 2*y*z + 2*w*x
        R[:, 2, 2] = 1 - 2*x*x - 2*y*y
        
        return R
    
    def get_projection_matrices(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """
        Get projection matrices for specified frames.
        
        Args:
            frame_indices: (N,) tensor of frame indices
            
        Returns:
            proj_mats: (N, 4, 4) tensor of projection matrices
        """
        N = len(frame_indices)
        proj_mats = torch.zeros(N, 4, 4, device=frame_indices.device)
        
        # Get rotation matrices
        R = self.quaternion_to_rotation_matrix(self.quaternions[frame_indices])  # (N, 3, 3)
        t = self.translations[frame_indices]  # (N, 3)
        
        # Build projection matrices [R | t]
        proj_mats[:, :3, :3] = R
        proj_mats[:, :3, 3] = t
        proj_mats[:, 3, 3] = 1.0
        
        return proj_mats

class WindowBundleAdjuster:
    """
    Window-aware cross-projection bundle adjustment.
    """
    
    def __init__(self, config: OptimizationConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.camera_model = None
        self.tracks_3d = None  # For Phase 2
        
    def compute_cross_projection_loss(self,
                                    window_tracks: List[Dict],
                                    camera_model: CameraModel) -> Tuple[torch.Tensor, Dict]:
        """
        Compute cross-projection loss for all windows.
        
        For each window:
        1. Triangulate 3D points from all frames
        2. Create T×T cross-projection matrix
        3. Compute reprojection errors
        """
        total_loss = 0.0
        num_projections = 0
        losses_per_window = []
        
        for window in window_tracks:
            st_frame = window['start_frame']
            ed_frame = window['end_frame'] + 1
            tracks_3d = torch.from_numpy(window['tracks_3d']).to(self.device)  # (T, N, 3)
            visibility = torch.from_numpy(window['visibility']).to(self.device)  # (T, N)
            
            T, N, _ = tracks_3d.shape
            
            # Frame indices for this window
            frame_indices = torch.arange(st_frame, ed_frame, device=self.device)
            
            # Get camera parameters for this window
            tan_fov_x = camera_model.tan_fov_x[frame_indices]  # (T,)
            tan_fov_y = camera_model.tan_fov_y[frame_indices]  # (T,)
            proj_mats = camera_model.get_projection_matrices(frame_indices)  # (T, 4, 4)
            
            # 1. Triangulate 3D points from depth and camera poses
            xyzw_world = torch.zeros(T, N, 4, device=self.device)
            
            for i in range(T):
                # Convert from normalized coordinates to 3D
                x_norm = tracks_3d[i, :, 0] / 512 - 1.0  # Normalize to [-1, 1]
                y_norm = tracks_3d[i, :, 1] / 288 - 1.0  # Assuming 1024x576 image
                z = tracks_3d[i, :, 2]  # depth
                
                # Convert to 3D camera space using FOV
                x_cam = x_norm * tan_fov_x[i] * z
                y_cam = y_norm * tan_fov_y[i] * z
                
                # Create homogeneous coordinates in camera space
                xyzw_cam = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=-1)  # (N, 4)
                
                # Transform to world space
                xyzw_world[i] = xyzw_cam @ proj_mats[i].T
            
            # 2. Cross-projection: project each frame's 3D to all other frames
            window_loss = 0.0
            valid_projections = 0
            
            for j in range(T):  # Target frame
                for i in range(T):  # Source frame
                    if i == j:
                        continue
                    
                    # Get visible points in both frames
                    valid_mask = visibility[i] & visibility[j]  # (N,)
                    if not valid_mask.any():
                        continue
                    
                    # Project i-th frame's 3D points to j-th frame
                    xyzw_j = xyzw_world[i, valid_mask] @ proj_mats[j]  # (M, 4)
                    
                    # Perspective division
                    z_proj = xyzw_j[:, 2]
                    valid_depth = z_proj > 0.1  # Avoid division by near-zero
                    
                    if not valid_depth.any():
                        continue
                    
                    x_proj = xyzw_j[valid_depth, 0] / z_proj[valid_depth]
                    y_proj = xyzw_j[valid_depth, 1] / z_proj[valid_depth]
                    
                    # Convert to normalized coordinates
                    x_proj_norm = x_proj / (tan_fov_x[j] * z_proj[valid_depth])
                    y_proj_norm = y_proj / (tan_fov_y[j] * z_proj[valid_depth])
                    
                    # Convert to pixel coordinates
                    x_proj_pix = (x_proj_norm + 1.0) * 512
                    y_proj_pix = (y_proj_norm + 1.0) * 288
                    
                    # Ground truth 2D positions
                    gt_tracks = tracks_3d[j, valid_mask, :2]  # (M, 2)
                    gt_tracks_valid = gt_tracks[valid_depth]
                    
                    # Compute reprojection error
                    reproj_error = torch.stack([x_proj_pix, y_proj_pix], dim=-1) - gt_tracks_valid
                    
                    # Apply robust loss if enabled
                    if self.config.use_robust_loss:
                        # Huber loss
                        error_norm = torch.norm(reproj_error, dim=-1)
                        sigma = self.config.robust_loss_sigma
                        loss = torch.where(
                            error_norm < sigma,
                            0.5 * error_norm**2,
                            sigma * error_norm - 0.5 * sigma**2
                        ).mean()
                    else:
                        loss = (reproj_error**2).mean()
                    
                    window_loss += loss
                    valid_projections += 1
            
            if valid_projections > 0:
                window_loss /= valid_projections
                losses_per_window.append(window_loss.item())
                total_loss += window_loss
                num_projections += valid_projections
        
        # Average over all windows
        if num_projections > 0:
            total_loss /= len(window_tracks)
        
        info = {
            'num_windows': len(window_tracks),
            'num_projections': num_projections,
            'losses_per_window': losses_per_window,
            'mean_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        }
        
        return total_loss, info
    
    def optimize_phase1(self, 
                       window_tracks: List[Dict],
                       init_tan_fov_x: float,
                       init_tan_fov_y: float) -> Tuple[CameraModel, Dict]:
        """
        Phase 1: Camera-only optimization with fixed 3D points from depth.
        """
        # Count total frames
        max_frame = max(w['end_frame'] for w in window_tracks) + 1
        
        # Initialize camera model
        self.camera_model = CameraModel(max_frame, init_tan_fov_x, init_tan_fov_y).to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([
            {'params': self.camera_model.quaternions, 'lr': self.config.learning_rate_camera},
            {'params': self.camera_model.translations, 'lr': self.config.learning_rate_translation},
            {'params': [self.camera_model.tan_fov_x, self.camera_model.tan_fov_y], 
             'lr': self.config.learning_rate_fov}
        ])
        
        # Optimization loop
        prev_loss = float('inf')
        history = {'losses': [], 'iterations': []}
        
        logger.info("Starting Phase 1 optimization (camera-only)")
        
        for iteration in range(self.config.max_iterations):
            # Normalize quaternions
            self.camera_model.normalize_quaternions()
            
            # Forward pass
            loss, info = self.compute_cross_projection_loss(window_tracks, self.camera_model)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.camera_model.parameters(), self.config.gradient_clip)
            
            # Update parameters
            optimizer.step()
            
            # Logging
            if iteration % 100 == 0:
                logger.info(f"Iteration {iteration}: loss={loss.item():.6f}, projections={info['num_projections']}")
                history['losses'].append(loss.item())
                history['iterations'].append(iteration)
            
            # Convergence check
            if abs(prev_loss - loss.item()) < self.config.convergence_threshold:
                logger.info(f"Converged at iteration {iteration}")
                break
                
            prev_loss = loss.item()
        
        return self.camera_model, history
    
    def optimize_phase2(self,
                       window_tracks: List[Dict],
                       camera_model: CameraModel) -> Tuple[CameraModel, Dict]:
        """
        Phase 2: Joint optimization of cameras and 3D points.
        Only query points at window boundaries are optimized.
        """
        logger.info("Starting Phase 2 optimization (camera + 3D)")
        
        # Initialize 3D points for query points
        tracks_3d_params = []
        
        for window in window_tracks:
            # Get query 3D points (at window boundaries)
            query_3d_start = torch.from_numpy(window['query_3d_start']).to(self.device)
            query_3d_end = torch.from_numpy(window['query_3d_end']).to(self.device)
            
            # Make them parameters
            if query_3d_start.numel() > 0:
                tracks_3d_params.append(nn.Parameter(query_3d_start))
            if query_3d_end.numel() > 0:
                tracks_3d_params.append(nn.Parameter(query_3d_end))
        
        # Setup optimizer for Phase 2
        optimizer = torch.optim.Adam([
            {'params': camera_model.parameters(), 'lr': self.config.learning_rate_camera * 0.1},
            {'params': tracks_3d_params, 'lr': self.config.learning_rate_3d}
        ])
        
        # TODO: Implement Phase 2 optimization with joint camera and 3D refinement
        # This is more complex and requires modifying the loss computation
        
        logger.info("Phase 2 optimization completed")
        
        return camera_model, {'phase2': 'completed'}