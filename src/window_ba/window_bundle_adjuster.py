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
    proj_loss_weight: float = 1.0
    depth_loss_weight: float = 0.0  # Following GeometryCrafter, disabled by default
    
class CameraModel(nn.Module):
    """
    Camera model with quaternion rotation, translation, and FOV.
    Following GeometryCrafter's parametrization.
    """
    
    def __init__(self, num_frames: int, init_fov_x: float, init_fov_y: float, 
                 single_camera: bool = False):
        super().__init__()
        
        self.num_frames = num_frames
        self.single_camera = single_camera
        
        # Initialize quaternions to identity [1, 0, 0, 0]
        quaternions = torch.zeros(num_frames, 4)
        quaternions[:, 0] = 1.0  # w component
        self.quaternions = nn.Parameter(quaternions)
        
        # Initialize translations to zero
        self.translations = nn.Parameter(torch.zeros(num_frames, 3))
        
        # Initialize FOVs
        if single_camera:
            # Single FOV shared across all frames
            self.tan_fov_x = nn.Parameter(torch.tensor(init_fov_x))
            self.tan_fov_y = nn.Parameter(torch.tensor(init_fov_y))
        else:
            # Per-frame FOVs
            self.tan_fov_x = nn.Parameter(torch.full((num_frames,), init_fov_x))
            self.tan_fov_y = nn.Parameter(torch.full((num_frames,), init_fov_y))
        
        logger.info(f"Initialized camera model for {num_frames} frames (single_camera={single_camera})")
    
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
        
        These are extrinsic matrices [R|t] that transform world coordinates to camera coordinates:
        X_camera = R * X_world + t
        
        Args:
            frame_indices: (N,) tensor of frame indices
            
        Returns:
            proj_mats: (N, 4, 4) tensor of world-to-camera transformation matrices
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
        
    def get_fov_params(self, frame_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get FOV parameters for specified frames.
        
        Args:
            frame_indices: (N,) tensor of frame indices, or None for all frames
            
        Returns:
            tan_fov_x: FOV parameters for x
            tan_fov_y: FOV parameters for y
        """
        if self.single_camera:
            # Return the same FOV for all frames
            if frame_indices is not None:
                N = len(frame_indices)
                tan_fov_x = self.tan_fov_x.expand(N)
                tan_fov_y = self.tan_fov_y.expand(N)
            else:
                tan_fov_x = self.tan_fov_x.expand(self.num_frames)
                tan_fov_y = self.tan_fov_y.expand(self.num_frames)
        else:
            # Return per-frame FOVs
            if frame_indices is not None:
                tan_fov_x = self.tan_fov_x[frame_indices]
                tan_fov_y = self.tan_fov_y[frame_indices]
            else:
                tan_fov_x = self.tan_fov_x
                tan_fov_y = self.tan_fov_y
        
        return tan_fov_x, tan_fov_y

class WindowBundleAdjuster:
    """
    Window-aware cross-projection bundle adjustment.
    """
    
    def __init__(self, config: OptimizationConfig, device: str = 'cuda',
                 image_width: int = 1024, image_height: int = 576,
                 single_camera: bool = False):
        self.config = config
        self.device = device
        self.image_width = image_width
        self.image_height = image_height
        self.single_camera = single_camera
        self.camera_model = None
        self.tracks_3d = None  # For Phase 2
        self.tracks_st_idx = {}  # Start indices for each window's boundary points
        self.tracks_ed_idx = {}  # End indices for each window's boundary points
        
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
            end_frame = window['end_frame']
            tracks_3d = torch.from_numpy(window['tracks_3d']).to(self.device)  # (T, N, 3)
            visibility = torch.from_numpy(window['visibility']).to(self.device)  # (T, N)
            
            T, N, _ = tracks_3d.shape
            
            # Frame indices for this window
            frame_indices = torch.arange(st_frame, end_frame + 1, device=self.device)
            
            # Get camera parameters for this window
            tan_fov_x, tan_fov_y = camera_model.get_fov_params(frame_indices)  # (T,), (T,)
            proj_mats = camera_model.get_projection_matrices(frame_indices)  # (T, 4, 4)
            
            # 1. Triangulate 3D points from depth and camera poses
            xyzw_world = torch.zeros(T, N, 4, device=self.device)
            
            for i in range(T):
                # Convert from normalized coordinates to 3D
                x_norm = tracks_3d[i, :, 0] / (self.image_width / 2) - 1.0  # Normalize to [-1, 1]
                y_norm = tracks_3d[i, :, 1] / (self.image_height / 2) - 1.0  # Normalize to [-1, 1]
                z = tracks_3d[i, :, 2]  # depth
                
                # Convert to 3D camera space using FOV
                x_cam = x_norm * tan_fov_x[i] * z
                y_cam = y_norm * tan_fov_y[i] * z
                
                # Create homogeneous coordinates in camera space
                xyzw_cam = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=-1)  # (N, 4)
                
                # Transform to world space using inverse of camera-to-world matrix
                # proj_mats is [R|t] where R,t transform world to camera
                # So for camera to world, we need the inverse
                xyzw_world[i] = xyzw_cam @ torch.linalg.inv(proj_mats[i].float()).T
            
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
                    xyzw_j = xyzw_world[i, valid_mask] @ proj_mats[j].T  # (M, 4)
                    
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
                    x_proj_pix = (x_proj_norm + 1.0) * (self.image_width / 2)
                    y_proj_pix = (y_proj_norm + 1.0) * (self.image_height / 2)
                    
                    # Ground truth 2D positions
                    gt_tracks = tracks_3d[j, valid_mask, :2]  # (M, 2)
                    gt_tracks_valid = gt_tracks[valid_depth]
                    
                    # Compute reprojection error
                    reproj_error = torch.stack([x_proj_pix, y_proj_pix], dim=-1) - gt_tracks_valid
                    
                    # XY projection loss (2D reprojection error)
                    if self.config.use_robust_loss:
                        # Huber loss
                        error_norm = torch.norm(reproj_error, dim=-1)
                        sigma = self.config.robust_loss_sigma
                        xy_loss = torch.where(
                            error_norm < sigma,
                            0.5 * error_norm**2,
                            sigma * error_norm - 0.5 * sigma**2
                        ).mean()
                    else:
                        xy_loss = (reproj_error**2).mean()
                    
                    # Depth consistency loss (GeometryCrafter style)
                    # This is NOT comparing with ground truth depth, but checking consistency
                    # between projected depth and GeometryCrafter's predicted depth at target frame
                    if self.config.depth_loss_weight > 0:
                        # Get GeometryCrafter's predicted depth at target frame j
                        predicted_depth_j = tracks_3d[j, valid_mask, 2][valid_depth]
                        
                        # Depth ratio should be close to 1 for consistent depths
                        # Following GeometryCrafter: proj_z/predicted_depth ≈ 1
                        depth_ratio = z_proj[valid_depth] / (predicted_depth_j + 1e-6)
                        z_loss = F.mse_loss(depth_ratio, torch.ones_like(depth_ratio))
                        
                        loss = self.config.proj_loss_weight * xy_loss + self.config.depth_loss_weight * z_loss
                    else:
                        loss = self.config.proj_loss_weight * xy_loss
                    
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
        self.camera_model = CameraModel(max_frame, init_tan_fov_x, init_tan_fov_y, 
                                        single_camera=self.single_camera).to(self.device)
        
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
    
    def setup_init_track(self, window_tracks: List[Dict], camera_model: CameraModel):
        """
        Initialize 3D points from window boundary frames.
        Following GeometryCrafter's approach: only use query points at boundaries.
        """
        self.tracks = []
        self.tracks_st_idx = {}
        self.tracks_ed_idx = {}
        cur_idx = 0
        
        for window in window_tracks:
            st_frame = window['start_frame']
            end_frame = window['end_frame']
            tracks_3d = torch.from_numpy(window['tracks_3d']).float().to(self.device)  # (T, N, 3)
            query_times = window.get('query_time', None)
            
            # If no query_times info, skip boundary initialization
            if query_times is None:
                logger.warning(f"No query_times for window {st_frame}-{end_frame}, skipping boundary init")
                continue
                
            query_times = torch.from_numpy(query_times).to(self.device)
            
            # Get camera parameters for boundary frames
            # First frame
            frame_idx = torch.tensor([st_frame], device=self.device)
            proj_mat_first = camera_model.get_projection_matrices(frame_idx)[0]  # (4, 4)
            tan_fov_x_first, tan_fov_y_first = camera_model.get_fov_params(frame_idx)
            tan_fov_x_first = tan_fov_x_first[0]
            tan_fov_y_first = tan_fov_y_first[0]
            
            # Track whether this window has any boundary points
            window_has_points = False
            window_start_idx = cur_idx
            
            # Extract boundary points at first frame (query_time == 0)
            mask_first = (query_times == 0)
            if mask_first.any():
                # Convert 2D + depth to 3D world coordinates
                tracks_first = tracks_3d[0, mask_first]  # (M, 3)
                x_norm = tracks_first[:, 0] / (self.image_width / 2) - 1.0
                y_norm = tracks_first[:, 1] / (self.image_height / 2) - 1.0
                z = tracks_first[:, 2]
                
                # Convert to camera space
                x_cam = x_norm * tan_fov_x_first * z
                y_cam = y_norm * tan_fov_y_first * z
                
                # Transform to world space
                xyzw_cam = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=-1)
                xyzw_world = xyzw_cam @ torch.linalg.inv(proj_mat_first.float()).T
                
                self.tracks.append(xyzw_world[:, :3])
                if not window_has_points:
                    self.tracks_st_idx[st_frame] = cur_idx
                    window_has_points = True
                cur_idx += xyzw_world.shape[0]
                logger.debug(f"Window {st_frame}: Added {xyzw_world.shape[0]} start boundary points")
                
            # Last frame
            frame_idx = torch.tensor([end_frame], device=self.device)
            proj_mat_last = camera_model.get_projection_matrices(frame_idx)[0]
            tan_fov_x_last, tan_fov_y_last = camera_model.get_fov_params(frame_idx)
            tan_fov_x_last = tan_fov_x_last[0]
            tan_fov_y_last = tan_fov_y_last[0]
            
            # Extract boundary points at last frame
            # Use window_size - 1 as the last frame index within window
            window_size = window.get('window_size', end_frame - st_frame + 1)
            mask_last = (query_times == window_size - 1)
            if mask_last.any():
                # Convert 2D + depth to 3D world coordinates
                tracks_last = tracks_3d[-1, mask_last]  # (M, 3)
                x_norm = tracks_last[:, 0] / (self.image_width / 2) - 1.0
                y_norm = tracks_last[:, 1] / (self.image_height / 2) - 1.0
                z = tracks_last[:, 2]
                
                # Convert to camera space
                x_cam = x_norm * tan_fov_x_last * z
                y_cam = y_norm * tan_fov_y_last * z
                
                # Transform to world space
                xyzw_cam = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=-1)
                xyzw_world = xyzw_cam @ torch.linalg.inv(proj_mat_last.float()).T
                
                self.tracks.append(xyzw_world[:, :3])
                if not window_has_points:
                    self.tracks_st_idx[st_frame] = window_start_idx
                    window_has_points = True
                cur_idx += xyzw_world.shape[0]
                logger.debug(f"Window {st_frame}: Added {xyzw_world.shape[0]} end boundary points")
            
            # Only set end index if window has any points
            if window_has_points:
                self.tracks_ed_idx[st_frame] = cur_idx
            else:
                logger.debug(f"Window {st_frame}: No boundary points found")
            
        # Concatenate all boundary 3D points
        if self.tracks:
            self.tracks_3d = nn.Parameter(torch.cat(self.tracks, dim=0))
            logger.info(f"Initialized {self.tracks_3d.shape[0]} boundary 3D points for optimization")
        else:
            logger.warning("No boundary points found for Phase 2 initialization")
            self.tracks_3d = None
    
    def compute_phase2_loss(self, window_tracks: List[Dict], camera_model: CameraModel) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss for Phase 2 using optimizable boundary 3D points.
        """
        if self.tracks_3d is None:
            return torch.tensor(0.0, device=self.device), {'error': 'No boundary points'}
            
        total_loss = 0.0
        num_projections = 0
        
        for window in window_tracks:
            st_frame = window['start_frame']
            end_frame = window['end_frame']
            
            # Skip if no boundary points for this window
            if st_frame not in self.tracks_st_idx:
                continue
                
            # Get boundary 3D points for this window
            st_idx = self.tracks_st_idx[st_frame]
            ed_idx = self.tracks_ed_idx[st_frame]
            boundary_3d = self.tracks_3d[st_idx:ed_idx]  # (M, 3)
            
            if boundary_3d.shape[0] == 0:
                logger.warning(f"No boundary points for window {st_frame}-{end_frame}")
                continue
            
            # Get 2D tracks and visibility
            tracks_2d = torch.from_numpy(window['tracks']).float().to(self.device)  # (T, N, 2)
            visibility = torch.from_numpy(window['visibility']).to(self.device)  # (T, N)
            query_times = torch.from_numpy(window.get('query_time', np.array([]))).to(self.device)
            
            if query_times.numel() == 0:
                continue
                
            # Split boundary points back into first/last frame groups
            mask_first = (query_times == 0)
            # Use window_size - 1 as the last frame index within window
            window_size = window.get('window_size', end_frame - st_frame + 1)
            mask_last = (query_times == window_size - 1)
            
            # Count points for each boundary
            n_first = mask_first.sum().item()
            n_last = mask_last.sum().item()
            
            # Project boundary 3D points to all frames in window
            T = end_frame - st_frame + 1
            frame_indices = torch.arange(st_frame, end_frame + 1, device=self.device)
            proj_mats = camera_model.get_projection_matrices(frame_indices)  # (T, 4, 4)
            tan_fov_x, tan_fov_y = camera_model.get_fov_params(frame_indices)  # Use proper method
            
            # Process first frame boundary points
            if n_first > 0:
                boundary_3d_first = boundary_3d[:n_first]  # (n_first, 3)
                
                for t in range(T):
                    # Project to frame t
                    xyzw = torch.cat([boundary_3d_first, torch.ones(n_first, 1, device=self.device)], dim=-1)
                    xyzw_proj = xyzw @ proj_mats[t].T
                    
                    # Perspective division
                    z_proj = xyzw_proj[:, 2]
                    valid = z_proj > 0.1
                    
                    if not valid.any():
                        continue
                        
                    x_proj = xyzw_proj[valid, 0] / z_proj[valid]
                    y_proj = xyzw_proj[valid, 1] / z_proj[valid]
                    
                    # Convert to normalized coordinates
                    x_norm = x_proj / (tan_fov_x[t] * z_proj[valid])
                    y_norm = y_proj / (tan_fov_y[t] * z_proj[valid])
                    
                    # Convert to pixel coordinates
                    x_pix = (x_norm + 1.0) * (self.image_width / 2)
                    y_pix = (y_norm + 1.0) * (self.image_height / 2)
                    
                    # Get ground truth 2D positions
                    gt_2d = tracks_2d[t, mask_first][valid]  # (M, 2)
                    vis = visibility[t, mask_first][valid]  # (M,)
                    
                    if vis.any():
                        # Compute reprojection error
                        proj_2d = torch.stack([x_pix, y_pix], dim=-1)
                        error = (proj_2d - gt_2d) * vis.unsqueeze(-1)
                        
                        if self.config.use_robust_loss:
                            error_norm = torch.norm(error, dim=-1)
                            sigma = self.config.robust_loss_sigma
                            loss = torch.where(
                                error_norm < sigma,
                                0.5 * error_norm**2,
                                sigma * error_norm - 0.5 * sigma**2
                            ).sum() / vis.sum()
                        else:
                            loss = (error**2).sum() / (2 * vis.sum())
                            
                        total_loss += loss
                        num_projections += vis.sum().item()
            
            # Process last frame boundary points (similar logic)
            if n_last > 0:
                # Check if we have enough points
                if boundary_3d.shape[0] < n_first + n_last:
                    logger.warning(f"Not enough boundary points: have {boundary_3d.shape[0]}, need {n_first + n_last}")
                    continue
                    
                boundary_3d_last = boundary_3d[n_first:n_first+n_last]  # (n_last, 3)
                
                for t in range(T):
                    # Project to frame t
                    xyzw = torch.cat([boundary_3d_last, torch.ones(n_last, 1, device=self.device)], dim=-1)
                    xyzw_proj = xyzw @ proj_mats[t].T
                    
                    # Perspective division
                    z_proj = xyzw_proj[:, 2]
                    valid = z_proj > 0.1
                    
                    if not valid.any():
                        continue
                        
                    x_proj = xyzw_proj[valid, 0] / z_proj[valid]
                    y_proj = xyzw_proj[valid, 1] / z_proj[valid]
                    
                    # Convert to normalized coordinates
                    x_norm = x_proj / (tan_fov_x[t] * z_proj[valid])
                    y_norm = y_proj / (tan_fov_y[t] * z_proj[valid])
                    
                    # Convert to pixel coordinates
                    x_pix = (x_norm + 1.0) * (self.image_width / 2)
                    y_pix = (y_norm + 1.0) * (self.image_height / 2)
                    
                    # Get ground truth 2D positions
                    gt_2d = tracks_2d[t, mask_last][valid]  # (M, 2)
                    vis = visibility[t, mask_last][valid]  # (M,)
                    
                    if vis.any():
                        # Compute reprojection error
                        proj_2d = torch.stack([x_pix, y_pix], dim=-1)
                        error = (proj_2d - gt_2d) * vis.unsqueeze(-1)
                        
                        if self.config.use_robust_loss:
                            error_norm = torch.norm(error, dim=-1)
                            sigma = self.config.robust_loss_sigma
                            loss = torch.where(
                                error_norm < sigma,
                                0.5 * error_norm**2,
                                sigma * error_norm - 0.5 * sigma**2
                            ).sum() / vis.sum()
                        else:
                            loss = (error**2).sum() / (2 * vis.sum())
                            
                        total_loss += loss
                        num_projections += vis.sum().item()
        
        # Scale by projection loss weight
        total_loss *= self.config.proj_loss_weight
        
        info = {
            'num_boundary_points': self.tracks_3d.shape[0] if self.tracks_3d is not None else 0,
            'num_projections': num_projections,
            'mean_loss': total_loss.item() / max(1, num_projections) if num_projections > 0 else 0
        }
        
        return total_loss, info
    
    def optimize_phase2(self,
                       window_tracks: List[Dict],
                       camera_model: CameraModel) -> Tuple[CameraModel, Dict]:
        """
        Phase 2: Joint optimization of cameras and 3D points.
        Only query points at window boundaries are optimized.
        
        This follows GeometryCrafter's approach:
        - In Phase 1: Fix 3D points from depth, optimize cameras only
        - In Phase 2: Jointly optimize both cameras and sparse 3D points (boundaries)
        
        The key insight is that we don't need to optimize ALL 3D points,
        just the ones at window boundaries to ensure consistency across windows.
        """
        logger.info("Starting Phase 2 optimization (camera + 3D)")
        
        # Initialize boundary 3D points from current camera parameters
        self.setup_init_track(window_tracks, camera_model)
        
        if self.tracks_3d is None:
            logger.warning("No boundary points found - skipping Phase 2")
            return camera_model, {'phase2': 'no_boundary_points'}
        
        # Setup optimizer for Phase 2
        # Reduce learning rates for cameras since they're already optimized
        optimizer = torch.optim.Adam([
            {'params': camera_model.quaternions, 'lr': self.config.learning_rate_camera * 0.1},
            {'params': camera_model.translations, 'lr': self.config.learning_rate_translation * 0.1},
            {'params': self.tracks_3d, 'lr': self.config.learning_rate_3d}
        ])
        
        # Optimization loop
        prev_loss = float('inf')
        history = {'losses': [], 'iterations': []}
        
        for iteration in range(self.config.max_iterations):  # Same iterations as Phase 1
            # Normalize quaternions
            camera_model.normalize_quaternions()
            
            # Forward pass
            loss, info = self.compute_phase2_loss(window_tracks, camera_model)
            
            if loss.item() == 0:
                logger.warning("Phase 2 loss is zero - stopping")
                break
                
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(camera_model.parameters()) + [self.tracks_3d], 
                self.config.gradient_clip
            )
            
            # Update parameters
            optimizer.step()
            
            # Logging
            if iteration % 50 == 0:
                logger.info(f"Phase 2 Iteration {iteration}: loss={loss.item():.6f}, "
                          f"boundary_points={info['num_boundary_points']}, "
                          f"projections={info['num_projections']}")
                history['losses'].append(loss.item())
                history['iterations'].append(iteration)
            
            # Convergence check
            if abs(prev_loss - loss.item()) < self.config.convergence_threshold:
                logger.info(f"Phase 2 converged at iteration {iteration}")
                break
                
            prev_loss = loss.item()
        
        # Store optimized boundary points back to window tracks
        for window in window_tracks:
            st_frame = window['start_frame']
            if st_frame in self.tracks_st_idx:
                st_idx = self.tracks_st_idx[st_frame]
                ed_idx = self.tracks_ed_idx[st_frame]
                window['boundary_3d_optimized'] = self.tracks_3d[st_idx:ed_idx].detach().cpu().numpy()
        
        return camera_model, history