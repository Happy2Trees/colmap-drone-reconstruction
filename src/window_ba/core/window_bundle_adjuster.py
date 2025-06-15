"""
Window-based Bundle Adjuster following GeometryCrafter's cross-projection approach.
Refactored version with cleaner structure.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import time
import random
from typing import List, Dict, Tuple
from tqdm import tqdm

from ..utils.config_manager import OptimizationConfig
from ..models.data_models import WindowTrackData, OptimizationResult
from ..models.camera_model import CameraModel
from ..utils.geometry_utils import (
    unproject_points_with_fov, project_points_with_fov,
    apply_robust_loss
)

logger = logging.getLogger(__name__)


class WindowBundleAdjuster:
    """
    Window-aware cross-projection bundle adjustment.
    
    This class implements the two-phase optimization:
    - Phase 1: Camera-only optimization with fixed 3D from depth
    - Phase 2: Joint optimization of cameras and boundary 3D points
    """
    
    def __init__(self, 
                 config: OptimizationConfig, 
                 device: str = 'cuda',
                 image_width: int = 1024, 
                 image_height: int = 576,
                 single_camera: bool = False):
        """
        Initialize bundle adjuster.
        
        Args:
            config: Optimization configuration
            device: Computation device (string)
            image_width: Image width in pixels
            image_height: Image height in pixels
            single_camera: Whether to use single camera model
        """
        self.config = config
        self.device = torch.device(device)
        self.image_width = image_width
        self.image_height = image_height
        self.single_camera = single_camera
        
        # Phase 2 specific attributes
        self.boundary_3d_points = None
        self.boundary_point_mapping = {}  # Maps window to boundary point indices
        
    def optimize_phase1(self, 
                       window_tracks: List[WindowTrackData],
                       init_tan_fov_x: float,
                       init_tan_fov_y: float) -> Tuple[CameraModel, OptimizationResult]:
        """
        Phase 1: Camera-only optimization with fixed 3D points from depth.
        
        Args:
            window_tracks: List of window track data
            init_tan_fov_x: Initial tangent of half horizontal FOV
            init_tan_fov_y: Initial tangent of half vertical FOV
            
        Returns:
            camera_model: Optimized camera model
            result: Optimization result with history
        """
        logger.info("Starting Phase 1 optimization (camera-only)")
        start_time = time.time()
        
        # Apply window sampling for debugging if configured
        sampled_tracks = self._sample_windows_for_debug(window_tracks)
        
        # Initialize camera model
        # Note: end_frame is exclusive, so no need to add 1
        max_frame = max(track.end_frame for track in window_tracks)
        logger.info(f"Max frame index: {max_frame}, Total windows: {len(window_tracks)}, Using: {len(sampled_tracks)}")
        logger.info(f"Frame ranges: {[(w.start_frame, w.end_frame) for w in sampled_tracks[:5]]}")  # Show first 5
        
        camera_model = CameraModel(
            max_frame, init_tan_fov_x, init_tan_fov_y, 
            single_camera=self.single_camera,
            device=self.device
        )
        
        # Setup optimizer
        logger.info("Setting up optimizer...")
        optimizer = self._setup_phase1_optimizer(camera_model)
        
        # Optimization loop
        history = {'losses': [], 'iterations': [], 'reprojection_errors': []}
        prev_loss = float('inf')
        loss = torch.tensor(0.0, device=self.device)  # Initialize loss
        iteration = 0  # Initialize iteration
        loss_info = {'mean_reprojection_error': 0.0}  # Initialize loss_info
        
        logger.info(f"Starting optimization with {self.config.max_iterations} iterations...")
        pbar = tqdm(range(self.config.max_iterations), 
                   desc=f"Phase 1 Optimization ({len(sampled_tracks)} windows)")
        
        for iteration in pbar:
            # Normalize quaternions
            camera_model.normalize_quaternions()
            
            # Compute loss
            loss, loss_info = self._compute_phase1_loss(sampled_tracks, camera_model)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                camera_model.parameters(), 
                self.config.gradient_clip
            )
            
            # Update parameters
            optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.6f}",
                'repr_err': f"{loss_info['mean_reprojection_error']:.1f}px"
            })
            
            # Logging
            if iteration % self.config.log_interval == 0:
                logger.info(
                    f"Iteration {iteration}: loss={loss.item():.6f}, "
                    f"reprojection_error={loss_info['mean_reprojection_error']:.3f} pixels"
                )
                history['losses'].append(loss.item())
                history['iterations'].append(iteration)
                history['reprojection_errors'].append(loss_info['mean_reprojection_error'])
            
            # Convergence check
            if abs(prev_loss - loss.item()) < self.config.convergence_threshold:
                logger.info(f"Converged at iteration {iteration}")
                break
            
            prev_loss = loss.item()
        
        pbar.close()
        
        # Report completion
        elapsed_time = time.time() - start_time
        logger.info(f"Phase 1 optimization completed in {elapsed_time:.1f} seconds")
        if 'loss_info' in locals():
            logger.info(f"Final loss: {loss.item():.6f}, Final reprojection error: {loss_info['mean_reprojection_error']:.3f} pixels")
        else:
            logger.info(f"Final loss: {loss.item():.6f}")
        
        # Create optimization result
        result = OptimizationResult(
            success=True,
            final_loss=loss.item(),
            iterations=iteration + 1,
            converged=iteration < self.config.max_iterations - 1,
            history=history,
            camera_params=camera_model.get_parameters_dict()
        )
        
        return camera_model, result
    
    def optimize_phase2(self,
                       window_tracks: List[WindowTrackData],
                       camera_model: CameraModel) -> Tuple[CameraModel, OptimizationResult]:
        """
        Phase 2: Joint optimization of cameras and 3D points.
        Only boundary points at window edges are optimized.
        
        Args:
            window_tracks: List of window track data
            camera_model: Camera model from Phase 1
            
        Returns:
            camera_model: Further optimized camera model
            result: Optimization result with history
        """
        logger.info("Starting Phase 2 optimization (camera + 3D)")
        start_time = time.time()
        
        # Apply window sampling for debugging if configured
        sampled_tracks = self._sample_windows_for_debug(window_tracks)
        
        # Initialize boundary 3D points
        self._initialize_boundary_points(sampled_tracks, camera_model)
        
        if self.boundary_3d_points is None:
            logger.warning("No boundary points found - skipping Phase 2")
            return camera_model, OptimizationResult(
                success=False,
                final_loss=0.0,
                iterations=0,
                converged=False,
                history={},
                camera_params=camera_model.get_parameters_dict()
            )
        
        # Setup optimizer for Phase 2
        logger.info(f"Setting up Phase 2 optimizer with {self.boundary_3d_points.shape[0]} boundary points...")
        optimizer = self._setup_phase2_optimizer(camera_model)
        
        # Optimization loop
        history = {'losses': [], 'iterations': [], 'reprojection_errors': []}
        prev_loss = float('inf')
        loss = torch.tensor(0.0, device=self.device)  # Initialize loss
        iteration = 0  # Initialize iteration
        loss_info = {'mean_reprojection_error': 0.0, 'num_boundary_points': 0}  # Initialize loss_info
        
        logger.info(f"Starting optimization with {self.config.max_iterations} iterations...")
        pbar = tqdm(range(self.config.max_iterations), 
                   desc=f"Phase 2 Optimization ({len(sampled_tracks)} windows)")
        
        for iteration in pbar:
            # Normalize quaternions
            camera_model.normalize_quaternions()
            
            # Compute loss
            loss, loss_info = self._compute_phase2_loss(sampled_tracks, camera_model)
            
            if loss.item() == 0:
                logger.warning("Phase 2 loss is zero - stopping")
                break
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(camera_model.parameters()) + [self.boundary_3d_points], 
                self.config.gradient_clip
            )
            
            # Update parameters
            optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.6f}",
                'repr_err': f"{loss_info['mean_reprojection_error']:.1f}px",
                'boundary_pts': loss_info['num_boundary_points']
            })
            
            # Logging
            if iteration % self.config.log_interval == 0:
                logger.info(
                    f"Phase 2 Iteration {iteration}: loss={loss.item():.6f}, "
                    f"boundary_points={loss_info['num_boundary_points']}, "
                    f"reprojection_error={loss_info['mean_reprojection_error']:.3f} pixels"
                )
                history['losses'].append(loss.item())
                history['iterations'].append(iteration)
                history['reprojection_errors'].append(loss_info['mean_reprojection_error'])
            
            # Convergence check
            if abs(prev_loss - loss.item()) < self.config.convergence_threshold:
                logger.info(f"Phase 2 converged at iteration {iteration}")
                break
            
            prev_loss = loss.item()
        
        pbar.close()
        
        # Report completion
        elapsed_time = time.time() - start_time
        logger.info(f"Phase 2 optimization completed in {elapsed_time:.1f} seconds")
        if 'loss_info' in locals():
            logger.info(f"Final loss: {loss.item():.6f}, Final reprojection error: {loss_info['mean_reprojection_error']:.3f} pixels")
        else:
            logger.info(f"Final loss: {loss.item():.6f}")
        
        # Store optimized boundary points back to sampled window tracks
        self._update_window_boundary_points(sampled_tracks)
        
        # Create optimization result
        result = OptimizationResult(
            success=True,
            final_loss=loss.item(),
            iterations=iteration + 1,
            converged=iteration < self.config.max_iterations - 1,
            history=history,
            camera_params=camera_model.get_parameters_dict(),
            boundary_3d_points=self.boundary_3d_points
        )
        
        return camera_model, result
    
    def _sample_windows_for_debug(self, window_tracks: List[WindowTrackData]) -> List[WindowTrackData]:
        """
        Sample a subset of windows for debugging if configured.
        
        Args:
            window_tracks: All available window tracks
            
        Returns:
            Sampled window tracks based on debug configuration
        """
        # If no debug sampling configured, return all windows
        if self.config.debug_num_windows is None:
            return window_tracks
        
        # Limit the number of windows
        num_windows = min(self.config.debug_num_windows, len(window_tracks))
        
        if num_windows == len(window_tracks):
            return window_tracks
        
        # Apply sampling strategy
        if self.config.debug_window_sampling == "first":
            sampled = window_tracks[:num_windows]
            logger.info(f"Debug mode: Using first {num_windows} windows out of {len(window_tracks)}")
        
        elif self.config.debug_window_sampling == "random":
            sampled = random.sample(window_tracks, num_windows)
            sampled.sort(key=lambda w: w.window_idx)  # Keep order for logging
            logger.info(f"Debug mode: Randomly sampled {num_windows} windows out of {len(window_tracks)}")
        
        elif self.config.debug_window_sampling == "evenly_spaced":
            # Select evenly spaced windows
            step = len(window_tracks) / num_windows
            indices = [int(i * step) for i in range(num_windows)]
            sampled = [window_tracks[i] for i in indices]
            logger.info(f"Debug mode: Evenly sampled {num_windows} windows out of {len(window_tracks)}")
        
        else:
            raise ValueError(f"Unknown debug_window_sampling: {self.config.debug_window_sampling}")
        
        # Log selected windows
        window_indices = [w.window_idx for w in sampled]
        logger.info(f"Selected window indices: {window_indices}")
        
        return sampled
    
    def _setup_phase1_optimizer(self, camera_model: CameraModel) -> torch.optim.Optimizer:
        """Setup optimizer for Phase 1."""
        return torch.optim.Adam([
            {'params': camera_model.quaternions, 'lr': self.config.learning_rate_camera},
            {'params': camera_model.translations, 'lr': self.config.learning_rate_translation},
            {'params': [camera_model.tan_fov_x, camera_model.tan_fov_y],
             'lr': self.config.learning_rate_fov}
        ], eps=1e-15)  # Following GeometryCrafter's epsilon setting
    
    def _setup_phase2_optimizer(self, camera_model: CameraModel) -> torch.optim.Optimizer:
        """Setup optimizer for Phase 2 with reduced learning rates for cameras."""
        return torch.optim.Adam([
            {'params': camera_model.quaternions, 'lr': self.config.learning_rate_camera * 0.1},
            {'params': camera_model.translations, 'lr': self.config.learning_rate_translation * 0.1},
            {'params': self.boundary_3d_points, 'lr': self.config.learning_rate_3d}
        ], eps=1e-15)  # Following GeometryCrafter's epsilon setting
    
    def _compute_phase1_loss(self, 
                            window_tracks: List[WindowTrackData],
                            camera_model: CameraModel) -> Tuple[torch.Tensor, Dict]:
        """
        Compute cross-projection loss for Phase 1.
        
        This implements the core cross-projection loss where each frame's
        3D points (from depth) are projected to all other frames in the window.
        """
        total_loss = 0.0
        total_reprojection_error = 0.0
        num_projections = 0
        losses_per_window = []
        
        for window_idx, window_data in enumerate(window_tracks):
            # Convert to torch tensors
            window = window_data  # Using WindowTrackData directly
            
            # Log window processing (only for first iteration to avoid spam)
            if not hasattr(self, '_first_window_log_done'):
                logger.debug(f"Processing window {window_idx+1}/{len(window_tracks)} (frames {window.start_frame}-{window.end_frame})")
                if window_idx == len(window_tracks) - 1:
                    self._first_window_log_done = True
            
            # Get tracks with depth
            if not window.has_depth:
                logger.warning(f"Window {window.window_idx} has no depth information")
                continue
            
            tracks_3d = torch.from_numpy(window.tracks_3d).to(self.device)  # (T, N, 3)
            visibility = torch.from_numpy(window.visibility).to(self.device)  # (T, N)
            
            # Get track dimensions (T: frames, N: points)
            _ = tracks_3d.shape
            
            # Frame indices for this window
            frame_indices = torch.arange(window.start_frame, window.end_frame, device=self.device)
            
            # Validate frame indices
            if frame_indices.max() >= camera_model.num_frames:
                logger.error(f"Frame index {frame_indices.max()} exceeds camera model size {camera_model.num_frames}")
                logger.error(f"Window {window.window_idx}: frames {window.start_frame}-{window.end_frame}")
                raise ValueError(f"Frame index out of bounds")
            
            # Get camera parameters for this window
            tan_fov_x, tan_fov_y = camera_model.get_fov_params(frame_indices)
            proj_mats = camera_model.get_projection_matrices(frame_indices)  # (T, 4, 4)
            
            # Compute world coordinates for all frames
            xyzw_world = self._tracks_to_world_coordinates(
                tracks_3d, proj_mats, tan_fov_x, tan_fov_y
            )
            
            # Cross-projection loss
            window_loss, window_info = self._compute_cross_projection_loss(
                xyzw_world, tracks_3d, visibility, proj_mats, tan_fov_x, tan_fov_y
            )
            
            if window_info['num_projections'] > 0:
                losses_per_window.append(window_loss.item())
                total_loss += window_loss
                total_reprojection_error += window_info['total_reprojection_error']
                num_projections += window_info['num_projections']
        
        # Compute mean reprojection error (but keep total loss as sum, not average)
        if num_projections > 0:
            # Note: Following GeometryCrafter, we sum losses across windows without averaging
            # This ensures sufficient gradient magnitude for optimization
            mean_reprojection_error = total_reprojection_error / num_projections
        else:
            mean_reprojection_error = 0.0
        
        info = {
            'num_windows': len(window_tracks),
            'num_projections': num_projections,
            'losses_per_window': losses_per_window,
            'mean_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            'mean_reprojection_error': mean_reprojection_error
        }
        
        # Ensure total_loss is a tensor
        if not isinstance(total_loss, torch.Tensor):
            total_loss = torch.tensor(total_loss, device=self.device)
        
        return total_loss, info
    
    def _tracks_to_world_coordinates(self,
                                   tracks_3d: torch.Tensor,
                                   proj_mats: torch.Tensor,
                                   tan_fov_x: torch.Tensor,
                                   tan_fov_y: torch.Tensor) -> torch.Tensor:
        """
        Convert tracks with depth to world coordinates.
        
        Args:
            tracks_3d: (T, N, 3) tensor with [x_pix, y_pix, depth]
            proj_mats: (T, 4, 4) world-to-camera matrices
            tan_fov_x: (T,) tangent of half horizontal FOV
            tan_fov_y: (T,) tangent of half vertical FOV
            
        Returns:
            xyzw_world: (T, N, 4) world coordinates
        """
        T, N, _ = tracks_3d.shape
        xyzw_world = torch.zeros(T, N, 4, device=self.device)
        
        for t in range(T):
            # Extract 2D points and depth
            points_2d = tracks_3d[t, :, :2]  # (N, 2)
            depths = tracks_3d[t, :, 2]  # (N,)
            
            # Unproject to camera space
            points_3d_cam = unproject_points_with_fov(
                points_2d, depths, tan_fov_x[t], tan_fov_y[t],
                self.image_width, self.image_height
            )
            
            # Transform to world space
            # proj_mats[t] is world-to-camera, so we need the inverse
            try:
                # Debug: Check matrix values before inversion
                if t == 0:  # Only log first iteration
                    logger.debug(f"proj_mats[{t}] shape: {proj_mats[t].shape}")
                    logger.debug(f"proj_mats[{t}]:\n{proj_mats[t]}")
                    logger.debug(f"proj_mats[{t}] determinant: {torch.det(proj_mats[t][:3, :3])}")
                
                # Check if matrix is singular before inverting
                det = torch.det(proj_mats[t][:3, :3])
                if torch.abs(det) < 1e-6:
                    logger.warning(f"Nearly singular matrix at frame {t}, det={det}")
                    # Use pseudo-inverse for numerical stability
                    c2w_mat = torch.linalg.pinv(proj_mats[t])
                else:
                    c2w_mat = torch.linalg.inv(proj_mats[t])
                    
            except RuntimeError:
                logger.error(f"Failed to invert proj_mats[{t}]:\n{proj_mats[t]}")
                logger.error(f"Matrix determinant: {torch.det(proj_mats[t][:3, :3])}")
                logger.error(f"Matrix diagonal: {torch.diag(proj_mats[t])}")
                # Try pseudo-inverse as fallback
                logger.warning("Using pseudo-inverse as fallback")
                c2w_mat = torch.linalg.pinv(proj_mats[t])
            
            # Create homogeneous coordinates
            ones = torch.ones(N, 1, device=self.device)
            points_cam_homo = torch.cat([points_3d_cam, ones], dim=-1)
            
            # Transform to world
            xyzw_world[t] = points_cam_homo @ c2w_mat.T
        
        return xyzw_world
    
    def _compute_cross_projection_loss(self,
                                     xyzw_world: torch.Tensor,
                                     tracks_3d: torch.Tensor,
                                     visibility: torch.Tensor,
                                     proj_mats: torch.Tensor,
                                     tan_fov_x: torch.Tensor,
                                     tan_fov_y: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute cross-projection loss within a window.
        
        For each pair of frames (i, j), project frame i's 3D points to frame j
        and compute reprojection error.
        """
        T = xyzw_world.shape[0]
        window_loss = 0.0
        total_reprojection_error = 0.0
        num_projections = 0
        
        for j in range(T):  # Target frame
            for i in range(T):  # Source frame
                if i == j:
                    continue
                
                # Get visible points in both frames
                valid_mask = visibility[i] & visibility[j]  # (N,)
                if not valid_mask.any():
                    continue
                
                # Project i-th frame's 3D points to j-th frame
                points_3d_world = xyzw_world[i, valid_mask, :3]  # (M, 3)
                
                # Transform to camera j
                points_homo = torch.cat([points_3d_world, torch.ones(len(points_3d_world), 1, device=self.device)], dim=-1)
                points_cam_j = points_homo @ proj_mats[j].T  # (M, 4)
                points_3d_cam_j = points_cam_j[:, :3]
                
                # Project to 2D
                points_2d_proj, valid_depth = project_points_with_fov(
                    points_3d_cam_j, tan_fov_x[j], tan_fov_y[j],
                    self.image_width, self.image_height
                )
                
                if not valid_depth.any():
                    continue
                
                # Ground truth 2D positions
                gt_tracks = tracks_3d[j, valid_mask, :2]  # (M, 2)
                gt_tracks_valid = gt_tracks[valid_depth]
                points_2d_proj_valid = points_2d_proj[valid_depth]
                
                # Compute reprojection error
                reproj_error = points_2d_proj_valid - gt_tracks_valid
                error_norm = torch.norm(reproj_error, dim=-1)
                
                # Apply robust loss if configured
                if self.config.use_robust_loss:
                    error_loss = apply_robust_loss(
                        error_norm, 
                        sigma=self.config.robust_loss_sigma,
                        loss_type='huber'
                    ).mean()
                else:
                    error_loss = (error_norm**2).mean()
                
                # Add depth consistency loss if configured
                if self.config.depth_loss_weight > 0:
                    # Compare projected depth with GeometryCrafter's depth
                    projected_depth = points_cam_j[valid_depth, 2]
                    predicted_depth = tracks_3d[j, valid_mask, 2][valid_depth]
                    
                    depth_ratio = projected_depth / (predicted_depth + 1e-6)
                    depth_loss = torch.nn.functional.mse_loss(
                        depth_ratio, torch.ones_like(depth_ratio)
                    )
                    
                    loss = (self.config.proj_loss_weight * error_loss + 
                           self.config.depth_loss_weight * depth_loss)
                else:
                    loss = self.config.proj_loss_weight * error_loss
                
                window_loss += loss
                total_reprojection_error += error_norm.sum().item()
                num_projections += len(error_norm)
        
        # Note: Following GeometryCrafter, we only average over visible points,
        # not over frame pairs. This ensures sufficient gradient magnitude.
        
        info = {
            'num_projections': num_projections,
            'total_reprojection_error': total_reprojection_error
        }
        
        # Ensure window_loss is a tensor
        if not isinstance(window_loss, torch.Tensor):
            window_loss = torch.tensor(window_loss, device=self.device)
        
        return window_loss, info
    
    def _initialize_boundary_points(self,
                                  window_tracks: List[WindowTrackData],
                                  camera_model: CameraModel):
        """Initialize boundary 3D points for Phase 2 optimization."""
        boundary_points_list = []
        self.boundary_point_mapping = {}
        
        for window in window_tracks:
            if window.query_time is None:
                continue
            
            # Get boundary masks
            mask_start = window.boundary_mask_start
            mask_end = window.boundary_mask_end
            
            # Initialize 3D points from boundary frames
            start_idx = len(boundary_points_list)
            
            # Process start boundary
            if mask_start.any():
                boundary_3d = self._extract_boundary_3d_points(
                    window, mask_start, 0, camera_model
                )
                boundary_points_list.extend(boundary_3d)
            
            # Process end boundary
            if mask_end.any():
                boundary_3d = self._extract_boundary_3d_points(
                    window, mask_end, window.num_frames - 1, camera_model
                )
                boundary_points_list.extend(boundary_3d)
            
            end_idx = len(boundary_points_list)
            
            if end_idx > start_idx:
                self.boundary_point_mapping[window.window_idx] = (start_idx, end_idx)
        
        if boundary_points_list:
            # Stack all boundary points
            all_boundary_points = torch.stack(boundary_points_list)
            self.boundary_3d_points = nn.Parameter(all_boundary_points)
            logger.info(f"Initialized {len(all_boundary_points)} boundary 3D points for optimization")
        else:
            logger.warning("No boundary points found for Phase 2 initialization")
            self.boundary_3d_points = None
    
    def _extract_boundary_3d_points(self,
                                  window: WindowTrackData,
                                  mask: np.ndarray,
                                  frame_idx_in_window: int,
                                  camera_model: CameraModel) -> List[torch.Tensor]:
        """Extract 3D points for boundary frame."""
        boundary_points = []
        
        # Get global frame index
        global_frame_idx = window.start_frame + frame_idx_in_window
        
        # Get tracks with depth
        if window.tracks_3d is None:
            return []
        tracks_3d = torch.from_numpy(window.tracks_3d[frame_idx_in_window, mask]).to(self.device)
        
        # Unproject to world coordinates
        points_3d_world = camera_model.unproject_points(
            tracks_3d[:, :2],  # 2D points
            tracks_3d[:, 2],   # depths
            global_frame_idx,
            self.image_width,
            self.image_height
        )
        
        for point in points_3d_world:
            boundary_points.append(point)
        
        return boundary_points
    
    def _compute_phase2_loss(self,
                           window_tracks: List[WindowTrackData],
                           camera_model: CameraModel) -> Tuple[torch.Tensor, Dict]:
        """Compute loss for Phase 2 using optimizable boundary 3D points."""
        if self.boundary_3d_points is None:
            return torch.tensor(0.0, device=self.device), {'error': 'No boundary points'}
        
        total_loss = 0.0
        total_reprojection_error = 0.0
        num_projections = 0
        
        for window in window_tracks:
            if window.window_idx not in self.boundary_point_mapping:
                continue
            
            # Get boundary points for this window
            start_idx, end_idx = self.boundary_point_mapping[window.window_idx]
            window_boundary_points = self.boundary_3d_points[start_idx:end_idx]
            
            # Compute reprojection loss for boundary points
            loss, info = self._compute_boundary_reprojection_loss(
                window, window_boundary_points, camera_model
            )
            
            total_loss += loss
            total_reprojection_error += info['total_reprojection_error']
            num_projections += info['num_projections']
        
        # Scale by projection loss weight
        total_loss *= self.config.proj_loss_weight
        
        info = {
            'num_boundary_points': len(self.boundary_3d_points),
            'num_projections': num_projections,
            'mean_reprojection_error': total_reprojection_error / max(1, num_projections)
        }
        
        # Ensure total_loss is a tensor
        if not isinstance(total_loss, torch.Tensor):
            total_loss = torch.tensor(total_loss, device=self.device)
        
        return total_loss, info
    
    def _compute_boundary_reprojection_loss(self,
                                          window: WindowTrackData,
                                          boundary_points: torch.Tensor,
                                          camera_model: CameraModel) -> Tuple[torch.Tensor, Dict]:
        """Compute reprojection loss for boundary points across all frames in window."""
        total_loss = 0.0
        total_error = 0.0
        num_projections = 0
        
        # Get 2D tracks and visibility
        tracks_2d = torch.from_numpy(window.tracks).float().to(self.device)
        visibility = torch.from_numpy(window.visibility).to(self.device)
        
        # Split boundary points back into start/end groups
        mask_start = window.boundary_mask_start
        mask_end = window.boundary_mask_end
        n_start = mask_start.sum()
        n_end = mask_end.sum()
        
        # Project boundary points to all frames
        for t in range(window.num_frames):
            global_frame_idx = window.start_frame + t
            
            # Project start boundary points
            if n_start > 0:
                loss_start, error_start, n_proj_start = self._project_and_compute_loss(
                    boundary_points[:n_start],
                    tracks_2d[t, mask_start],
                    visibility[t, mask_start],
                    global_frame_idx,
                    camera_model
                )
                total_loss += loss_start
                total_error += error_start
                num_projections += n_proj_start
            
            # Project end boundary points
            if n_end > 0:
                loss_end, error_end, n_proj_end = self._project_and_compute_loss(
                    boundary_points[n_start:n_start+n_end],
                    tracks_2d[t, mask_end],
                    visibility[t, mask_end],
                    global_frame_idx,
                    camera_model
                )
                total_loss += loss_end
                total_error += error_end
                num_projections += n_proj_end
        
        info = {
            'total_reprojection_error': total_error,
            'num_projections': num_projections
        }
        
        # Ensure total_loss is a tensor
        if not isinstance(total_loss, torch.Tensor):
            total_loss = torch.tensor(total_loss, device=self.device)
        
        return total_loss, info
    
    def _project_and_compute_loss(self,
                                 points_3d: torch.Tensor,
                                 gt_2d: torch.Tensor,
                                 visibility: torch.Tensor,
                                 frame_idx: int,
                                 camera_model: CameraModel) -> Tuple[torch.Tensor, float, int]:
        """Project 3D points and compute loss against ground truth."""
        # Project to 2D
        points_2d_proj, valid_mask = camera_model.project_points(
            points_3d, frame_idx, self.image_width, self.image_height
        )
        
        # Apply visibility mask
        valid_mask = valid_mask & visibility
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=self.device), 0.0, 0
        
        # Compute error
        error_vec = points_2d_proj[valid_mask] - gt_2d[valid_mask]
        error_norm = torch.norm(error_vec, dim=-1)
        
        # Apply robust loss
        if self.config.use_robust_loss:
            loss = apply_robust_loss(
                error_norm,
                sigma=self.config.robust_loss_sigma,
                loss_type='huber'
            ).mean()
        else:
            loss = (error_norm**2).mean()
        
        total_error = error_norm.sum().item()
        num_valid = valid_mask.sum().item()
        
        # Ensure loss is a tensor
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=self.device)
        
        return loss, total_error, int(num_valid)
    
    def _update_window_boundary_points(self, window_tracks: List[WindowTrackData]):
        """Store optimized boundary points back to window tracks."""
        for window in window_tracks:
            if window.window_idx not in self.boundary_point_mapping:
                continue
            
            start_idx, end_idx = self.boundary_point_mapping[window.window_idx]
            if self.boundary_3d_points is not None:
                boundary_points = self.boundary_3d_points[start_idx:end_idx].detach().cpu().numpy()
            else:
                continue
            
            window.boundary_3d_optimized = boundary_points
            
        logger.info("Updated window tracks with optimized boundary points")