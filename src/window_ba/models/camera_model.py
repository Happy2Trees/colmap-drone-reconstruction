"""
Camera model for Window-based Bundle Adjustment.

This module provides the camera parametrization with quaternion rotation,
translation, and FOV following GeometryCrafter's approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

from .data_models import CameraParameters

logger = logging.getLogger(__name__)


class CameraModel(nn.Module):
    """
    Camera model with quaternion rotation, translation, and FOV.
    
    Following GeometryCrafter's parametrization:
    - Rotation: Quaternion [w, x, y, z] 
    - Translation: 3D vector
    - FOV: Tangent of half field-of-view angles
    """
    
    def __init__(self, 
                 num_frames: int, 
                 init_tan_fov_x: float, 
                 init_tan_fov_y: float,
                 single_camera: bool = False,
                 device: Optional[torch.device] = None):
        """
        Initialize camera model.
        
        Args:
            num_frames: Total number of camera frames
            init_tan_fov_x: Initial tangent of half horizontal FOV
            init_tan_fov_y: Initial tangent of half vertical FOV
            single_camera: If True, all frames share the same FOV parameters
            device: Device to create tensors on (if None, use CPU)
        """
        super().__init__()
        
        self.num_frames = num_frames
        self.single_camera = single_camera
        
        # Use provided device or default to CPU
        if device is None:
            device = torch.device('cpu')
        
        # Initialize quaternions to identity [1, 0, 0, 0]
        quaternions = torch.zeros(num_frames, 4, device=device)
        quaternions[:, 0] = 1.0  # w component
        self.quaternions = nn.Parameter(quaternions)
        
        # Initialize translations to zero
        self.translations = nn.Parameter(torch.zeros(num_frames, 3, device=device))
        
        # Initialize FOV parameters
        if single_camera:
            # Single FOV shared across all frames
            self.tan_fov_x = nn.Parameter(torch.tensor(init_tan_fov_x, device=device))
            self.tan_fov_y = nn.Parameter(torch.tensor(init_tan_fov_y, device=device))
        else:
            # Per-frame FOVs
            self.tan_fov_x = nn.Parameter(torch.full((num_frames,), init_tan_fov_x, device=device))
            self.tan_fov_y = nn.Parameter(torch.full((num_frames,), init_tan_fov_y, device=device))
        
        logger.info(f"Initialized camera model for {num_frames} frames (single_camera={single_camera}) on device {device}")
    
    def normalize_quaternions(self):
        """Normalize quaternions to unit length (in-place)."""
        with torch.no_grad():
            self.quaternions.data = F.normalize(self.quaternions.data, p=2, dim=1)
    
    @staticmethod
    def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
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
    
    def get_camera_to_world_matrices(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """
        Get camera-to-world transformation matrices (inverse of projection matrices).
        
        Args:
            frame_indices: (N,) tensor of frame indices
            
        Returns:
            c2w_mats: (N, 4, 4) tensor of camera-to-world transformation matrices
        """
        proj_mats = self.get_projection_matrices(frame_indices)
        c2w_mats = torch.linalg.inv(proj_mats)
        return c2w_mats
    
    def get_fov_params(self, frame_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get FOV parameters for specified frames.
        
        Args:
            frame_indices: (N,) tensor of frame indices, or None for all frames
            
        Returns:
            tan_fov_x: Tangent of half horizontal FOV
            tan_fov_y: Tangent of half vertical FOV
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
    
    def project_points(self, 
                      points_3d_world: torch.Tensor, 
                      frame_idx: int,
                      image_width: int,
                      image_height: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D world points to 2D image coordinates.
        
        Args:
            points_3d_world: (N, 3) tensor of 3D points in world coordinates
            frame_idx: Frame index for camera parameters
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            points_2d: (N, 2) tensor of 2D points in pixel coordinates
            valid_mask: (N,) boolean tensor indicating valid projections
        """
        # Get camera parameters
        frame_tensor = torch.tensor([frame_idx], device=points_3d_world.device)
        proj_mat = self.get_projection_matrices(frame_tensor)[0]  # (4, 4)
        tan_fov_x, tan_fov_y = self.get_fov_params(frame_tensor)
        tan_fov_x = tan_fov_x[0]
        tan_fov_y = tan_fov_y[0]
        
        # Transform to camera coordinates
        points_homo = torch.cat([points_3d_world, torch.ones(len(points_3d_world), 1, device=points_3d_world.device)], dim=-1)
        points_cam = points_homo @ proj_mat.T  # (N, 4)
        
        # Extract x, y, z in camera space
        x_cam = points_cam[:, 0]
        y_cam = points_cam[:, 1]
        z_cam = points_cam[:, 2]
        
        # Check validity (points in front of camera)
        valid_mask = z_cam > 0.1
        
        # Perspective division and normalization
        x_norm = x_cam / (tan_fov_x * z_cam)  # Normalized to [-1, 1]
        y_norm = y_cam / (tan_fov_y * z_cam)
        
        # Convert to pixel coordinates
        x_pix = (x_norm + 1.0) * (image_width / 2)
        y_pix = (y_norm + 1.0) * (image_height / 2)
        
        points_2d = torch.stack([x_pix, y_pix], dim=-1)
        
        return points_2d, valid_mask
    
    def unproject_points(self,
                        points_2d: torch.Tensor,
                        depths: torch.Tensor,
                        frame_idx: int,
                        image_width: int,
                        image_height: int) -> torch.Tensor:
        """
        Unproject 2D points with depth to 3D world coordinates.
        
        Args:
            points_2d: (N, 2) tensor of 2D points in pixel coordinates
            depths: (N,) tensor of depth values
            frame_idx: Frame index for camera parameters
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            points_3d_world: (N, 3) tensor of 3D points in world coordinates
        """
        # Get camera parameters
        frame_tensor = torch.tensor([frame_idx], device=points_2d.device)
        c2w_mat = self.get_camera_to_world_matrices(frame_tensor)[0]  # (4, 4)
        tan_fov_x, tan_fov_y = self.get_fov_params(frame_tensor)
        tan_fov_x = tan_fov_x[0]
        tan_fov_y = tan_fov_y[0]
        
        # Convert pixel to normalized coordinates
        x_norm = points_2d[:, 0] / (image_width / 2) - 1.0
        y_norm = points_2d[:, 1] / (image_height / 2) - 1.0
        
        # Convert to camera space
        x_cam = x_norm * tan_fov_x * depths
        y_cam = y_norm * tan_fov_y * depths
        z_cam = depths
        
        # Create homogeneous coordinates
        points_cam_homo = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(depths)], dim=-1)
        
        # Transform to world space
        points_world_homo = points_cam_homo @ c2w_mat.T
        points_3d_world = points_world_homo[:, :3]
        
        return points_3d_world
    
    def get_parameters_dict(self) -> CameraParameters:
        """Get camera parameters as a structured object."""
        return CameraParameters(
            quaternions=self.quaternions.detach(),
            translations=self.translations.detach(),
            tan_fov_x=self.tan_fov_x.detach(),
            tan_fov_y=self.tan_fov_y.detach()
        )
    
    def load_parameters(self, params: CameraParameters):
        """Load camera parameters from a structured object."""
        with torch.no_grad():
            self.quaternions.copy_(params.quaternions)
            self.translations.copy_(params.translations)
            
            if self.single_camera:
                # Handle scalar vs tensor for single camera
                if isinstance(params.tan_fov_x, torch.Tensor):
                    if params.tan_fov_x.numel() > 1:
                        self.tan_fov_x.copy_(params.tan_fov_x[0:1])  # Take first element as tensor
                    else:
                        self.tan_fov_x.copy_(params.tan_fov_x)
                else:
                    # Convert float to tensor
                    self.tan_fov_x.copy_(torch.tensor(params.tan_fov_x, device=self.tan_fov_x.device))
                    
                if isinstance(params.tan_fov_y, torch.Tensor):
                    if params.tan_fov_y.numel() > 1:
                        self.tan_fov_y.copy_(params.tan_fov_y[0:1])  # Take first element as tensor
                    else:
                        self.tan_fov_y.copy_(params.tan_fov_y)
                else:
                    # Convert float to tensor
                    self.tan_fov_y.copy_(torch.tensor(params.tan_fov_y, device=self.tan_fov_y.device))
            else:
                # For multiple cameras
                if isinstance(params.tan_fov_x, torch.Tensor):
                    self.tan_fov_x.copy_(params.tan_fov_x)
                else:
                    # Convert float to tensor and expand to match number of frames
                    self.tan_fov_x.copy_(torch.full_like(self.tan_fov_x, params.tan_fov_x))
                    
                if isinstance(params.tan_fov_y, torch.Tensor):
                    self.tan_fov_y.copy_(params.tan_fov_y)
                else:
                    # Convert float to tensor and expand to match number of frames
                    self.tan_fov_y.copy_(torch.full_like(self.tan_fov_y, params.tan_fov_y))