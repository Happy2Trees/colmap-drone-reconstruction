"""
Geometric utilities for Window-based Bundle Adjustment.

This module provides common geometric operations used throughout the pipeline.
"""

import torch
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def normalize_coordinates(points_2d: torch.Tensor, 
                        image_width: int, 
                        image_height: int) -> torch.Tensor:
    """
    Convert pixel coordinates to normalized coordinates [-1, 1].
    
    Args:
        points_2d: (N, 2) or (..., N, 2) tensor of 2D points in pixel coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        points_norm: Same shape as input, normalized to [-1, 1]
    """
    x_norm = 2.0 * points_2d[..., 0] / image_width - 1.0
    y_norm = 2.0 * points_2d[..., 1] / image_height - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def denormalize_coordinates(points_norm: torch.Tensor,
                          image_width: int,
                          image_height: int) -> torch.Tensor:
    """
    Convert normalized coordinates [-1, 1] to pixel coordinates.
    
    Args:
        points_norm: (N, 2) or (..., N, 2) tensor of normalized 2D points
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        points_2d: Same shape as input, in pixel coordinates
    """
    x_pix = (points_norm[..., 0] + 1.0) * (image_width / 2)
    y_pix = (points_norm[..., 1] + 1.0) * (image_height / 2)
    return torch.stack([x_pix, y_pix], dim=-1)


def project_points_with_fov(points_3d_cam: torch.Tensor,
                          tan_fov_x: torch.Tensor,
                          tan_fov_y: torch.Tensor,
                          image_width: int,
                          image_height: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D camera points to 2D image coordinates using FOV parameters.
    
    Args:
        points_3d_cam: (..., N, 3) tensor of 3D points in camera coordinates
        tan_fov_x: Tangent of half horizontal FOV (scalar or tensor)
        tan_fov_y: Tangent of half vertical FOV (scalar or tensor)
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        points_2d: (..., N, 2) tensor of 2D points in pixel coordinates
        valid_mask: (..., N) boolean tensor indicating valid projections
    """
    # Extract x, y, z coordinates
    x_cam = points_3d_cam[..., 0]
    y_cam = points_3d_cam[..., 1]
    z_cam = points_3d_cam[..., 2]
    
    # Check validity (points in front of camera)
    valid_mask = z_cam > 0.1
    
    # Perspective division and normalization
    # Handle broadcasting for tan_fov values
    if tan_fov_x.dim() == 0:  # Scalar
        x_norm = x_cam / (tan_fov_x * z_cam)
        y_norm = y_cam / (tan_fov_y * z_cam)
    else:
        # Ensure proper broadcasting
        x_norm = x_cam / (tan_fov_x.unsqueeze(-1) * z_cam)
        y_norm = y_cam / (tan_fov_y.unsqueeze(-1) * z_cam)
    
    # Convert to pixel coordinates
    x_pix = (x_norm + 1.0) * (image_width / 2)
    y_pix = (y_norm + 1.0) * (image_height / 2)
    
    points_2d = torch.stack([x_pix, y_pix], dim=-1)
    
    return points_2d, valid_mask


def unproject_points_with_fov(points_2d: torch.Tensor,
                            depths: torch.Tensor,
                            tan_fov_x: torch.Tensor,
                            tan_fov_y: torch.Tensor,
                            image_width: int,
                            image_height: int) -> torch.Tensor:
    """
    Unproject 2D points with depth to 3D camera coordinates using FOV parameters.
    
    Args:
        points_2d: (..., N, 2) tensor of 2D points in pixel coordinates
        depths: (..., N) tensor of depth values
        tan_fov_x: Tangent of half horizontal FOV (scalar or tensor)
        tan_fov_y: Tangent of half vertical FOV (scalar or tensor)
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        points_3d_cam: (..., N, 3) tensor of 3D points in camera coordinates
    """
    # Convert pixel to normalized coordinates
    points_norm = normalize_coordinates(points_2d, image_width, image_height)
    x_norm = points_norm[..., 0]
    y_norm = points_norm[..., 1]
    
    # Convert to camera space
    # Handle broadcasting for tan_fov values
    if tan_fov_x.dim() == 0:  # Scalar
        x_cam = x_norm * tan_fov_x * depths
        y_cam = y_norm * tan_fov_y * depths
    else:
        x_cam = x_norm * tan_fov_x.unsqueeze(-1) * depths
        y_cam = y_norm * tan_fov_y.unsqueeze(-1) * depths
    
    z_cam = depths
    
    points_3d_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    
    return points_3d_cam


def transform_points_3d(points_3d: torch.Tensor,
                       transform_matrix: torch.Tensor) -> torch.Tensor:
    """
    Transform 3D points using a 4x4 transformation matrix.
    
    Args:
        points_3d: (..., N, 3) tensor of 3D points
        transform_matrix: (4, 4) or (..., 4, 4) transformation matrix
        
    Returns:
        points_3d_transformed: (..., N, 3) tensor of transformed 3D points
    """
    # Create homogeneous coordinates
    ones = torch.ones(points_3d.shape[:-1] + (1,), device=points_3d.device)
    points_homo = torch.cat([points_3d, ones], dim=-1)  # (..., N, 4)
    
    # Apply transformation
    if transform_matrix.dim() == 2:
        # Single transformation matrix
        points_transformed = points_homo @ transform_matrix.T
    else:
        # Batched transformation
        points_transformed = torch.matmul(points_homo, transform_matrix.transpose(-2, -1))
    
    # Return non-homogeneous coordinates
    return points_transformed[..., :3]


def compute_reprojection_error(points_2d_pred: torch.Tensor,
                             points_2d_gt: torch.Tensor,
                             valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute reprojection error between predicted and ground truth 2D points.
    
    Args:
        points_2d_pred: (..., N, 2) tensor of predicted 2D points
        points_2d_gt: (..., N, 2) tensor of ground truth 2D points
        valid_mask: (..., N) boolean tensor for valid points
        
    Returns:
        error: Scalar tensor of mean reprojection error
    """
    # Compute L2 distance
    error_vec = points_2d_pred - points_2d_gt
    error_norm = torch.norm(error_vec, dim=-1)  # (..., N)
    
    # Apply mask if provided
    if valid_mask is not None:
        error_norm = error_norm * valid_mask.float()
        num_valid = valid_mask.sum()
        if num_valid > 0:
            error = error_norm.sum() / num_valid
        else:
            error = torch.tensor(0.0, device=error_norm.device)
    else:
        error = error_norm.mean()
    
    return error


def apply_robust_loss(errors: torch.Tensor,
                     sigma: float = 1.0,
                     loss_type: str = 'huber') -> torch.Tensor:
    """
    Apply robust loss function to errors.
    
    Args:
        errors: (...,) tensor of error values (typically L2 norms)
        sigma: Threshold parameter for robust loss
        loss_type: Type of robust loss ('huber' or 'cauchy')
        
    Returns:
        robust_errors: Same shape as input with robust loss applied
    """
    if loss_type == 'huber':
        # Huber loss: quadratic for small errors, linear for large
        robust_errors = torch.where(
            errors < sigma,
            0.5 * errors**2,
            sigma * errors - 0.5 * sigma**2
        )
    elif loss_type == 'cauchy':
        # Cauchy loss: log(1 + (error/sigma)^2)
        robust_errors = sigma**2 * torch.log(1 + (errors / sigma)**2)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return robust_errors


def triangulate_points_dlt(points_2d_list: list,
                         proj_matrices: list,
                         image_width: int,
                         image_height: int) -> torch.Tensor:
    """
    Triangulate 3D points from multiple 2D observations using DLT.
    
    Args:
        points_2d_list: List of (N, 2) tensors of 2D points from each view
        proj_matrices: List of (3, 4) or (4, 4) projection matrices
        image_width: Image width for normalization
        image_height: Image height for normalization
        
    Returns:
        points_3d: (N, 3) tensor of triangulated 3D points
    """
    num_views = len(points_2d_list)
    num_points = points_2d_list[0].shape[0]
    device = points_2d_list[0].device
    
    # Normalize 2D points for numerical stability
    points_2d_norm = []
    for points_2d in points_2d_list:
        norm_pts = normalize_coordinates(points_2d, image_width, image_height)
        points_2d_norm.append(norm_pts)
    
    # Build linear system for each point
    points_3d = torch.zeros(num_points, 3, device=device)
    
    for i in range(num_points):
        A = []
        
        for view_idx in range(num_views):
            P = proj_matrices[view_idx]
            if P.shape[0] == 4:
                P = P[:3, :]  # Use only first 3 rows
            
            x, y = points_2d_norm[view_idx][i]
            
            # DLT equations: x * P[2] - P[0] = 0, y * P[2] - P[1] = 0
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        
        A = torch.stack(A)  # (2*num_views, 4)
        
        # Solve using SVD
        _, _, V = torch.linalg.svd(A)
        X_homo = V[-1, :]  # Last row of V
        
        # Convert from homogeneous to 3D
        points_3d[i] = X_homo[:3] / X_homo[3]
    
    return points_3d


def compute_fundamental_matrix(points1: torch.Tensor,
                             points2: torch.Tensor) -> torch.Tensor:
    """
    Compute fundamental matrix from corresponding points using 8-point algorithm.
    
    Args:
        points1: (N, 2) tensor of 2D points in first image
        points2: (N, 2) tensor of 2D points in second image
        
    Returns:
        F: (3, 3) fundamental matrix
    """
    N = points1.shape[0]
    
    # Build matrix A for linear system
    A = torch.zeros(N, 9, device=points1.device)
    
    for i in range(N):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        
        A[i] = torch.tensor([
            x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1
        ], device=points1.device)
    
    # Solve using SVD
    _, _, V = torch.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    
    # Enforce rank-2 constraint
    U, S, Vt = torch.linalg.svd(F)
    S[2] = 0  # Set smallest singular value to 0
    F = U @ torch.diag(S) @ Vt
    
    return F