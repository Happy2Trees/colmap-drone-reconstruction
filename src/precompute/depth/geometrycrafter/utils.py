"""Utility functions for GeometryCrafter depth extraction."""

import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional
import cv2


def normalize_depth(depth: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalize depth map to [0, 1] range.
    
    Args:
        depth: Raw depth map
        mask: Optional valid pixel mask
        
    Returns:
        Normalized depth map
    """
    if mask is not None:
        valid_depth = depth[mask > 0]
        if len(valid_depth) > 0:
            min_depth = np.percentile(valid_depth, 1)
            max_depth = np.percentile(valid_depth, 99)
        else:
            min_depth, max_depth = 0, 1
    else:
        min_depth = np.percentile(depth, 1)
        max_depth = np.percentile(depth, 99)
    
    depth_normalized = (depth - min_depth) / (max_depth - min_depth + 1e-6)
    return np.clip(depth_normalized, 0, 1)


def point_map_to_depth(point_map: torch.Tensor) -> torch.Tensor:
    """Extract depth from point map.
    
    Args:
        point_map: Point map tensor of shape (..., 3) where last dimension is [x, y, z]
        
    Returns:
        Depth map (z coordinate)
    """
    return point_map[..., 2]


def create_depth_colormap(depth: np.ndarray, 
                         mask: Optional[np.ndarray] = None,
                         colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Create a colored visualization of depth map.
    
    Args:
        depth: Depth map
        mask: Optional valid pixel mask
        colormap: OpenCV colormap to use
        
    Returns:
        RGB visualization
    """
    # Normalize depth
    depth_norm = normalize_depth(depth, mask)
    
    # Apply colormap
    depth_vis = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), colormap)
    
    # Set invalid pixels to black
    if mask is not None:
        depth_vis[mask == 0] = 0
    
    return depth_vis


def resize_to_multiple_of_64(image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Resize image to have dimensions that are multiples of 64.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (resized_image, original_size)
    """
    h, w = image.shape[:2]
    original_size = (h, w)
    
    # Calculate new dimensions
    new_h = (h // 64) * 64
    new_w = (w // 64) * 64
    
    if new_h != h or new_w != w:
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    return image, original_size


def batch_images(image_paths: List[Path], batch_size: int) -> List[List[Path]]:
    """Split image paths into batches.
    
    Args:
        image_paths: List of image paths
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        batches.append(batch)
    return batches


def save_depth_visualization_video(depth_maps: List[np.ndarray],
                                 valid_masks: Optional[List[np.ndarray]],
                                 output_path: Path,
                                 fps: int = 30) -> None:
    """Save depth maps as a visualization video.
    
    Args:
        depth_maps: List of depth maps
        valid_masks: Optional list of valid masks
        output_path: Output video path
        fps: Frames per second
    """
    if not depth_maps:
        return
    
    # Get dimensions from first frame
    h, w = depth_maps[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    # Write frames
    for i, depth in enumerate(depth_maps):
        mask = valid_masks[i] if valid_masks else None
        vis = create_depth_colormap(depth, mask)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        writer.write(vis_bgr)
    
    writer.release()