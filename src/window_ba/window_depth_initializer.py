"""
Window-based depth initializer following GeometryCrafter's approach.
Initializes 3D points using depth maps and camera intrinsics.
"""

import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class WindowDepthInitializer:
    """Initialize 3D points from depth maps for each window independently."""
    
    def __init__(self, intrinsics: np.ndarray, device: str = 'cuda'):
        self.intrinsics = intrinsics
        self.device = device
        
        # Extract camera parameters
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.cx = intrinsics[0, 2]
        self.cy = intrinsics[1, 2]
        
        logger.info(f"Initialized with intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
    
    def unproject_to_3d(self, points_2d: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        """
        Unproject 2D points to 3D using depth values.
        
        Args:
            points_2d: (N, 2) tensor of 2D points in pixel coordinates
            depths: (N,) tensor of depth values
            
        Returns:
            points_3d: (N, 3) tensor of 3D points in camera space
        """
        x_2d = points_2d[:, 0]
        y_2d = points_2d[:, 1]
        
        # Unproject to 3D camera space
        X = (x_2d - self.cx) * depths / self.fx
        Y = (y_2d - self.cy) * depths / self.fy
        Z = depths
        
        points_3d = torch.stack([X, Y, Z], dim=-1)
        return points_3d
    
    def triangulate_window_tracks(self, 
                                window_tracks: List[Dict], 
                                cameras: Optional[Dict] = None) -> List[Dict]:
        """
        Triangulate 3D points for each window using depth.
        
        For GeometryCrafter style:
        - Each frame gets its own 3D points from depth
        - No multi-view triangulation, just depth-based unprojection
        
        Args:
            window_tracks: List of window track dictionaries with tracks_3d
            cameras: Optional camera parameters (not used in Phase 1)
            
        Returns:
            Updated window tracks with 3D points
        """
        for window in window_tracks:
            if 'tracks_3d' not in window:
                logger.warning(f"Window {window['window_idx']} missing tracks_3d, skipping")
                continue
            
            tracks_3d = torch.from_numpy(window['tracks_3d']).to(self.device)  # (T, N, 3)
            T, N, _ = tracks_3d.shape
            
            # Initialize 3D points for each frame in the window
            xyzw_world = torch.zeros(T, N, 4, device=self.device)
            
            for t in range(T):
                # Extract 2D points and depths
                points_2d = tracks_3d[t, :, :2]  # (N, 2)
                depths = tracks_3d[t, :, 2]  # (N,)
                
                # Unproject to 3D camera space
                points_3d = self.unproject_to_3d(points_2d, depths)  # (N, 3)
                
                # Convert to homogeneous coordinates
                xyzw_world[t, :, :3] = points_3d
                xyzw_world[t, :, 3] = 1.0
                
                # If cameras are provided, transform to world space
                if cameras is not None and t in cameras:
                    # TODO: Apply camera transformation
                    pass
            
            # Store 3D points
            window['xyzw_world'] = xyzw_world.cpu().numpy()  # (T, N, 4)
            
            # For query points (used in Phase 2 refinement)
            # Extract 3D points only at window boundaries
            # This is a key concept from GeometryCrafter: only optimize boundary 3D points
            query_time = window['query_time']
            query_mask_start = (query_time == 0)
            query_mask_end = (query_time == window['window_size'] - 1)
            
            # Extract 3D points at window boundaries
            # For bidirectional tracking:
            # - Points from start frame (query_time = 0) use their 3D positions at frame 0
            # - Points from end frame (query_time = T-1) use their 3D positions at frame T-1
            # This provides 3D anchors at both window boundaries for cross-window optimization
            
            # Start boundary points: extracted at frame 0, use 3D position at frame 0
            if query_mask_start.any():
                query_3d_start = xyzw_world[0, query_mask_start, :3]  # (N_start, 3)
                window['query_3d_start'] = query_3d_start.cpu().numpy()
            else:
                window['query_3d_start'] = np.zeros((0, 3), dtype=np.float32)
            
            # End boundary points: extracted at frame T-1, use 3D position at frame T-1
            if query_mask_end.any():
                end_frame_idx = window['window_size'] - 1
                query_3d_end = xyzw_world[end_frame_idx, query_mask_end, :3]  # (N_end, 3)
                window['query_3d_end'] = query_3d_end.cpu().numpy()
            else:
                window['query_3d_end'] = np.zeros((0, 3), dtype=np.float32)
            
        logger.info("Triangulated 3D points for all windows using depth")
        return window_tracks
    
    def compute_intrinsics_from_tracks(self, window_tracks: List[Dict], 
                                     image_width: int, image_height: int) -> Tuple[float, float]:
        """
        Estimate camera intrinsics from tracks if not provided.
        Following GeometryCrafter's point_map_xy2intrinsic approach.
        
        Args:
            window_tracks: List of window tracks
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            (tanFovX, tanFovY) for optimization
        """
        # If we already have intrinsics, convert to FOV
        fov_x = 2 * np.arctan(image_width / (2 * self.fx))
        fov_y = 2 * np.arctan(image_height / (2 * self.fy))
        
        tan_fov_x = np.tan(fov_x / 2)
        tan_fov_y = np.tan(fov_y / 2)
        
        logger.info(f"Computed FOV from intrinsics: tanFovX={tan_fov_x:.4f}, tanFovY={tan_fov_y:.4f}")
        
        # Alternative: estimate from track distribution (GeometryCrafter style)
        # all_tracks = []
        # for window in window_tracks:
        #     tracks = window['tracks']  # (T, N, 2)
        #     all_tracks.append(tracks.reshape(-1, 2))
        # 
        # all_tracks = np.concatenate(all_tracks, axis=0)  # (M, 2)
        # 
        # # Compute extent
        # x_min, x_max = all_tracks[:, 0].min(), all_tracks[:, 0].max()
        # y_min, y_max = all_tracks[:, 1].min(), all_tracks[:, 1].max()
        # 
        # # Estimate FOV based on track coverage
        # ...
        
        return tan_fov_x, tan_fov_y
    
