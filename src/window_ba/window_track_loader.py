"""
Window-based track loader following GeometryCrafter's approach.
No track merging - each window's tracks are kept independent.
"""

import numpy as np
import torch
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class WindowTrackLoader:
    """Load and manage window-based tracks without merging."""
    
    def __init__(self, scene_dir: Path, device: str = 'cuda'):
        self.scene_dir = Path(scene_dir)
        self.device = device
        self.intrinsics = self._load_intrinsics()
        self.distortion = self._load_distortion()
        
    def _load_intrinsics(self) -> np.ndarray:
        """Load camera intrinsic matrix from K.txt."""
        K_path = self.scene_dir / 'K.txt'
        if not K_path.exists():
            logger.warning(f"K.txt not found at {K_path}, using default intrinsics")
            # Default intrinsics (can be adjusted)
            return np.array([[1000, 0, 512],
                           [0, 1000, 288],
                           [0, 0, 1]], dtype=np.float32)
        
        K = np.loadtxt(K_path).reshape(3, 3).astype(np.float32)
        logger.info(f"Loaded intrinsics from {K_path}")
        logger.info(f"K matrix:\n{K}")
        return K
    
    def _load_distortion(self) -> Optional[np.ndarray]:
        """Load distortion coefficients from dist.txt."""
        dist_path = self.scene_dir / 'dist.txt'
        if not dist_path.exists():
            logger.warning(f"dist.txt not found at {dist_path}, assuming no distortion")
            return None
            
        dist = np.loadtxt(dist_path).astype(np.float32)
        logger.info(f"Loaded distortion coefficients: {dist}")
        return dist
    
    def get_fov_from_intrinsics(self, width: int, height: int) -> Tuple[float, float]:
        """Convert intrinsic matrix to FOV angles."""
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        
        # GeometryCrafter uses actual image dimensions
        fov_x = 2 * np.arctan(width / (2 * fx))
        fov_y = 2 * np.arctan(height / (2 * fy))
        
        logger.info(f"Computed FOV from intrinsics: FovX={np.rad2deg(fov_x):.2f}°, FovY={np.rad2deg(fov_y):.2f}°")
        return fov_x, fov_y
    
    def load_window_tracks(self, track_dir: Path, track_pattern: str = "*.npy") -> List[Dict]:
        """
        Load all window tracks independently (no merging).
        
        Returns:
            List of window track dictionaries, each containing:
            - start_frame: int
            - end_frame: int
            - tracks: np.ndarray (T, N, 2) where T=window_size, N=num_points
            - visibility: np.ndarray (T, N) boolean
            - query_time: np.ndarray (N,) indicating when point was queried
        """
        track_dir = Path(track_dir)
        track_files = sorted(track_dir.glob(track_pattern))
        
        if not track_files:
            raise ValueError(f"No track files found in {track_dir} with pattern {track_pattern}")
        
        window_tracks = []
        
        for track_file in track_files:
            logger.info(f"Loading track file: {track_file}")
            
            # Load track data
            track_data = np.load(track_file, allow_pickle=True).item()
            
            # Check if this is the expected format
            if 'tracks' not in track_data or 'metadata' not in track_data:
                logger.error(f"Unexpected format in {track_file}. Expected 'tracks' and 'metadata' keys.")
                continue
            
            # Extract metadata
            metadata = track_data['metadata']
            window_size = metadata.get('window_size', 50)
            interval = metadata.get('interval', 10)
            
            # Process each window in the tracks list
            tracks_list = track_data['tracks']
            logger.info(f"Found {len(tracks_list)} windows in {track_file}")
            
            for window_data in tracks_list:
                # Each window_data is a dictionary with window information
                window_id = window_data.get('window_id', window_data.get('window_idx', 0))
                start_frame = window_data['start_frame']
                end_frame = window_data['end_frame']
                tracks = window_data['tracks']  # (T, N, 2)
                visibility = window_data['visibility']  # (T, N)
                
                # Query time indicates when points were extracted (0=start, T-1=end)
                # In GeometryCrafter, query points are only at window boundaries
                num_points = tracks.shape[1]
                query_time = np.zeros(num_points, dtype=np.int32)
                
                # Assume first half are from start frame, second half from end frame
                # (This should be adjusted based on actual data structure)
                query_time[num_points//2:] = tracks.shape[0] - 1
                
                window_tracks.append({
                    'window_idx': window_id,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'tracks': tracks,
                    'visibility': visibility,
                    'query_time': query_time,
                    'window_size': window_size,
                    'interval': interval
                })
        
        if window_tracks:
            logger.info(f"Loaded {len(window_tracks)} window tracks")
            logger.info(f"First window: frames {window_tracks[0]['start_frame']}-{window_tracks[0]['end_frame']}")
            logger.info(f"Last window: frames {window_tracks[-1]['start_frame']}-{window_tracks[-1]['end_frame']}")
        else:
            logger.error("No window tracks were loaded!")
        
        return window_tracks
    
    def prepare_depth_sampling(self, tracks: torch.Tensor, depths: List[torch.Tensor]) -> torch.Tensor:
        """
        Sample depth values at track locations using grid sampling.
        
        Args:
            tracks: (T, N, 2) tensor of track positions in pixel coordinates
            depths: List of T depth maps, each (H, W)
            
        Returns:
            depth_values: (T, N) tensor of sampled depth values
        """
        T, N, _ = tracks.shape
        depth_values = torch.zeros(T, N, device=tracks.device)
        
        for t in range(T):
            if t >= len(depths):
                break
                
            depth_map = depths[t]  # (H, W)
            H, W = depth_map.shape
            
            # Normalize coordinates to [-1, 1] for grid_sample
            # tracks are in pixel coordinates [0, W-1] x [0, H-1]
            x = tracks[t, :, 0]  # (N,)
            y = tracks[t, :, 1]  # (N,)
            
            x_norm = 2.0 * x / (W - 1) - 1.0
            y_norm = 2.0 * y / (H - 1) - 1.0
            
            # Create grid for sampling
            grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
            
            # Add batch and channel dimensions to depth
            depth_4d = depth_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            
            # Sample depth values
            sampled = F.grid_sample(depth_4d, grid, mode='bilinear', align_corners=True)  # (1, 1, 1, N)
            depth_values[t] = sampled.squeeze()
        
        return depth_values
    
    def create_tracks_with_depth(self, window_tracks: List[Dict], depth_dir: Path) -> List[Dict]:
        """
        Add depth channel to tracks by sampling from depth maps.
        
        Updates each window track dict to include:
        - tracks_3d: (T, N, 3) with [x, y, depth]
        """
        depth_dir = Path(depth_dir)
        
        for window in window_tracks:
            start_frame = window['start_frame']
            end_frame = window['end_frame']
            tracks = torch.from_numpy(window['tracks']).to(self.device)  # (T, N, 2)
            
            # Load depth maps for this window
            depths = []
            for frame_idx in range(start_frame, end_frame + 1):
                # Assuming depth files are named like 00000.npz, 00005.npz, etc.
                depth_file = depth_dir / f"{frame_idx:05d}.npz"
                if depth_file.exists():
                    depth_data = np.load(depth_file)
                    depth = torch.from_numpy(depth_data['depth']).to(self.device)
                    depths.append(depth)
                else:
                    logger.warning(f"Depth file not found: {depth_file}")
                    # Use dummy depth
                    H, W = 576, 1024  # Default size
                    depths.append(torch.ones(H, W, device=self.device))
            
            # Sample depth at track locations
            depth_values = self.prepare_depth_sampling(tracks, depths)  # (T, N)
            
            # Create 3D tracks [x, y, depth]
            tracks_3d = torch.cat([tracks, depth_values.unsqueeze(-1)], dim=-1)  # (T, N, 3)
            
            window['tracks_3d'] = tracks_3d.cpu().numpy()
            window['depth_sampled'] = True
        
        logger.info("Added depth channel to all window tracks")
        return window_tracks