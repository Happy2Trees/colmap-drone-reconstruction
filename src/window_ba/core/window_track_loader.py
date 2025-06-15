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
    
    def __init__(self, scene_dir: Path, device: str = 'cuda', 
                 image_width: int = 1024, image_height: int = 576):
        self.scene_dir = Path(scene_dir)
        self.device = device
        self.image_width = image_width
        self.image_height = image_height
        self.intrinsics = self._load_intrinsics()
        self.distortion = self._load_distortion()
        
    def _load_intrinsics(self) -> np.ndarray:
        """Load camera intrinsic matrix from K.txt."""
        K_path = self.scene_dir / 'K.txt'
        if not K_path.exists():
            logger.warning(f"K.txt not found at {K_path}, using default intrinsics")
            # Default intrinsics with principal point at image center
            cx = self.image_width / 2
            cy = self.image_height / 2
            return np.array([[1000, 0, cx],
                           [0, 1000, cy],
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
                
                # Query time indicates when points were extracted within the window
                # query_time = 0 means extracted from window's first frame (relative)
                # query_time = T-1 means extracted from window's last frame
                
                # Get query times (should always be present in new data)
                if 'query_times' in window_data:
                    query_time = window_data['query_times'].astype(np.int32)
                else:
                    # Legacy data without query_times - assume all from first frame
                    num_points = tracks.shape[1]
                    query_time = np.zeros(num_points, dtype=np.int32)
                    logger.warning(f"Window {window_id}: No query_times found, assuming legacy unidirectional tracking")
                
                # Check if bidirectional tracking was used
                is_bidirectional = window_data.get('bidirectional', False)
                if is_bidirectional or np.any(query_time > 0):
                    # Bidirectional tracking confirmed
                    num_start_queries = np.sum(query_time == 0)
                    num_end_queries = np.sum(query_time > 0)
                    logger.info(f"Window {window_id}: Bidirectional tracks with {num_start_queries} start queries and {num_end_queries} end queries")
                else:
                    # Unidirectional tracking
                    logger.info(f"Window {window_id}: Unidirectional tracks with {len(query_time)} points from start frame")
                
                # GeometryCrafter uses points from both boundaries:
                # - Some from window start (query_time = 0)
                # - Some from window end (query_time = window_size - 1)
                # This enables better cross-window consistency in Phase 2
                
                window_tracks.append({
                    'window_idx': window_id,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'tracks': tracks,
                    'visibility': visibility,
                    'query_time': query_time,
                    'window_size': tracks.shape[0],  # Use actual frame count
                    'interval': interval,
                    'bidirectional': is_bidirectional
                })
        
        if window_tracks:
            logger.info(f"Loaded {len(window_tracks)} window tracks")
            logger.info(f"First window: frames {window_tracks[0]['start_frame']}-{window_tracks[0]['end_frame']-1}")
            logger.info(f"Last window: frames {window_tracks[-1]['start_frame']}-{window_tracks[-1]['end_frame']-1}")
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
            
            # Get list of available depth files to determine naming pattern
            depth_files = sorted(depth_dir.glob("*.npz"))
            if depth_files and len(depth_files) > 1:
                # Extract frame numbers from filenames
                frame_numbers = []
                for f in depth_files[:10]:  # Check first 10 files
                    if f.stem.isdigit():
                        frame_numbers.append(int(f.stem))
                
                # Detect frame interval (e.g., 5 for 00000, 00005, 00010...)
                if len(frame_numbers) > 1:
                    frame_interval = frame_numbers[1] - frame_numbers[0]
                else:
                    frame_interval = 1
            else:
                frame_interval = 1
            
            # Load depth maps with proper frame mapping
            for local_idx in range(end_frame - start_frame):
                frame_idx = start_frame + local_idx
                # Map sequence index to actual frame number
                actual_frame_num = frame_idx * frame_interval
                depth_file = depth_dir / f"{actual_frame_num:05d}.npz"
                
                if depth_file.exists():
                    depth_data = np.load(depth_file)
                    depth = torch.from_numpy(depth_data['depth']).to(self.device)
                    depths.append(depth)
                else:
                    logger.warning(f"Depth file not found: {depth_file} (frame index: {frame_idx})")
                    # Use dummy depth with configured dimensions
                    H, W = self.image_height, self.image_width
                    depths.append(torch.ones(H, W, device=self.device))
            
            # Sample depth at track locations
            depth_values = self.prepare_depth_sampling(tracks, depths)  # (T, N)
            
            # Create 3D tracks [x, y, depth]
            tracks_3d = torch.cat([tracks, depth_values.unsqueeze(-1)], dim=-1)  # (T, N, 3)
            
            window['tracks_3d'] = tracks_3d.cpu().numpy()
            window['depth_sampled'] = True
        
        logger.info("Added depth channel to all window tracks")
        return window_tracks