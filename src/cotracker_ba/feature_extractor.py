"""Co-Tracker Feature Extraction Module

This module implements sliding window feature tracking using Co-Tracker.
"""

import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# Add CoTracker to path
sys.path.append(str(Path(__file__).parent.parent.parent / "submodules" / "co-tracker"))

try:
    from cotracker.predictor import CoTrackerPredictor
    from cotracker.utils.visualizer import Visualizer, read_video_from_path
except ImportError:
    print("Co-Tracker not found. Please ensure it's in submodules/co-tracker/")
    raise

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a feature track across multiple frames"""
    track_id: int
    points: np.ndarray  # Shape: (n_frames, 2) - (x, y) coordinates
    visibility: np.ndarray  # Shape: (n_frames,) - visibility flag for each frame
    confidence: np.ndarray  # Shape: (n_frames,) - confidence score for each frame
    start_frame: int
    end_frame: int


class CoTrackerExtractor:
    """Extracts feature tracks from video using Co-Tracker with sliding window approach"""
    
    def __init__(self, 
                 window_size: int = 30,
                 overlap: int = 15,
                 grid_size: int = 50,
                 confidence_threshold: float = 0.8,
                 device: str = 'cuda',
                 checkpoint_path: Optional[str] = None):
        """
        Initialize Co-Tracker extractor
        
        Args:
            window_size: Number of frames per window
            overlap: Number of overlapping frames between windows
            grid_size: Initial grid sampling size for feature points
            confidence_threshold: Minimum confidence score for valid tracks
            device: Device to run inference on ('cuda' or 'cpu')
            checkpoint_path: Path to model checkpoint (if None, downloads from torch.hub)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        self.grid_size = grid_size
        self.confidence_threshold = confidence_threshold
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Initialize Co-Tracker model
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model = CoTrackerPredictor(checkpoint=checkpoint_path)
        else:
            # Download from torch hub
            logger.info("Loading CoTracker from torch hub...")
            self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Initialized CoTrackerExtractor with window_size={window_size}, "
                   f"overlap={overlap}, grid_size={grid_size}")
    
    def load_video_frames(self, video_path: str, max_frames: Optional[int] = None) -> np.ndarray:
        """Load video frames into memory
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load (None for all)
            
        Returns:
            frames: Array of shape (n_frames, height, width, 3)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames could be loaded from {video_path}")
        
        frames = np.array(frames)
        logger.info(f"Loaded {len(frames)} frames from video")
        return frames
    
    def get_initial_queries(self, frame: np.ndarray, query_frame: int = 0) -> torch.Tensor:
        """Generate initial query points for tracking
        
        Args:
            frame: Frame to sample points from (H, W, 3)
            query_frame: Frame index where points are sampled
            
        Returns:
            queries: Tensor of shape (n_points, 3) with (t, x, y) coordinates
        """
        h, w = frame.shape[:2]
        
        # Create grid of points
        x_step = max(1, w // self.grid_size)
        y_step = max(1, h // self.grid_size)
        
        queries = []
        for y in range(y_step // 2, h - y_step // 2, y_step):
            for x in range(x_step // 2, w - x_step // 2, x_step):
                queries.append([float(query_frame), float(x), float(y)])
        
        queries = torch.tensor(queries, dtype=torch.float32)
        logger.debug(f"Generated {len(queries)} query points at frame {query_frame}")
        return queries
    
    def track_window(self, frames: np.ndarray, start_idx: int) -> List[Track]:
        """Track features in a single window
        
        Args:
            frames: All video frames (n_frames, H, W, 3)
            start_idx: Starting frame index for this window
            
        Returns:
            tracks: List of Track objects for this window
        """
        # Extract window frames
        end_idx = min(start_idx + self.window_size, len(frames))
        window_frames = frames[start_idx:end_idx]
        n_window_frames = len(window_frames)
        
        # Convert to tensor with correct format
        video_tensor = torch.from_numpy(window_frames).float()
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension -> (B, T, C, H, W)
        video_tensor = video_tensor.to(self.device)
        
        # Sample points from the middle frame for better tracking
        query_frame = n_window_frames // 2
        queries = self.get_initial_queries(window_frames[query_frame], query_frame)
        queries = queries.to(self.device)
        queries = queries.unsqueeze(0)  # Add batch dimension -> (B, N, 3)
        
        # Run Co-Tracker with backward tracking
        with torch.no_grad():
            pred_tracks, pred_visibility = self.model(
                video_tensor, 
                queries=queries,
                backward_tracking=True
            )
        
        # Convert predictions to numpy
        pred_tracks = pred_tracks.squeeze(0).cpu().numpy()  # (N, T, 2)
        pred_visibility = pred_visibility.squeeze(0).cpu().numpy()  # (N, T)
        
        # Create Track objects
        tracks = []
        for i in range(len(pred_tracks)):
            # Check visibility
            visible_frames = pred_visibility[i] > 0.5
            
            if np.sum(visible_frames) < 5:  # Require at least 5 visible frames
                continue
            
            # Use visibility as confidence since CoTracker doesn't return separate confidence
            track = Track(
                track_id=i,  # Will be reassigned globally later
                points=pred_tracks[i],
                visibility=pred_visibility[i],
                confidence=pred_visibility[i],  # Use visibility as confidence
                start_frame=start_idx,
                end_frame=start_idx + n_window_frames - 1
            )
            tracks.append(track)
        
        logger.info(f"Window [{start_idx}:{end_idx}] - Tracked {len(tracks)} features")
        return tracks
    
    def extract_tracks(self, video_path: str) -> List[Track]:
        """Extract feature tracks from entire video using sliding windows
        
        Args:
            video_path: Path to video file
            
        Returns:
            all_tracks: List of all Track objects across the video
        """
        # Load video frames
        frames = self.load_video_frames(video_path)
        n_frames = len(frames)
        
        # Process video in sliding windows
        all_tracks = []
        global_track_id = 0
        
        for start_idx in range(0, n_frames, self.stride):
            # Track features in this window
            window_tracks = self.track_window(frames, start_idx)
            
            # Assign global track IDs
            for track in window_tracks:
                track.track_id = global_track_id
                global_track_id += 1
                all_tracks.append(track)
            
            # Check if we've processed all frames
            if start_idx + self.window_size >= n_frames:
                break
        
        logger.info(f"Extracted {len(all_tracks)} tracks from {n_frames} frames")
        return all_tracks
    
    def visualize_tracks(self, 
                        video_path: str,
                        tracks: List[Track],
                        output_path: str,
                        max_tracks: int = 100):
        """Visualize tracks overlaid on video
        
        Args:
            video_path: Path to input video
            tracks: List of Track objects
            output_path: Path to save visualization video
            max_tracks: Maximum number of tracks to visualize
        """
        # Load video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Select tracks to visualize
        vis_tracks = tracks[:max_tracks]
        
        # Generate random colors for each track
        colors = np.random.randint(0, 255, (len(vis_tracks), 3))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw tracks
            for track_idx, track in enumerate(vis_tracks):
                if track.start_frame <= frame_idx <= track.end_frame:
                    local_idx = frame_idx - track.start_frame
                    if track.visibility[local_idx] > 0.5:
                        pt = track.points[local_idx].astype(int)
                        color = colors[track_idx].tolist()
                        cv2.circle(frame, tuple(pt), 3, color, -1)
                        
                        # Draw trail
                        for i in range(max(0, local_idx - 10), local_idx):
                            if track.visibility[i] > 0.5:
                                pt_prev = track.points[i].astype(int)
                                alpha = (i - max(0, local_idx - 10)) / 10
                                cv2.line(frame, tuple(pt_prev), tuple(pt), 
                                       [int(c * alpha) for c in color], 1)
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        logger.info(f"Saved track visualization to {output_path}")