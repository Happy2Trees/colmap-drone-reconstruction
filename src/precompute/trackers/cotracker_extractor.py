"""Extract point tracks using CoTracker with interval-based windowing"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
from PIL import Image
import logging
import cv2
import imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from .base_tracker import BaseTracker
from .feature_initializers import (
    GridInitializer,
    SIFTInitializer, 
    SuperPointInitializer
)


class CoTrackerExtractor(BaseTracker):
    """Extract point tracks using CoTracker with interval-based windowing"""
    
    def __init__(self, 
                 window_size: int = 48,
                 interval: int = 10,
                 initialization_method: str = 'grid',
                 grid_size: int = 20,
                 device: str = 'cuda',
                 max_features: int = 400,
                 superpoint_weights: Optional[str] = None):
        """
        Args:
            window_size: Number of frames per window
            interval: Frame interval between window starts
            initialization_method: 'grid', 'sift', or 'superpoint'
            grid_size: Grid size for grid initialization (grid_size x grid_size points)
            device: Device to run on
            max_features: Maximum number of features for SIFT/SuperPoint
            superpoint_weights: Path to SuperPoint weights file (optional)
        """
        super().__init__(device)
        self.window_size = window_size
        self.interval = interval
        self.initialization_method = initialization_method
        self.grid_size = grid_size
        self.max_features = max_features
        
        # Load model
        self.model = self._load_model()
        
        # Initialize feature extractor
        self.feature_initializer = self._get_feature_initializer(superpoint_weights)
    
    def _load_model(self):
        """Load CoTracker model"""
        logging.info("Loading CoTracker3 OFFLINE model...")
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        model = model.to(self.device)
        model.eval()
        logging.info("Model loaded successfully")
        return model
    
    def _get_feature_initializer(self, superpoint_weights: Optional[str] = None):
        """Get the appropriate feature initializer based on method"""
        if self.initialization_method == 'grid':
            return GridInitializer(grid_size=self.grid_size, max_features=self.grid_size * self.grid_size)
        elif self.initialization_method == 'sift':
            return SIFTInitializer(max_features=self.max_features, grid_filter=True)
        elif self.initialization_method == 'superpoint':
            return SuperPointInitializer(
                max_features=self.max_features,
                grid_filter=True,
                weights_path=superpoint_weights,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown initialization method: {self.initialization_method}")
    
    def _generate_windows(self, total_frames: int) -> List[Tuple[int, int]]:
        """Generate interval-based windows"""
        windows = []
        window_start = 0
        
        while window_start < total_frames:
            window_end = min(window_start + self.window_size, total_frames)
            
            # Only add window if it has at least 2 frames
            if window_end - window_start >= 2:
                windows.append((window_start, window_end))
            
            # Move to next window start
            window_start += self.interval
            
            # Stop if next window would start beyond sequence
            if window_start >= total_frames:
                break
        
        logging.info(f"Generated {len(windows)} windows for {total_frames} frames")
        for i, (start, end) in enumerate(windows):
            logging.info(f"  Window {i}: frames [{start}, {end})")
        
        return windows
    
    def _get_query_points(self, 
                         image_path: Path,
                         window_frame_offset: int = 0) -> Tuple[np.ndarray, Optional[Dict]]:
        """Generate query points based on initialization method
        
        Args:
            image_path: Path to the image
            window_frame_offset: Time offset for query points
            
        Returns:
            query_points: Array of shape (N, 3) with [time, x, y]
            extra_info: Additional feature information (keypoints, descriptors, etc.)
        """
        return self.feature_initializer.extract_features(image_path, window_frame_offset)
    
    
    def _load_images_for_window(self, image_paths: List[Path], 
                               start_idx: int, end_idx: int) -> np.ndarray:
        """Load images for a specific window"""
        window_paths = image_paths[start_idx:end_idx]
        images = []
        
        for img_path in window_paths:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            images.append(img_array)
        
        return np.stack(images)
    
    def extract_tracks(self, 
                      image_dir: str,
                      output_path: Optional[str] = None) -> Dict:
        """
        Extract tracks for all windows and save as .npy file
        
        Args:
            image_dir: Directory containing images
            output_path: Output path for .npy file. If None, auto-generated.
            
        Returns:
            Dictionary containing tracks and metadata
        """
        image_dir = Path(image_dir)
        
        # Get image paths
        image_paths = self._get_image_paths(image_dir)
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        total_frames = len(image_paths)
        logging.info(f"Found {total_frames} images")
        
        # Get image dimensions
        img_height, img_width = self._get_image_dimensions(image_paths[0])
        logging.info(f"Image dimensions: {img_width}x{img_height}")
        
        # Generate windows
        windows = self._generate_windows(total_frames)
        
        # Process each window
        all_window_tracks = []
        
        for window_idx, (start, end) in enumerate(windows):
            logging.info(f"Processing window {window_idx}/{len(windows)}: frames [{start}, {end})")
            
            # Load window frames
            window_frames = self._load_images_for_window(image_paths, start, end)
            video_tensor = torch.from_numpy(window_frames).permute(0, 3, 1, 2).float()
            video_tensor = video_tensor.unsqueeze(0).to(self.device)
            
            # Get query points for this window
            first_frame_path = image_paths[start]
            query_points, extra_info = self._get_query_points(first_frame_path, window_frame_offset=0)
            queries = torch.from_numpy(query_points).float().to(self.device)
            queries = queries.unsqueeze(0)  # Add batch dimension
            
            # Run tracking
            with torch.no_grad():
                pred_tracks, pred_visibility = self.model(video_tensor, queries=queries)
            
            # Store window tracks - only essential data
            window_data = {
                'window_id': window_idx,
                'start_frame': start,
                'end_frame': end,
                'tracks': pred_tracks[0].cpu().numpy(),  # Remove batch dim
                'visibility': pred_visibility[0].cpu().numpy(),  # Remove batch dim
            }
            
            all_window_tracks.append(window_data)
            
            # Clear GPU memory
            del window_frames, video_tensor, pred_tracks, pred_visibility
            torch.cuda.empty_cache()
        
        # Prepare output data - minimal structure
        output_data = {
            'tracks': all_window_tracks,
            'metadata': {
                'window_size': self.window_size,
                'interval': self.interval,
                'initialization_method': self.initialization_method,
                'total_frames': total_frames,
            }
        }
        
        # Save to file
        if output_path is None:
            scene_dir = image_dir.parent
            cotracker_dir = scene_dir / 'cotracker'
            cotracker_dir.mkdir(exist_ok=True)
            
            filename = f"{self.window_size}_{self.interval}_{self.initialization_method}.npy"
            output_path = cotracker_dir / filename
        
        np.save(output_path, output_data, allow_pickle=True)
        logging.info(f"Saved tracks to {output_path}")
        
        return output_data
    
    def load_tracks(self, track_path: str) -> Dict:
        """Load tracks from saved file"""
        data = np.load(track_path, allow_pickle=True).item()
        return data
    
    def visualize_tracks(self, 
                        image_dir: str,
                        track_data: Optional[Dict] = None,
                        track_path: Optional[str] = None,
                        output_dir: Optional[str] = None,
                        save_video: bool = True,
                        save_frames: bool = False,
                        fps: int = 10) -> None:
        """
        Visualize tracks as video and/or frames
        
        Args:
            image_dir: Directory containing images
            track_data: Track data dictionary (if None, loads from track_path)
            track_path: Path to saved tracks (if track_data is None)
            output_dir: Output directory for visualization (if None, auto-generated)
            save_video: Save as MP4 video
            save_frames: Save sample frames
            fps: Output video FPS
        """
        image_dir = Path(image_dir)
        
        # Load track data if not provided
        if track_data is None:
            if track_path is None:
                raise ValueError("Either track_data or track_path must be provided")
            track_data = self.load_tracks(track_path)
        
        # Setup output directory
        if output_dir is None:
            scene_dir = image_dir.parent
            viz_dir = scene_dir / 'visualizations'
            viz_dir.mkdir(exist_ok=True)
            
            metadata = track_data['metadata']
            suffix = f"{metadata['window_size']}_{metadata['interval']}_{metadata['initialization_method']}"
            output_dir = viz_dir / f"cotracker_{suffix}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Creating visualizations in {output_dir}")
        
        # Get image paths
        image_paths = self._get_image_paths(image_dir)
        total_frames = len(image_paths)
        
        # Build trajectories for visualization
        window_trajectories = self._build_window_trajectories(track_data, total_frames)
        
        # Generate colors for each window
        window_colors = []
        for window_data in window_trajectories:
            num_points = len(window_data['trajectories'])
            colors = self._generate_distinct_colors(num_points)
            window_colors.append(colors)
        
        if save_video:
            self._create_video(image_paths, window_trajectories, window_colors, 
                             output_dir, fps, save_frames)
        
        # Create summary plot
        self._create_summary_plot(track_data, output_dir)
        
        logging.info(f"Visualizations saved to {output_dir}")
    
    def _build_window_trajectories(self, track_data: Dict, total_frames: int) -> List[Dict]:
        """Build trajectories for visualization"""
        window_trajectories = []
        
        for window_idx, window_tracks in enumerate(track_data['tracks']):
            trajectories = []
            
            tracks = window_tracks['tracks']  # Shape: (T, N, 2)
            visibility = window_tracks['visibility']  # Shape: (T, N)
            start = window_tracks['start_frame']
            end = window_tracks['end_frame']
            num_points = tracks.shape[1]
            
            # Create trajectories for this window
            for i in range(num_points):
                traj = {
                    'positions': np.full((total_frames, 2), np.nan),
                    'visibility': np.zeros(total_frames, dtype=bool),
                    'window_idx': window_idx,
                    'start_frame': start,
                    'end_frame': end
                }
                
                # Fill in positions for frames in this window
                for local_frame_idx in range(tracks.shape[0]):
                    frame_idx = start + local_frame_idx
                    if frame_idx < total_frames:
                        traj['positions'][frame_idx] = tracks[local_frame_idx, i]
                        traj['visibility'][frame_idx] = visibility[local_frame_idx, i]
                
                trajectories.append(traj)
            
            window_trajectories.append({
                'trajectories': trajectories,
                'window_idx': window_idx,
                'start_frame': start,
                'end_frame': end,
                'num_tracks': num_points
            })
        
        return window_trajectories
    
    def _generate_distinct_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n visually distinct colors"""
        if n <= 10:
            # Use tab10 colormap for small number of colors
            cmap = cm.get_cmap('tab10')
            colors = [cmap(i) for i in range(n)]
        else:
            # Use HSV colormap for larger number of colors
            colors = []
            for i in range(n):
                hue = i / n
                # Convert HSV to RGB (full saturation and value)
                rgb = mcolors.hsv_to_rgb([hue, 1.0, 0.9])
                colors.append(rgb[:3])  # Take only RGB, ignore alpha if present
        
        # Convert to 0-255 range
        colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]
        return colors
    
    def _draw_trajectories_on_frame(self, frame: np.ndarray, 
                                   window_trajectories: List[Dict],
                                   window_colors: List[List[Tuple]],
                                   current_frame_idx: int,
                                   point_size: int = 6,
                                   line_thickness: int = 2,
                                   trail_length: int = 20) -> np.ndarray:
        """Draw trajectories on a single frame"""
        frame_vis = frame.copy()
        
        # Draw trajectories from each window
        for window_idx, (window_data, colors) in enumerate(zip(window_trajectories, window_colors)):
            trajectories = window_data['trajectories']
            window_start = window_data['start_frame']
            window_end = window_data['end_frame']
            
            # Only draw if current frame is within this window's range
            if window_start <= current_frame_idx < window_end:
                # Draw trails and current points for each trajectory in this window
                for traj_idx, (traj, color) in enumerate(zip(trajectories, colors)):
                    positions = traj['positions']
                    visibility = traj['visibility']
                    
                    # Draw trail (past positions within this window)
                    trail_start = max(window_start, current_frame_idx - trail_length)
                    valid_positions = []
                    
                    for t in range(trail_start, current_frame_idx + 1):
                        if visibility[t] and not np.isnan(positions[t]).any():
                            valid_positions.append(positions[t].astype(int))
                    
                    # Draw trail as connected lines
                    if len(valid_positions) > 1:
                        pts = np.array(valid_positions, dtype=np.int32)
                        # Draw with gradual fade
                        for i in range(len(pts) - 1):
                            alpha = (i + 1) / len(pts)  # Fade effect
                            thickness = max(1, int(line_thickness * alpha))
                            cv2.line(frame_vis, tuple(pts[i]), tuple(pts[i+1]), color, thickness)
                    
                    # Draw current point
                    if visibility[current_frame_idx]:
                        if not np.isnan(positions[current_frame_idx]).any():
                            pt = positions[current_frame_idx].astype(int)
                            # Draw point with white outline
                            cv2.circle(frame_vis, tuple(pt), point_size + 1, (255, 255, 255), -1)
                            cv2.circle(frame_vis, tuple(pt), point_size, color, -1)
        
        # Add window info
        self._add_frame_info(frame_vis, current_frame_idx, window_trajectories)
        
        return frame_vis
    
    def _add_frame_info(self, frame: np.ndarray, current_frame_idx: int, 
                       window_trajectories: List[Dict]) -> None:
        """Add frame and window information to the visualization"""
        h, w = frame.shape[:2]
        
        # Frame counter
        text = f"Frame: {current_frame_idx}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   (255, 255, 255), 2, cv2.LINE_AA)
        
        # Active windows
        active_windows = []
        for window_data in window_trajectories:
            if window_data['start_frame'] <= current_frame_idx < window_data['end_frame']:
                active_windows.append(window_data['window_idx'])
        
        if active_windows:
            window_text = f"Windows: {', '.join(map(str, active_windows))}"
            cv2.putText(frame, window_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (255, 255, 255), 2, cv2.LINE_AA)
    
    def _create_video(self, image_paths: List[Path], 
                     window_trajectories: List[Dict],
                     window_colors: List[List[Tuple]],
                     output_dir: Path,
                     fps: int = 10,
                     save_frames: bool = False) -> None:
        """Create video visualization"""
        video_path = output_dir / 'tracking_result.mp4'
        video_writer = imageio.get_writer(str(video_path), fps=fps, codec='libx264', quality=8)
        
        # Process each frame
        total_frames = len(image_paths)
        sample_frame_indices = [0, total_frames//4, total_frames//2, 
                               3*total_frames//4, total_frames-1]
        
        for frame_idx in range(total_frames):
            if frame_idx % 10 == 0:
                logging.info(f"Processing frame {frame_idx}/{total_frames}")
            
            # Load current frame
            img = Image.open(image_paths[frame_idx])
            frame = np.array(img)
            img.close()
            
            # Draw trajectories
            frame_vis = self._draw_trajectories_on_frame(
                frame, window_trajectories, window_colors, frame_idx
            )
            
            # Write frame to video (imageio expects RGB)
            video_writer.append_data(frame_vis)
            
            # Save sample frames
            if save_frames and frame_idx in sample_frame_indices:
                frame_bgr = cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_dir / f'frame_{frame_idx:04d}.png'), frame_bgr)
        
        video_writer.close()
        logging.info(f"Video saved to: {video_path}")
    
    def _create_summary_plot(self, track_data: Dict, output_dir: Path) -> None:
        """Create summary visualization plot"""
        metadata = track_data['metadata']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'CoTracker Precompute Summary - {metadata["initialization_method"].upper()}', 
                    fontsize=16)
        
        # 1. Window timeline
        ax = axes[0, 0]
        windows = track_data['tracks']
        for i, window in enumerate(windows):
            start = window['start_frame']
            end = window['end_frame']
            ax.barh(i, end - start, left=start, height=0.8, 
                   alpha=0.7, label=f'Window {i}' if i < 5 else '')
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Window')
        ax.set_title(f'Window Timeline (size={metadata["window_size"]}, interval={metadata["interval"]})')
        if len(windows) <= 5:
            ax.legend()
        
        # 2. Track count per window
        ax = axes[0, 1]
        track_counts = [w['tracks'].shape[1] for w in windows]
        ax.bar(range(len(track_counts)), track_counts)
        ax.set_xlabel('Window')
        ax.set_ylabel('Number of Tracks')
        ax.set_title('Tracks per Window')
        
        # 3. Configuration info
        ax = axes[1, 0]
        ax.axis('off')
        info_text = "Configuration:\n\n"
        for key, value in metadata.items():
            info_text += f"{key}: {value}\n"
        
        ax.text(0.1, 0.5, info_text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # 4. Statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        total_tracks = sum(track_counts)
        avg_tracks = np.mean(track_counts)
        
        stats_text = "Statistics:\n\n"
        stats_text += f"Total windows: {len(windows)}\n"
        stats_text += f"Total tracks: {total_tracks}\n"
        stats_text += f"Average tracks/window: {avg_tracks:.1f}\n"
        stats_text += f"Total frames: {metadata['total_frames']}\n"
        
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tracking_summary.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Summary plot saved to: {output_dir / 'tracking_summary.png'}")