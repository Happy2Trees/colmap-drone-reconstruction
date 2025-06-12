"""Base class for depth estimation modules supporting both image and video models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class BaseDepthEstimator(ABC):
    """Abstract base class for depth estimation modules.
    
    Supports both single-image and video-based depth estimation models.
    Video models can leverage temporal consistency across frames.
    """
    
    def __init__(self, config: Dict):
        """Initialize depth estimator with configuration.
        
        Args:
            config: Configuration dictionary containing model-specific parameters
        """
        self.config = config
        self.device = config.get('device', 'cuda')
        self.batch_size = config.get('batch_size', 1)
        self.output_format = config.get('output_format', 'npy')
        self.save_visualization = config.get('save_visualization', False)
        
        # Video-specific parameters
        self.is_video_model = config.get('is_video_model', False)
        self.temporal_window = config.get('temporal_window', 1)
        self.temporal_stride = config.get('temporal_stride', 1)
    
    @abstractmethod
    def extract_depth(self, 
                     image_dir: Path, 
                     output_path: Optional[Path] = None) -> Dict:
        """Extract depth maps from all images in a directory.
        
        For video models, this should handle temporal windowing appropriately.
        
        Args:
            image_dir: Path to directory containing input images
            output_path: Optional path to save depth maps. If None, uses default location
            
        Returns:
            Dictionary containing:
                - 'depth_dir': Path to directory containing depth maps
                - 'num_frames': Number of frames processed
                - 'metadata': Additional metadata about the extraction
        """
        pass
    
    @abstractmethod
    def process_sequence(self, 
                        images: Union[List[np.ndarray], np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Process a sequence of images to extract depth maps.
        
        This is the main processing method that handles both single images
        and video sequences depending on the model type.
        
        Args:
            images: Either:
                - List of images as numpy arrays (H, W, 3) in RGB format
                - Single numpy array of shape (T, H, W, 3) for video sequences
            
        Returns:
            Tuple of (depth_maps, masks) where:
                - depth_maps: List of depth maps as numpy arrays (H, W)
                - masks: List of valid pixel masks as numpy arrays (H, W)
        """
        pass
    
    def create_temporal_windows(self, 
                              image_paths: List[Path],
                              window_size: int,
                              stride: int = 1) -> List[List[Path]]:
        """Create overlapping temporal windows for video processing.
        
        Args:
            image_paths: List of image file paths in temporal order
            window_size: Size of temporal window
            stride: Stride between windows
            
        Returns:
            List of windows, where each window is a list of image paths
        """
        windows = []
        num_frames = len(image_paths)
        
        # Create sliding windows
        for start_idx in range(0, num_frames, stride):
            end_idx = min(start_idx + window_size, num_frames)
            window = image_paths[start_idx:end_idx]
            
            # Pad last window if needed
            if len(window) < window_size and start_idx > 0:
                # Use frames from the end to maintain window size
                start_idx = max(0, num_frames - window_size)
                window = image_paths[start_idx:num_frames]
            
            windows.append(window)
            
            # Stop if we've processed all frames
            if end_idx >= num_frames:
                break
        
        return windows
    
    def merge_windowed_depths(self,
                            windowed_depths: List[Dict[int, np.ndarray]],
                            window_size: int,
                            stride: int,
                            total_frames: int) -> List[np.ndarray]:
        """Merge depth maps from overlapping temporal windows.
        
        Args:
            windowed_depths: List of dictionaries mapping frame indices to depth maps
            window_size: Size of temporal window used
            stride: Stride between windows
            total_frames: Total number of frames
            
        Returns:
            List of merged depth maps
        """
        # Initialize storage for all frames
        merged_depths = [None] * total_frames
        frame_counts = np.zeros(total_frames)
        
        # Accumulate depths from all windows
        for window_idx, depth_dict in enumerate(windowed_depths):
            for frame_idx, depth in depth_dict.items():
                if merged_depths[frame_idx] is None:
                    merged_depths[frame_idx] = np.zeros_like(depth, dtype=np.float64)
                
                # Add depth contribution
                merged_depths[frame_idx] += depth.astype(np.float64)
                frame_counts[frame_idx] += 1
        
        # Average overlapping predictions
        for i in range(total_frames):
            if merged_depths[i] is not None and frame_counts[i] > 0:
                merged_depths[i] = (merged_depths[i] / frame_counts[i]).astype(np.float32)
        
        return merged_depths
    
    def get_output_path(self, image_dir: Path, output_path: Optional[Path] = None) -> Path:
        """Get the output path for depth maps.
        
        Args:
            image_dir: Path to input image directory
            output_path: Optional custom output path
            
        Returns:
            Path to output directory for depth maps
        """
        if output_path is not None:
            return output_path
        
        # Default: create depth/<model_name> directory in the same parent as images
        model_name = self.__class__.__name__.replace('Extractor', '')
        return image_dir.parent / 'depth' / model_name
    
    def save_depth_map(self, depth: np.ndarray, output_file: Path, 
                      mask: Optional[np.ndarray] = None,
                      metadata: Optional[Dict] = None) -> None:
        """Save a single depth map to file.
        
        Args:
            depth: Depth map as numpy array (H, W)
            output_file: Path to save the depth map
            mask: Optional valid pixel mask
            metadata: Optional metadata to save with depth
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if self.output_format == 'npy':
            # Save as numpy array with optional mask and metadata
            if mask is not None or metadata is not None:
                save_dict = {'depth': depth}
                if mask is not None:
                    save_dict['mask'] = mask
                if metadata is not None:
                    save_dict['metadata'] = metadata
                np.savez_compressed(output_file.with_suffix('.npz'), **save_dict)
            else:
                np.save(output_file.with_suffix('.npy'), depth)
        
        elif self.output_format == 'png':
            # Normalize and save as 16-bit PNG
            import cv2
            depth_normalized = self._normalize_depth_for_png(depth, mask)
            cv2.imwrite(str(output_file.with_suffix('.png')), depth_normalized)
        
        elif self.output_format == 'pfm':
            # Save as PFM format (floating point)
            self._save_pfm(output_file.with_suffix('.pfm'), depth)
        
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")
    
    def _normalize_depth_for_png(self, depth: np.ndarray, 
                                 mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Normalize depth map for PNG output.
        
        Args:
            depth: Raw depth map
            mask: Optional valid pixel mask
            
        Returns:
            16-bit normalized depth map
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
        
        # Normalize to 16-bit range
        depth_normalized = (depth - min_depth) / (max_depth - min_depth + 1e-6)
        depth_normalized = np.clip(depth_normalized * 65535, 0, 65535).astype(np.uint16)
        
        if mask is not None:
            depth_normalized[mask == 0] = 0
        
        return depth_normalized
    
    def _save_pfm(self, filename: Path, data: np.ndarray) -> None:
        """Save data in PFM format.
        
        Args:
            filename: Output file path
            data: Data to save
        """
        height, width = data.shape[:2]
        with open(filename, 'wb') as f:
            # Write header
            if len(data.shape) == 3:
                f.write(b'PF\n')  # Color
            else:
                f.write(b'Pf\n')  # Grayscale
            
            f.write(f'{width} {height}\n'.encode())
            f.write(b'-1.0\n')  # Little-endian
            
            # Write data
            data.astype(np.float32).tobytes()
    
    def create_visualization(self, depth: np.ndarray, 
                           mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Create a visualization of the depth map.
        
        Args:
            depth: Depth map
            mask: Optional valid pixel mask
            
        Returns:
            RGB visualization image
        """
        import cv2
        
        # Apply mask if provided
        if mask is not None:
            depth = depth.copy()
            depth[mask == 0] = np.nan
        
        # Normalize depth for visualization
        valid_depth = depth[~np.isnan(depth)]
        if len(valid_depth) > 0:
            vmin = np.percentile(valid_depth, 5)
            vmax = np.percentile(valid_depth, 95)
            depth_norm = (depth - vmin) / (vmax - vmin + 1e-6)
            depth_norm = np.clip(depth_norm, 0, 1)
        else:
            depth_norm = depth
        
        # Apply colormap
        depth_norm[np.isnan(depth_norm)] = 0
        depth_vis = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), 
                                     cv2.COLORMAP_INFERNO)
        
        # Set invalid pixels to black
        if mask is not None:
            depth_vis[mask == 0] = 0
        
        return depth_vis