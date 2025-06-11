"""Abstract base class for all tracking extractors"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class BaseTracker(ABC):
    """Abstract base class for point tracking extractors"""
    
    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
    
    @abstractmethod
    def extract_tracks(self, 
                      image_dir: str,
                      output_path: Optional[str] = None) -> Dict:
        """
        Extract tracks from images and save to disk
        
        Args:
            image_dir: Directory containing images
            output_path: Output path for saved tracks. If None, auto-generated.
            
        Returns:
            Dictionary containing tracks and metadata
        """
        pass
    
    @abstractmethod
    def load_tracks(self, track_path: str) -> Dict:
        """
        Load tracks from saved file
        
        Args:
            track_path: Path to saved track file
            
        Returns:
            Dictionary containing tracks and metadata
        """
        pass
    
    def _get_image_paths(self, image_dir: Path) -> List[Path]:
        """
        Get sorted list of image paths from directory
        
        Args:
            image_dir: Directory containing images
            
        Returns:
            Sorted list of image paths
        """
        # Try common image extensions
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
            image_paths = sorted(image_dir.glob(ext))
            if image_paths:
                return image_paths
        
        raise ValueError(f"No images found in {image_dir}")
    
    def _get_image_dimensions(self, image_path: Path) -> Tuple[int, int]:
        """
        Get image dimensions without loading full image
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple of (height, width)
        """
        from PIL import Image
        
        with Image.open(image_path) as img:
            return img.height, img.width