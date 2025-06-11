"""Grid-based feature initializer"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from PIL import Image

from .base_initializer import BaseFeatureInitializer


class GridInitializer(BaseFeatureInitializer):
    """Initialize features using a regular grid pattern"""
    
    def __init__(self, grid_size: int = 20, max_features: int = 400):
        """
        Args:
            grid_size: Number of grid points in each dimension
            max_features: Maximum number of features (grid_size * grid_size)
        """
        super().__init__(max_features)
        self.grid_size = grid_size
    
    def extract_features(self, 
                        image_path: Path,
                        window_frame_offset: int = 0) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Extract grid-based features
        
        Args:
            image_path: Path to the image
            window_frame_offset: Time offset for query points
            
        Returns:
            query_points: Grid points in [time, x, y] format
            extra_info: None for grid method
        """
        # Load image to get dimensions
        img = Image.open(image_path)
        h, w = img.height, img.width
        img.close()
        
        # Create grid with slight randomization
        np.random.seed(None)  # Random seed for variety
        margin = 0.1
        
        y_coords = np.linspace(h * margin, h * (1-margin), self.grid_size)
        x_coords = np.linspace(w * margin, w * (1-margin), self.grid_size)
        
        # Add small random perturbation (up to 5 pixels)
        y_coords += np.random.uniform(-5, 5, size=self.grid_size)
        x_coords += np.random.uniform(-5, 5, size=self.grid_size)
        
        # Ensure points stay within bounds
        y_coords = np.clip(y_coords, 0, h-1)
        x_coords = np.clip(x_coords, 0, w-1)
        
        # Generate all combinations
        points = []
        for y in y_coords:
            for x in x_coords:
                points.append([window_frame_offset, x, y])  # [time, x, y]
        
        return np.array(points), None
    
    def get_method_name(self) -> str:
        """Return the name of the initialization method"""
        return "grid"