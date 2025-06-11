"""Base abstract class for feature initializers"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict
import numpy as np
from pathlib import Path


class BaseFeatureInitializer(ABC):
    """Abstract base class for feature initialization methods"""
    
    def __init__(self, max_features: int = 400):
        """
        Args:
            max_features: Maximum number of features to extract
        """
        self.max_features = max_features
    
    @abstractmethod
    def extract_features(self, 
                        image_path: Path,
                        window_frame_offset: int = 0) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Extract features from an image
        
        Args:
            image_path: Path to the image
            window_frame_offset: Time offset for query points (usually 0 for first frame)
            
        Returns:
            query_points: Numpy array of shape (N, 3) with [time, x, y] format
            extra_info: Optional dictionary with additional feature information
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of the initialization method"""
        pass
    
    def filter_features_by_region(self, points: np.ndarray, 
                                 image_shape: Tuple[int, int], 
                                 grid_size: int = 20) -> np.ndarray:
        """
        Filter features to ensure good spatial distribution
        
        Divides image into grid cells and selects best features from each cell
        
        Args:
            points: Array of (x, y) coordinates
            image_shape: (height, width) of the image
            grid_size: Number of grid cells in each dimension
            
        Returns:
            Filtered points array
        """
        h, w = image_shape
        cell_h = h // grid_size
        cell_w = w // grid_size
        
        filtered_points = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Define cell boundaries
                y_min = i * cell_h
                y_max = (i + 1) * cell_h if i < grid_size - 1 else h
                x_min = j * cell_w
                x_max = (j + 1) * cell_w if j < grid_size - 1 else w
                
                # Find points in this cell
                mask = (points[:, 0] >= x_min) & (points[:, 0] < x_max) & \
                       (points[:, 1] >= y_min) & (points[:, 1] < y_max)
                cell_points = points[mask]
                
                # Select up to 2 points from this cell
                if len(cell_points) > 0:
                    filtered_points.extend(cell_points[:2])
        
        filtered_array = np.array(filtered_points) if filtered_points else points[:self.max_features]
        
        # Ensure we don't exceed max_features
        if len(filtered_array) > self.max_features:
            filtered_array = filtered_array[:self.max_features]
            
        return filtered_array