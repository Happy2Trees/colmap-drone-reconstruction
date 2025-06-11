"""SIFT-based feature initializer"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

from .base_initializer import BaseFeatureInitializer


class SIFTInitializer(BaseFeatureInitializer):
    """Initialize features using SIFT keypoint detection"""
    
    def __init__(self, 
                 max_features: int = 400,
                 contrast_threshold: float = 0.01,
                 grid_filter: bool = True):
        """
        Args:
            max_features: Maximum number of features to extract
            contrast_threshold: Minimum response threshold for filtering weak features
            grid_filter: Whether to filter features for spatial distribution
        """
        super().__init__(max_features)
        self.contrast_threshold = contrast_threshold
        self.grid_filter = grid_filter
        
        # Create SIFT detector
        try:
            self.sift = cv2.SIFT_create(
                nfeatures=max_features * 2,  # Extract more, then filter
                contrastThreshold=contrast_threshold
            )
        except AttributeError:
            # For older OpenCV versions
            self.sift = cv2.xfeatures2d.SIFT_create(
                nfeatures=max_features * 2,
                contrastThreshold=contrast_threshold
            )
    
    def extract_features(self, 
                        image_path: Path,
                        window_frame_offset: int = 0) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Extract SIFT features from an image
        
        Args:
            image_path: Path to the image
            window_frame_offset: Time offset for query points
            
        Returns:
            query_points: SIFT keypoints in [time, x, y] format
            extra_info: Dictionary containing keypoints and descriptors
        """
        # Load image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect and compute SIFT features
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        
        if len(keypoints) == 0:
            logging.warning(f"No SIFT features found in {image_path}, falling back to grid")
            # Fallback to grid points
            h, w = img.shape
            return self._generate_grid_fallback(h, w, window_frame_offset), None
        
        # Extract coordinates
        points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        
        # Filter for spatial distribution if requested
        if self.grid_filter and len(points) > self.max_features:
            points = self.filter_features_by_region(points, img.shape)
        
        # Limit to max_features
        if len(points) > self.max_features:
            # SIFT already orders by response, so take first ones
            points = points[:self.max_features]
            keypoints = keypoints[:self.max_features]
            descriptors = descriptors[:self.max_features] if descriptors is not None else None
        
        # Add time dimension
        queries = np.hstack([
            np.full((len(points), 1), window_frame_offset),
            points
        ])
        
        # Don't store extra info to avoid serialization issues with cv2.KeyPoint
        extra_info = None
        
        logging.info(f"Extracted {len(keypoints)} SIFT features from {image_path.name}")
        
        return queries, extra_info
    
    def _generate_grid_fallback(self, height: int, width: int, 
                               window_frame_offset: int = 0,
                               grid_size: int = 20) -> np.ndarray:
        """Generate grid points as fallback when no SIFT features found"""
        margin = 0.1
        
        y_coords = np.linspace(height * margin, height * (1-margin), grid_size)
        x_coords = np.linspace(width * margin, width * (1-margin), grid_size)
        
        points = []
        for y in y_coords:
            for x in x_coords:
                points.append([window_frame_offset, x, y])
        
        return np.array(points)
    
    def get_method_name(self) -> str:
        """Return the name of the initialization method"""
        return "sift"