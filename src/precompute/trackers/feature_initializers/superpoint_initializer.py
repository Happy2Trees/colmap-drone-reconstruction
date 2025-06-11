"""SuperPoint-based feature initializer"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging
import sys

from .base_initializer import BaseFeatureInitializer


class SuperPointInitializer(BaseFeatureInitializer):
    """Initialize features using SuperPoint deep learning feature detector"""
    
    def __init__(self, 
                 max_features: int = 400,
                 conf_threshold: float = 0.015,
                 nms_dist: int = 4,
                 grid_filter: bool = True,
                 weights_path: Optional[str] = None,
                 device: str = 'cuda'):
        """
        Args:
            max_features: Maximum number of features to extract
            conf_threshold: Confidence threshold for feature detection
            nms_dist: Non-maximum suppression distance
            grid_filter: Whether to filter features for spatial distribution
            weights_path: Path to SuperPoint weights file
            device: Device to run on ('cuda' or 'cpu')
        """
        super().__init__(max_features)
        self.conf_threshold = conf_threshold
        self.nms_dist = nms_dist
        self.grid_filter = grid_filter
        self.device = device
        
        # Initialize SuperPoint
        self.superpoint = self._init_superpoint(weights_path)
    
    def _init_superpoint(self, weights_path: Optional[str] = None):
        """Initialize SuperPoint model"""
        try:
            # Try to import SuperPoint from submodules
            superpoint_path = Path(__file__).parent.parent.parent.parent.parent.parent / "submodules/SuperPoint"
            if superpoint_path.exists():
                sys.path.insert(0, str(superpoint_path))
                from demo_superpoint import SuperPointFrontend
                
                # Default weights path
                if weights_path is None:
                    weights_path = superpoint_path / "superpoint_v1.pth"
                else:
                    weights_path = Path(weights_path)
                
                if not weights_path.exists():
                    raise FileNotFoundError(f"SuperPoint weights not found at {weights_path}")
                
                # Initialize SuperPoint
                superpoint = SuperPointFrontend(
                    weights_path=str(weights_path),
                    nms_dist=self.nms_dist,
                    conf_thresh=self.conf_threshold,
                    nn_thresh=0.7,
                    cuda=(self.device == 'cuda')
                )
                
                logging.info("SuperPoint initialized successfully")
                return superpoint
            else:
                raise ImportError("SuperPoint submodule not found")
                
        except Exception as e:
            logging.error(f"Failed to initialize SuperPoint: {e}")
            logging.warning("SuperPoint initialization failed, will fall back to SIFT")
            return None
    
    def extract_features(self, 
                        image_path: Path,
                        window_frame_offset: int = 0) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Extract SuperPoint features from an image
        
        Args:
            image_path: Path to the image
            window_frame_offset: Time offset for query points
            
        Returns:
            query_points: SuperPoint keypoints in [time, x, y] format
            extra_info: Dictionary containing corners, descriptors, and confidence
        """
        # If SuperPoint not available, fall back to SIFT
        if self.superpoint is None:
            logging.info("Using SIFT as fallback for SuperPoint")
            from .sift_initializer import SIFTInitializer
            sift = SIFTInitializer(max_features=self.max_features, grid_filter=self.grid_filter)
            return sift.extract_features(image_path, window_frame_offset)
        
        # Load image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to float32 in range [0, 1]
        img_float = img.astype(np.float32) / 255.0
        
        # Run SuperPoint
        try:
            corners, descriptors, heatmap = self.superpoint.run(img_float)
        except Exception as e:
            logging.error(f"SuperPoint inference failed: {e}")
            # Fallback to grid
            h, w = img.shape
            return self._generate_grid_fallback(h, w, window_frame_offset), None
        
        if corners is None or corners.shape[1] == 0:
            logging.warning(f"No SuperPoint features found in {image_path}, falling back to grid")
            h, w = img.shape
            return self._generate_grid_fallback(h, w, window_frame_offset), None
        
        # Extract coordinates (transpose for consistency)
        points = corners[:2, :].T  # Shape: (N, 2) with [x, y]
        confidences = corners[2, :]
        
        # Filter for spatial distribution if requested
        if self.grid_filter and len(points) > self.max_features:
            # Sort by confidence before filtering
            sorted_indices = np.argsort(confidences)[::-1]
            points = points[sorted_indices]
            confidences = confidences[sorted_indices]
            
            # Filter by region
            points = self.filter_features_by_region(points, img.shape)
            
            # Update confidences to match filtered points
            # Find which points were kept
            kept_indices = []
            for i, p in enumerate(points):
                for j, orig_p in enumerate(corners[:2, :].T[sorted_indices]):
                    if np.allclose(p, orig_p):
                        kept_indices.append(sorted_indices[j])
                        break
            
            if kept_indices:
                confidences = corners[2, kept_indices]
                if descriptors is not None:
                    descriptors = descriptors[:, kept_indices]
        
        # Limit to max_features
        if len(points) > self.max_features:
            points = points[:self.max_features]
            confidences = confidences[:self.max_features]
            if descriptors is not None:
                descriptors = descriptors[:, :self.max_features]
        
        # Add time dimension
        queries = np.hstack([
            np.full((len(points), 1), window_frame_offset),
            points
        ])
        
        # Don't store extra info to avoid unnecessary data storage
        extra_info = None
        
        logging.info(f"Extracted {len(points)} SuperPoint features from {image_path.name}")
        
        return queries, extra_info
    
    def _generate_grid_fallback(self, height: int, width: int, 
                               window_frame_offset: int = 0,
                               grid_size: int = 20) -> np.ndarray:
        """Generate grid points as fallback when no SuperPoint features found"""
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
        return "superpoint"