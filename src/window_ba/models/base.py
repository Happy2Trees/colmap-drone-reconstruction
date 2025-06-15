"""
Base classes and interfaces for Window-based Bundle Adjustment.

This module provides abstract base classes and common interfaces
to ensure consistent implementation across the window BA pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


@dataclass
class ImageDimensions:
    """Container for image dimensions."""
    width: int
    height: int
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height
    
    def to_normalized(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert pixel coordinates to normalized coordinates [-1, 1]."""
        x_norm = 2.0 * x / (self.width - 1) - 1.0
        y_norm = 2.0 * y / (self.height - 1) - 1.0
        return x_norm, y_norm
    
    def from_normalized(self, x_norm: torch.Tensor, y_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert normalized coordinates [-1, 1] to pixel coordinates."""
        x = (x_norm + 1.0) * (self.width / 2)
        y = (y_norm + 1.0) * (self.height / 2)
        return x, y


@dataclass
class WindowData:
    """Container for window tracking data."""
    window_idx: int
    start_frame: int
    end_frame: int
    tracks: Union[np.ndarray, torch.Tensor]  # (T, N, 2) or (T, N, 3) with depth
    visibility: Union[np.ndarray, torch.Tensor]  # (T, N)
    query_time: Optional[Union[np.ndarray, torch.Tensor]] = None  # (N,)
    tracks_3d: Optional[Union[np.ndarray, torch.Tensor]] = None  # (T, N, 3)
    xyzw_world: Optional[Union[np.ndarray, torch.Tensor]] = None  # (T, N, 4)
    boundary_3d_optimized: Optional[Union[np.ndarray, torch.Tensor]] = None  # (M, 3)
    
    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1
    
    @property
    def num_points(self) -> int:
        return self.tracks.shape[1]
    
    @property
    def has_depth(self) -> bool:
        return self.tracks_3d is not None
    
    @property
    def has_3d_world(self) -> bool:
        return self.xyzw_world is not None
    
    def to_torch(self, device: str = 'cuda') -> 'WindowData':
        """Convert numpy arrays to torch tensors."""
        def _to_torch(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return torch.from_numpy(x).to(device)
        
        # Convert required fields
        tracks = _to_torch(self.tracks)
        visibility = _to_torch(self.visibility)
        
        # These should never be None
        assert tracks is not None, "tracks cannot be None"
        assert visibility is not None, "visibility cannot be None"
        
        return WindowData(
            window_idx=self.window_idx,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            tracks=tracks,
            visibility=visibility,
            query_time=_to_torch(self.query_time),
            tracks_3d=_to_torch(self.tracks_3d),
            xyzw_world=_to_torch(self.xyzw_world),
            boundary_3d_optimized=_to_torch(self.boundary_3d_optimized)
        )


class BaseOptimizer(ABC):
    """Abstract base class for optimization modules."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
    
    @abstractmethod
    def optimize(self, *args, **kwargs):
        """Run optimization process."""
        pass
    
    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Compute optimization loss."""
        pass


class BaseLoader(ABC):
    """Abstract base class for data loading modules."""
    
    def __init__(self, scene_dir: Path, device: str = 'cuda'):
        self.scene_dir = Path(scene_dir)
        self.device = device
    
    @abstractmethod
    def load(self, *args, **kwargs):
        """Load data from disk."""
        pass


class BaseInitializer(ABC):
    """Abstract base class for 3D initialization modules."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
    
    @abstractmethod
    def initialize_3d(self, *args, **kwargs):
        """Initialize 3D points from observations."""
        pass


class BaseCameraModel(nn.Module):
    """Abstract base class for camera models."""
    
    def __init__(self, num_frames: int):
        super().__init__()
        self.num_frames = num_frames
    
    @abstractmethod
    def get_projection_matrix(self, frame_idx: int) -> torch.Tensor:
        """Get projection matrix for a specific frame."""
        pass
    
    @abstractmethod
    def project_points(self, points_3d: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """Project 3D points to 2D for a specific frame."""
        pass