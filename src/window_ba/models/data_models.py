"""
Data models for Window-based Bundle Adjustment.

This module provides structured data models to replace dictionaries
and ensure type safety throughout the pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import torch

if TYPE_CHECKING:
    from .camera_model import CameraModel


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: Optional[np.ndarray] = None
    
    @classmethod
    def from_matrix(cls, K: np.ndarray, width: int, height: int, 
                    distortion: Optional[np.ndarray] = None) -> 'CameraIntrinsics':
        """Create from intrinsic matrix."""
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=width,
            height=height,
            distortion=distortion
        )
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 intrinsic matrix."""
        K = np.array([[self.fx, 0, self.cx],
                      [0, self.fy, self.cy],
                      [0, 0, 1]], dtype=np.float32)
        return K
    
    def get_fov(self) -> Tuple[float, float]:
        """Get field of view angles in radians."""
        fov_x = 2 * np.arctan(self.width / (2 * self.fx))
        fov_y = 2 * np.arctan(self.height / (2 * self.fy))
        return fov_x, fov_y
    
    def get_tan_fov(self) -> Tuple[float, float]:
        """Get tangent of half FOV angles."""
        fov_x, fov_y = self.get_fov()
        return np.tan(fov_x / 2), np.tan(fov_y / 2)


@dataclass
class WindowTrackData:
    """Container for window tracking data."""
    window_idx: int
    start_frame: int
    end_frame: int
    tracks: np.ndarray  # (T, N, 2) pixel coordinates
    visibility: np.ndarray  # (T, N) boolean
    query_time: np.ndarray  # (N,) frame indices when points were queried
    window_size: int
    interval: int
    bidirectional: bool = False
    
    # Optional fields populated during processing
    tracks_3d: Optional[np.ndarray] = None  # (T, N, 3) with depth
    xyzw_world: Optional[np.ndarray] = None  # (T, N, 4) world coordinates
    boundary_3d_optimized: Optional[np.ndarray] = None  # (M, 3) Phase 2 optimized
    query_3d_start: Optional[np.ndarray] = None  # (N_start, 3)
    query_3d_end: Optional[np.ndarray] = None  # (N_end, 3)
    
    def __post_init__(self):
        """Validate data consistency."""
        T, N, _ = self.tracks.shape
        assert self.visibility.shape == (T, N), f"Visibility shape mismatch: {self.visibility.shape} vs expected {(T, N)}"
        assert len(self.query_time) == N, f"Query time length mismatch: {len(self.query_time)} vs expected {N}"
        assert self.end_frame - self.start_frame == T, f"Frame count mismatch"
    
    @property
    def num_frames(self) -> int:
        """Number of frames in window."""
        return self.end_frame - self.start_frame
    
    @property
    def num_points(self) -> int:
        """Number of tracked points."""
        return self.tracks.shape[1]
    
    @property
    def has_depth(self) -> bool:
        """Check if depth information is available."""
        return self.tracks_3d is not None
    
    @property
    def has_world_coords(self) -> bool:
        """Check if world coordinates are available."""
        return self.xyzw_world is not None
    
    @property
    def boundary_mask_start(self) -> np.ndarray:
        """Boolean mask for points queried at window start."""
        return self.query_time == 0
    
    @property
    def boundary_mask_end(self) -> np.ndarray:
        """Boolean mask for points queried at window end."""
        return self.query_time == self.window_size - 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for backward compatibility."""
        d = {
            'window_idx': self.window_idx,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'tracks': self.tracks,
            'visibility': self.visibility,
            'query_time': self.query_time,
            'window_size': self.window_size,
            'interval': self.interval,
            'bidirectional': self.bidirectional
        }
        
        # Add optional fields
        if self.tracks_3d is not None:
            d['tracks_3d'] = self.tracks_3d
        if self.xyzw_world is not None:
            d['xyzw_world'] = self.xyzw_world
        if self.boundary_3d_optimized is not None:
            d['boundary_3d_optimized'] = self.boundary_3d_optimized
        if self.query_3d_start is not None:
            d['query_3d_start'] = self.query_3d_start
        if self.query_3d_end is not None:
            d['query_3d_end'] = self.query_3d_end
            
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'WindowTrackData':
        """Create from dictionary."""
        return cls(
            window_idx=d['window_idx'],
            start_frame=d['start_frame'],
            end_frame=d['end_frame'],
            tracks=d['tracks'],
            visibility=d['visibility'],
            query_time=d['query_time'],
            window_size=d.get('window_size', d['end_frame'] - d['start_frame'] + 1),
            interval=d.get('interval', 1),
            bidirectional=d.get('bidirectional', False),
            tracks_3d=d.get('tracks_3d'),
            xyzw_world=d.get('xyzw_world'),
            boundary_3d_optimized=d.get('boundary_3d_optimized'),
            query_3d_start=d.get('query_3d_start'),
            query_3d_end=d.get('query_3d_end')
        )


@dataclass
class CameraParameters:
    """Optimized camera parameters."""
    quaternions: torch.Tensor  # (N, 4) rotation quaternions
    translations: torch.Tensor  # (N, 3) translation vectors
    tan_fov_x: Union[torch.Tensor, float]  # FOV parameters
    tan_fov_y: Union[torch.Tensor, float]
    
    @property
    def num_frames(self) -> int:
        """Number of camera frames."""
        return len(self.quaternions)
    
    def to_numpy(self) -> Dict[str, Union[np.ndarray, float]]:
        """Convert to numpy arrays."""
        return {
            'quaternions': self.quaternions.cpu().numpy(),
            'translations': self.translations.cpu().numpy(),
            'tan_fov_x': self.tan_fov_x.cpu().numpy() if isinstance(self.tan_fov_x, torch.Tensor) else self.tan_fov_x,
            'tan_fov_y': self.tan_fov_y.cpu().numpy() if isinstance(self.tan_fov_y, torch.Tensor) else self.tan_fov_y
        }
    
    @classmethod
    def from_numpy(cls, data: Dict[str, np.ndarray], device: str = 'cuda') -> 'CameraParameters':
        """Create from numpy arrays."""
        return cls(
            quaternions=torch.from_numpy(data['quaternions']).to(device),
            translations=torch.from_numpy(data['translations']).to(device),
            tan_fov_x=torch.from_numpy(data['tan_fov_x']).to(device) if isinstance(data['tan_fov_x'], np.ndarray) else torch.tensor(data['tan_fov_x']).to(device),
            tan_fov_y=torch.from_numpy(data['tan_fov_y']).to(device) if isinstance(data['tan_fov_y'], np.ndarray) else torch.tensor(data['tan_fov_y']).to(device)
        )


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    success: bool
    final_loss: float
    iterations: int
    converged: bool
    history: Dict[str, List[float]]
    camera_params: Optional[CameraParameters] = None
    boundary_3d_points: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = {
            'success': self.success,
            'final_loss': self.final_loss,
            'iterations': self.iterations,
            'converged': self.converged,
            'history': self.history
        }
        
        if self.camera_params is not None:
            d['camera_params'] = self.camera_params.to_numpy()
        if self.boundary_3d_points is not None:
            d['boundary_3d_points'] = self.boundary_3d_points.cpu().numpy()
            
        return d


@dataclass
class SceneConfiguration:
    """Scene-specific configuration."""
    scene_dir: Path
    image_dir: Path
    track_dir: Path
    depth_dir: Path
    intrinsics: CameraIntrinsics
    track_pattern: str = "*_sift_bidirectional.npy"
    
    def __post_init__(self):
        """Convert paths to Path objects."""
        self.scene_dir = Path(self.scene_dir)
        self.image_dir = Path(self.image_dir)
        self.track_dir = Path(self.track_dir)
        self.depth_dir = Path(self.depth_dir)


@dataclass
class PipelineState:
    """State of the pipeline execution."""
    current_step: int = 0
    completed_steps: List[str] = field(default_factory=list)
    window_tracks: Optional[List[WindowTrackData]] = None
    camera_model: Optional['CameraModel'] = None
    phase1_result: Optional[OptimizationResult] = None
    phase2_result: Optional[OptimizationResult] = None
    
    def is_step_completed(self, step_name: str) -> bool:
        """Check if a step has been completed."""
        return step_name in self.completed_steps
    
    def mark_step_completed(self, step_name: str):
        """Mark a step as completed."""
        if step_name not in self.completed_steps:
            self.completed_steps.append(step_name)


@dataclass
class ReprojectionError:
    """Container for reprojection error statistics."""
    mean_error: float
    median_error: float
    std_error: float
    max_error: float
    num_points: int
    errors_by_window: Dict[int, float]
    
    def summary_string(self) -> str:
        """Get summary string."""
        return (f"Reprojection Error - Mean: {self.mean_error:.3f}, "
                f"Median: {self.median_error:.3f}, Std: {self.std_error:.3f}, "
                f"Max: {self.max_error:.3f} (pixels)")