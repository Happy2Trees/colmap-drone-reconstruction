"""
Configuration management for Window-based Bundle Adjustment.

This module provides centralized configuration handling with validation,
defaults, and environment-specific overrides.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class TrackLoaderConfig:
    """Configuration for track loading."""
    track_mode: str = "sift"  # Options: sift, superpoint, grid
    depth_subdir: str = "depth/GeometryCrafter"
    bidirectional_priority: bool = True  # Prefer bidirectional tracks if available
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_modes = ["sift", "superpoint", "grid"]
        if self.track_mode not in valid_modes:
            raise ValueError(f"Invalid track_mode: {self.track_mode}. Must be one of {valid_modes}")


@dataclass 
class OptimizationConfig:
    """Configuration for bundle adjustment optimization."""
    # Phase 1: Camera-only optimization
    max_iterations: int = 10000
    learning_rate_camera: float = 1e-3
    learning_rate_translation: float = 1e-2
    learning_rate_fov: float = 1e-4
    
    # Phase 2: Joint optimization
    learning_rate_3d: float = 1e-2
    
    # Convergence and regularization
    convergence_threshold: float = 1e-6
    gradient_clip: float = 1.0
    
    # Loss configuration
    use_robust_loss: bool = True
    robust_loss_sigma: float = 1.0
    proj_loss_weight: float = 1.0
    depth_loss_weight: float = 0.0  # GeometryCrafter default
    
    # Logging
    log_interval: int = 100
    
    # Debug options for faster testing
    debug_num_windows: Optional[int] = None  # Use only first N windows (None = use all)
    debug_window_sampling: str = "first"  # Options: first, random, evenly_spaced
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.convergence_threshold <= 0:
            raise ValueError("convergence_threshold must be positive")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        
        # Validate debug options
        if self.debug_num_windows is not None and self.debug_num_windows <= 0:
            raise ValueError("debug_num_windows must be positive or None")
        valid_sampling = ["first", "random", "evenly_spaced"]
        if self.debug_window_sampling not in valid_sampling:
            raise ValueError(f"debug_window_sampling must be one of {valid_sampling}")


@dataclass
class CameraConfig:
    """Configuration for camera model."""
    image_width: int = 1024
    image_height: int = 576
    single_camera: bool = True  # Share FOV across all frames
    default_fov_degrees: float = 60.0
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        if self.default_fov_degrees <= 0 or self.default_fov_degrees >= 180:
            raise ValueError("default_fov_degrees must be between 0 and 180")


@dataclass
class OutputConfig:
    """Configuration for output options."""
    save_intermediate: bool = True
    save_colmap: bool = True
    colmap_format: str = "binary"  # binary or text
    save_visualizations: bool = True
    visualization_dpi: int = 150
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_formats = ["binary", "text"]
        if self.colmap_format not in valid_formats:
            raise ValueError(f"Invalid colmap_format: {self.colmap_format}. Must be one of {valid_formats}")


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    enabled: bool = True
    camera_trajectory: bool = True
    point_cloud: bool = True
    reprojection_errors: bool = True
    summary_plot: bool = True
    plot_every_n_cameras: int = 10  # For camera orientation visualization
    point_size: int = 20
    figure_size: Tuple[int, int] = (12, 8)
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.plot_every_n_cameras <= 0:
            raise ValueError("plot_every_n_cameras must be positive")
        if self.point_size <= 0:
            raise ValueError("point_size must be positive")
        if len(self.figure_size) != 2 or self.figure_size[0] <= 0 or self.figure_size[1] <= 0:
            raise ValueError("figure_size must be a tuple of two positive integers")


@dataclass
class WindowBAConfig:
    """Complete configuration for Window BA pipeline."""
    device: str = "cuda"
    track_loader: TrackLoaderConfig = field(default_factory=TrackLoaderConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Runtime options
    use_checkpoint: bool = True
    verbose: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_devices = ["cuda", "cpu"]
        if self.device not in valid_devices:
            # Check if it's a specific CUDA device
            if not self.device.startswith("cuda:"):
                raise ValueError(f"Invalid device: {self.device}. Must be one of {valid_devices} or cuda:N")


class ConfigManager:
    """Manager for loading and merging configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to user configuration file, or None for defaults
        """
        self.default_config_path = Path(__file__).parent.parent.parent / "config" / "window_ba.yaml"
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> WindowBAConfig:
        """Load and merge configuration from files."""
        # Start with default configuration
        config_dict = self._get_default_config_dict()
        
        # Load default YAML if it exists
        if self.default_config_path.exists():
            default_yaml = self._load_yaml(self.default_config_path)
            config_dict = self._merge_configs(config_dict, default_yaml)
            logger.info(f"Loaded default config from {self.default_config_path}")
        
        # Load user YAML if provided
        if self.config_path and self.config_path.exists():
            user_yaml = self._load_yaml(self.config_path)
            config_dict = self._merge_configs(config_dict, user_yaml)
            logger.info(f"Loaded user config from {self.config_path}")
        
        # Create structured config object
        return self._dict_to_config(config_dict)
    
    def _get_default_config_dict(self) -> Dict[str, Any]:
        """Get default configuration as dictionary."""
        return asdict(WindowBAConfig())
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Configuration to override with
            
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._merge_configs(result[key], value)
            else:
                # Override value
                result[key] = value
        
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> WindowBAConfig:
        """Convert dictionary to structured configuration object."""
        # Extract nested configurations
        track_loader_dict = config_dict.get('track_loader', {})
        optimization_dict = config_dict.get('optimization', {})
        camera_dict = config_dict.get('camera', {})
        output_dict = config_dict.get('output', {})
        visualization_dict = config_dict.get('visualization', {})
        
        # Create nested config objects
        track_loader = TrackLoaderConfig(**track_loader_dict)
        optimization = OptimizationConfig(**optimization_dict)
        camera = CameraConfig(**camera_dict)
        output = OutputConfig(**output_dict)
        visualization = VisualizationConfig(**visualization_dict)
        
        # Create main config
        return WindowBAConfig(
            device=config_dict.get('device', 'cuda'),
            track_loader=track_loader,
            optimization=optimization,
            camera=camera,
            output=output,
            visualization=visualization,
            use_checkpoint=config_dict.get('use_checkpoint', True),
            verbose=config_dict.get('verbose', False)
        )
    
    def save_config(self, path: Path) -> None:
        """Save current configuration to YAML file."""
        config_dict = asdict(self.config)
        
        # Convert tuple to list for YAML serialization
        if 'visualization' in config_dict and 'figure_size' in config_dict['visualization']:
            config_dict['visualization']['figure_size'] = list(config_dict['visualization']['figure_size'])
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved configuration to {path}")
    
    def update_from_args(self, **kwargs) -> None:
        """Update configuration from command-line arguments."""
        for key, value in kwargs.items():
            if value is not None:
                # Handle nested attributes
                if '.' in key:
                    parts = key.split('.')
                    obj = self.config
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], value)
                else:
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
    
    def validate(self) -> None:
        """Validate the complete configuration."""
        # This triggers __post_init__ validation for all nested configs
        self.config.__post_init__()
        self.config.track_loader.__post_init__()
        self.config.optimization.__post_init__()
        self.config.camera.__post_init__()
        self.config.output.__post_init__()
        self.config.visualization.__post_init__()