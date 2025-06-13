"""
Window-based Bundle Adjustment module following GeometryCrafter's approach.

This module implements:
- Window-based track processing without merging
- Cross-projection bundle adjustment
- Depth-based 3D initialization
- Two-phase optimization (camera-only, then camera+3D)
"""

from .window_track_loader import WindowTrackLoader
from .window_depth_initializer import WindowDepthInitializer
from .window_bundle_adjuster import WindowBundleAdjuster
from .pipeline import WindowBAPipeline

__all__ = [
    'WindowTrackLoader',
    'WindowDepthInitializer',
    'WindowBundleAdjuster',
    'WindowBAPipeline',
]