"""Co-Tracker + Global Bundle Adjustment Pipeline

This package implements a video-based Structure-from-Motion pipeline using:
- Meta's Co-Tracker for high-quality feature tracking
- GTSAM for global bundle adjustment
"""

from .feature_extractor import CoTrackerExtractor
from .track_manager import TrackManager
from .camera_model import CameraModel
from .bundle_adjustment import GlobalBundleAdjustment
from .pipeline import CoTrackerBAPipeline

__all__ = [
    'CoTrackerExtractor',
    'TrackManager', 
    'CameraModel',
    'GlobalBundleAdjustment',
    'CoTrackerBAPipeline'
]