"""
Checkpoint management for Window-based Bundle Adjustment.

This module handles saving and loading of intermediate results
to enable pipeline resumption and debugging.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from ..models.data_models import (
    WindowTrackData, CameraParameters, PipelineState
)
from ..models.camera_model import CameraModel

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manager for saving and loading pipeline checkpoints."""
    
    # Checkpoint file names
    TRACKS_FILE = "checkpoint_tracks.npz"
    TRACKS_3D_FILE = "checkpoint_tracks_3d.npz" 
    CAMERAS_PHASE1_FILE = "cameras_phase1.npz"
    CAMERAS_FINAL_FILE = "cameras_final.npz"
    WINDOW_TRACKS_FILE = "window_tracks_3d.npz"
    PHASE1_HISTORY_FILE = "phase1_history.json"
    PHASE2_HISTORY_FILE = "phase2_history.json"
    PIPELINE_STATE_FILE = "pipeline_state.json"
    
    def __init__(self, output_dir: Path):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Directory to save/load checkpoints
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def detect_completed_steps(self) -> PipelineState:
        """
        Automatically detect which steps have been completed by checking output files.
        
        Returns:
            PipelineState object with completed steps information
        """
        state = PipelineState()
        
        # Check for pipeline state file first
        state_file = self.output_dir / self.PIPELINE_STATE_FILE
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    saved_state = json.load(f)
                state.completed_steps = saved_state.get('completed_steps', [])
                state.current_step = saved_state.get('current_step', 0)
                logger.info(f"Loaded pipeline state: step {state.current_step}, completed: {state.completed_steps}")
            except Exception as e:
                logger.warning(f"Failed to load pipeline state: {e}")
        
        # Verify based on actual files
        # Step 1: Basic tracks loaded
        if (self.output_dir / self.TRACKS_FILE).exists():
            if "track_loading" not in state.completed_steps:
                state.completed_steps.append("track_loading")
            state.current_step = max(state.current_step, 1)
            
        # Step 2: 3D initialized tracks exist
        if (self.output_dir / self.TRACKS_3D_FILE).exists():
            if "depth_initialization" not in state.completed_steps:
                state.completed_steps.append("depth_initialization")
            state.current_step = max(state.current_step, 2)
            
        # Step 3: Phase 1 camera results exist
        if (self.output_dir / self.CAMERAS_PHASE1_FILE).exists():
            if "phase1_optimization" not in state.completed_steps:
                state.completed_steps.append("phase1_optimization")
            state.current_step = max(state.current_step, 3)
            
        # Step 4: Phase 2 results exist
        if (self.output_dir / self.WINDOW_TRACKS_FILE).exists():
            data = np.load(self.output_dir / self.WINDOW_TRACKS_FILE, allow_pickle=True)
            if 'windows' in data:
                windows = data['windows']
                # Check if any window has boundary_3d_optimized (Phase 2 marker)
                if any('boundary_3d_optimized' in w for w in windows):
                    if "phase2_optimization" not in state.completed_steps:
                        state.completed_steps.append("phase2_optimization")
                    state.current_step = max(state.current_step, 4)
                    
        # Step 5: COLMAP export exists
        colmap_dir = self.output_dir / 'colmap'
        if (colmap_dir / 'cameras.bin').exists() or (colmap_dir / 'cameras.txt').exists():
            if "colmap_export" not in state.completed_steps:
                state.completed_steps.append("colmap_export")
            state.current_step = max(state.current_step, 5)
            
        logger.info(f"Detected pipeline state: step {state.current_step}, completed steps: {state.completed_steps}")
        
        return state
    
    def save_tracks(self, window_tracks: List[WindowTrackData], include_3d: bool = False):
        """
        Save window tracks to checkpoint.
        
        Args:
            window_tracks: List of WindowTrackData objects
            include_3d: Whether to include 3D information
        """
        filename = self.TRACKS_3D_FILE if include_3d else self.TRACKS_FILE
        
        # Convert to dictionary format for saving
        save_data = []
        for track in window_tracks:
            # Convert to dict but exclude 3D data if not requested
            track_dict = track.to_dict()
            
            if not include_3d:
                # Remove 3D-related fields
                for key in ['tracks_3d', 'xyzw_world', 'boundary_3d_optimized', 
                           'query_3d_start', 'query_3d_end']:
                    track_dict.pop(key, None)
            
            save_data.append(track_dict)
        
        np.savez_compressed(self.output_dir / filename, windows=save_data)
        logger.info(f"Saved {len(window_tracks)} window tracks to {filename}")
    
    def load_tracks(self, include_3d: bool = False) -> Optional[List[WindowTrackData]]:
        """
        Load window tracks from checkpoint.
        
        Args:
            include_3d: Whether to load 3D information
            
        Returns:
            List of WindowTrackData objects or None if not found
        """
        filename = self.TRACKS_3D_FILE if include_3d else self.TRACKS_FILE
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            return None
        
        try:
            data = np.load(filepath, allow_pickle=True)
            window_tracks = []
            
            for track_dict in data['windows']:
                # Convert dict to WindowTrackData
                window_tracks.append(WindowTrackData.from_dict(track_dict))
            
            logger.info(f"Loaded {len(window_tracks)} window tracks from {filename}")
            return window_tracks
            
        except Exception as e:
            logger.error(f"Failed to load tracks from {filename}: {e}")
            return None
    
    def save_camera_parameters(self, camera_params: CameraParameters, phase: str = "final"):
        """
        Save camera parameters to checkpoint.
        
        Args:
            camera_params: CameraParameters object
            phase: "phase1" or "final"
        """
        filename = self.CAMERAS_PHASE1_FILE if phase == "phase1" else self.CAMERAS_FINAL_FILE
        
        # Convert to numpy
        save_data = camera_params.to_numpy()
        
        np.savez_compressed(self.output_dir / filename, **save_data)
        logger.info(f"Saved camera parameters to {filename}")
    
    def load_camera_parameters(self, phase: str = "final", device: str = 'cuda') -> Optional[CameraParameters]:
        """
        Load camera parameters from checkpoint.
        
        Args:
            phase: "phase1" or "final"
            device: Device to load tensors to
            
        Returns:
            CameraParameters object or None if not found
        """
        filename = self.CAMERAS_PHASE1_FILE if phase == "phase1" else self.CAMERAS_FINAL_FILE
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            return None
        
        try:
            data = np.load(filepath)
            camera_params = CameraParameters.from_numpy(
                {key: data[key] for key in data.files},
                device=device
            )
            logger.info(f"Loaded camera parameters from {filename}")
            return camera_params
            
        except Exception as e:
            logger.error(f"Failed to load camera parameters from {filename}: {e}")
            return None
    
    def save_optimization_history(self, history: Dict[str, Any], phase: str):
        """
        Save optimization history.
        
        Args:
            history: Dictionary containing optimization history
            phase: "phase1" or "phase2"
        """
        filename = self.PHASE1_HISTORY_FILE if phase == "phase1" else self.PHASE2_HISTORY_FILE
        
        with open(self.output_dir / filename, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Saved {phase} optimization history")
    
    def load_optimization_history(self, phase: str) -> Optional[Dict[str, Any]]:
        """
        Load optimization history.
        
        Args:
            phase: "phase1" or "phase2"
            
        Returns:
            History dictionary or None if not found
        """
        filename = self.PHASE1_HISTORY_FILE if phase == "phase1" else self.PHASE2_HISTORY_FILE
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r') as f:
                history = json.load(f)
            logger.info(f"Loaded {phase} optimization history")
            return history
            
        except Exception as e:
            logger.error(f"Failed to load {phase} history: {e}")
            return None
    
    def save_pipeline_state(self, state: PipelineState):
        """
        Save pipeline state.
        
        Args:
            state: PipelineState object
        """
        save_data = {
            'current_step': state.current_step,
            'completed_steps': state.completed_steps,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / self.PIPELINE_STATE_FILE, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Saved pipeline state: step {state.current_step}")
    
    def save_final_window_tracks(self, window_tracks: List[WindowTrackData]):
        """
        Save final window tracks with all 3D information.
        
        Args:
            window_tracks: List of WindowTrackData objects
        """
        save_data = []
        for track in window_tracks:
            save_data.append(track.to_dict())
        
        np.savez_compressed(
            self.output_dir / self.WINDOW_TRACKS_FILE,
            windows=save_data
        )
        logger.info(f"Saved final window tracks to {self.WINDOW_TRACKS_FILE}")
    
    def create_camera_model(self, 
                          camera_params: CameraParameters,
                          num_frames: int,
                          init_tan_fov_x: float,
                          init_tan_fov_y: float,
                          single_camera: bool = False,
                          device: str = 'cuda') -> CameraModel:
        """
        Create and initialize a CameraModel from saved parameters.
        
        Args:
            camera_params: Saved camera parameters
            num_frames: Total number of frames
            init_tan_fov_x: Initial FOV (will be overridden)
            init_tan_fov_y: Initial FOV (will be overridden)
            single_camera: Whether using single camera mode
            device: Device to create model on
            
        Returns:
            Initialized CameraModel
        """
        # Create model with initial FOV values
        camera_model = CameraModel(
            num_frames=num_frames,
            init_tan_fov_x=init_tan_fov_x,
            init_tan_fov_y=init_tan_fov_y,
            single_camera=single_camera
        ).to(device)
        
        # Load saved parameters
        camera_model.load_parameters(camera_params)
        
        return camera_model
    
    def checkpoint_exists(self, checkpoint_name: str) -> bool:
        """
        Check if a specific checkpoint file exists.
        
        Args:
            checkpoint_name: Name of checkpoint to check
            
        Returns:
            True if checkpoint exists
        """
        checkpoint_files = {
            'tracks': self.TRACKS_FILE,
            'tracks_3d': self.TRACKS_3D_FILE,
            'cameras_phase1': self.CAMERAS_PHASE1_FILE,
            'cameras_final': self.CAMERAS_FINAL_FILE,
            'phase1_history': self.PHASE1_HISTORY_FILE,
            'phase2_history': self.PHASE2_HISTORY_FILE,
            'pipeline_state': self.PIPELINE_STATE_FILE,
            'window_tracks': self.WINDOW_TRACKS_FILE
        }
        
        if checkpoint_name not in checkpoint_files:
            return False
        
        return (self.output_dir / checkpoint_files[checkpoint_name]).exists()
    
    def clear_checkpoints(self):
        """Clear all checkpoint files."""
        checkpoint_files = [
            self.TRACKS_FILE,
            self.TRACKS_3D_FILE,
            self.CAMERAS_PHASE1_FILE,
            self.CAMERAS_FINAL_FILE,
            self.WINDOW_TRACKS_FILE,
            self.PHASE1_HISTORY_FILE,
            self.PHASE2_HISTORY_FILE,
            self.PIPELINE_STATE_FILE
        ]
        
        for filename in checkpoint_files:
            filepath = self.output_dir / filename
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Removed checkpoint: {filename}")
        
        logger.info("Cleared all checkpoints")