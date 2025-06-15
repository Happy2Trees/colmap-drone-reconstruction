"""
Window-based Bundle Adjustment pipeline following GeometryCrafter.
Refactored version with cleaner architecture.
"""

import numpy as np
import torch
import logging
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from .utils.config_manager import ConfigManager
from .models.data_models import (
    WindowTrackData, CameraIntrinsics,
    PipelineState
)
from .utils.checkpoint_manager import CheckpointManager
from .core.window_track_loader import WindowTrackLoader
from .core.window_depth_initializer import WindowDepthInitializer
from .core.window_bundle_adjuster import WindowBundleAdjuster
from .utils.colmap_exporter import COLMAPExporter
from .utils.visualization import WindowBAVisualizer

logger = logging.getLogger(__name__)


class WindowBAPipeline:
    """
    Complete pipeline for window-based bundle adjustment.
    
    This class orchestrates the entire pipeline:
    1. Load tracks and camera parameters
    2. Initialize 3D points from depth
    3. Phase 1: Camera-only optimization
    4. Phase 2: Joint camera + 3D optimization (optional)
    5. Export results to COLMAP format
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        logger.info(f"Initialized WindowBA pipeline with device: {self.config.device}")
    
    def run(self, scene_dir: Path, output_dir: Path, use_refine: bool = False) -> Dict:
        """
        Run the complete window-based bundle adjustment pipeline.
        
        Args:
            scene_dir: Directory containing images, tracks, depth, and camera params
            output_dir: Directory to save results
            use_refine: Whether to run Phase 2 (camera + 3D refinement)
            
        Returns:
            Dictionary with pipeline results
        """
        scene_dir = Path(scene_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store scene_dir for use in methods
        self.scene_dir = scene_dir
        
        # Setup logging
        self._setup_logging(output_dir)
        
        logger.info(f"Starting Window BA pipeline for scene: {scene_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Use refinement (Phase 2): {use_refine}")
        
        # Initialize results
        results = {
            'scene_dir': str(scene_dir),
            'output_dir': str(output_dir),
            'start_time': datetime.now().isoformat(),
            'config': self.config_manager.config.__dict__
        }
        
        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(output_dir)
        
        # Detect and load pipeline state
        pipeline_state = checkpoint_manager.detect_completed_steps()
        
        try:
            # Execute pipeline steps
            if not pipeline_state.is_step_completed("track_loading"):
                pipeline_state = self._step1_load_tracks(
                    scene_dir, pipeline_state, checkpoint_manager
                )
            
            if not pipeline_state.is_step_completed("depth_initialization"):
                pipeline_state = self._step2_initialize_3d(
                    scene_dir, pipeline_state, checkpoint_manager
                )
            
            if not pipeline_state.is_step_completed("phase1_optimization"):
                pipeline_state = self._step3_optimize_phase1(
                    pipeline_state, checkpoint_manager
                )
            
            if use_refine and not pipeline_state.is_step_completed("phase2_optimization"):
                pipeline_state = self._step4_optimize_phase2(
                    pipeline_state, checkpoint_manager
                )
            
            if not pipeline_state.is_step_completed("colmap_export"):
                pipeline_state = self._step5_export_results(
                    scene_dir, pipeline_state, output_dir
                )
            
            # Create visualizations if enabled
            if self.config.visualization.enabled:
                self._create_visualizations(pipeline_state, output_dir)
            
            # Finalize results
            results['end_time'] = datetime.now().isoformat()
            results['success'] = True
            results['phase1'] = pipeline_state.phase1_result.to_dict() if pipeline_state.phase1_result else None
            results['phase2'] = pipeline_state.phase2_result.to_dict() if pipeline_state.phase2_result else None
            
            # Save summary
            self._save_summary(results, output_dir)
            
            logger.info("Window BA pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            results['error'] = str(e)
            results['success'] = False
            raise
        
        return results
    
    def _step1_load_tracks(self, 
                          scene_dir: Path,
                          pipeline_state: PipelineState,
                          checkpoint_manager: CheckpointManager) -> PipelineState:
        """Step 1: Load tracks and camera parameters."""
        logger.info("Step 1: Loading tracks and camera parameters")
        
        # Initialize track loader
        track_loader = WindowTrackLoader(
            scene_dir,
            device=self.config.device,
            image_width=self.config.camera.image_width,
            image_height=self.config.camera.image_height
        )
        
        # Load window tracks
        track_dir = scene_dir / 'cotracker'
        track_pattern = self._get_track_pattern()
        
        window_tracks_dict = track_loader.load_window_tracks(track_dir, track_pattern)
        
        # Convert to WindowTrackData objects
        window_tracks = [WindowTrackData.from_dict(track) for track in window_tracks_dict]
        
        # Add depth to tracks
        depth_dir = scene_dir / self.config.track_loader.depth_subdir
        window_tracks_dict = track_loader.create_tracks_with_depth(window_tracks_dict, depth_dir)
        window_tracks = [WindowTrackData.from_dict(track) for track in window_tracks_dict]
        
        # Update pipeline state
        pipeline_state.window_tracks = window_tracks
        pipeline_state.mark_step_completed("track_loading")
        
        # Save checkpoint
        checkpoint_manager.save_tracks(window_tracks, include_3d=False)
        checkpoint_manager.save_pipeline_state(pipeline_state)
        
        logger.info(f"Loaded {len(window_tracks)} window tracks")
        return pipeline_state
    
    def _step2_initialize_3d(self,
                            scene_dir: Path,
                            pipeline_state: PipelineState,
                            checkpoint_manager: CheckpointManager) -> PipelineState:
        """Step 2: Initialize 3D points from depth."""
        logger.info("Step 2: Initializing 3D points from depth")
        
        # Load tracks if not in state
        if pipeline_state.window_tracks is None:
            pipeline_state.window_tracks = checkpoint_manager.load_tracks(include_3d=False)
        
        # Initialize depth initializer
        track_loader = WindowTrackLoader(
            scene_dir,
            device=self.config.device,
            image_width=self.config.camera.image_width,
            image_height=self.config.camera.image_height
        )
        
        depth_initializer = WindowDepthInitializer(
            track_loader.intrinsics,
            device=self.config.device
        )
        
        # Triangulate tracks
        if pipeline_state.window_tracks is None:
            raise ValueError("Window tracks are not loaded")
        
        window_tracks_dict = [track.to_dict() for track in pipeline_state.window_tracks]
        window_tracks_dict = depth_initializer.triangulate_window_tracks(window_tracks_dict)
        pipeline_state.window_tracks = [WindowTrackData.from_dict(track) for track in window_tracks_dict]
        
        # Update state
        pipeline_state.mark_step_completed("depth_initialization")
        
        # Save checkpoint
        checkpoint_manager.save_tracks(pipeline_state.window_tracks, include_3d=True)
        checkpoint_manager.save_pipeline_state(pipeline_state)
        
        logger.info("Initialized 3D points for all windows")
        return pipeline_state
    
    def _step3_optimize_phase1(self,
                              pipeline_state: PipelineState,
                              checkpoint_manager: CheckpointManager) -> PipelineState:
        """Step 3: Phase 1 - Camera-only optimization."""
        logger.info("Step 3: Phase 1 - Camera-only optimization")
        
        # Load tracks if not in state
        if pipeline_state.window_tracks is None:
            pipeline_state.window_tracks = checkpoint_manager.load_tracks(include_3d=True)
        
        # Get FOV from intrinsics
        if pipeline_state.window_tracks is None or len(pipeline_state.window_tracks) == 0:
            raise ValueError("Window tracks are not available")
        
        track_loader = WindowTrackLoader(
            self.scene_dir,
            device=self.config.device,
            image_width=self.config.camera.image_width,
            image_height=self.config.camera.image_height
        )
        fov_x, fov_y = track_loader.get_fov_from_intrinsics(
            self.config.camera.image_width,
            self.config.camera.image_height
        )
        tan_fov_x = np.tan(fov_x / 2)
        tan_fov_y = np.tan(fov_y / 2)
        
        # Initialize bundle adjuster
        bundle_adjuster = WindowBundleAdjuster(
            self.config.optimization,
            device=self.config.device,
            image_width=self.config.camera.image_width,
            image_height=self.config.camera.image_height,
            single_camera=self.config.camera.single_camera
        )
        
        # Run Phase 1 optimization
        if pipeline_state.window_tracks is None:
            raise ValueError("Window tracks are not loaded for Phase 1")
        
        camera_model, phase1_result = bundle_adjuster.optimize_phase1(
            pipeline_state.window_tracks,
            tan_fov_x,
            tan_fov_y
        )
        
        # Update state
        pipeline_state.camera_model = camera_model
        pipeline_state.phase1_result = phase1_result
        pipeline_state.mark_step_completed("phase1_optimization")
        
        # Save checkpoints
        if phase1_result.camera_params is not None:
            checkpoint_manager.save_camera_parameters(phase1_result.camera_params, phase="phase1")
        checkpoint_manager.save_optimization_history(phase1_result.history, phase="phase1")
        checkpoint_manager.save_pipeline_state(pipeline_state)
        
        logger.info(f"Phase 1 completed: final loss = {phase1_result.final_loss:.6f}")
        return pipeline_state
    
    def _step4_optimize_phase2(self,
                              pipeline_state: PipelineState,
                              checkpoint_manager: CheckpointManager) -> PipelineState:
        """Step 4: Phase 2 - Joint camera and 3D optimization."""
        logger.info("Step 4: Phase 2 - Joint camera and 3D optimization")
        
        # Load camera model if not in state
        if pipeline_state.camera_model is None:
            camera_params = checkpoint_manager.load_camera_parameters(phase="phase1")
            if camera_params and pipeline_state.window_tracks:
                # Recreate camera model
                max_frame = max(track.end_frame for track in pipeline_state.window_tracks) + 1
                tan_fov_x, tan_fov_y = camera_params.tan_fov_x, camera_params.tan_fov_y
                
                # Convert tensor/array to float if needed
                if isinstance(tan_fov_x, torch.Tensor):
                    tan_fov_x_float = float(tan_fov_x.item())
                elif isinstance(tan_fov_x, np.ndarray):
                    tan_fov_x_float = float(tan_fov_x)
                else:
                    tan_fov_x_float = float(tan_fov_x)
                    
                if isinstance(tan_fov_y, torch.Tensor):
                    tan_fov_y_float = float(tan_fov_y.item())
                elif isinstance(tan_fov_y, np.ndarray):
                    tan_fov_y_float = float(tan_fov_y)
                else:
                    tan_fov_y_float = float(tan_fov_y)
                
                pipeline_state.camera_model = checkpoint_manager.create_camera_model(
                    camera_params,
                    max_frame,
                    tan_fov_x_float,
                    tan_fov_y_float,
                    self.config.camera.single_camera,
                    self.config.device
                )
        
        # Initialize bundle adjuster
        bundle_adjuster = WindowBundleAdjuster(
            self.config.optimization,
            device=self.config.device,
            image_width=self.config.camera.image_width,
            image_height=self.config.camera.image_height,
            single_camera=self.config.camera.single_camera
        )
        
        # Run Phase 2 optimization
        if pipeline_state.window_tracks is None:
            raise ValueError("Window tracks are not loaded for Phase 2")
        if pipeline_state.camera_model is None:
            raise ValueError("Camera model is not initialized for Phase 2")
        
        camera_model, phase2_result = bundle_adjuster.optimize_phase2(
            pipeline_state.window_tracks,
            pipeline_state.camera_model
        )
        
        # Update state
        pipeline_state.camera_model = camera_model
        pipeline_state.phase2_result = phase2_result
        pipeline_state.mark_step_completed("phase2_optimization")
        
        # Save checkpoints
        if phase2_result.camera_params is not None:
            checkpoint_manager.save_camera_parameters(phase2_result.camera_params, phase="final")
        checkpoint_manager.save_optimization_history(phase2_result.history, phase="phase2")
        if pipeline_state.window_tracks is not None:
            checkpoint_manager.save_final_window_tracks(pipeline_state.window_tracks)
        checkpoint_manager.save_pipeline_state(pipeline_state)
        
        logger.info(f"Phase 2 completed: final loss = {phase2_result.final_loss:.6f}")
        return pipeline_state
    
    def _step5_export_results(self,
                             scene_dir: Path,
                             pipeline_state: PipelineState,
                             output_dir: Path) -> PipelineState:
        """Step 5: Export results to COLMAP format."""
        logger.info("Step 5: Exporting results to COLMAP format")
        
        # Load camera model if needed
        if pipeline_state.camera_model is None:
            phase = "final" if pipeline_state.is_step_completed("phase2_optimization") else "phase1"
            camera_params = CheckpointManager(output_dir).load_camera_parameters(phase=phase)
            
            if camera_params and pipeline_state.window_tracks:
                max_frame = max(track.end_frame for track in pipeline_state.window_tracks) + 1
                tan_fov_x, tan_fov_y = camera_params.tan_fov_x, camera_params.tan_fov_y
                
                # Convert tensor/array to float if needed
                if isinstance(tan_fov_x, torch.Tensor):
                    tan_fov_x_float = float(tan_fov_x.item())
                elif isinstance(tan_fov_x, np.ndarray):
                    tan_fov_x_float = float(tan_fov_x)
                else:
                    tan_fov_x_float = float(tan_fov_x)
                    
                if isinstance(tan_fov_y, torch.Tensor):
                    tan_fov_y_float = float(tan_fov_y.item())
                elif isinstance(tan_fov_y, np.ndarray):
                    tan_fov_y_float = float(tan_fov_y)
                else:
                    tan_fov_y_float = float(tan_fov_y)
                
                pipeline_state.camera_model = CheckpointManager(output_dir).create_camera_model(
                    camera_params,
                    max_frame,
                    tan_fov_x_float,
                    tan_fov_y_float,
                    self.config.camera.single_camera,
                    self.config.device
                )
        
        # Load intrinsics
        track_loader = WindowTrackLoader(
            scene_dir,
            device=self.config.device,
            image_width=self.config.camera.image_width,
            image_height=self.config.camera.image_height
        )
        intrinsics = CameraIntrinsics.from_matrix(
            track_loader.intrinsics,
            self.config.camera.image_width,
            self.config.camera.image_height
        )
        
        # Export to COLMAP
        if self.config.output.save_colmap:
            if pipeline_state.camera_model is None:
                logger.warning("Camera model not available for COLMAP export")
                return pipeline_state
            if pipeline_state.window_tracks is None:
                logger.warning("Window tracks not available for COLMAP export")
                return pipeline_state
            
            exporter = COLMAPExporter(
                output_dir / 'colmap',
                binary_format=(self.config.output.colmap_format == 'binary')
            )
            exporter.export(
                pipeline_state.camera_model,
                pipeline_state.window_tracks,
                intrinsics
            )
        
        # Update state
        pipeline_state.mark_step_completed("colmap_export")
        CheckpointManager(output_dir).save_pipeline_state(pipeline_state)
        
        logger.info("Export completed")
        return pipeline_state
    
    def _get_track_pattern(self) -> str:
        """Get track file pattern based on configuration."""
        track_mode = self.config.track_loader.track_mode
        
        if self.config.track_loader.bidirectional_priority:
            # Try bidirectional first
            bidirectional_patterns = {
                'sift': '*_sift_bidirectional.npy',
                'superpoint': '*_superpoint_bidirectional.npy',
                'grid': '*_grid_bidirectional.npy'
            }
            return bidirectional_patterns.get(track_mode, '*_sift_bidirectional.npy')
        else:
            # Use unidirectional
            unidirectional_patterns = {
                'sift': '*_sift.npy',
                'superpoint': '*_superpoint.npy',
                'grid': '*_grid.npy'
            }
            return unidirectional_patterns.get(track_mode, '*_sift.npy')
    
    def _setup_logging(self, output_dir: Path):
        """Setup file logging for the pipeline."""
        log_file = output_dir / 'window_ba.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def _create_visualizations(self, pipeline_state: PipelineState, output_dir: Path):
        """Create visualizations if enabled."""
        logger.info("Creating visualizations (saving as PNG files)")
        
        try:
            visualizer = WindowBAVisualizer(
                output_dir / 'visualizations',
                self.config.camera.image_width,
                self.config.camera.image_height
            )
            
            if self.config.visualization.camera_trajectory:
                if pipeline_state.camera_model and pipeline_state.window_tracks:
                    window_tracks_dict = [track.to_dict() for track in pipeline_state.window_tracks]
                    visualizer.visualize_camera_trajectory(
                        pipeline_state.camera_model,
                        window_tracks_dict
                    )
            
            if self.config.visualization.point_cloud:
                if pipeline_state.window_tracks:
                    window_tracks_dict = [track.to_dict() for track in pipeline_state.window_tracks]
                    visualizer.visualize_3d_points(
                        window_tracks_dict,
                        pipeline_state.camera_model
                    )
            
            if self.config.visualization.reprojection_errors:
                if pipeline_state.window_tracks and pipeline_state.camera_model:
                    window_tracks_dict = [track.to_dict() for track in pipeline_state.window_tracks]
                    visualizer.visualize_reprojection_errors(
                        window_tracks_dict,
                        pipeline_state.camera_model
                    )
            
            if self.config.visualization.summary_plot:
                if pipeline_state.camera_model and pipeline_state.window_tracks:
                    window_tracks_dict = [track.to_dict() for track in pipeline_state.window_tracks]
                    phase1_history = pipeline_state.phase1_result.history if pipeline_state.phase1_result else {}
                    phase2_history = pipeline_state.phase2_result.history if pipeline_state.phase2_result else {}
                    visualizer.create_summary_visualization(
                        pipeline_state.camera_model,
                        window_tracks_dict,
                        phase1_history,
                        phase2_history
                    )
            
            logger.info("Visualizations created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")
    
    def _save_summary(self, results: Dict, output_dir: Path):
        """Save pipeline summary."""
        # Save JSON summary
        with open(output_dir / 'pipeline_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create human-readable summary
        with open(output_dir / 'summary.txt', 'w') as f:
            f.write("Window-based Bundle Adjustment Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Scene: {results['scene_dir']}\n")
            f.write(f"Output: {results['output_dir']}\n")
            f.write(f"Start time: {results['start_time']}\n")
            f.write(f"End time: {results.get('end_time', 'N/A')}\n\n")
            
            if 'phase1' in results and results['phase1']:
                f.write("Phase 1 Results:\n")
                f.write(f"  - Final loss: {results['phase1']['final_loss']:.6f}\n")
                f.write(f"  - Iterations: {results['phase1']['iterations']}\n")
                f.write(f"  - Converged: {results['phase1']['converged']}\n\n")
            
            if 'phase2' in results and results['phase2']:
                f.write("Phase 2 Results:\n")
                f.write(f"  - Final loss: {results['phase2']['final_loss']:.6f}\n")
                f.write(f"  - Iterations: {results['phase2']['iterations']}\n")
                f.write(f"  - Converged: {results['phase2']['converged']}\n\n")
            
            f.write(f"Success: {results.get('success', False)}\n")