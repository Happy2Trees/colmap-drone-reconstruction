"""
Window-based Bundle Adjustment pipeline following GeometryCrafter.
"""

import numpy as np
import torch
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import sys

# Add parent directory to path for colmap_utils
sys.path.append(str(Path(__file__).parent.parent))
from colmap_utils import read_write_model

from .window_track_loader import WindowTrackLoader
from .window_depth_initializer import WindowDepthInitializer
from .window_bundle_adjuster import WindowBundleAdjuster, OptimizationConfig

logger = logging.getLogger(__name__)

class WindowBAPipeline:
    """Complete pipeline for window-based bundle adjustment."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initialized WindowBA pipeline with device: {self.device}")
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration from YAML file or use defaults."""
        default_config = {
            'device': 'cuda',
            'track_loader': {
                'track_pattern': '*_sift.npy',
                'depth_subdir': 'depth/GeometryCrafter'
            },
            'optimization': {
                'max_iterations': 10000,
                'learning_rate_camera': 1e-3,
                'learning_rate_translation': 1e-2,
                'learning_rate_fov': 1e-4,
                'convergence_threshold': 1e-6,
                'use_robust_loss': True,
                'robust_loss_sigma': 1.0
            },
            'output': {
                'save_intermediate': True,
                'save_colmap': True,
                'colmap_format': 'binary'
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            # Merge with defaults
            default_config.update(user_config)
            logger.info(f"Loaded configuration from {config_path}")
        
        return default_config
    
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
        
        # Setup logging
        log_file = output_dir / 'window_ba.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Starting Window BA pipeline for scene: {scene_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        results = {
            'scene_dir': str(scene_dir),
            'output_dir': str(output_dir),
            'start_time': datetime.now().isoformat(),
            'config': self.config
        }
        
        try:
            # Step 1: Load tracks and camera parameters
            logger.info("Step 1: Loading tracks and camera parameters")
            track_loader = WindowTrackLoader(scene_dir, device=self.device)
            
            # Load window tracks (no merging)
            track_dir = scene_dir / 'cotracker'
            track_pattern = self.config['track_loader']['track_pattern']
            window_tracks = track_loader.load_window_tracks(track_dir, track_pattern)
            
            results['num_windows'] = len(window_tracks)
            results['total_frames'] = max(w['end_frame'] for w in window_tracks) + 1
            
            # Add depth to tracks
            depth_dir = scene_dir / self.config['track_loader']['depth_subdir']
            window_tracks = track_loader.create_tracks_with_depth(window_tracks, depth_dir)
            
            # Get image dimensions (assuming all images have same size)
            # For now, using hardcoded values from the data
            image_width, image_height = 1024, 576
            
            # Get FOV from intrinsics
            fov_x, fov_y = track_loader.get_fov_from_intrinsics(image_width, image_height)
            tan_fov_x = np.tan(fov_x / 2)
            tan_fov_y = np.tan(fov_y / 2)
            
            # Step 2: Initialize 3D points from depth
            logger.info("Step 2: Initializing 3D points from depth")
            depth_initializer = WindowDepthInitializer(track_loader.intrinsics, device=self.device)
            window_tracks = depth_initializer.triangulate_window_tracks(window_tracks)
            window_tracks = depth_initializer.compute_depth_confidence(window_tracks)
            
            # Step 3: Phase 1 - Camera-only optimization
            logger.info("Step 3: Phase 1 - Camera-only optimization")
            opt_config = OptimizationConfig(**self.config['optimization'])
            bundle_adjuster = WindowBundleAdjuster(opt_config, device=self.device)
            
            camera_model, phase1_history = bundle_adjuster.optimize_phase1(
                window_tracks, tan_fov_x, tan_fov_y
            )
            
            results['phase1'] = {
                'final_loss': phase1_history['losses'][-1] if phase1_history['losses'] else 0,
                'iterations': len(phase1_history['losses']),
                'converged': len(phase1_history['losses']) < opt_config.max_iterations
            }
            
            # Save Phase 1 results
            if self.config['output']['save_intermediate']:
                self._save_cameras(camera_model, output_dir / 'cameras_phase1.npz')
                self._save_history(phase1_history, output_dir / 'phase1_history.json')
            
            # Step 4: Phase 2 - Joint optimization (optional)
            if use_refine:
                logger.info("Step 4: Phase 2 - Joint camera and 3D optimization")
                camera_model, phase2_history = bundle_adjuster.optimize_phase2(
                    window_tracks, camera_model
                )
                results['phase2'] = phase2_history
            
            # Step 5: Save final results
            logger.info("Step 5: Saving final results")
            
            # Save optimized cameras
            self._save_cameras(camera_model, output_dir / 'cameras_final.npz')
            
            # Save window tracks with 3D
            self._save_window_tracks(window_tracks, output_dir / 'window_tracks_3d.npz')
            
            # Export to COLMAP format if requested
            if self.config['output']['save_colmap']:
                logger.info("Exporting to COLMAP format")
                self._export_to_colmap(camera_model, window_tracks, track_loader.intrinsics, 
                                     output_dir / 'colmap', image_width, image_height)
            
            # Save pipeline summary
            results['end_time'] = datetime.now().isoformat()
            results['success'] = True
            
            with open(output_dir / 'pipeline_summary.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Create human-readable summary
            self._create_summary(results, output_dir / 'summary.txt')
            
            logger.info("Window BA pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            results['error'] = str(e)
            results['success'] = False
            raise
        
        return results
    
    def _save_cameras(self, camera_model: torch.nn.Module, output_path: Path):
        """Save camera parameters to file."""
        with torch.no_grad():
            cameras = {
                'quaternions': camera_model.quaternions.cpu().numpy(),
                'translations': camera_model.translations.cpu().numpy(),
                'tan_fov_x': camera_model.tan_fov_x.cpu().numpy(),
                'tan_fov_y': camera_model.tan_fov_y.cpu().numpy(),
            }
        np.savez_compressed(output_path, **cameras)
        logger.info(f"Saved camera parameters to {output_path}")
    
    def _save_window_tracks(self, window_tracks: List[Dict], output_path: Path):
        """Save window tracks with 3D information."""
        # Convert to saveable format
        save_data = []
        for window in window_tracks:
            save_window = {
                'window_idx': window['window_idx'],
                'start_frame': window['start_frame'],
                'end_frame': window['end_frame'],
                'tracks': window['tracks'],
                'visibility': window['visibility'],
                'tracks_3d': window.get('tracks_3d', None),
                'xyzw_world': window.get('xyzw_world', None),
            }
            save_data.append(save_window)
        
        np.savez_compressed(output_path, windows=save_data)
        logger.info(f"Saved window tracks to {output_path}")
    
    def _save_history(self, history: Dict, output_path: Path):
        """Save optimization history."""
        with open(output_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _export_to_colmap(self, camera_model, window_tracks, intrinsics, 
                         output_dir: Path, width: int, height: int):
        """Export results to COLMAP format."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Prepare cameras dictionary
        cameras = self._create_colmap_cameras(intrinsics, width, height)
        
        # 2. Prepare images dictionary
        images = self._create_colmap_images(camera_model, window_tracks)
        
        # 3. Prepare points3D dictionary
        points3D = self._create_colmap_points3D(window_tracks)
        
        # 4. Write COLMAP model
        ext = '.bin' if self.config['output']['colmap_format'] == 'binary' else '.txt'
        read_write_model.write_model(cameras, images, points3D, str(output_dir), ext)
        
        logger.info(f"Exported COLMAP model to {output_dir} (format: {ext})")
        logger.info(f"  - Cameras: {len(cameras)}")
        logger.info(f"  - Images: {len(images)}")
        logger.info(f"  - Points3D: {len(points3D)}")
    
    def _create_colmap_cameras(self, intrinsics: np.ndarray, width: int, height: int) -> Dict:
        """Create COLMAP cameras dictionary."""
        # For simplicity, use a single camera model
        # COLMAP uses PINHOLE model: fx, fy, cx, cy
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        cameras = {
            1: read_write_model.Camera(
                id=1,
                model='PINHOLE',
                width=width,
                height=height,
                params=np.array([fx, fy, cx, cy])
            )
        }
        
        return cameras
    
    def _create_colmap_images(self, camera_model: torch.nn.Module, 
                             window_tracks: List[Dict]) -> Dict:
        """Create COLMAP images dictionary from optimized camera poses."""
        images = {}
        
        with torch.no_grad():
            # Get total number of frames
            max_frame = max(w['end_frame'] for w in window_tracks) + 1
            
            # Process each frame
            for frame_idx in range(max_frame):
                # Get quaternion and translation
                qvec = camera_model.quaternions[frame_idx].cpu().numpy()
                tvec = camera_model.translations[frame_idx].cpu().numpy()
                
                # Normalize quaternion
                qvec = qvec / np.linalg.norm(qvec)
                
                # Collect 2D-3D correspondences for this frame
                xys = []
                point3D_ids = []
                
                # Find tracks visible in this frame
                for window in window_tracks:
                    if window['start_frame'] <= frame_idx <= window['end_frame']:
                        # Frame index within window
                        window_frame_idx = frame_idx - window['start_frame']
                        
                        # Get visibility for this frame
                        visibility = window['visibility'][window_frame_idx]
                        
                        # Get 2D points
                        tracks_2d = window['tracks'][window_frame_idx]  # (N, 2)
                        
                        # Add visible points
                        for point_idx in range(len(visibility)):
                            if visibility[point_idx]:
                                # Create unique point3D ID
                                # Using window_idx * 10000 + point_idx to ensure uniqueness
                                point3D_id = window['window_idx'] * 10000 + point_idx
                                
                                xys.append(tracks_2d[point_idx])
                                point3D_ids.append(point3D_id)
                
                # Create image entry
                image_name = f"{frame_idx:05d}.jpg"  # Assuming jpg format
                
                if len(xys) > 0:
                    xys = np.array(xys)
                    point3D_ids = np.array(point3D_ids)
                else:
                    xys = np.zeros((0, 2))
                    point3D_ids = np.zeros(0, dtype=np.int32)
                
                images[frame_idx + 1] = read_write_model.Image(
                    id=frame_idx + 1,  # COLMAP uses 1-based indexing
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=1,  # Single camera model
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids
                )
        
        return images
    
    def _create_colmap_points3D(self, window_tracks: List[Dict]) -> Dict:
        """Create COLMAP points3D dictionary from window tracks."""
        points3D = {}
        
        for window in window_tracks:
            # Get 3D points for this window
            if 'xyzw_world' not in window:
                continue
                
            xyzw_world = window['xyzw_world']  # (T, N, 4)
            visibility = window['visibility']  # (T, N)
            
            T, N, _ = xyzw_world.shape
            
            # For each point in the window
            for point_idx in range(N):
                # Create unique point3D ID
                point3D_id = window['window_idx'] * 10000 + point_idx
                
                # Find frames where this point is visible
                visible_frames = []
                for t in range(T):
                    if visibility[t, point_idx]:
                        global_frame_idx = window['start_frame'] + t
                        visible_frames.append(global_frame_idx)
                
                if len(visible_frames) == 0:
                    continue
                
                # Use median 3D position across all observations
                xyz_observations = []
                image_ids = []
                point2D_idxs = []
                
                for t in range(T):
                    if visibility[t, point_idx]:
                        xyz = xyzw_world[t, point_idx, :3]
                        xyz_observations.append(xyz)
                        
                        # Image ID (1-based)
                        global_frame_idx = window['start_frame'] + t
                        image_ids.append(global_frame_idx + 1)
                        
                        # Point2D index in the image
                        # This needs to match the order in the image's xys array
                        # For simplicity, using the point index
                        point2D_idxs.append(point_idx)
                
                # Median 3D position
                xyz_median = np.median(xyz_observations, axis=0)
                
                # Error estimation (using std dev of observations)
                if len(xyz_observations) > 1:
                    error = np.std(xyz_observations)
                else:
                    error = 1.0  # Default error
                
                # Color (default to gray)
                rgb = np.array([128, 128, 128], dtype=np.uint8)
                
                points3D[point3D_id] = read_write_model.Point3D(
                    id=point3D_id,
                    xyz=xyz_median,
                    rgb=rgb,
                    error=error,
                    image_ids=np.array(image_ids, dtype=np.int32),
                    point2D_idxs=np.array(point2D_idxs, dtype=np.int32)
                )
        
        logger.info(f"Created {len(points3D)} 3D points from window tracks")
        
        return points3D
    
    def _create_summary(self, results: Dict, output_path: Path):
        """Create human-readable summary."""
        with open(output_path, 'w') as f:
            f.write("Window-based Bundle Adjustment Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Scene: {results['scene_dir']}\n")
            f.write(f"Output: {results['output_dir']}\n")
            f.write(f"Start time: {results['start_time']}\n")
            f.write(f"End time: {results.get('end_time', 'N/A')}\n\n")
            
            f.write(f"Data Statistics:\n")
            f.write(f"  - Number of windows: {results.get('num_windows', 0)}\n")
            f.write(f"  - Total frames: {results.get('total_frames', 0)}\n\n")
            
            if 'phase1' in results:
                f.write(f"Phase 1 Results:\n")
                f.write(f"  - Final loss: {results['phase1']['final_loss']:.6f}\n")
                f.write(f"  - Iterations: {results['phase1']['iterations']}\n")
                f.write(f"  - Converged: {results['phase1']['converged']}\n\n")
            
            if 'phase2' in results:
                f.write(f"Phase 2: Completed\n\n")
            
            f.write(f"Success: {results.get('success', False)}\n")