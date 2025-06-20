"""Main pipeline for precomputing all features for 3D reconstruction"""

import argparse
import logging
from pathlib import Path
from typing import Dict
import yaml
import json
import sys

from .trackers.cotracker_extractor import CoTrackerExtractor
from ..preprocessing import preprocess_scene, slice_fps_core

# Conditional imports for optional modules
try:
    from .depth.geometrycrafter import GeometryCrafterExtractor
    GEOMETRYCRAFTER_AVAILABLE = True
except ImportError:
    GEOMETRYCRAFTER_AVAILABLE = False
    
# Future imports:
# from .optical_flow.raft_extractor import RAFTExtractor


class PrecomputePipeline:
    """Main pipeline for precomputing all features"""
    
    def __init__(self, config: dict):
        self.config = config
        self.extractors = {}
        
        # Initialize extractors based on config
        if 'cotracker' in config['features']:
            logging.info("Initializing CoTracker extractor...")
            cotracker_config = config.get('cotracker', {})
            self.extractors['cotracker'] = CoTrackerExtractor(
                window_size=cotracker_config.get('window_size', 48),
                interval=cotracker_config.get('interval', 10),
                initialization_method=cotracker_config.get('initialization_method', 'grid'),
                device=cotracker_config.get('device', 'cuda'),
                max_features=cotracker_config.get('max_features', 400),
                superpoint_weights=cotracker_config.get('superpoint_weights', None)
            )
        
        # Initialize depth extractor
        if 'depth' in config['features']:
            if GEOMETRYCRAFTER_AVAILABLE:
                logging.info("Initializing GeometryCrafter depth extractor...")
                depth_config = config.get('depth', {})
                try:
                    logging.info("Initializing GeometryCrafter implementation")
                    self.extractors['depth'] = GeometryCrafterExtractor(depth_config)
                except Exception as e:
                    logging.error(f"Failed to initialize GeometryCrafter: {e}")
                    logging.warning("Depth extraction will be skipped")
            else:
                logging.warning("GeometryCrafter not available. Make sure GeometryCrafter is in submodules/")
            
        if 'flow' in config['features']:
            logging.warning("Optical flow extraction not yet implemented")
    
    def _is_preprocessing_valid(self, preprocessed_dir: Path, target_width: int, target_height: int) -> bool:
        """Check if existing preprocessed directory is valid"""
        # Check essential files exist
        required_files = ['K.txt', 'dist.txt', 'preprocessing_info.yaml']
        for file in required_files:
            if not (preprocessed_dir / file).exists():
                logging.debug(f"Missing required file: {file}")
                return False
        
        # Check images directory
        images_dir = preprocessed_dir / 'images'
        if not images_dir.exists() or not list(images_dir.glob('*.jpg')):
            logging.debug("Images directory missing or empty")
            return False
        
        # Check preprocessing info matches
        try:
            with open(preprocessed_dir / 'preprocessing_info.yaml', 'r') as f:
                info = yaml.safe_load(f)
            
            saved_width = info.get('target_size', {}).get('width')
            saved_height = info.get('target_size', {}).get('height')
            
            if saved_width != target_width or saved_height != target_height:
                logging.debug(f"Resolution mismatch: saved {saved_width}x{saved_height}, requested {target_width}x{target_height}")
                return False
                
            return True
            
        except Exception as e:
            logging.debug(f"Error reading preprocessing info: {e}")
            return False
    
    def _copy_camera_params(self, source_dir: Path, target_dir: Path):
        """Copy camera parameter files (K.txt, dist.txt) from source to target"""
        for param_file in ['K.txt', 'dist.txt']:
            source_file = source_dir / param_file
            if source_file.exists():
                import shutil
                shutil.copy2(source_file, target_dir / param_file)
                logging.debug(f"Copied {param_file} to {target_dir}")
    
    def _determine_final_output_dir(self, scene_dir: Path, preprocess_config: dict) -> Path:
        """Determine the final output directory name based on all preprocessing steps"""
        output_name = scene_dir.name
        
        # Add frame sampling suffix if enabled
        fs_config = preprocess_config.get('frame_sampling', {})
        if fs_config.get('enabled', False):
            if 'interval' in fs_config:
                output_name += f"_interval{fs_config['interval']}"
            else:
                output_name += f"_fps{fs_config['target_fps']}"
        
        # Add resolution suffix if enabled
        target_width = preprocess_config.get('target_width')
        target_height = preprocess_config.get('target_height')
        if target_width and target_height:
            output_name += f"_processed_{target_width}x{target_height}"
        
        return scene_dir.parent / output_name

    def _preprocess_if_needed(self, scene_dir: Path) -> Path:
        """Preprocess scene if enabled in config"""
        preprocess_config = self.config.get('preprocessing', {})
        
        if not preprocess_config.get('enabled', False):
            logging.info("Preprocessing disabled, using original scene")
            return scene_dir
        
        # Determine final output directory based on all preprocessing steps
        final_output_dir = self._determine_final_output_dir(scene_dir, preprocess_config)
        force_overwrite = preprocess_config.get('force_overwrite', False)
        
        # Check if final output already exists and is valid
        if final_output_dir.exists() and not force_overwrite:
            # Check if preprocessing info matches
            fs_config = preprocess_config.get('frame_sampling', {})
            target_width = preprocess_config.get('target_width')
            target_height = preprocess_config.get('target_height')
            
            # For now, just check if images directory exists
            images_dir = final_output_dir / 'images'
            if images_dir.exists() and list(images_dir.glob('*.jpg')):
                # If resolution preprocessing is enabled, verify it
                if target_width and target_height:
                    if self._is_preprocessing_valid(final_output_dir, target_width, target_height):
                        logging.info(f"Using existing preprocessed scene: {final_output_dir}")
                        return final_output_dir
                else:
                    # Just frame sampling, check if it exists
                    logging.info(f"Using existing preprocessed scene: {final_output_dir}")
                    return final_output_dir
            
            # Invalid, remove and re-process
            logging.warning(f"Existing preprocessed directory invalid, will re-process")
            import shutil
            shutil.rmtree(final_output_dir)
        
        # Apply preprocessing steps directly to final output directory
        logging.info(f"Preprocessing scene to: {final_output_dir}")
        
        # Step 1: Frame sampling (if enabled)
        fs_config = preprocess_config.get('frame_sampling', {})
        if fs_config.get('enabled', False):
            logging.info(f"Applying frame sampling: fps={fs_config.get('target_fps')}, interval={fs_config.get('interval')}")
            
            # If also doing resize, we need a temporary location for frame sampling
            target_width = preprocess_config.get('target_width')
            target_height = preprocess_config.get('target_height')
            
            if target_width and target_height:
                # Use temporary directory for frame sampling
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir) / "sampled"
                    
                    # Apply frame sampling to temp directory
                    result = slice_fps_core(
                        scene_dir / "images",
                        temp_path / "images",
                        target_fps=fs_config.get('target_fps'),
                        source_fps=fs_config.get('source_fps'),
                        interval=fs_config.get('interval')
                    )
                    
                    if result < 0:
                        raise RuntimeError("Frame sampling failed")
                    
                    # Copy camera parameters to temp directory
                    self._copy_camera_params(scene_dir, temp_path)
                    
                    # Apply resize and crop from temp to final
                    logging.info(f"Applying resize and crop to {target_width}x{target_height}...")
                    output_dir = preprocess_scene(
                        scene_dir=temp_path,
                        output_dir=final_output_dir,
                        target_size=(target_width, target_height),
                        force=True
                    )
                    
                    # Save combined preprocessing info
                    info = {
                        'source_dir': str(scene_dir),
                        'frame_sampling': fs_config,
                        'num_frames_sampled': result,
                        'target_size': {'width': target_width, 'height': target_height}
                    }
                    with open(final_output_dir / 'preprocessing_info.yaml', 'w') as f:
                        yaml.dump(info, f)
            else:
                # Only frame sampling, apply directly to final directory
                result = slice_fps_core(
                    scene_dir / "images",
                    final_output_dir / "images",
                    target_fps=fs_config.get('target_fps'),
                    source_fps=fs_config.get('source_fps'),
                    interval=fs_config.get('interval')
                )
                
                if result < 0:
                    raise RuntimeError("Frame sampling failed")
                
                # Copy camera parameters
                self._copy_camera_params(scene_dir, final_output_dir)
                
                # Save frame sampling info
                info = {
                    'source_dir': str(scene_dir),
                    'frame_sampling': fs_config,
                    'num_frames_sampled': result
                }
                with open(final_output_dir / 'frame_sampling_info.yaml', 'w') as f:
                    yaml.dump(info, f)
        
        # Step 2: Only resize and crop (no frame sampling)
        elif target_width := preprocess_config.get('target_width'):
            target_height = preprocess_config.get('target_height')
            if target_height:
                logging.info(f"Applying resize and crop to {target_width}x{target_height}...")
                output_dir = preprocess_scene(
                    scene_dir=scene_dir,
                    output_dir=final_output_dir,
                    target_size=(target_width, target_height),
                    force=force_overwrite
                )
        
        logging.info("Preprocessing completed successfully")
        return final_output_dir
    
    def run(self, scene_dir: str):
        """Run all configured extractors on the scene"""
        scene_dir_path = Path(scene_dir)
        
        if not scene_dir_path.exists():
            raise ValueError(f"Scene directory not found: {scene_dir_path}")
            
        # Preprocess scene if enabled
        processed_scene_dir = self._preprocess_if_needed(scene_dir_path)
        
        image_dir = processed_scene_dir / 'images'
        if not image_dir.exists():
            raise ValueError(f"Images directory not found: {image_dir}")
        
        # Create log file in processed scene directory
        log_path = processed_scene_dir / 'precompute.log'
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Starting precompute pipeline for scene: {processed_scene_dir}")
        if processed_scene_dir != scene_dir_path:
            logging.info(f"Original scene: {scene_dir_path}")
        logging.info(f"Configuration: {json.dumps(self.config, indent=2)}")


        # Extract features
        results = {}
        
        # Extract CoTracker tracks
        if 'cotracker' in self.extractors:
            logging.info("Extracting CoTracker tracks...")
            try:
                # Get bidirectional setting from config (default to True for better tracking)
                cotracker_config = self.config.get('cotracker', {})
                bidirectional = cotracker_config.get('bidirectional', True)
                
                track_data = self.extractors['cotracker'].extract_tracks(
                    image_dir, 
                    bidirectional=bidirectional
                )
                results['cotracker'] = {
                    'status': 'success',
                    'num_windows': len(track_data['tracks']),
                    'metadata': track_data['metadata'],
                    'track_data': track_data  # Store for visualization
                }
                
                # Check visualization settings
                viz_config = self.config.get('visualization', {})
                if viz_config.get('enabled', False):
                    logging.info("Creating visualization...")
                    # Pass visualization parameters
                    # Note: point_size and trail_length would need to be added to visualize_tracks
                    # For now, using default values in the visualization method
                    self.extractors['cotracker'].visualize_tracks(
                        image_dir, 
                        track_data=track_data,
                        save_video=viz_config.get('save_video', True),
                        save_frames=viz_config.get('save_frames', False),
                        fps=viz_config.get('fps', 10)
                    )
                    
            except Exception as e:
                logging.error(f"CoTracker extraction failed: {e}")
                results['cotracker'] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Extract depth maps
        if 'depth' in self.extractors:
            logging.info("Extracting depth maps with GeometryCrafter...")
            try:
                # Get total number of frames
                image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
                total_frames = len(image_paths)
                
                # Get segment size from config
                depth_config = self.config.get('depth', {})
                segment_size = depth_config.get('segment_size', 1000)
                
                logging.info(f"Total frames: {total_frames}, processing in segments of {segment_size}")
                
                # Process in segments
                all_segment_results = []
                frame_idx = 0
                
                while frame_idx < total_frames:
                    segment_end = min(frame_idx + segment_size, total_frames)
                    logging.info(f"Processing frames {frame_idx} to {segment_end-1}...")
                    
                    try:
                        segment_results = self.extractors['depth'].extract_depth(
                            image_dir, 
                            frame_start=frame_idx,
                            frame_end=segment_end
                        )
                        all_segment_results.append(segment_results)
                    except Exception as e:
                        logging.error(f"Segment {frame_idx}-{segment_end} failed: {e}")
                        raise
                    
                    frame_idx = segment_end
                
                # Combine results
                total_processed = sum(r['num_frames'] for r in all_segment_results)
                
                # Create consolidated metadata file
                if all_segment_results:
                    output_dir = all_segment_results[0]['depth_dir']
                    consolidated_metadata = {
                        'num_frames': total_processed,
                        'total_frames': total_frames,
                        'num_segments': len(all_segment_results),
                        'segment_size': segment_size,
                        'model': all_segment_results[0]['metadata']['model'],
                        'model_type': all_segment_results[0]['metadata']['model_type'],
                        'window_size': all_segment_results[0]['metadata']['window_size'],
                        'overlap': all_segment_results[0]['metadata']['overlap'],
                        'height': all_segment_results[0]['metadata']['height'],
                        'width': all_segment_results[0]['metadata']['width'],
                        'output_format': all_segment_results[0]['metadata']['output_format'],
                        'downsample_ratio': all_segment_results[0]['metadata']['downsample_ratio'],
                        'segments': [{
                            'start': r['metadata']['segment_start'],
                            'end': r['metadata']['segment_end'],
                            'frames': r['num_frames']
                        } for r in all_segment_results]
                    }
                    
                    # Save consolidated metadata
                    with open(output_dir.parent / 'depth_metadata.json', 'w') as f:
                        json.dump(consolidated_metadata, f, indent=2)
                    logging.info("Saved consolidated depth metadata")
                
                results['depth'] = {
                    'status': 'success',
                    'num_frames': total_processed,
                    'total_frames': total_frames,
                    'num_segments': len(all_segment_results),
                    'segment_size': segment_size,
                    'output_dir': str(all_segment_results[0]['depth_dir']) if all_segment_results else None,
                    'metadata': {
                        'model': all_segment_results[0]['metadata']['model'] if all_segment_results else 'GeometryCrafter',
                        'segments': [{
                            'start': r['metadata']['segment_start'],
                            'end': r['metadata']['segment_end'],
                            'frames': r['num_frames']
                        } for r in all_segment_results]
                    }
                }
                logging.info(f"Depth extraction complete: {total_processed} frames processed in {len(all_segment_results)} segments")
            except Exception as e:
                logging.error(f"Depth extraction failed: {e}")
                results['depth'] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Future: Extract optical flow
        
        # Save results summary
        summary_path = processed_scene_dir / 'precompute_summary.json'
        # Don't save track_data to summary (too large)
        summary_results = {}
        for feature, result in results.items():
            summary_results[feature] = {k: v for k, v in result.items() if k != 'track_data'}
        
        with open(summary_path, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        logging.info(f"Precompute pipeline complete! Summary saved to {summary_path}")
        return results


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Precompute features for 3D reconstruction')
    parser.add_argument('scene_dir', type=str, help='Path to scene directory')
    parser.add_argument('--config', type=str, 
                       default='/hdd2/0321_block_drone_video/colmap/config/precompute.yaml',
                       help='Path to config file (default: config/precompute.yaml)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization (overrides config file setting)')
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        sys.exit(1)
    
    # Override visualization settings from command line if specified
    if args.no_visualize:
        if 'visualization' not in config:
            config['visualization'] = {}
        config['visualization']['enabled'] = False
    
    # Setup logging
    log_level = config.get('log_level', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Run pipeline
    try:
        pipeline = PrecomputePipeline(config)
        results = pipeline.run(args.scene_dir)
        
        # Print summary
        print("\nPrecompute Summary:")
        print("-" * 50)
        for feature, result in results.items():
            print(f"{feature}: {result['status']}")
            if result['status'] == 'success' and 'metadata' in result:
                metadata = result['metadata']
                print(f"  Total frames: {metadata.get('total_frames', 'N/A')}")
                print(f"  Number of windows: {metadata.get('num_windows', 'N/A')}")
                if feature == 'cotracker':
                    print(f"  Window size: {metadata.get('window_size', 'N/A')}")
                    print(f"  Interval: {metadata.get('interval', 'N/A')}")
                    print(f"  Method: {metadata.get('initialization_method', 'N/A')}")
            elif result['status'] == 'failed':
                print(f"  Error: {result['error']}")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()