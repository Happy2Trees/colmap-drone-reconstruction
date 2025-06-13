#!/usr/bin/env python3
"""
Main script to run window-based bundle adjustment following GeometryCrafter.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.window_ba.pipeline import WindowBAPipeline

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    parser = argparse.ArgumentParser(
        description="Window-based Bundle Adjustment following GeometryCrafter approach"
    )
    
    # Required arguments
    parser.add_argument(
        'scene_dir',
        type=Path,
        help='Path to scene directory containing images, tracks, depth, and camera parameters'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=None,
        help='Output directory for results (default: scene_dir/window_ba_output)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--use_refine',
        action='store_true',
        help='Enable Phase 2 optimization (camera + 3D refinement)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not args.scene_dir.exists():
        logger.error(f"Scene directory does not exist: {args.scene_dir}")
        sys.exit(1)
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = args.scene_dir / 'window_ba_output'
    
    # Log configuration
    logger.info("Window-based Bundle Adjustment")
    logger.info(f"Scene directory: {args.scene_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    if args.config:
        logger.info(f"Config file: {args.config}")
    logger.info(f"Use refinement: {args.use_refine}")
    
    try:
        # Create pipeline
        pipeline = WindowBAPipeline(config_path=args.config)
        
        # Run pipeline
        results = pipeline.run(
            scene_dir=args.scene_dir,
            output_dir=args.output_dir,
            use_refine=args.use_refine
        )
        
        # Print summary
        if results['success']:
            logger.info("Pipeline completed successfully!")
            if 'phase1' in results:
                logger.info(f"Phase 1 final loss: {results['phase1']['final_loss']:.6f}")
        else:
            logger.error("Pipeline failed!")
            if 'error' in results:
                logger.error(f"Error: {results['error']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()