"""Camera Model Module

This module handles camera intrinsic parameters and distortion coefficients.
"""

import numpy as np
import os
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CameraModel:
    """Manages camera intrinsic parameters and distortion models"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize Camera Model
        
        Args:
            config_dir: Directory containing camera configuration files
                       If None, uses default config/intrinsic directory
        """
        if config_dir is None:
            # Default to project config directory
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config" / "intrinsic"
        
        self.config_dir = Path(config_dir)
        self.camera_params = {}
        
        # Load available camera configurations
        self._load_camera_configs()
        
    def _load_camera_configs(self):
        """Load all available camera configurations"""
        if not self.config_dir.exists():
            logger.warning(f"Config directory {self.config_dir} does not exist")
            return
        
        # Find all subdirectories with camera configs
        for cam_dir in self.config_dir.iterdir():
            if cam_dir.is_dir():
                k_file = cam_dir / "K.txt"
                dist_file = cam_dir / "dist.txt"
                
                if k_file.exists() and dist_file.exists():
                    cam_name = cam_dir.name
                    try:
                        K = self._load_intrinsic_matrix(k_file)
                        dist = self._load_distortion_coeffs(dist_file)
                        
                        self.camera_params[cam_name] = {
                            'K': K,
                            'dist': dist,
                            'fx': K[0, 0],
                            'fy': K[1, 1],
                            'cx': K[0, 2],
                            'cy': K[1, 2],
                            'width': None,  # Will be set when processing images
                            'height': None
                        }
                        
                        logger.info(f"Loaded camera parameters for '{cam_name}'")
                        logger.debug(f"  fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, "
                                   f"cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
                        logger.debug(f"  distortion: {dist}")
                    except Exception as e:
                        logger.error(f"Failed to load camera params for {cam_name}: {e}")
    
    def _load_intrinsic_matrix(self, filepath: Path) -> np.ndarray:
        """Load camera intrinsic matrix from file
        
        Args:
            filepath: Path to K.txt file
            
        Returns:
            3x3 intrinsic matrix
        """
        K = np.zeros((3, 3))
        with open(filepath, 'r') as f:
            lines = f.readlines()
            # Skip first line (header) and parse matrix
            for i in range(3):
                # Each line has format: "line_number value1 value2 value3"
                values = lines[i+1].strip().split()[1:]  # Skip line number
                K[i] = [float(v) for v in values]
        return K
    
    def _load_distortion_coeffs(self, filepath: Path) -> np.ndarray:
        """Load distortion coefficients from file
        
        Args:
            filepath: Path to dist.txt file
            
        Returns:
            Array of distortion coefficients [k1, k2, p1, p2, k3]
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
            # Second line contains the coefficients (first line is header)
            values = lines[1].strip().split()[1:]  # Skip line number
            dist = np.array([float(v) for v in values])
        return dist
    
    def get_camera_params(self, camera_name: str) -> Dict:
        """Get camera parameters by name
        
        Args:
            camera_name: Name of camera configuration (e.g., 'x3', 'x7')
            
        Returns:
            Dictionary with camera parameters
        """
        if camera_name not in self.camera_params:
            raise ValueError(f"Camera '{camera_name}' not found. "
                           f"Available cameras: {list(self.camera_params.keys())}")
        return self.camera_params[camera_name]
    
    def set_image_size(self, camera_name: str, width: int, height: int):
        """Set image dimensions for a camera
        
        Args:
            camera_name: Name of camera configuration
            width: Image width in pixels
            height: Image height in pixels
        """
        if camera_name in self.camera_params:
            self.camera_params[camera_name]['width'] = width
            self.camera_params[camera_name]['height'] = height
            logger.debug(f"Set image size for {camera_name}: {width}x{height}")
    
    def get_colmap_camera_model(self, camera_name: str) -> Dict:
        """Get camera parameters in COLMAP format
        
        Args:
            camera_name: Name of camera configuration
            
        Returns:
            Dictionary with COLMAP camera model parameters
        """
        params = self.get_camera_params(camera_name)
        
        # COLMAP SIMPLE_RADIAL model uses: f, cx, cy, k
        # where f is the average of fx and fy, and k is the first radial distortion
        f = (params['fx'] + params['fy']) / 2.0
        
        colmap_params = {
            'model': 'SIMPLE_RADIAL',
            'width': params.get('width', 0),
            'height': params.get('height', 0),
            'params': [f, params['cx'], params['cy'], params['dist'][0]]
        }
        
        return colmap_params
    
    def get_gtsam_calibration(self, camera_name: str):
        """Get camera calibration for GTSAM
        
        Args:
            camera_name: Name of camera configuration
            
        Returns:
            GTSAM calibration object (requires gtsam to be installed)
        """
        try:
            import gtsam
        except ImportError:
            raise ImportError("GTSAM is required for this function. "
                            "Install with: pip install gtsam")
        
        params = self.get_camera_params(camera_name)
        
        # Create GTSAM calibration
        # Using Cal3_S2 (simple calibration with single focal length and no distortion)
        # For more accurate results, could use Cal3DS2_Base with distortion
        fx = params['fx']
        fy = params['fy']
        cx = params['cx']
        cy = params['cy']
        
        # For simplicity, use average focal length
        f = (fx + fy) / 2.0
        
        # Create calibration object
        calibration = gtsam.Cal3_S2(f, f, 0.0, cx, cy)
        
        return calibration
    
    def undistort_points(self, points: np.ndarray, camera_name: str) -> np.ndarray:
        """Undistort image points
        
        Args:
            points: Array of shape (N, 2) with distorted points
            camera_name: Name of camera configuration
            
        Returns:
            Array of undistorted points
        """
        import cv2
        
        params = self.get_camera_params(camera_name)
        K = params['K']
        dist = params['dist']
        
        # Convert to format expected by OpenCV
        points = points.reshape(-1, 1, 2).astype(np.float32)
        
        # Undistort points
        undistorted = cv2.undistortPoints(points, K, dist, P=K)
        
        return undistorted.reshape(-1, 2)
    
    def project_points(self, points_3d: np.ndarray, 
                      camera_pose: np.ndarray,
                      camera_name: str) -> np.ndarray:
        """Project 3D points to image plane
        
        Args:
            points_3d: Array of shape (N, 3) with 3D points
            camera_pose: 4x4 camera pose matrix (world to camera)
            camera_name: Name of camera configuration
            
        Returns:
            Array of shape (N, 2) with projected 2D points
        """
        params = self.get_camera_params(camera_name)
        K = params['K']
        
        # Transform points to camera coordinate system
        points_3d_hom = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        points_cam = (camera_pose @ points_3d_hom.T).T[:, :3]
        
        # Project to image plane
        points_2d_hom = (K @ points_cam.T).T
        points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]
        
        return points_2d
    
    def save_colmap_cameras_txt(self, output_path: str, camera_name: str):
        """Save camera parameters in COLMAP cameras.txt format
        
        Args:
            output_path: Path to save cameras.txt
            camera_name: Name of camera configuration
        """
        colmap_params = self.get_colmap_camera_model(camera_name)
        
        with open(output_path, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write("# Number of cameras: 1\n")
            
            camera_id = 1
            model = colmap_params['model']
            width = colmap_params['width']
            height = colmap_params['height']
            params_str = ' '.join(map(str, colmap_params['params']))
            
            f.write(f"{camera_id} {model} {width} {height} {params_str}\n")
        
        logger.info(f"Saved COLMAP cameras to {output_path}")