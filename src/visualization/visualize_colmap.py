import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

def read_cameras_txt(filepath):
    """
    Parses the cameras.txt file.
    Returns a dictionary mapping camera_id to camera parameters.
    """
    cameras = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

def read_images_txt(filepath):
    """
    Parses the images.txt file.
    Returns a list of dictionaries, each representing an image with its pose.
    """
    images = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith('#'):
                i += 1
                continue
            
            # Image pose line
            parts = lines[i].strip().split()
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]
            
            # Convert quaternion to rotation matrix
            # COLMAP quaternion order is W, X, Y, Z
            # Scipy Rotation expects X, Y, Z, W
            r = Rotation.from_quat([qx, qy, qz, qw])
            rotation_matrix = r.as_matrix()
            
            # Camera position (center of projection) is -R.T @ t
            # Or, more directly, COLMAP provides T_world_cam, so t_world_cam is the camera center
            # However, COLMAP's t is C_w = -R * T_cw, where T_cw is camera center in world. So t = -R * T_cw
            # T_cw = -R.T * t
            # For plotting, we often need the camera center C and the orientation R (world to camera)
            # The provided t is the translation vector of the camera in the world frame.
            # The camera center C = -R^T * t (from R, t which transforms world to camera)
            # COLMAP's R, t are such that X_cam = R * X_world + t
            # So, camera center in world coords C_w satisfies R * C_w + t = 0 => C_w = -R_transpose * t
            
            # However, a simpler interpretation from COLMAP docs:
            # "The transformation [R | t] is specified such that a point X in world coordinate system is transformed to x in camera coordinate system according to x = R * X + t."
            # Thus, the camera center in world coordinates is C = -R^T * t.
            # And the provided t is the translation part of the world-to-camera transformation.

            # For plotting camera axes, we need the rotation from camera to world (R_cw = R.T)
            # and the camera center in world (C_w = -R.T * t_wc).
            # The given R, t are R_wc, t_wc.
            # So, R_cw = rotation_matrix.T
            # Camera center C_w = -rotation_matrix.T @ np.array([tx, ty, tz])

            images.append({
                'id': image_id,
                'q': [qw, qx, qy, qz],
                't': np.array([tx, ty, tz]),
                'R_wc': rotation_matrix, # World to Camera rotation
                't_wc': np.array([tx, ty, tz]), # World to Camera translation
                'camera_id': camera_id,
                'name': name
            })
            i += 2 # Skip the POINTS2D line
    return images

def read_points3D_txt(filepath):
    """
    Parses the points3D.txt file.
    Returns a NumPy array of 3D points (X, Y, Z) and an array of colors (R, G, B).
    """
    points = []
    colors = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
            points.append([float(parts[1]), float(parts[2]), float(parts[3])])
            colors.append([int(parts[4]), int(parts[5]), int(parts[6])])
    return np.array(points), np.array(colors) / 255.0 # Normalize colors to [0,1]

def plot_colmap_data(images, points3D, colors3D, cameras_data=None, scale=0.5, output_path=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D points
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c=colors3D, s=1, marker='.')

    # Plot camera poses
    for image_info in images:
        R_wc = image_info['R_wc'] # Rotation: World to Camera
        t_wc = image_info['t_wc'] # Translation: World to Camera

        # Camera center in world coordinates C_w = -R_wc^T * t_wc
        cam_center_world = -R_wc.T @ t_wc
        
        # Rotation from camera to world R_cw = R_wc^T
        R_cw = R_wc.T

        # Plot camera center
        ax.scatter(cam_center_world[0], cam_center_world[1], cam_center_world[2], c='red', marker='o', s=30)

        # Plot camera axes (scaled)
        # X-axis (red)
        x_axis = R_cw @ np.array([scale, 0, 0])
        ax.plot([cam_center_world[0], cam_center_world[0] + x_axis[0]],
                [cam_center_world[1], cam_center_world[1] + x_axis[1]],
                [cam_center_world[2], cam_center_world[2] + x_axis[2]], 'r-')
        # Y-axis (green)
        y_axis = R_cw @ np.array([0, scale, 0])
        ax.plot([cam_center_world[0], cam_center_world[0] + y_axis[0]],
                [cam_center_world[1], cam_center_world[1] + y_axis[1]],
                [cam_center_world[2], cam_center_world[2] + y_axis[2]], 'g-')
        # Z-axis (blue) - principal axis
        z_axis = R_cw @ np.array([0, 0, scale])
        ax.plot([cam_center_world[0], cam_center_world[0] + z_axis[0]],
                [cam_center_world[1], cam_center_world[1] + z_axis[1]],
                [cam_center_world[2], cam_center_world[2] + z_axis[2]], 'b-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('COLMAP 3D Reconstruction')
    
    # Auto-scale axes to fit data
    all_points = np.vstack((points3D, np.array([(-R_wc.T @ img['t_wc']) for img in images for R_wc in [img['R_wc']]] )))
    
    if all_points.size > 0:
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        ax.auto_scale_xyz(min_coords, max_coords)

        # Try to make axes equal for a better 3D representation
        # Get ranges for each axis
        x_range = max_coords[0] - min_coords[0]
        y_range = max_coords[1] - min_coords[1]
        z_range = max_coords[2] - min_coords[2]
        max_range = max(x_range, y_range, z_range)

        mid_x = (max_coords[0] + min_coords[0]) * 0.5
        mid_y = (max_coords[1] + min_coords[1]) * 0.5
        mid_z = (max_coords[2] + min_coords[2]) * 0.5

        ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
        ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)

    else:
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])


    # plt.show()
    output_filename = output_path if output_path else 'colmap_visualization.png'
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Visualize COLMAP reconstruction results")
    parser.add_argument("sparse_path", help="Path to COLMAP sparse reconstruction directory")
    parser.add_argument("--output", default="outputs/visualizations/colmap_visualization.png", 
                        help="Output image path (default: outputs/visualizations/colmap_visualization.png)")
    
    args = parser.parse_args()
    
    base_path = args.sparse_path.rstrip('/') + '/'
    cameras_file = base_path + 'cameras.txt'
    images_file = base_path + 'images.txt'
    points3D_file = base_path + 'points3D.txt'

    print(f"Loading cameras from: {cameras_file}")
    cameras = read_cameras_txt(cameras_file)
    
    print(f"Loading images from: {images_file}")
    images_data = read_images_txt(images_file)
    
    print(f"Loading 3D points from: {points3D_file}")
    points3D_xyz, points3D_rgb = read_points3D_txt(points3D_file)

    print(f"Found {len(cameras)} camera(s).")
    print(f"Found {len(images_data)} image pose(s).")
    print(f"Found {points3D_xyz.shape[0]} 3D point(s).")

    if points3D_xyz.shape[0] == 0 and not images_data:
        print("No 3D points or camera poses to visualize.")
    else:
        plot_colmap_data(images_data, points3D_xyz, points3D_rgb, cameras_data=cameras, output_path=args.output)
