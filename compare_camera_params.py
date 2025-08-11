import sys
import numpy as np
from src.colmap_utils import read_write_model

# Original camera parameters from the script
# Format: fx, fy, cx, cy, k1, k2, p1, p2
original_params = np.array([19872.643351, 19873.226698, 2123.585064, 1499.441091, 
                           -0.208057, -5.646531, 0.002627, 0.003169])

# Read optimized model
sparse_path = "/hdd2/0321_block_drone_video/colmap/outputs/workspaces/section2_7x_window_sift_48_12_30fps/sparse/0"
cameras, images, points3D = read_write_model.read_model(sparse_path, ext=".bin")

print("=" * 80)
print("Camera Parameter Comparison")
print("=" * 80)

# Get the camera (assuming single camera)
camera_id = list(cameras.keys())[0]
camera = cameras[camera_id]

print(f"\nCamera Model: {camera.model}")
print(f"Image Dimensions: {camera.width} x {camera.height}")

optimized_params = camera.params

# Parameter names for OPENCV model
param_names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]

print("\n{:<10} {:<20} {:<20} {:<15} {:<10}".format(
    "Parameter", "Original", "Optimized", "Difference", "% Change"))
print("-" * 80)

for i, name in enumerate(param_names):
    orig = original_params[i]
    opt = optimized_params[i]
    diff = opt - orig
    pct_change = (diff / orig) * 100 if orig != 0 else 0
    
    print("{:<10} {:<20.6f} {:<20.6f} {:<15.6f} {:<10.2f}%".format(
        name, orig, opt, diff, pct_change))

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)

# Focal lengths
fx_change = ((optimized_params[0] - original_params[0]) / original_params[0]) * 100
fy_change = ((optimized_params[1] - original_params[1]) / original_params[1]) * 100
print(f"Focal length changes: fx={fx_change:.2f}%, fy={fy_change:.2f}%")

# Principal point
cx_change = optimized_params[2] - original_params[2]
cy_change = optimized_params[3] - original_params[3]
print(f"Principal point shift: Δcx={cx_change:.2f} pixels, Δcy={cy_change:.2f} pixels")

# Distortion
k1_change = ((optimized_params[4] - original_params[4]) / abs(original_params[4])) * 100 if original_params[4] != 0 else 0
k2_change = ((optimized_params[5] - original_params[5]) / abs(original_params[5])) * 100 if original_params[5] != 0 else 0
print(f"Radial distortion changes: k1={k1_change:.2f}%, k2={k2_change:.2f}%")

# Total number of images and 3D points
print(f"\nReconstruction statistics:")
print(f"Number of images: {len(images)}")
print(f"Number of 3D points: {len(points3D)}")

# Additional analysis
print(f"\nAspect ratio analysis:")
print(f"Original fx/fy ratio: {original_params[0]/original_params[1]:.6f}")
print(f"Optimized fx/fy ratio: {optimized_params[0]/optimized_params[1]:.6f}")
print(f"Image aspect ratio: {camera.width/camera.height:.6f}")

# Check if parameters look reasonable
print(f"\n⚠️  Warning: fy changed dramatically from {original_params[1]:.2f} to {optimized_params[1]:.2f}")
print(f"This suggests the camera parameters may not have been properly frozen during optimization.")