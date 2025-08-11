import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch

# Add DeepLSD to Python path
sys.path.insert(0, '/hdd2/0321_block_drone_video/colmap/submodules/DeepLSD')

# DeepLSD imports
from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd_inference import DeepLSD
from deeplsd.geometry.viz_2d import plot_images, plot_lines

# Load the image
image_path = os.path.join(os.path.dirname(__file__), 'sample1.png')
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ROI coordinates (x1, y1, x2, y2)
x1, y1, x2, y2 = 2147, 282, 2280, 516

# Crop the ROI
roi = image_rgb[y1:y2, x1:x2]
roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

# Model config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf = {
    'detect_lines': True,  # Whether to detect lines or only DF/AF
    'line_detection_params': {
        'merge': True,  # Merge close-by lines to get longer segments
        'filtering': True,  # Normal filtering
        'grad_thresh': 3,  # Lower threshold for more lines
        'grad_nfa': False,  # Don't use NFA for challenging images
    }
}

# Load the model
ckpt_path = '/hdd2/0321_block_drone_video/colmap/submodules/DeepLSD/weights/deeplsd_md.tar'
if not os.path.exists(ckpt_path):
    print(f"Warning: Model checkpoint not found at {ckpt_path}")
    print("Please download the model weights from DeepLSD repository")
    exit(1)

ckpt = torch.load(str(ckpt_path), map_location='cpu')
net = DeepLSD(conf)
net.load_state_dict(ckpt['model'])
net = net.to(device).eval()

# Detect lines in ROI
inputs = {'image': torch.tensor(roi_gray, dtype=torch.float, device=device)[None, None] / 255.}
with torch.no_grad():
    out = net(inputs)
    pred_lines = out['lines'][0]

# Initialize variables
filtered_lines = []
filtered_lines_full = []

# Convert lines from ROI coordinates to full image coordinates
if pred_lines is not None and len(pred_lines) > 0:
    # Lines are in format [[x1, y1], [x2, y2]]
    # Add offset to map back to full image
    pred_lines_full = pred_lines.copy()
    pred_lines_full[:, :, 0] += x1  # Add x offset
    pred_lines_full[:, :, 1] += y1  # Add y offset
    
    print(f"\n Detected {len(pred_lines)} lines in ROI")
    print("Line endpoints (in full image coordinates):")
    for idx, line in enumerate(pred_lines_full[:5]):  # Show first 5 lines
        start = line[0]
        end = line[1]
        print(f"  Line {idx+1}: ({start[0]:.1f}, {start[1]:.1f}) -> ({end[0]:.1f}, {end[1]:.1f})")
    
    # Filter lines by minimum length
    min_length = 25  # Minimum line length in pixels (only long lines)
    filtered_lines = []
    filtered_lines_full = []
    
    for i, line in enumerate(pred_lines):
        length = np.linalg.norm(line[1] - line[0])
        if length >= min_length:
            filtered_lines.append(line)
            filtered_lines_full.append(pred_lines_full[i])
    
    print(f"  Filtered to {len(filtered_lines)} lines (min length: {min_length} pixels)")
    
    # Draw lines on full image with thinner lines
    result_img = image_rgb.copy()
    for line in filtered_lines_full:
        start_point = tuple(line[0].astype(int))
        end_point = tuple(line[1].astype(int))
        cv2.line(result_img, start_point, end_point, (255, 0, 0), 1)  # Thinner line (1 pixel)
    
    # Draw lines on ROI for visualization
    roi_with_lines = roi.copy()
    for line in filtered_lines:
        start_point = tuple(line[0].astype(int))
        end_point = tuple(line[1].astype(int))
        cv2.line(roi_with_lines, start_point, end_point, (255, 0, 0), 1)  # Thinner line
    
    # Save ROI with lines
    roi_output_path = os.path.join(os.path.dirname(__file__), 'result_roi_lines.png')
    cv2.imwrite(roi_output_path, cv2.cvtColor(roi_with_lines, cv2.COLOR_RGB2BGR))
    print(f"ROI with lines saved to: {roi_output_path}")
else:
    print("No lines detected in the ROI")
    result_img = image_rgb.copy()
    roi_with_lines = roi.copy()

# Save the result image
output_path = os.path.join(os.path.dirname(__file__), 'result_lines.png')
cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
print(f"Result saved to: {output_path}")

# Display the results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original image with ROI
axes[0, 0].set_title('Original Image with ROI')
axes[0, 0].imshow(image_rgb)
rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='green', linewidth=2)
axes[0, 0].add_patch(rect)

# ROI grayscale
axes[0, 1].set_title('ROI Grayscale')
axes[0, 1].imshow(roi_gray, cmap='gray')

# ROI with detected lines
axes[1, 0].set_title('ROI with Detected Lines')
axes[1, 0].imshow(roi_with_lines)

# Full image with lines
axes[1, 1].set_title('Full Image with Lines')
axes[1, 1].imshow(result_img)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'line_detection_results.png'), dpi=150, bbox_inches='tight')
plt.close()

# Alternative visualization using DeepLSD's built-in plotting
if filtered_lines is not None and len(filtered_lines) > 0:
    fig2 = plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    
    # Plot ROI with DeepLSD visualization
    plt.subplot(1, 3, 2)
    plot_images([roi], ['ROI (Filtered Lines)'], cmaps='gray')
    plot_lines([np.array(filtered_lines)], indices=[0])
    
    # Plot full image with lines
    plt.subplot(1, 3, 3)
    plot_images([image_rgb], ['Full Image with Filtered Lines'], cmaps='gray')
    plot_lines([np.array(filtered_lines_full)], indices=[0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'line_detection_deeplsd_viz.png'), dpi=150, bbox_inches='tight')
    plt.close()