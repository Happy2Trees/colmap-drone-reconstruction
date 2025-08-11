import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Add SAM2 to path
sys.path.append('/hdd2/0321_block_drone_video/colmap/submodules/sam2')

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Set up device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Enable autocast for CUDA
if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# Visualization functions
def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', 
                               facecolor=(0, 0, 0, 0), lw=2))

def generate_half_ellipse_points(roi_center, roi_width, roi_height, num_points=3, left_side=True):
    """
    Generate point prompts for a half ellipse (semicircle) within the ROI.
    Includes both positive and negative prompts.
    
    Args:
        roi_center: (x, y) center of the ROI
        roi_width: width of the ROI
        roi_height: height of the ROI
        num_points: number of total positive points to generate
        left_side: if True, generate points for left half; if False, for right half
    
    Returns:
        points: numpy array of shape (num_points + 2, 2) with point coordinates
        labels: numpy array of shape (num_points + 2,) with 1s (foreground) and 0s (background)
    """
    cx, cy = roi_center
    
    points = []
    labels = []
    
    # Start with center point (positive)
    points.append([cx, cy])
    labels.append(1)
    
    # Generate positive points along the left half of an ellipse
    if left_side:
        # Focused positive points on left semicircle from 130° to 230° (well within bounds)
        positive_angles = [
            13*np.pi/18,  # 130°
            7*np.pi/9,    # 140°
            5*np.pi/6,    # 150°
            8*np.pi/9,    # 160°
            17*np.pi/18,  # 170°
            np.pi,        # 180°
            19*np.pi/18,  # 190°
            10*np.pi/9,   # 200°
            7*np.pi/6,    # 210°
            11*np.pi/9,   # 220°
            23*np.pi/18   # 230°
        ]
        # Negative points: top, bottom, and right side (0°, 45°, -45°)
        negative_angles = [np.pi/2, 3*np.pi/2, 0, np.pi/4, -np.pi/4]  # 90°, 270°, 0°, 45°, -45°
    else:
        # Right semicircle (not used in this case)
        positive_angles = [-np.pi/4, np.pi/4]  # -45° and 45°
        negative_angles = [np.pi/2, 3*np.pi/2]  # 90° and 270°
    
    # Different radius for positive and negative points
    # Moderate radius for positive points to cover the half ellipse
    positive_radius_x = roi_width * 0.20  # 20% of ROI width
    positive_radius_y = roi_height * 0.25  # 25% of ROI height
    
    # Larger radius for negative points at boundaries
    negative_radius_x = roi_width * 0.35  # 35% of ROI width
    negative_radius_y = roi_height * 0.4  # 40% of ROI height
    
    # Add positive points with consistent small radius
    for angle in positive_angles:
        x = cx + positive_radius_x * np.cos(angle)
        y = cy + positive_radius_y * np.sin(angle)
        points.append([x, y])
        labels.append(1)
    
    # Add vertical line of positive points on the left side
    # X coordinate is fixed at left edge of half ellipse
    left_x = cx - positive_radius_x  # At the left edge
    # Y coordinates distributed from top to bottom of half ellipse
    y_positions = np.linspace(cy - positive_radius_y * 0.7, cy + positive_radius_y * 0.7, 7)
    for y in y_positions:
        points.append([left_x, y])
        labels.append(1)
    
    # Add negative points at top and bottom
    for angle in negative_angles:
        x = cx + negative_radius_x * np.cos(angle)
        y = cy + negative_radius_y * np.sin(angle)
        points.append([x, y])
        labels.append(0)  # Negative prompt
    
    points = np.array(points)
    labels = np.array(labels, dtype=np.int32)
    
    return points, labels

def main():
    # Change to SAM2 directory for config loading
    original_dir = os.getcwd()
    os.chdir('/hdd2/0321_block_drone_video/colmap/submodules/sam2')
    
    # Load image
    image_path = '/hdd2/0321_block_drone_video/colmap/testtest/sample1.png'
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    
    # ROI coordinates from ellipse_extract.py
    x1, y1, x2, y2 = 2147, 282, 2280, 516
    
    # Crop ROI
    roi = image[y1:y2, x1:x2]
    roi_height, roi_width = roi.shape[:2]
    
    # Calculate ROI center in original image coordinates
    roi_center_x = (x1 + x2) // 2
    roi_center_y = (y1 + y2) // 2
    
    # Load SAM2 model
    sam2_checkpoint = "/hdd2/0321_block_drone_video/colmap/submodules/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    print("Loading SAM2 model...")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Set the cropped ROI image for SAM2
    predictor.set_image(roi)
    
    # Generate point prompts for left half-ellipse
    # Points are in ROI coordinates (relative to cropped image)
    roi_center_in_roi = (roi_width // 2, roi_height // 2)
    input_points, input_labels = generate_half_ellipse_points(
        roi_center_in_roi, 
        roi_width, 
        roi_height, 
        num_points=3,  # 3 total points: center + 2 middle points
        left_side=True
    )
    
    print(f"Generated {len(input_points)} point prompts:")
    for i, (pt, label) in enumerate(zip(input_points, input_labels)):
        prompt_type = "Positive" if label == 1 else "Negative"
        print(f"  Point {i}: ({pt[0]:.1f}, {pt[1]:.1f}), Type: {prompt_type}")
    
    # Predict masks
    print("Running SAM2 prediction...")
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,  # Get multiple masks to choose best
    )
    
    # Sort masks by score
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    
    print(f"Generated {len(masks)} masks with scores: {scores}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 1. Original image with ROI
    axes[0].imshow(image)
    axes[0].set_title('Original Image with ROI')
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                        edgecolor='green', linewidth=2)
    axes[0].add_patch(rect)
    axes[0].axis('off')
    
    # 2. ROI with point prompts
    axes[1].imshow(roi)
    axes[1].set_title('ROI with Point Prompts')
    # Points are already in ROI coordinates
    show_points(input_points, input_labels, axes[1], marker_size=200)
    axes[1].axis('off')
    
    # 3-5. Show top 3 masks on ROI
    for i in range(min(3, len(masks))):
        axes[i+2].imshow(roi)
        show_mask(masks[i], axes[i+2])
        show_points(input_points, input_labels, axes[i+2], marker_size=200)
        axes[i+2].set_title(f'Mask {i+1} (Score: {scores[i]:.3f})')
        axes[i+2].axis('off')
    
    # 6. Best mask on full image (composite)
    # Create full image with mask overlay in ROI region
    full_image_with_mask = image.copy()
    best_mask = masks[0]
    # Create a colored overlay for the mask
    mask_overlay = np.zeros_like(roi)
    mask_color = np.array([30, 144, 255])  # Blue color
    for c in range(3):
        mask_overlay[:, :, c] = best_mask * mask_color[c]
    
    # Blend mask with ROI region
    alpha = 0.5
    roi_with_mask = (1 - alpha) * roi + alpha * mask_overlay
    full_image_with_mask[y1:y2, x1:x2] = roi_with_mask.astype(np.uint8)
    
    axes[5].imshow(full_image_with_mask)
    axes[5].set_title('Best Mask on Full Image')
    axes[5].axis('off')
    
    plt.tight_layout()
    
    # Save results
    output_path = '/hdd2/0321_block_drone_video/colmap/testtest/sam2_half_ellipse_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to: {output_path}")
    
    # Save individual mask on ROI
    mask_output_path = '/hdd2/0321_block_drone_video/colmap/testtest/sam2_best_mask.png'
    plt.figure(figsize=(10, 10))
    plt.imshow(roi)
    show_mask(masks[0], plt.gca())
    show_points(input_points, input_labels, plt.gca())
    plt.axis('off')
    plt.savefig(mask_output_path, dpi=150, bbox_inches='tight')
    print(f"Best mask saved to: {mask_output_path}")
    
    plt.show()
    
    # Change back to original directory
    os.chdir(original_dir)

if __name__ == "__main__":
    main()