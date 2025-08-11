import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
from PIL import Image

# Add paths
sys.path.append('/hdd2/0321_block_drone_video/colmap/submodules/sam2')
sys.path.insert(0, '/hdd2/0321_block_drone_video/colmap/submodules/DeepLSD')

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# DeepLSD imports (not used, using HoughLines instead)
# from deeplsd.utils.tensor import batch_to_device
# from deeplsd.models.deeplsd_inference import DeepLSD
# from deeplsd.geometry.viz_2d import plot_images, plot_lines

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enable autocast for CUDA (will disable for DeepLSD)
autocast_enabled = False
if device.type == "cuda":
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def generate_half_ellipse_points(roi_center, roi_width, roi_height, num_points=3, left_side=True):
    """Generate point prompts for half ellipse segmentation (exact copy from sam2_roi_half_ellipse.py)"""
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
    
    # ROI coordinates
    x1, y1, x2, y2 = 2147, 282, 2280, 516
    roi = image[y1:y2, x1:x2]
    roi_height, roi_width = roi.shape[:2]
    
    # Load SAM2 model
    sam2_checkpoint = "/hdd2/0321_block_drone_video/colmap/submodules/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    print("Loading SAM2 model...")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Set ROI image and get mask
    predictor.set_image(roi)
    roi_center_in_roi = (roi_width // 2, roi_height // 2)
    input_points, input_labels = generate_half_ellipse_points(
        roi_center_in_roi, roi_width, roi_height, num_points=3, left_side=True
    )
    
    print("Running SAM2 prediction...")
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
    else:
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
    
    # Use the first mask (best one)
    mask1 = masks[0].astype(np.uint8)
    print(f"Generated mask with score: {scores[0]:.3f}")
    
    # Apply mask to ROI for line detection
    roi_masked = roi.copy()
    roi_masked[mask1 == 0] = 0  # Black out non-mask areas
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask1 * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze the mask to understand what we're getting
    print(f"\nMask shape: {mask1.shape}, unique values: {np.unique(mask1)}")
    print(f"Number of white pixels: {np.sum(mask1 > 0)}")
    
    if contours:
        print(f"Found {len(contours)} contours")
        
        # Sort contours by area and analyze top ones
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for i, cnt in enumerate(sorted_contours[:3]):
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            print(f"  Contour {i}: area={area:.1f}, perimeter={perimeter:.1f}, points={len(cnt)}")
        
        # Use the largest contour
        largest_contour = sorted_contours[0]
        
        # Try to approximate the contour to find straight segments
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        print(f"Approximated contour has {len(approx)} vertices")
        
        # Find the longest edge in the approximated contour
        pred_lines = []
        for i in range(len(approx)):
            pt1 = approx[i][0]
            pt2 = approx[(i + 1) % len(approx)][0]
            length = np.linalg.norm(pt2 - pt1)
            if length > 20:  # Minimum length threshold
                pred_lines.append(np.array([pt1, pt2], dtype=np.float32))
                print(f"  Edge {i}: length={length:.1f}")
        
        # Also try standard HoughLines
        mask_edges = cv2.Canny(mask1 * 255, 50, 150)
        lines = cv2.HoughLinesP(mask_edges, 1, np.pi/180, threshold=20, minLineLength=15, maxLineGap=20)
        
        if lines is not None:
            print(f"\nHoughLines detected {len(lines)} lines")
            for line in lines[:5]:  # Show first 5
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                print(f"  Line: ({x1},{y1})-({x2},{y2}), length={length:.1f}")
                pred_lines.append(np.array([[x1, y1], [x2, y2]], dtype=np.float32))
        
        pred_lines = np.array(pred_lines) if pred_lines else None
    else:
        print("No contours found in mask")
        pred_lines = None
    
    # Save edge image for debugging
    edge_debug_output = os.path.join(os.path.dirname(__file__), 'sam2_mask_edges.png')
    cv2.imwrite(edge_debug_output, mask_edges if 'mask_edges' in locals() else np.zeros_like(mask1))
    print(f"Edge image saved to: {edge_debug_output}")
    
    # Skip DeepLSD since we're using HoughLines
    os.chdir(original_dir)  # Change back to original directory
    
    if pred_lines is not None and len(pred_lines) > 0:
        print(f"\nDetected {len(pred_lines)} lines in masked region")
        
        # All lines are valid since we're detecting on masked image
        valid_lines = pred_lines
        
        # Print line information
        for i, line in enumerate(pred_lines[:10]):  # Show first 10 lines
            line_length = np.linalg.norm(line[1] - line[0])
            print(f"  Line {i}: length={line_length:.1f} pixels")
        
        if len(valid_lines) > 0:
            # Find the longest line
            longest_line = None
            max_length = 0
            
            for line in valid_lines:
                length = np.linalg.norm(line[1] - line[0])
                if length > max_length:
                    max_length = length
                    longest_line = line
            
            print(f"Longest line length: {max_length:.1f} pixels")
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            # 1. Original ROI
            axes[0].imshow(roi)
            axes[0].set_title('Original ROI')
            axes[0].axis('off')
            
            # 2. SAM2 Mask
            axes[1].imshow(mask1, cmap='gray')
            axes[1].set_title(f'SAM2 Mask (Score: {scores[0]:.3f})')
            axes[1].axis('off')
            
            # 3. Masked ROI
            axes[2].imshow(roi_masked)
            axes[2].set_title('Masked ROI')
            axes[2].axis('off')
            
            # 4. All valid lines
            roi_all_lines = roi.copy()
            for line in valid_lines:
                pt1 = tuple(line[0].astype(int))
                pt2 = tuple(line[1].astype(int))
                cv2.line(roi_all_lines, pt1, pt2, (0, 255, 0), 1)
            axes[3].imshow(roi_all_lines)
            axes[3].set_title(f'All Valid Lines ({len(valid_lines)})')
            axes[3].axis('off')
            
            # 5. Longest line only
            roi_longest = roi.copy()
            if longest_line is not None:
                pt1 = tuple(longest_line[0].astype(int))
                pt2 = tuple(longest_line[1].astype(int))
                cv2.line(roi_longest, pt1, pt2, (255, 0, 0), 2)
                # Add endpoints
                cv2.circle(roi_longest, pt1, 5, (0, 255, 0), -1)
                cv2.circle(roi_longest, pt2, 5, (0, 0, 255), -1)
            axes[4].imshow(roi_longest)
            axes[4].set_title(f'Longest Line ({max_length:.1f} px)')
            axes[4].axis('off')
            
            # 6. Full image with longest line
            full_image_result = image.copy()
            if longest_line is not None:
                # Convert to full image coordinates
                pt1_full = (int(longest_line[0][0] + x1), int(longest_line[0][1] + y1))
                pt2_full = (int(longest_line[1][0] + x1), int(longest_line[1][1] + y1))
                cv2.line(full_image_result, pt1_full, pt2_full, (255, 0, 0), 2)
                # Draw ROI box
                cv2.rectangle(full_image_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            axes[5].imshow(full_image_result)
            axes[5].set_title('Full Image with Longest Line')
            axes[5].axis('off')
            
            plt.tight_layout()
            output_path = os.path.join(os.path.dirname(__file__), 'sam2_mask_deeplsd_result.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nResults saved to: {output_path}")
            
            # Save individual result with longest line
            individual_output = os.path.join(os.path.dirname(__file__), 'sam2_mask_longest_line.png')
            cv2.imwrite(individual_output, cv2.cvtColor(roi_longest, cv2.COLOR_RGB2BGR))
            print(f"Longest line result saved to: {individual_output}")
            
            plt.show()
        else:
            print("No valid lines found within the mask")
    else:
        print("No lines detected")
        
    # Always save mask visualization for debugging
    debug_output = os.path.join(os.path.dirname(__file__), 'sam2_mask_debug.png')
    fig_debug, axes_debug = plt.subplots(1, 3, figsize=(15, 5))
    
    axes_debug[0].imshow(roi)
    axes_debug[0].set_title('ROI')
    axes_debug[0].axis('off')
    
    axes_debug[1].imshow(mask1, cmap='gray')
    axes_debug[1].set_title(f'Mask (Score: {scores[0]:.3f})')
    axes_debug[1].axis('off')
    
    axes_debug[2].imshow(roi_masked)
    axes_debug[2].set_title('Masked ROI')
    axes_debug[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(debug_output, dpi=150, bbox_inches='tight')
    print(f"Debug visualization saved to: {debug_output}")
    plt.close()
    
    # Save additional debug image with all detected lines
    if pred_lines is not None and len(pred_lines) > 0:
        roi_all_lines = roi.copy()
        roi_with_mask_overlay = roi.copy()
        
        # Overlay mask in red
        mask_overlay = np.zeros_like(roi)
        mask_overlay[:, :, 0] = mask1 * 255  # Red channel
        roi_with_mask_overlay = cv2.addWeighted(roi_with_mask_overlay, 0.7, mask_overlay, 0.3, 0)
        
        # Draw all lines
        for i, line in enumerate(pred_lines):
            pt1 = tuple(line[0].astype(int))
            pt2 = tuple(line[1].astype(int))
            cv2.line(roi_with_mask_overlay, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(roi_with_mask_overlay, str(i), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        lines_debug_output = os.path.join(os.path.dirname(__file__), 'sam2_mask_lines_debug.png')
        cv2.imwrite(lines_debug_output, cv2.cvtColor(roi_with_mask_overlay, cv2.COLOR_RGB2BGR))
        print(f"Lines debug visualization saved to: {lines_debug_output}")

if __name__ == "__main__":
    main()