import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the image
image_path = os.path.join(os.path.dirname(__file__), 'sample1.png')
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ROI coordinates (x1, y1, x2, y2)
x1, y1, x2, y2 = 2147, 282, 2280, 516

# Crop the ROI
roi = image_rgb[y1:y2, x1:x2]
roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(roi_gray, (9, 9), 2)

# Apply HoughCircles with adjusted parameters - wide range for exploration
# Parameters: image, method, dp, minDist, param1, param2, minRadius, maxRadius
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,              # Inverse ratio of accumulator resolution
    minDist=30,        # Minimum distance between detected centers
    param1=50,         # Upper threshold for Canny edge detector
    param2=20,         # Threshold for center detection (lower = more sensitive)
    minRadius=10,      # Minimum circle radius (very small)
    maxRadius=110      # Maximum circle radius (very large)
)

if circles is not None:
    # Convert circles to integer values
    circles = np.uint16(np.around(circles))
    
    # Find the circle with largest radius
    if len(circles[0]) > 0:
        # Sort circles by radius (descending)
        sorted_circles = circles[0][circles[0][:, 2].argsort()[::-1]]
        
        # Get the largest circle
        largest_circle = sorted_circles[0]
        
        # Circle center and radius in ROI coordinates
        center_x_roi, center_y_roi, radius = largest_circle[0], largest_circle[1], largest_circle[2]
        
        # Convert ROI coordinates to full image coordinates
        center_x = center_x_roi + x1
        center_y = center_y_roi + y1
        
        # Draw circle on full image (red color)
        cv2.circle(image_rgb, (center_x, center_y), radius, (255, 0, 0), 2)
        # Draw center point
        cv2.circle(image_rgb, (center_x, center_y), 2, (255, 0, 0), 3)
        
        # Also draw on ROI for visualization
        roi_with_circle = roi.copy()
        cv2.circle(roi_with_circle, (center_x_roi, center_y_roi), radius, (255, 0, 0), 2)
        cv2.circle(roi_with_circle, (center_x_roi, center_y_roi), 2, (255, 0, 0), 3)
        
        print(f"\nTotal circles detected: {len(circles[0])}")
        print("All detected circles (sorted by radius):")
        for idx, circle in enumerate(sorted_circles[:5]):  # Show top 5 circles
            cx, cy, r = circle[0] + x1, circle[1] + y1, circle[2]
            print(f"  {idx+1}. Center: ({cx}, {cy}), Radius: {r}")
        print(f"\n✓ Selected largest circle at center ({center_x}, {center_y}) with radius {radius}")
    
    # Save the ROI with circle
    roi_output_path = os.path.join(os.path.dirname(__file__), 'result_roi_circle.png')
    cv2.imwrite(roi_output_path, cv2.cvtColor(roi_with_circle, cv2.COLOR_RGB2BGR))
    print(f"ROI with circle saved to: {roi_output_path}")
else:
    print("No circles found in the ROI")

# Save the result image
output_path = os.path.join(os.path.dirname(__file__), 'result_circle.png')
cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
print(f"Result saved to: {output_path}")

# Display the results
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(12, 4))

ax1.set_title('Original Image with ROI')
ax1.imshow(image_rgb)
# Draw ROI rectangle
rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='green', linewidth=2)
ax1.add_patch(rect)

ax2.set_title('ROI Grayscale')
ax2.imshow(roi_gray, cmap='gray')

ax3.set_title('Result with Circle')
ax3.imshow(image_rgb)

plt.tight_layout()
plt.show()