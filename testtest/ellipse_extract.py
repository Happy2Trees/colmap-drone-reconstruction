import matplotlib.pyplot as plt
import numpy as np
from skimage import color, img_as_ubyte, io
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import os

# Load the image
image_path = os.path.join(os.path.dirname(__file__), 'sample1.png')
image_rgb = io.imread(image_path)

# ROI coordinates (x1, y1, x2, y2)
x1, y1, x2, y2 = 2147, 282, 2280, 516

# Crop the ROI
roi = image_rgb[y1:y2, x1:x2]

# Convert ROI to grayscale
roi_gray = color.rgb2gray(roi)

# Detect edges in ROI
# Lower thresholds to detect more edges
edges = canny(roi_gray, sigma=1.5, low_threshold=0.3, high_threshold=0.6)

# Perform Hough Transform
# Adjust parameters based on expected ellipse size in ROI
# Lower accuracy for more robust detection, lower threshold for more candidates
result = hough_ellipse(edges, accuracy=10, threshold=30, min_size=30, max_size=120)

if len(result) > 0:
    # Sort by accumulator value and get the best ellipse
    result.sort(order='accumulator')
    best = list(result[-1])
    
    # Extract ellipse parameters (coordinates are relative to ROI)
    yc_roi, xc_roi, a, b = (int(round(x)) for x in best[1:5])
    orientation = best[5]
    
    # Convert ROI coordinates to original image coordinates
    yc = yc_roi + y1
    xc = xc_roi + x1
    
    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    
    # Filter out points outside image bounds
    mask = (cy >= 0) & (cy < image_rgb.shape[0]) & (cx >= 0) & (cx < image_rgb.shape[1])
    cy, cx = cy[mask], cx[mask]
    
    # Draw in red on full image
    image_rgb[cy, cx] = (255, 0, 0)
    
    # Also draw ellipse on cropped ROI for visualization
    roi_with_ellipse = roi.copy()
    cy_roi, cx_roi = ellipse_perimeter(yc_roi, xc_roi, a, b, orientation)
    
    # Filter ROI ellipse points
    mask_roi = (cy_roi >= 0) & (cy_roi < roi.shape[0]) & (cx_roi >= 0) & (cx_roi < roi.shape[1])
    cy_roi, cx_roi = cy_roi[mask_roi], cx_roi[mask_roi]
    
    # Draw in red on ROI
    roi_with_ellipse[cy_roi, cx_roi] = (255, 0, 0)
    
    # Save the ROI with ellipse
    roi_output_path = os.path.join(os.path.dirname(__file__), 'result_roi_ellipse.png')
    io.imsave(roi_output_path, roi_with_ellipse)
    print(f"ROI with ellipse saved to: {roi_output_path}")
    
    print(f"Ellipse found at center ({xc}, {yc}) with axes a={a}, b={b}, orientation={orientation:.2f}")
else:
    print("No ellipse found in the ROI")

# Save the result image
output_path = os.path.join(os.path.dirname(__file__), 'result_ellipse.png')
io.imsave(output_path, image_rgb)
print(f"Result saved to: {output_path}")

# Display the results
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(12, 4))

ax1.set_title('Original Image with ROI')
ax1.imshow(image_rgb)
# Draw ROI rectangle
rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='green', linewidth=2)
ax1.add_patch(rect)

ax2.set_title('ROI Edges')
ax2.imshow(edges, cmap='gray')

ax3.set_title('Result with Ellipse')
ax3.imshow(image_rgb)

plt.tight_layout()
plt.show()