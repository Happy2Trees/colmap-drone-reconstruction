#!/usr/bin/env python3
"""
SIFT-based CoTracker sliding window test
Uses SIFT feature points instead of uniform grid for tracking
This should provide better tracking in textured regions and avoid textureless areas
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import imageio

# Add the submodules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../submodules/co-tracker'))

try:
    from cotracker.predictor import CoTrackerPredictor
    from cotracker.utils.visualizer import Visualizer
except ImportError:
    print("Error: CoTracker not found. Please ensure submodules are initialized:")
    print("  cd /hdd2/0321_block_drone_video/colmap")
    print("  git submodule update --init --recursive")
    sys.exit(1)


def get_image_paths(image_dir, start_idx=0, num_frames=100):
    """Get paths of images without loading them"""
    image_files = sorted(Path(image_dir).glob("*.jpg"))
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    # Select subset
    image_files = image_files[start_idx:start_idx + num_frames]
    return image_files


def load_images_for_window(image_paths, start_idx, end_idx):
    """Load only the images needed for current window"""
    window_paths = image_paths[start_idx:end_idx]
    images = []
    
    for img_path in window_paths:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        images.append(img_array)
    
    return np.stack(images)


def extract_sift_features(image, max_features=500, min_response=0.01):
    """Extract SIFT features from an image
    
    Args:
        image: Input image as numpy array (RGB)
        max_features: Maximum number of features to extract
        min_response: Minimum response threshold for filtering weak features
    
    Returns:
        keypoints: List of cv2.KeyPoint objects
        descriptors: SIFT descriptors
        points: Numpy array of (x, y) coordinates
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create SIFT detector
    try:
        sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=min_response)
    except AttributeError:
        # For older OpenCV versions
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=max_features, contrastThreshold=min_response)
    
    # Detect and compute
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Extract coordinates
    points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    
    print(f"  Extracted {len(keypoints)} SIFT features")
    
    return keypoints, descriptors, points


def filter_features_by_region(points, image_shape, grid_size=20):
    """Filter features to ensure good spatial distribution
    
    Divides image into grid cells and selects best features from each cell
    """
    h, w = image_shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    filtered_points = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Define cell boundaries
            y_min = i * cell_h
            y_max = (i + 1) * cell_h if i < grid_size - 1 else h
            x_min = j * cell_w
            x_max = (j + 1) * cell_w if j < grid_size - 1 else w
            
            # Find points in this cell
            mask = (points[:, 0] >= x_min) & (points[:, 0] < x_max) & \
                   (points[:, 1] >= y_min) & (points[:, 1] < y_max)
            cell_points = points[mask]
            
            # Select up to 2 points from this cell (best responses)
            if len(cell_points) > 0:
                # SIFT already orders by response, so take first ones
                filtered_points.extend(cell_points[:2])
    
    return np.array(filtered_points) if filtered_points else points[:100]  # Fallback


def get_sift_query_points(image, max_features=400, grid_filter=True):
    """Generate query points based on SIFT features
    
    Args:
        image: Input image as numpy array
        max_features: Maximum number of features to use
        grid_filter: Whether to filter features for spatial distribution
    
    Returns:
        Numpy array of query points in format [time, x, y]
    """
    # Extract SIFT features
    keypoints, descriptors, points = extract_sift_features(image, max_features=max_features*2)
    
    if len(points) == 0:
        print("  Warning: No SIFT features found, falling back to grid")
        # Fallback to grid
        return get_grid_points(image.shape, 20)
    
    # Filter for spatial distribution if requested
    if grid_filter and len(points) > max_features:
        points = filter_features_by_region(points, image.shape)
    
    # Limit to max_features
    if len(points) > max_features:
        points = points[:max_features]
    
    # Add time dimension (0 for first frame of window)
    queries = np.hstack([np.zeros((len(points), 1)), points])
    
    return queries


def get_grid_points(image_shape, grid_size):
    """Generate a grid of query points (fallback)"""
    h, w = image_shape[:2]
    
    # Create grid
    y_coords = np.linspace(h * 0.1, h * 0.9, grid_size)
    x_coords = np.linspace(w * 0.1, w * 0.9, grid_size)
    
    # Generate all combinations
    points = []
    for y in y_coords:
        for x in x_coords:
            points.append([0, x, y])  # [time, x, y]
    
    return np.array(points)


def run_sift_based_tracking(image_paths, window_size=24, overlap=12, device='cuda', max_features=400):
    """Run CoTracker with SIFT-based query points
    
    Args:
        image_paths: List of image file paths
        window_size: Number of frames per window
        overlap: Number of overlapping frames between windows
        device: Device to run on
        max_features: Maximum number of SIFT features per window
    """
    
    # Initialize model using torch.hub for automatic download
    print("Initializing CoTracker3 OFFLINE model...")
    print("Downloading model weights if needed (this may take a while on first run)...")
    
    # Use torch.hub to load the model with automatic download
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device)
    print("Model loaded successfully!")
    
    # Generate windows with overlap
    stride = window_size - overlap
    windows = []
    total_frames = len(image_paths)
    
    print(f"\nWindow configuration:")
    print(f"  Total frames: {total_frames}")
    print(f"  Window size: {window_size} frames")
    print(f"  Overlap: {overlap} frames ({overlap/window_size*100:.1f}%)")
    print(f"  Stride: {stride} frames")
    print(f"  Max SIFT features per window: {max_features}")
    
    for start in range(0, total_frames, stride):
        end = min(start + window_size, total_frames)
        windows.append((start, end))
        if end >= total_frames:
            break
    
    print(f"\nGenerated {len(windows)} windows:")
    for i, (start, end) in enumerate(windows):
        overlap_with_prev = 0 if i == 0 else windows[i-1][1] - start
        print(f"  Window {i+1}: frames [{start:3d}-{end:3d}), overlap with previous: {overlap_with_prev} frames")
    
    print(f"\nProcessing {len(windows)} windows with SIFT features...")
    
    # Process each window
    all_tracks = []
    
    for i, (start, end) in enumerate(windows):
        print(f"\nWindow {i+1}/{len(windows)}: frames {start}-{end-1}")
        
        # Load only window frames to save memory
        window_frames = load_images_for_window(image_paths, start, end)
        
        # Extract SIFT features from the first frame of the window
        print("  Extracting SIFT features from first frame...")
        first_frame = window_frames[0]
        queries = get_sift_query_points(first_frame, max_features=max_features, grid_filter=True)
        print(f"  Using {len(queries)} SIFT-based query points")
        
        # Convert to tensors
        video_tensor = torch.from_numpy(window_frames).permute(0, 3, 1, 2).float()
        video_tensor = video_tensor.unsqueeze(0).to(device)  # Add batch dimension
        queries_tensor = torch.from_numpy(queries).float().to(device)
        queries_tensor = queries_tensor.unsqueeze(0)  # [B=1, N, 3]
        
        # Run tracking
        with torch.no_grad():
            pred_tracks, pred_visibility = model(video_tensor, queries=queries_tensor)
        
        # Store results with SIFT feature info
        all_tracks.append({
            'tracks': pred_tracks.cpu(),
            'visibility': pred_visibility.cpu(),
            'start_frame': start,
            'end_frame': end,
            'sift_points': queries[:, 1:],  # Store original SIFT points (x, y)
            'num_features': len(queries)
        })
        
        # Clear GPU memory after each window
        del window_frames, video_tensor, pred_tracks, pred_visibility, queries_tensor
        torch.cuda.empty_cache()
        
    return all_tracks, windows


def visualize_sift_features(image, sift_points, output_path):
    """Visualize SIFT feature points on an image"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Plot SIFT points
    if len(sift_points) > 0:
        ax.scatter(sift_points[:, 0], sift_points[:, 1], 
                   c='red', s=30, alpha=0.7, edgecolors='yellow', linewidth=1)
    
    ax.set_title(f'SIFT Features ({len(sift_points)} points)')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def build_window_trajectories(all_tracks, total_frames):
    """Build trajectories for each window independently"""
    window_trajectories = []
    
    for window_idx, track_data in enumerate(all_tracks):
        tracks = track_data['tracks'][0]  # Remove batch dimension
        visibility = track_data['visibility'][0]  # Remove batch dimension
        start = track_data['start_frame']
        end = track_data['end_frame']
        num_points = tracks.shape[1]
        sift_points = track_data.get('sift_points', None)
        
        # Create trajectories for this window
        trajectories = []
        for i in range(num_points):
            traj = {
                'positions': np.full((total_frames, 2), np.nan),
                'visibility': np.zeros(total_frames, dtype=bool),
                'window_idx': window_idx,
                'start_frame': start,
                'end_frame': end,
                'is_sift': True,
                'sift_location': sift_points[i] if sift_points is not None else None
            }
            
            # Fill in the positions for this window
            for t in range(tracks.shape[0]):
                frame_idx = start + t
                if frame_idx < total_frames and visibility[t, i]:
                    traj['positions'][frame_idx] = tracks[t, i, :2].cpu().numpy()
                    traj['visibility'][frame_idx] = True
            
            trajectories.append(traj)
        
        window_trajectories.append({
            'trajectories': trajectories,
            'window_idx': window_idx,
            'start_frame': start,
            'end_frame': end,
            'num_sift_features': track_data.get('num_features', 0)
        })
    
    return window_trajectories


def generate_distinct_colors(n):
    """Generate n visually distinct colors"""
    if n <= 10:
        # Use tab10 colormap for small number of colors
        cmap = cm.get_cmap('tab10')
        colors = [cmap(i) for i in range(n)]
    else:
        # Use HSV colormap for larger number of colors
        colors = []
        for i in range(n):
            hue = i / n
            # Convert HSV to RGB (full saturation and value)
            rgb = mcolors.hsv_to_rgb([hue, 1.0, 0.9])
            colors.append(rgb[:3])  # Take only RGB, ignore alpha if present
    
    # Convert to 0-255 range
    colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]
    return colors


def draw_window_trajectories_on_frame(frame, window_trajectories, window_colors, current_frame_idx, 
                                      point_size=6, line_thickness=2, trail_length=20):
    """Draw trajectories from multiple windows on a single frame"""
    frame_vis = frame.copy()
    
    # Draw trajectories from each window
    for window_idx, (window_data, colors) in enumerate(zip(window_trajectories, window_colors)):
        trajectories = window_data['trajectories']
        window_start = window_data['start_frame']
        window_end = window_data['end_frame']
        
        # Only draw if current frame is within this window's range
        if window_start <= current_frame_idx < window_end:
            # Draw trails and current points for each trajectory in this window
            for traj_idx, (traj, color) in enumerate(zip(trajectories, colors)):
                positions = traj['positions']
                visibility = traj['visibility']
                
                # Draw trail (past positions within this window)
                trail_start = max(window_start, current_frame_idx - trail_length)
                valid_positions = []
                
                for t in range(trail_start, current_frame_idx + 1):
                    if visibility[t] and not np.isnan(positions[t]).any():
                        valid_positions.append(positions[t].astype(int))
                
                # Draw trail as connected lines
                if len(valid_positions) > 1:
                    pts = np.array(valid_positions, dtype=np.int32)
                    # Draw with gradual fade
                    for i in range(len(pts) - 1):
                        alpha = (i + 1) / len(pts)  # Fade effect
                        thickness = max(1, int(line_thickness * alpha))
                        cv2.line(frame_vis, tuple(pts[i]), tuple(pts[i+1]), color, thickness)
                
                # Draw current point
                if visibility[current_frame_idx]:
                    if not np.isnan(positions[current_frame_idx]).any():
                        pt = positions[current_frame_idx].astype(int)
                        # Draw with a distinctive style for SIFT points
                        cv2.circle(frame_vis, tuple(pt), point_size + 1, (255, 255, 255), -1)
                        cv2.circle(frame_vis, tuple(pt), point_size, color, -1)
                        
                        # Add "S" label for SIFT
                        cv2.putText(frame_vis, "S", 
                                   (pt[0] + point_size + 2, pt[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return frame_vis


def create_sift_summary_plot(image_paths, window_trajectories, window_colors, all_tracks, output_dir):
    """Create summary visualization for SIFT-based tracking"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CoTracker SIFT-Based Tracking Summary', fontsize=16)
    
    # 1. SIFT features visualization for first window
    ax = axes[0, 0]
    first_img = Image.open(image_paths[0])
    ax.imshow(first_img)
    
    # Plot SIFT points from first window
    if all_tracks and 'sift_points' in all_tracks[0]:
        sift_points = all_tracks[0]['sift_points']
        ax.scatter(sift_points[:, 0], sift_points[:, 1], 
                   c='red', s=20, alpha=0.7, edgecolors='yellow', linewidth=0.5)
        ax.set_title(f'Window 1 SIFT Features ({len(sift_points)} points)')
    else:
        ax.set_title('Window 1 SIFT Features')
    ax.axis('off')
    
    # 2. SIFT feature count per window
    ax = axes[0, 1]
    window_indices = []
    feature_counts = []
    
    for i, track_data in enumerate(all_tracks):
        window_indices.append(i + 1)
        feature_counts.append(track_data.get('num_features', 0))
    
    ax.bar(window_indices, feature_counts)
    ax.set_xlabel('Window Number')
    ax.set_ylabel('Number of SIFT Features')
    ax.set_title('SIFT Features Extracted per Window')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Feature distribution heatmap
    ax = axes[0, 2]
    h, w = np.array(Image.open(image_paths[0])).shape[:2]
    heatmap = np.zeros((h//10, w//10))
    
    # Accumulate all SIFT points
    for track_data in all_tracks:
        if 'sift_points' in track_data:
            points = track_data['sift_points']
            for pt in points:
                x_bin = int(pt[0] / 10)
                y_bin = int(pt[1] / 10)
                if 0 <= x_bin < heatmap.shape[1] and 0 <= y_bin < heatmap.shape[0]:
                    heatmap[y_bin, x_bin] += 1
    
    im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear')
    ax.set_title('SIFT Feature Distribution Heatmap')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 4. Tracking success rate
    ax = axes[1, 0]
    window_success_rates = []
    
    for window_data in window_trajectories:
        trajectories = window_data['trajectories']
        success_count = 0
        
        for traj in trajectories:
            # Count as successful if tracked for >50% of window duration
            vis_frames = np.sum(traj['visibility'][traj['start_frame']:traj['end_frame']])
            window_duration = traj['end_frame'] - traj['start_frame']
            if vis_frames > window_duration * 0.5:
                success_count += 1
        
        success_rate = success_count / len(trajectories) if trajectories else 0
        window_success_rates.append(success_rate * 100)
    
    ax.plot(range(1, len(window_success_rates) + 1), window_success_rates, 'o-', linewidth=2)
    ax.set_xlabel('Window Number')
    ax.set_ylabel('Tracking Success Rate (%)')
    ax.set_title('SIFT Feature Tracking Success Rate')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # 5. Sample trajectories with SIFT initial positions
    ax = axes[1, 1]
    first_img_for_bg = Image.open(image_paths[0])
    ax.imshow(first_img_for_bg, alpha=0.3)
    first_img_for_bg.close()
    
    # Plot sample trajectories from first 3 windows
    samples_per_window = 30
    for window_idx, (window_data, colors) in enumerate(zip(window_trajectories[:3], window_colors[:3])):
        trajectories = window_data['trajectories']
        
        # Sample some trajectories
        sample_indices = np.linspace(0, len(trajectories)-1, 
                                   min(samples_per_window, len(trajectories)), dtype=int)
        
        for idx in sample_indices:
            traj = trajectories[idx]
            color_norm = tuple(c/255 for c in colors[idx])
            
            # Get valid positions
            valid_mask = traj['visibility'] & ~np.isnan(traj['positions'][:, 0])
            if np.any(valid_mask):
                valid_pos = traj['positions'][valid_mask]
                ax.plot(valid_pos[:, 0], valid_pos[:, 1], 
                       color=color_norm, linewidth=1, alpha=0.5)
                
                # Mark SIFT initial position with a special marker
                if traj.get('sift_location') is not None:
                    sift_loc = traj['sift_location']
                    ax.plot(sift_loc[0], sift_loc[1], 'r*', markersize=4)
    
    ax.set_title('Sample SIFT-Based Trajectories (Red * = SIFT locations)')
    ax.axis('off')
    
    # 6. Comparison info text
    ax = axes[1, 2]
    ax.axis('off')
    
    info_text = "SIFT-Based Tracking Advantages:\n\n"
    info_text += "• Features detected in textured regions\n"
    info_text += "• Avoids textureless/uniform areas\n"
    info_text += "• More stable tracking on edges/corners\n"
    info_text += "• Adaptive to image content\n\n"
    info_text += f"Total SIFT features used: {sum(feature_counts)}\n"
    info_text += f"Average per window: {np.mean(feature_counts):.1f}\n"
    info_text += f"Tracking success rate: {np.mean(window_success_rates):.1f}%"
    
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sift_tracking_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    first_img.close()


def visualize_tracks(image_paths, all_tracks, output_dir):
    """Visualize SIFT-based tracking results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating SIFT-based visualizations...")
    
    # Get image dimensions from first image
    first_img = Image.open(image_paths[0])
    w, h = first_img.size
    first_img.close()
    
    # Visualize SIFT features from first window
    if all_tracks and 'sift_points' in all_tracks[0]:
        first_frame = np.array(Image.open(image_paths[0]))
        visualize_sift_features(first_frame, all_tracks[0]['sift_points'], 
                                output_dir / 'sift_features_window1.png')
    
    # Prepare video writer with imageio
    fps = 10  # Output video FPS
    video_path = output_dir / 'sift_tracking_result.mp4'
    video_writer = imageio.get_writer(str(video_path), fps=fps, codec='libx264', quality=8)
    
    # Build trajectories
    window_trajectories = build_window_trajectories(all_tracks, len(image_paths))
    
    # Generate colors for each window
    window_colors = []
    for window_data in window_trajectories:
        num_points = len(window_data['trajectories'])
        colors = generate_distinct_colors(num_points)
        window_colors.append(colors)
    
    # Process each frame
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(exist_ok=True)
    
    for frame_idx in range(len(image_paths)):
        print(f"\rVisualizing frame {frame_idx + 1}/{len(image_paths)}", end='', flush=True)
        
        # Load current frame
        img = Image.open(image_paths[frame_idx])
        frame = np.array(img)
        img.close()
        
        # Draw trajectories
        frame_vis = draw_window_trajectories_on_frame(frame, window_trajectories, window_colors, frame_idx)
        
        # Write frame to video (imageio expects RGB)
        video_writer.append_data(frame_vis)
        
        # Save sample frames
        if frame_idx in [0, len(image_paths)//4, len(image_paths)//2, 
                        3*len(image_paths)//4, len(image_paths)-1]:
            frame_bgr = cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f'sift_frame_{frame_idx:04d}.png'), frame_bgr)
    
    print()  # New line after progress
    video_writer.close()
    print(f"\nVideo saved to: {video_path}")
    
    # Create summary plot
    create_sift_summary_plot(image_paths, window_trajectories, window_colors, all_tracks, output_dir)
    
    print(f"\nSIFT-based visualizations saved to {output_dir}")


def main():
    """Main test function for SIFT-based tracking"""
    # Configuration
    script_dir = Path(__file__).parent.parent.parent  # Go up to colmap root
    image_dir = script_dir / "data/light_emitter_block_x3/section2"
    output_dir = script_dir / "outputs/cotracker_sift_test"
    num_frames = 100  # Test with first 100 frames
    window_size = 48  # Window size
    overlap = 24  # 50% overlap
    max_sift_features = 400  # Maximum SIFT features per window
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return
    
    # Get image paths
    print(f"Getting image paths from {image_dir}...")
    try:
        image_paths = get_image_paths(str(image_dir), num_frames=num_frames)
        print(f"Found {len(image_paths)} images")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Run SIFT-based tracking
    start_time = time.time()
    all_tracks, windows = run_sift_based_tracking(
        image_paths, 
        window_size=window_size,
        overlap=overlap,
        device=device,
        max_features=max_sift_features
    )
    end_time = time.time()
    
    print(f"\nSIFT-based tracking completed in {end_time - start_time:.2f} seconds")
    print(f"FPS: {len(image_paths) / (end_time - start_time):.2f}")
    
    # Visualize results
    visualize_tracks(image_paths, all_tracks, output_dir)
    
    print("\nSIFT-based test completed successfully!")
    print(f"Results saved to: {output_dir}")
    
    # Print comparison instructions
    print("\nTo compare with uniform grid results:")
    print(f"  Grid-based: outputs/cotracker_test/")
    print(f"  SIFT-based: {output_dir}/")


if __name__ == "__main__":
    main()