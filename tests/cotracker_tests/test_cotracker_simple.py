#!/usr/bin/env python3
"""
Simple CoTracker sliding window test with visualization
Runs CoTracker inference on windows and visualizes the results
Uses OFFLINE model for better tracking quality
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


def run_sliding_window_tracking(image_paths, window_size=24, overlap=12, device='cuda'):
    """Run CoTracker with sliding windows using OFFLINE model
    
    Args:
        image_paths: List of image file paths
        window_size: Number of frames per window
        overlap: Number of overlapping frames between windows
        device: Device to run on
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
    
    for start in range(0, total_frames, stride):
        end = min(start + window_size, total_frames)
        windows.append((start, end))
        if end >= total_frames:
            break
    
    print(f"\nGenerated {len(windows)} windows:")
    for i, (start, end) in enumerate(windows):
        overlap_with_prev = 0 if i == 0 else windows[i-1][1] - start
        print(f"  Window {i+1}: frames [{start:3d}-{end:3d}), overlap with previous: {overlap_with_prev} frames")
    
    print(f"\nProcessing {len(windows)} windows...")
    
    # Process each window
    all_tracks = []
    
    # Get first frame size to initialize queries
    first_img = Image.open(image_paths[0])
    img_height, img_width = first_img.height, first_img.width
    first_img.close()
    
    for i, (start, end) in enumerate(windows):
        print(f"\nWindow {i+1}/{len(windows)}: frames {start}-{end-1}")
        
        # Load only window frames to save memory
        window_frames = load_images_for_window(image_paths, start, end)
        video_tensor = torch.from_numpy(window_frames).permute(0, 3, 1, 2).float()
        video_tensor = video_tensor.unsqueeze(0).to(device)  # Add batch dimension
        
        # Run tracking
        with torch.no_grad():
            # Initialize NEW grid of points for EACH window
            grid_size = 20  # Increased for better coverage
            # Create grid points at the FIRST frame of current window
            grid_pts = get_grid_points_for_window((img_height, img_width), grid_size, 
                                                 window_frame_offset=0)  # Always use first frame of window
            queries = torch.from_numpy(grid_pts).float().to(device)
            
            # Add batch dimension only
            queries = queries.unsqueeze(0)  # [B=1, N, 3] where N is number of points
            
            # Run offline model (processes entire window at once)
            # Note: The model returned by torch.hub is already a predictor
            pred_tracks, pred_visibility = model(video_tensor, queries=queries)
            
        # Store results
        all_tracks.append({
            'tracks': pred_tracks.cpu(),
            'visibility': pred_visibility.cpu(),
            'start_frame': start,
            'end_frame': end
        })
        
        # Clear GPU memory after each window
        del window_frames, video_tensor, pred_tracks, pred_visibility
        torch.cuda.empty_cache()
        
    return all_tracks, windows


def get_grid_points(image_shape, grid_size):
    """Generate a grid of query points"""
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


def get_grid_points_for_window(image_shape, grid_size, window_frame_offset=0):
    """Generate a grid of query points for a specific window
    
    Args:
        image_shape: (height, width) of the image
        grid_size: Number of points in each dimension
        window_frame_offset: Frame offset within the window (typically 0 for first frame)
    """
    h, w = image_shape[:2]
    
    # Create grid with slight randomization to avoid tracking same points
    # Add small random offset to grid positions for variety
    np.random.seed(None)  # Use current time for randomness
    margin = 0.1
    
    y_coords = np.linspace(h * margin, h * (1-margin), grid_size)
    x_coords = np.linspace(w * margin, w * (1-margin), grid_size)
    
    # Add small random perturbation (up to 5 pixels)
    y_coords += np.random.uniform(-5, 5, size=grid_size)
    x_coords += np.random.uniform(-5, 5, size=grid_size)
    
    # Ensure points stay within valid bounds
    y_coords = np.clip(y_coords, 0, h-1)
    x_coords = np.clip(x_coords, 0, w-1)
    
    # Generate all combinations
    points = []
    for y in y_coords:
        for x in x_coords:
            points.append([window_frame_offset, x, y])  # [time, x, y]
    
    return np.array(points)


def build_window_trajectories(all_tracks, total_frames):
    """Build trajectories for each window independently
    
    Returns:
        List of window trajectory sets, where each set contains trajectories for that window
    """
    window_trajectories = []
    
    for window_idx, track_data in enumerate(all_tracks):
        tracks = track_data['tracks'][0]  # Remove batch dimension
        visibility = track_data['visibility'][0]  # Remove batch dimension
        start = track_data['start_frame']
        end = track_data['end_frame']
        num_points = tracks.shape[1]
        
        # Create trajectories for this window
        trajectories = []
        for i in range(num_points):
            traj = {
                'positions': np.full((total_frames, 2), np.nan),
                'visibility': np.zeros(total_frames, dtype=bool),
                'window_idx': window_idx,
                'start_frame': start,
                'end_frame': end
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
            'end_frame': end
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
                                      point_size=8, line_thickness=3, trail_length=30):
    """Draw trajectories from multiple windows on a single frame
    
    Args:
        frame: Input image as numpy array
        window_trajectories: List of window trajectory sets
        window_colors: List of color sets for each window
        current_frame_idx: Current frame index
        point_size: Size of the tracking points
        line_thickness: Thickness of trajectory lines
        trail_length: Number of past frames to show in trail
    """
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
                        # Draw white border for better visibility
                        cv2.circle(frame_vis, tuple(pt), point_size + 2, (255, 255, 255), -1)
                        cv2.circle(frame_vis, tuple(pt), point_size, color, -1)
                        
                        # Add window index label
                        cv2.putText(frame_vis, f"W{window_idx}", 
                                   (pt[0] + point_size + 3, pt[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame_vis


def create_enhanced_window_summary_plot(image_paths, window_trajectories, window_colors, output_dir):
    """Create enhanced summary visualization for window-based tracking"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CoTracker Window-Based Tracking Summary', fontsize=16)
    
    # 1. First window starting points
    ax = axes[0, 0]
    first_img = Image.open(image_paths[0])
    ax.imshow(first_img)
    
    # Plot starting points of first window
    if window_trajectories:
        first_window = window_trajectories[0]
        colors = window_colors[0]
        for traj_idx, (traj, color) in enumerate(zip(first_window['trajectories'], colors)):
            if traj['visibility'][0]:
                pt = traj['positions'][0]
                if not np.isnan(pt).any():
                    color_norm = tuple(c/255 for c in color)
                    ax.plot(pt[0], pt[1], 'o', color=color_norm, markersize=8, 
                           markeredgecolor='white', markeredgewidth=1)
    
    ax.set_title(f'Window 1 Starting Points (400 points, 20x20 grid)')
    ax.axis('off')
    
    # 2. Window coverage visualization
    ax = axes[0, 1]
    coverage = np.zeros(len(image_paths))
    window_boundaries = []
    
    for window_data in window_trajectories:
        start = window_data['start_frame']
        end = window_data['end_frame']
        coverage[start:end] += 1
        window_boundaries.append((start, end))
    
    ax.plot(coverage, linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Window Coverage')
    ax.set_title(f'Sliding Window Coverage ({len(window_trajectories)} windows)')
    ax.grid(True, alpha=0.3)
    
    # Add window boundaries
    for i, (start, end) in enumerate(window_boundaries):
        ax.axvspan(start, end, alpha=0.1, color=f'C{i%10}')
        ax.text((start + end) / 2, ax.get_ylim()[1] * 0.9, f'W{i+1}', 
                ha='center', fontsize=8)
    
    # 3. Multiple windows visualization
    ax = axes[0, 2]
    # Show frame where multiple windows overlap
    overlap_frame = None
    for i in range(len(image_paths)):
        if coverage[i] > 1:
            overlap_frame = i
            break
    
    if overlap_frame is not None:
        img = Image.open(image_paths[overlap_frame])
        frame = np.array(img)
        img.close()
        frame_vis = draw_window_trajectories_on_frame(frame, window_trajectories, 
                                                     window_colors, overlap_frame)
        ax.imshow(frame_vis)
        ax.set_title(f'Frame {overlap_frame} (Multiple Windows)')
    else:
        ax.text(0.5, 0.5, 'No overlapping windows', ha='center', va='center')
    ax.axis('off')
    
    # 4. Tracking statistics per window
    ax = axes[1, 0]
    window_indices = []
    avg_displacements = []
    
    for window_idx, window_data in enumerate(window_trajectories):
        displacements = []
        for traj in window_data['trajectories']:
            start_idx = window_data['start_frame']
            end_idx = window_data['end_frame'] - 1
            
            if (start_idx < len(traj['visibility']) and end_idx < len(traj['visibility']) and
                traj['visibility'][start_idx] and traj['visibility'][end_idx]):
                if not np.isnan(traj['positions'][start_idx]).any() and not np.isnan(traj['positions'][end_idx]).any():
                    disp = np.linalg.norm(traj['positions'][end_idx] - traj['positions'][start_idx])
                    displacements.append(disp)
        
        if displacements:
            window_indices.append(window_idx + 1)
            avg_displacements.append(np.mean(displacements))
    
    if window_indices:
        ax.bar(window_indices, avg_displacements)
        ax.set_xlabel('Window Number')
        ax.set_ylabel('Average Point Displacement (pixels)')
        ax.set_title('Average Displacement per Window')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Visibility over time for all windows
    ax = axes[1, 1]
    total_visibility = np.zeros(len(image_paths))
    
    for window_data in window_trajectories:
        for traj in window_data['trajectories']:
            total_visibility += traj['visibility']
    
    ax.plot(total_visibility, linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Total Visible Points')
    ax.set_title('Total Tracking Points Over Time')
    ax.grid(True, alpha=0.3)
    
    # Add window boundaries
    for i, (start, end) in enumerate(window_boundaries[:5]):  # Show first 5 windows
        ax.axvspan(start, end, alpha=0.05, color=f'C{i%10}')
    
    # 6. Sample trajectories from different windows
    ax = axes[1, 2]
    first_img_for_bg = Image.open(image_paths[0])
    ax.imshow(first_img_for_bg, alpha=0.3)
    first_img_for_bg.close()
    
    # Plot sample trajectories from different windows
    samples_per_window = 20
    for window_idx, (window_data, colors) in enumerate(zip(window_trajectories[:3], window_colors[:3])):
        trajectories = window_data['trajectories']
        
        # Sample some trajectories from this window
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
                       color=color_norm, linewidth=1.5, alpha=0.6)
                # Mark start with window number
                ax.text(valid_pos[0, 0], valid_pos[0, 1], f'{window_idx+1}', 
                       fontsize=6, color='white', 
                       bbox=dict(boxstyle='circle,pad=0.1', facecolor=color_norm, alpha=0.8))
    
    ax.set_title('Sample Trajectories from Windows 1-3')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'enhanced_window_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    first_img.close()




def visualize_tracks(image_paths, all_tracks, output_dir):
    """Visualize tracking results with custom enhanced visualization
    
    Args:
        image_paths: List of image file paths
        all_tracks: List of track dictionaries from each window
        output_dir: Output directory for visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating enhanced visualizations...")
    
    # Get image dimensions from first image
    first_img = Image.open(image_paths[0])
    w, h = first_img.size
    first_img.close()
    
    # Prepare video writer with imageio
    fps = 10  # Output video FPS
    video_path = output_dir / 'tracking_result_enhanced.mp4'
    video_writer = imageio.get_writer(str(video_path), fps=fps, codec='libx264', quality=8)
    
    # Build trajectories for each window independently
    window_trajectories = build_window_trajectories(all_tracks, len(image_paths))
    
    # Generate colors for each window (different colors per window)
    window_colors = []
    for window_data in window_trajectories:
        num_points = len(window_data['trajectories'])
        colors = generate_distinct_colors(num_points)
        window_colors.append(colors)
    
    # Process each frame
    for frame_idx in range(len(image_paths)):
        print(f"\rVisualizing frame {frame_idx + 1}/{len(image_paths)}", end='', flush=True)
        
        # Load current frame
        img = Image.open(image_paths[frame_idx])
        frame = np.array(img)
        img.close()
        
        # Draw trajectories from all windows that are active at this frame
        frame_vis = draw_window_trajectories_on_frame(frame, window_trajectories, window_colors, frame_idx)
        
        # Write frame to video (imageio expects RGB)
        video_writer.append_data(frame_vis)
        
        # Save sample frames as higher quality images
        if frame_idx in [0, len(image_paths)//4, len(image_paths)//2, 
                        3*len(image_paths)//4, len(image_paths)-1]:
            plt.figure(figsize=(12, 10))
            plt.imshow(frame_vis)
            plt.title(f'Frame {frame_idx} with Trajectories')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / f'enhanced_frame_{frame_idx:04d}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    print()  # New line after progress
    video_writer.close()
    print(f"\nVideo saved to: {video_path}")
    
    # Create summary plot
    create_enhanced_window_summary_plot(image_paths, window_trajectories, window_colors, output_dir)
    
    print(f"\nVisualizations saved to {output_dir}")




def main():
    """Main test function"""
    # Configuration
    # Use relative path from script location
    script_dir = Path(__file__).parent.parent.parent  # Go up to colmap root
    image_dir = script_dir / "data/light_emitter_block_x3/section2"
    output_dir = script_dir / "outputs/cotracker_test"
    num_frames = 100  # Test with first 100 frames
    window_size = 48  # Increased window size for better tracking
    overlap = 24  # 50% overlap
    
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
    
    # Run tracking
    start_time = time.time()
    all_tracks, windows = run_sliding_window_tracking(
        image_paths, 
        window_size=window_size,
        overlap=overlap,
        device=device
    )
    end_time = time.time()
    
    print(f"\nTracking completed in {end_time - start_time:.2f} seconds")
    print(f"FPS: {len(image_paths) / (end_time - start_time):.2f}")
    
    # Visualize results
    visualize_tracks(image_paths, all_tracks, output_dir)
    
    print("\nTest completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()