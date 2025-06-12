#!/usr/bin/env python3
"""Check depth scale consistency across segments."""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_depth_statistics(depth_dir: Path, segment_size: int = 100):
    """Load depth maps and compute statistics per segment."""
    
    # Get all depth files
    depth_files = sorted(depth_dir.glob("*.npz"))
    if not depth_files:
        raise ValueError(f"No depth files found in {depth_dir}")
    
    print(f"Found {len(depth_files)} depth files")
    
    # Compute statistics per segment
    segments = []
    segment_stats = []
    
    for i in range(0, len(depth_files), segment_size):
        segment_end = min(i + segment_size, len(depth_files))
        segment_files = depth_files[i:segment_end]
        
        print(f"\nProcessing segment {i//segment_size + 1}: frames {i} to {segment_end-1}")
        
        # Collect all depth values for this segment
        all_depths = []
        valid_counts = []
        
        for depth_file in tqdm(segment_files, desc="Loading depth maps"):
            # Load npz file (contains 'depth' and 'mask' arrays)
            depth_npz = np.load(depth_file)
            
            # Extract depth and mask
            if 'depth' in depth_npz and 'mask' in depth_npz:
                depth_data = depth_npz['depth']
                valid_mask = depth_npz['mask']
                depths = depth_data[valid_mask]
            else:
                print(f"Unexpected data keys in {depth_file.name}: {list(depth_npz.keys())}")
                continue
            
            all_depths.append(depths)
            valid_counts.append(np.sum(valid_mask))
        
        # Concatenate all depths for this segment
        segment_depths = np.concatenate(all_depths)
        
        # Compute statistics
        stats = {
            'segment': f"{i}-{segment_end-1}",
            'start_frame': int(i),
            'end_frame': int(segment_end - 1),
            'num_frames': int(len(segment_files)),
            'total_valid_pixels': int(sum(valid_counts)),
            'mean': float(np.mean(segment_depths)),
            'std': float(np.std(segment_depths)),
            'median': float(np.median(segment_depths)),
            'min': float(np.min(segment_depths)),
            'max': float(np.max(segment_depths)),
            'percentile_25': float(np.percentile(segment_depths, 25)),
            'percentile_75': float(np.percentile(segment_depths, 75)),
            'percentile_5': float(np.percentile(segment_depths, 5)),
            'percentile_95': float(np.percentile(segment_depths, 95)),
        }
        
        segments.append(f"{i}-{segment_end-1}")
        segment_stats.append(stats)
        
        print(f"Segment {stats['segment']}:")
        print(f"  Mean depth: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"  Median depth: {stats['median']:.3f}")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  IQR: [{stats['percentile_25']:.3f}, {stats['percentile_75']:.3f}]")
    
    return segments, segment_stats

def analyze_consistency(segment_stats):
    """Analyze consistency across segments."""
    
    print("\n" + "="*50)
    print("CONSISTENCY ANALYSIS")
    print("="*50)
    
    # Extract statistics
    means = [s['mean'] for s in segment_stats]
    stds = [s['std'] for s in segment_stats]
    medians = [s['median'] for s in segment_stats]
    
    # Global statistics
    global_mean = np.mean(means)
    global_std = np.std(means)
    
    print(f"\nGlobal mean of segment means: {global_mean:.3f} ± {global_std:.3f}")
    print(f"Coefficient of variation (CV) of means: {global_std/global_mean:.3%}")
    
    # Check for significant changes
    print("\nSegment-to-segment changes:")
    for i in range(1, len(segment_stats)):
        prev = segment_stats[i-1]
        curr = segment_stats[i]
        
        mean_change = (curr['mean'] - prev['mean']) / prev['mean'] * 100
        median_change = (curr['median'] - prev['median']) / prev['median'] * 100
        
        if abs(mean_change) > 10:  # More than 10% change
            print(f"  ⚠️  Segment {prev['segment']} → {curr['segment']}: "
                  f"mean changed by {mean_change:+.1f}%")
        else:
            print(f"  ✓  Segment {prev['segment']} → {curr['segment']}: "
                  f"mean changed by {mean_change:+.1f}%")
    
    # Scale consistency score
    cv = global_std / global_mean
    if cv < 0.05:
        consistency = "EXCELLENT (CV < 5%)"
    elif cv < 0.10:
        consistency = "GOOD (CV < 10%)"
    elif cv < 0.20:
        consistency = "MODERATE (CV < 20%)"
    else:
        consistency = "POOR (CV > 20%)"
    
    print(f"\nOverall scale consistency: {consistency}")
    
    return means, stds, medians

def plot_statistics(segments, segment_stats, output_path):
    """Create visualization of depth statistics across segments."""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Extract data
    x = range(len(segments))
    means = [s['mean'] for s in segment_stats]
    stds = [s['std'] for s in segment_stats]
    medians = [s['median'] for s in segment_stats]
    p25 = [s['percentile_25'] for s in segment_stats]
    p75 = [s['percentile_75'] for s in segment_stats]
    p5 = [s['percentile_5'] for s in segment_stats]
    p95 = [s['percentile_95'] for s in segment_stats]
    
    # Plot 1: Mean and standard deviation
    ax1 = axes[0]
    ax1.errorbar(x, means, yerr=stds, fmt='o-', capsize=5)
    ax1.set_ylabel('Depth Mean ± Std')
    ax1.set_title('Depth Statistics Across Segments')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Median and IQR
    ax2 = axes[1]
    ax2.plot(x, medians, 'o-', label='Median')
    ax2.fill_between(x, p25, p75, alpha=0.3, label='IQR (25-75%)')
    ax2.set_ylabel('Depth Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Range (5-95 percentile)
    ax3 = axes[2]
    ax3.fill_between(x, p5, p95, alpha=0.3, label='5-95% Range')
    ax3.plot(x, means, 'r-', label='Mean')
    ax3.set_xlabel('Segment')
    ax3.set_ylabel('Depth Value')
    ax3.set_xticks(x)
    ax3.set_xticklabels([s['segment'] for s in segment_stats], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nVisualization saved to: {output_path}")

def main():
    # Path to depth directory
    depth_dir = Path("data/3x_section2_processed_1024x576/depth/GeometryCrafter")
    
    if not depth_dir.exists():
        print(f"Error: Depth directory not found: {depth_dir}")
        return
    
    # Analyze with 100-frame segments
    segments, segment_stats = load_depth_statistics(depth_dir, segment_size=100)
    
    # Analyze consistency
    means, stds, medians = analyze_consistency(segment_stats)
    
    # Save detailed statistics
    stats_file = depth_dir.parent.parent / "depth_consistency_analysis.json"
    with open(stats_file, 'w') as f:
        json.dump({
            'segment_size': 100,
            'segments': segment_stats,
            'summary': {
                'total_segments': len(segment_stats),
                'global_mean': float(np.mean(means)),
                'global_std': float(np.std(means)),
                'coefficient_of_variation': float(np.std(means) / np.mean(means)),
            }
        }, f, indent=2)
    print(f"\nDetailed statistics saved to: {stats_file}")
    
    # Create visualization
    plot_path = depth_dir.parent.parent / "depth_consistency_plot.png"
    plot_statistics(segments, segment_stats, plot_path)

if __name__ == "__main__":
    main()