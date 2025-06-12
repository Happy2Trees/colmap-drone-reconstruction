#!/usr/bin/env python3
"""Analyze depth consistency and save results as CSV."""

import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm

def analyze_depth_segments(depth_dir: Path, segment_size: int = 100):
    """Analyze depth maps by segments and save statistics."""
    
    # Get all depth files
    depth_files = sorted(depth_dir.glob("*.npz"))
    if not depth_files:
        raise ValueError(f"No depth files found in {depth_dir}")
    
    print(f"Found {len(depth_files)} depth files")
    
    # Prepare CSV output
    csv_path = depth_dir.parent.parent / "depth_segment_analysis.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Write header
    csv_writer.writerow([
        'Segment', 'Start_Frame', 'End_Frame', 'Num_Frames',
        'Mean', 'Std', 'Median', 'Min', 'Max',
        'P5', 'P25', 'P75', 'P95',
        'IQR', 'Valid_Pixels'
    ])
    
    all_means = []
    all_medians = []
    
    # Process each segment
    for i in range(0, len(depth_files), segment_size):
        segment_end = min(i + segment_size, len(depth_files))
        segment_files = depth_files[i:segment_end]
        
        print(f"\nAnalyzing segment: frames {i} to {segment_end-1}")
        
        # Collect depth values
        all_depths = []
        valid_count = 0
        
        for depth_file in tqdm(segment_files, desc="Processing"):
            depth_npz = np.load(depth_file)
            
            if 'depth' in depth_npz and 'mask' in depth_npz:
                depth_data = depth_npz['depth']
                valid_mask = depth_npz['mask']
                depths = depth_data[valid_mask]
                all_depths.append(depths)
                valid_count += np.sum(valid_mask)
        
        # Concatenate all depths
        segment_depths = np.concatenate(all_depths)
        
        # Calculate statistics
        mean = np.mean(segment_depths)
        std = np.std(segment_depths)
        median = np.median(segment_depths)
        min_val = np.min(segment_depths)
        max_val = np.max(segment_depths)
        p5 = np.percentile(segment_depths, 5)
        p25 = np.percentile(segment_depths, 25)
        p75 = np.percentile(segment_depths, 75)
        p95 = np.percentile(segment_depths, 95)
        iqr = p75 - p25
        
        all_means.append(mean)
        all_medians.append(median)
        
        # Write to CSV
        csv_writer.writerow([
            f"{i}-{segment_end-1}", i, segment_end-1, len(segment_files),
            f"{mean:.6f}", f"{std:.6f}", f"{median:.6f}", 
            f"{min_val:.6f}", f"{max_val:.6f}",
            f"{p5:.6f}", f"{p25:.6f}", f"{p75:.6f}", f"{p95:.6f}",
            f"{iqr:.6f}", valid_count
        ])
        
        print(f"  Mean: {mean:.3f} ± {std:.3f}")
        print(f"  Median: {median:.3f}")
        print(f"  Range: [{min_val:.3f}, {max_val:.3f}]")
    
    # Add summary statistics
    csv_writer.writerow([])
    csv_writer.writerow(['SUMMARY STATISTICS'])
    csv_writer.writerow(['Metric', 'Value'])
    csv_writer.writerow(['Global Mean', f"{np.mean(all_means):.6f}"])
    csv_writer.writerow(['Global Std of Means', f"{np.std(all_means):.6f}"])
    csv_writer.writerow(['Coefficient of Variation', f"{np.std(all_means)/np.mean(all_means):.6f}"])
    csv_writer.writerow(['Min Mean', f"{np.min(all_means):.6f}"])
    csv_writer.writerow(['Max Mean', f"{np.max(all_means):.6f}"])
    
    csv_file.close()
    
    print(f"\n{'='*50}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print(f"Results saved to: {csv_path}")
    print(f"\nGlobal statistics:")
    print(f"  Mean of segment means: {np.mean(all_means):.6f}")
    print(f"  CV of means: {np.std(all_means)/np.mean(all_means)*100:.2f}%")
    
    # Check consistency
    cv = np.std(all_means) / np.mean(all_means)
    if cv < 0.05:
        print(f"  Consistency: EXCELLENT (CV < 5%)")
    elif cv < 0.10:
        print(f"  Consistency: GOOD (CV < 10%)")
    elif cv < 0.20:
        print(f"  Consistency: MODERATE (CV < 20%)")
    else:
        print(f"  Consistency: POOR (CV > 20%)")

if __name__ == "__main__":
    depth_dir = Path("data/3x_section2_processed_1024x576/depth/GeometryCrafter")
    analyze_depth_segments(depth_dir, segment_size=100)