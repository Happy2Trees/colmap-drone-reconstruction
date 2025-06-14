#!/bin/bash
# Example script to run Window-based Bundle Adjustment with Phase 2 optimization

# Test scene path
SCENE_PATH="/hdd2/0321_block_drone_video/colmap/data/3x_section2_fps12_processed_1024x576"

# Output directory with timestamp
OUTPUT_DIR="outputs/window_ba_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Running Window BA on test scene"
echo "Scene: $SCENE_PATH"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Run with Phase 2 optimization enabled
python -m src.window_ba "$SCENE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --use_refine \
    --verbose

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Window BA completed successfully!"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo "  - Cameras: $OUTPUT_DIR/cameras_final.npz"
    echo "  - COLMAP model: $OUTPUT_DIR/colmap/"
    echo "  - Visualizations: $OUTPUT_DIR/visualizations/"
    echo "  - Summary: $OUTPUT_DIR/summary.txt"
    
    # List visualization files
    if [ -d "$OUTPUT_DIR/visualizations" ]; then
        echo ""
        echo "Generated visualizations:"
        ls -la "$OUTPUT_DIR/visualizations/"*.png
    fi
else
    echo ""
    echo "✗ Window BA failed!"
    echo "Check the logs for details."
fi