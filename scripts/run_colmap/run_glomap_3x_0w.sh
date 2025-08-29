#!/bin/bash
set -e

# Define paths relative to the script location
IMAGE_PATH="./data/not_finished/light_emitter_block_x3/section1_sliced"
WORKSPACE_PATH="./outputs/workspaces/glomap_section1_3x_seq"
DATABASE_PATH="$WORKSPACE_PATH/database.db"

# Camera intrinsics (fx, fy, cx, cy, k1, k2, p1, p2)
# Extracted from intrinsic/x3/K.txt and intrinsic/x3/dist.txt
CAMERA_PARAMS="9660.362404,9713.707651,1355.300874,1632.943811,-0.274332,0.435356,0.006031,-0.021247"

# Create workspace directory if it doesn't exist
mkdir -p "$WORKSPACE_PATH/sparse"

echo "Running feature extraction with provided intrinsics..."
colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model OPENCV \
    --ImageReader.camera_params "$CAMERA_PARAMS" \
    --ImageReader.single_camera 1

echo "Running exhaustive matching..."
colmap exhaustive_matcher \
    --database_path "$DATABASE_PATH"

echo "Running sparse reconstruction (mapping)..."
glomap mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$WORKSPACE_PATH/sparse"

echo "COLMAP reconstruction finished."
echo "Workspace: $WORKSPACE_PATH"
echo "Sparse model stored in: $WORKSPACE_PATH/sparse"
