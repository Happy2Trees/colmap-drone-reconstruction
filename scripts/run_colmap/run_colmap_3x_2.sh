#!/bin/bash
set -e

# Define paths relative to the script location
IMAGE_PATH="./data/light_emitter_block_x3/section3_sliced"
WORKSPACE_PATH="./outputs/workspaces/section3_3x"
DATABASE_PATH="$WORKSPACE_PATH/database.db"

# Create workspace directory if it doesn't exist
mkdir -p "$WORKSPACE_PATH/sparse"

echo "Running feature extraction..."
colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.single_camera 1

echo "Running exhaustive matching..."
colmap exhaustive_matcher \
    --database_path "$DATABASE_PATH"

echo "Running sparse reconstruction (mapping)..."
colmap mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$WORKSPACE_PATH/sparse"

echo "COLMAP reconstruction finished."
echo "Workspace: $WORKSPACE_PATH"
echo "Sparse model stored in: $WORKSPACE_PATH/sparse"
