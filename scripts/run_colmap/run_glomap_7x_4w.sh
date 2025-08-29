#!/bin/bash
set -e

# Define paths relative to the script location
IMAGE_PATH="data/not_finished/light_emitter_block_x7_only_1section_frames/DJI_20250328122059_0049_D_selected_frames"
WORKSPACE_PATH="./outputs/workspaces/glomap_section1_7x_seq"
DATABASE_PATH="$WORKSPACE_PATH/database.db"

# Camera intrinsics (fx, fy, cx, cy, k1, k2, p1, p2, k3)
# Extracted from intrinsic/x7/K.txt and intrinsic/x7/dist.txt
CAMERA_PARAMS="19872.643351,19873.226698,2123.585064,1499.441091,-0.208057,-5.646531,0.002627,0.003169"

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