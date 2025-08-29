#!/usr/bin/env bash
GPU_INDEX=0,1,2,3
IMAGE_PATH="data/sec2_7x"
WORKSPACE_PATH="outputs/workspaces/glomap_sec2_7x_sequential"
DATABASE_PATH="$WORKSPACE_PATH/database.db"

mkdir -p "$WORKSPACE_PATH"
mkdir -p "$WORKSPACE_PATH/sparse"

# Camera intrinsics (fx, fy, cx, cy, k1, k2, p1, p2)
# Extracted from intrinsic/x3/K.txt and intrinsic/x3/dist.txt
# CAMERA_PARAMS="19872.643351,19873.226698,2123.585064,1499.441091,-0.208057,-5.646531,0.002627,0.003169"
CAMERA_PARAMS="19872.9350245,2123.585064,1499.441091,0.0"
echo "Running feature extraction with provided intrinsics..."
CUDA_VISIBLE_DEVICES=$GPU_INDEX colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.camera_params "$CAMERA_PARAMS" \
    --ImageReader.single_camera 1

echo "Running sequential matching..."
colmap sequential_matcher \
    --database_path "$DATABASE_PATH"

echo "Running sparse reconstruction (mapping)..."
glomap mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$WORKSPACE_PATH/sparse"

echo "Script finished."