#!/usr/bin/env bash
GPU_INDEX=0
IMAGE_PATH="data/sec4_3x_20fps"
WORKSPACE_PATH="outputs/workspaces/hierarchical_mapper_sec4_3x_20fps"
DATABASE_PATH="$WORKSPACE_PATH/database.db"

mkdir -p "$WORKSPACE_PATH"
mkdir -p "$WORKSPACE_PATH/sparse"

# Camera intrinsics (fx, fy, cx, cy, k1, k2, p1, p2)
# Extracted from intrinsic/x3/K.txt and intrinsic/x3/dist.txt
CAMERA_PARAMS="9660.362404,9713.707651,1355.300874,1632.943811,-0.274332,0.435356,0.006031,-0.021247"

echo "Running feature extraction with provided intrinsics..."
CUDA_VISIBLE_DEVICES=$GPU_INDEX colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model OPENCV \
    --ImageReader.camera_params "$CAMERA_PARAMS" \
    --ImageReader.single_camera 1

echo "Running exhaustive matching..."
colmap exhaustive_matcher \
    --database_path "$DATABASE_PATH"

echo "Running sparse reconstruction (mapping)..."
colmap hierarchical_mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$WORKSPACE_PATH/sparse"

echo "Script finished."