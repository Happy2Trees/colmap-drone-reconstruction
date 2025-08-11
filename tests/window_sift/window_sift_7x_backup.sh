#!/usr/bin/env bash
GPU_INDEX=0
WINDOW_SIZE=48
STRIDE=12
# IMAGE_PATH="/home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2/images"
IMAGE_PATH="/home/sanggyun/feature_matching/data/sec2_7x"
WORKSPACE_PATH="/home/sanggyun/hdseo/colmap-drone-reconstruction/tests/window_sift/workspace/section2_7x_window_sift_${WINDOW_SIZE}_${STRIDE}_30fps_hierarchical_mapper"
DATABASE_PATH="$WORKSPACE_PATH/database.db"

mkdir -p "$WORKSPACE_PATH"
mkdir -p "$WORKSPACE_PATH/sparse"

python generate_pairs.py \
    --image_dir "$IMAGE_PATH" \
    --window_size "$WINDOW_SIZE" \
    --stride "$STRIDE" \
    --output_path "$WORKSPACE_PATH/pairs.txt"
# Camera intrinsics (fx, fy, cx, cy, k1, k2, p1, p2)
# Extracted from intrinsic/x3/K.txt and intrinsic/x3/dist.txt
CAMERA_PARAMS="19872.643351,19873.226698,2123.585064,1499.441091,-0.208057,-5.646531,0.002627,0.003169"

echo "Running feature extraction with provided intrinsics..."
CUDA_VISIBLE_DEVICES=$GPU_INDEX colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model OPENCV \
    --ImageReader.camera_params "$CAMERA_PARAMS" \
    --ImageReader.single_camera 1

echo "Importing and matching specified pairs from pairs.txt..."
CUDA_VISIBLE_DEVICES=$GPU_INDEX colmap matches_importer \
    --database_path "$DATABASE_PATH" \
    --match_list_path "$WORKSPACE_PATH/pairs.txt" \
    --match_type pairs

echo "Running sparse reconstruction (mapping)..."
CUDA_VISIBLE_DEVICES=$GPU_INDEX colmap hierarchical_mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$WORKSPACE_PATH/sparse" \
    --Mapper.num_threads -1

echo "WORKSPACE_PATH = $WORKSPACE_PATH"
echo "Script finished."