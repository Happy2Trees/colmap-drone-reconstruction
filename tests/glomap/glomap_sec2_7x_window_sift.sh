#!/usr/bin/env bash
GPU_INDEX=2
WINDOW_SIZE=48
STRIDE=12
# IMAGE_PATH="/home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2/images"
IMAGE_PATH="data/sec2_7x"
WORKSPACE_PATH="outputs/workspaces/section2_7x_window_sift_${WINDOW_SIZE}_${STRIDE}_glomap"
DATABASE_PATH="$WORKSPACE_PATH/database.db"

mkdir -p "$WORKSPACE_PATH"
mkdir -p "$WORKSPACE_PATH/sparse"

# python tests/window_sift/generate_pairs.py \
#     --image_dir "$IMAGE_PATH" \
#     --window_size "$WINDOW_SIZE" \
#     --stride "$STRIDE" \
#     --output_path "$WORKSPACE_PATH/pairs.txt"
# # Camera intrinsics (f, cx, cy, k)
# # CAMERA_PARAMS="19872.643351,19873.226698,2123.585064,1499.441091,-0.208057,-5.646531,0.002627,0.003169"
# CAMERA_PARAMS="19872.9350245,2123.585064,1499.441091,0.0"
# echo "Running feature extraction with provided intrinsics..."
# CUDA_VISIBLE_DEVICES=$GPU_INDEX colmap feature_extractor \
#     --database_path "$DATABASE_PATH" \
#     --image_path "$IMAGE_PATH" \
#     --ImageReader.camera_model SIMPLE_RADIAL \
#     --ImageReader.camera_params "$CAMERA_PARAMS" \
#     --ImageReader.single_camera 1

# echo "Importing and matching specified pairs from pairs.txt..."
# CUDA_VISIBLE_DEVICES=$GPU_INDEX colmap matches_importer \
#     --database_path "$DATABASE_PATH" \
#     --match_list_path "$WORKSPACE_PATH/pairs.txt" \
#     --match_type pairs

echo "Running sparse reconstruction (mapping) with fixed intrinsics..."
CUDA_VISIBLE_DEVICES=$GPU_INDEX glomap mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$WORKSPACE_PATH/sparse" \
    # --Mapper.num_threads -1
    # --Mapper.ba_refine_focal_length 0 \
    # --Mapper.ba_refine_principal_point 0 \
    # --Mapper.ba_refine_extra_params 0

echo "WORKSPACE_PATH = $WORKSPACE_PATH"
echo "Script finished."