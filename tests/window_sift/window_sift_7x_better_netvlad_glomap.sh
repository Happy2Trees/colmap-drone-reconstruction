#!/usr/bin/env bash
GPU_INDEX=0,1,2,3
WINDOW_SIZE=48
STRIDE=12
# IMAGE_PATH="/home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2/images"
IMAGE_PATH="data/sec2_7x"
# Build a NetVLAD-centered subset under data/, then use it for reconstruction
NEW_IMAGE_PATH="data/sec2_7x_netvlad_subset_${WINDOW_SIZE}"
WORKSPACE_PATH="outputs/workspaces/section2_7x_window_sift_${WINDOW_SIZE}_${STRIDE}_netvlad_glomap_better"
DATABASE_PATH="$WORKSPACE_PATH/database.db"

mkdir -p "$WORKSPACE_PATH"
mkdir -p "$WORKSPACE_PATH/sparse"

echo "Preparing NetVLAD-centered subset at $NEW_IMAGE_PATH ..."
python tests/window_sift/build_subset_from_netvlad.py \
    --image_dir "$IMAGE_PATH" \
    --retrieval_result_path "image_retrival_result.txt" \
    --output_dir "$NEW_IMAGE_PATH" \
    --overwrite

# Use the subset as the working image path
IMAGE_PATH="$NEW_IMAGE_PATH"

python tests/window_sift/generate_pairs_loop.py \
    --image_dir "$IMAGE_PATH" \
    --window_size "$WINDOW_SIZE" \
    --stride "$STRIDE" \
    --print_windows \
    --output_path "$WORKSPACE_PATH/pairs.txt" | tee "$WORKSPACE_PATH/window_debug.txt"
# Camera intrinsics (f, cx, cy, k)
# Extracted from intrinsic/x3/K.txt and intrinsic/x3/dist.txt
CAMERA_PARAMS="19872.9350245,2123.585064,1499.441091,0.0"

echo "Running feature extraction with provided intrinsics..."
CUDA_VISIBLE_DEVICES=$GPU_INDEX colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.camera_params "$CAMERA_PARAMS" \
    --ImageReader.single_camera 1

echo "Importing and matching specified pairs from pairs.txt..."
CUDA_VISIBLE_DEVICES=$GPU_INDEX colmap matches_importer \
    --database_path "$DATABASE_PATH" \
    --match_list_path "$WORKSPACE_PATH/pairs.txt" \
    --match_type pairs

echo "Running sparse reconstruction (mapping)..."
CUDA_VISIBLE_DEVICES=$GPU_INDEX glomap mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$WORKSPACE_PATH/sparse"

echo "WORKSPACE_PATH = $WORKSPACE_PATH"
echo "Script finished."
