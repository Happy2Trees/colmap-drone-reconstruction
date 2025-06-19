# colmap image_undistorter \
#     --image_path /home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2_processed_1024x576_24_6_2000/images \
#     --input_path /home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2_processed_1024x576_24_6_2000/sparse/9 \
#     --output_path /home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2_processed_1024x576_24_6_2000/dense9/ \
#     --output_type COLMAP \


# colmap patch_match_stereo \
#     --workspace_path /home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2_processed_1024x576_24_6_2000/dense4 \
#     --workspace_format COLMAP \
#     --PatchMatchStereo.geom_consistency true


# colmap stereo_fusion \
#     --workspace_path /home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2_processed_1024x576_24_6_2000/dense4 \
#     --workspace_format COLMAP \
#     --input_type geometric \
#     --output_path /home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2_processed_1024x576_24_6_2000/dense4/fused.ply

#/home/sanggyun/hdseo/colmap-drone-reconstruction/data/window4_processed_1024x576
#/home/sanggyun/hdseo/colmap-drone-reconstruction/data/sec2_sliced_1_processed_1024x576

BASE="/home/sanggyun/hdseo/colmap-drone-reconstruction/data/window1"
DENSE_BASE="$BASE/dense_2"

colmap image_undistorter \
    --image_path $BASE/images \
    --input_path $BASE/sparse/2 \
    --output_path $DENSE_BASE/ \
    --output_type COLMAP \


colmap patch_match_stereo \
    --workspace_path $DENSE_BASE \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true


colmap stereo_fusion \
    --workspace_path $DENSE_BASE \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DENSE_BASE/fused.ply \



#!/bin/bash

# BASE="/home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2_processed_1024x576_24_2"
# IMAGE_PATH="$BASE/images"
# SPARSE_DIR="$BASE/sparse"
# DENSE_BASE="$BASE/dense"

# for SPARSE_PATH in "$SPARSE_DIR"/*; do
#   if [ -d "$SPARSE_PATH" ]; then
#     NAME=$(basename "$SPARSE_PATH")
#     DENSE_PATH="$DENSE_BASE/$NAME"

#     echo "▶️ Processing sparse model: $NAME"

#     # 1. Undistort images
#     colmap image_undistorter \
#       --image_path "$IMAGE_PATH" \
#       --input_path "$SPARSE_PATH" \
#       --output_path "$DENSE_PATH" \
#       --output_type COLMAP

#     # 2. PatchMatch stereo
#     colmap patch_match_stereo \
#       --workspace_path "$DENSE_PATH" \
#       --workspace_format COLMAP \
#       --PatchMatchStereo.geom_consistency true

#     # 3. Stereo fusion
#     colmap stereo_fusion \
#       --workspace_path "$DENSE_PATH" \
#       --workspace_format COLMAP \
#       --input_type geometric \
#       --output_path "$DENSE_PATH/fused.ply"
    
#     echo "✅ Done: $DENSE_PATH/fused.ply"
#   fi
# done
