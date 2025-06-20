#!/usr/bin/env bash
DATASET_PATH="/home/sanggyun/hdseo/colmap-drone-reconstruction/data/window4"
# Run COLMAP sparse reconstruction using CoTracker-generated DB


colmap feature_extractor \
  --database_path $DATASET_PATH/database.db \
  --image_path $DATASET_PATH/images

colmap exhaustive_matcher \
  --database_path $DATASET_PATH/database.db

mkdir $DATASET_PATH/sparse

colmap mapper \
  --database_path $DATASET_PATH/database.db \
  --image_path $DATASET_PATH/images \
  --output_path $DATASET_PATH/sparse \
  --Mapper.num_threads 8

mkdir $DATASET_PATH/dense

colmap image_undistorter \
  --image_path $DATASET_PATH/images \
  --input_path $DATASET_PATH/sparse/0 \
  --output_path $DATASET_PATH/dense \
  --output_type COLMAP \

colmap patch_match_stereo \
  --workspace_path $DATASET_PATH/dense \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
  --workspace_path $DATASET_PATH/dense \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path $DATASET_PATH/dense/fused.ply

# colmap poisson_mesher \
#   --input_path $DATASET_PATH/dense/fused.ply \
#   --output_path $DATASET_PATH/dense/meshed-poisson.ply

# colmap delaunay_mesher \
#   --input_path $DATASET_PATH/dense \
#   --output_path $DATASET_PATH/dense/meshed-delaunay.ply