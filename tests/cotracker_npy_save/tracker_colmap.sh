#!/usr/bin/env bash
BASE="/home/sanggyun/hdseo/colmap-drone-reconstruction/data/window3_processed_1024x576"
# Run COLMAP sparse reconstruction using CoTracker-generated DB

#/home/sanggyun/hdseo/colmap-drone-reconstruction/data/sec2_4_processed_1024x576
#/home/sanggyun/hdseo/colmap-drone-reconstruction/data/window1_processed_1024x576

colmap mapper \
  --database_path   $BASE/outputs/x3_section2_cotracker_sift.db \
  --image_path      $BASE/images \
  --output_path     $BASE/sparse \
  --Mapper.num_threads 8
  # --Mapper.min_num_matches 2 \
  # --Mapper.tri_min_angle 0.5 \
  # --Mapper.multiple_models 0 \
  # --Mapper.tri_ignore_two_view_tracks 0  \
  # --Mapper.tri_complete_max_reproj_error 6.0 \
  # --Mapper.ba_local_max_num_iterations 5 \
  # --Mapper.abs_pose_min_num_inliers 3 \
  # --Mapper.abs_pose_min_inlier_ratio 0.1 \
  # --Mapper.init_min_num_inliers 30 \
  # --Mapper.init_max_error 6 \
  # --Mapper.filter_max_reproj_error 6 \
  # --Mapper.max_reg_trials 10 \
  # --Mapper.ba_global_use_pba 0  \
  # --Mapper.ba_global_max_num_iterations 5 \
  # --Mapper.ba_refine_focal_length 0 \
  # --Mapper.ba_refine_principal_point 0 \
  # --Mapper.ba_refine_extra_params 0 \
