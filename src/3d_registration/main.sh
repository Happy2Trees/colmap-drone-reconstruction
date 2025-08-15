# python -m src.3d_registration.main \
#   --json_dir /path/to/detect/json \
#   --model_dir /home/sanggyun/hdseo/colmap-drone-reconstruction/tests/window_sift/refine/section2_7x_window_sift_48_12_30fps_better/sparse/1 \
#   --exp_dir experiments/my_exp \
#   --xyz_npy /hdd2/0321_block_drone_video/colmap/data/measurement_xyz.npy \
#   --candidate_txt /hdd2/0321_block_drone_video/colmap/data/candidate_list.txt


python -m main \
  --skip_tracking --skip_triangulation \
  --reuse_triangulated_ply /home/sanggyun/hdseo/colmap-drone-reconstruction/src/3d_registration/results/triangulated_ransac_better_points.ply \
  --xyz_npy /home/sanggyun/hdseo/colmap-drone-reconstruction/src/3d_registration/measurement_xyz.npy \
  --candidate_txt /home/sanggyun/hdseo/colmap-drone-reconstruction/src/3d_registration/candidate_list.txt \
  --eval_methods "L2,huber,cauchy" \
  --eval_with_scale "true,false" \
  --voxel_size 0.05 \
  --aligned_ply_out outputs/aligned_source.ply \
  --reg_debug_dir outputs/register_debug
#   --gt_order "23,22,25,26,2,20,19,1,21" \