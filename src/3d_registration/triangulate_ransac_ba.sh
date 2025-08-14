python triangulate_ransac_ba.py \
  --model_dir /home/sanggyun/hdseo/colmap-drone-reconstruction/tests/window_sift/refine/section2_7x_window_sift_48_12_30fps_better/sparse/1 \
  --csv /home/sanggyun/hdseo/colmap-drone-reconstruction/src/3d_registration/tracks_modifiy.csv \
  --out_prefix output/triangulated_ransac_better \
  --ransac_thresh 3.0 \
  --ransac_iters 400 \
  --min_inliers 3  \
  --do_ba