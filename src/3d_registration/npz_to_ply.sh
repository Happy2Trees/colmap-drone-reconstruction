python npz_to_ply.py \
  --npz /home/sanggyun/hdseo/colmap-drone-reconstruction/src/3d_registration/output/triangulated_ransac_better_ba_results.npz \
  --out output/triangulated_ba_better_points.ply \
  --color id
# # 흰색 단색
# python npz_to_ply.py --npz triangulated_ba_results.npz --out points_white.ply --color white
# # z(깊이) 그레이스케일
# python npz_to_ply.py --npz triangulated_ba_results.npz --out points_zgray.ply --color z
