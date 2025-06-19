# colmap model_converter \
#     --input_path  /home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2_processed_1024x576_24_6_2000/sparse/0\
#     --output_path /home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2_processed_1024x576_24_6_2000/sparse/0/points.ply \
#     --output_type PLY

#/home/sanggyun/hdseo/colmap-drone-reconstruction/data/sec2_sliced_1_processed_1024x576
#/home/sanggyun/hdseo/colmap-drone-reconstruction/data/window4_processed_1024x576
BASE=/home/sanggyun/hdseo/colmap-drone-reconstruction/data/window4
for m in "$BASE"/sparse/*; do
  name=$(basename "$m")
  colmap model_converter \
      --input_path  "$m" \
      --output_path "$BASE/sparse/$name/points.ply" \
      --output_type PLY
done


# colmap model_converter \
#   --input_path "$BASE/sparse/0" \
#   --output_path "$BASE/sparse-txt/0" \
#   --output_type TXT
