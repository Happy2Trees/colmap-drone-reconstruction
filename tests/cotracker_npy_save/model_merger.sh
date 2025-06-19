# # #!/usr/bin/env bash

# # BASE=/home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2_processed_1024x576_24_6_2000
# # SPARSE=$BASE/sparse
# # MERGED=$SPARSE/merged

# # colmap model_merger \
# #   --input_path $SPARSE \
# #   --output_path $MERGED
# #!/usr/bin/env bash

# BASE=/home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2_processed_1024x576_24_6_2000
# SPARSE=$BASE/sparse
# OUT=$SPARSE/merged

# mkdir -p $SPARSE/tmp

# # Step 1: merge 0 and 1 → tmp/1
# colmap model_merger \
#   --input_path1 $SPARSE/0 \
#   --input_path2 $SPARSE/1 \
#   --output_path $SPARSE/tmp/1

# # Step 2: merge tmp/1 and 2 → tmp/2
# colmap model_merger \
#   --input_path1 $SPARSE/tmp/1 \
#   --input_path2 $SPARSE/2 \
#   --output_path $SPARSE/tmp/2

# # Step 3: merge tmp/2 and 3 → tmp/3
# colmap model_merger \
#   --input_path1 $SPARSE/tmp/2 \
#   --input_path2 $SPARSE/3 \
#   --output_path $SPARSE/tmp/3

# # ... 계속해서
# colmap model_merger --input_path1 $SPARSE/tmp/3 --input_path2 $SPARSE/4 --output_path $SPARSE/tmp/4
# colmap model_merger --input_path1 $SPARSE/tmp/4 --input_path2 $SPARSE/5 --output_path $SPARSE/tmp/5
# colmap model_merger --input_path1 $SPARSE/tmp/5 --input_path2 $SPARSE/6 --output_path $SPARSE/tmp/6
# colmap model_merger --input_path1 $SPARSE/tmp/6 --input_path2 $SPARSE/7 --output_path $SPARSE/tmp/7
# colmap model_merger --input_path1 $SPARSE/tmp/7 --input_path2 $SPARSE/8 --output_path $SPARSE/tmp/8
# colmap model_merger --input_path1 $SPARSE/tmp/8 --input_path2 $SPARSE/9 --output_path $OUT

BASE1 = "/home/sanggyun/hdseo/colmap-drone-reconstruction/data/window1_processed_1024x576"
BASE2 = "/home/sanggyun/hdseo/colmap-drone-reconstruction/data/window2_processed_1024x576"

colmap model_merger \
  --input_path1 /home/sanggyun/hdseo/colmap-drone-reconstruction/data/window1_processed_1024x576/sparse/0 \
  --input_path2 /home/sanggyun/hdseo/colmap-drone-reconstruction/data/window2_processed_1024x576/sparse/0 \
  --output_path /home/sanggyun/hdseo/colmap-drone-reconstruction/data/window2_processed_1024x576/sparse/merged

#!/bin/bash

# BASE_PATH="/home/sanggyun/hdseo/colmap-drone-reconstruction/data/3x_section2_processed_1024x576_24_6_2000/sparse"
# MERGE_BASE="$BASE_PATH/merged_pairs"
# mkdir -p "$MERGE_BASE"

# # 모델 리스트 가져오기 (sparse/ 디렉토리 안의 숫자 폴더들만)
# models=()
# for d in "$BASE_PATH"/*; do
#     if [[ -d "$d" && "$(basename "$d")" =~ ^[0-9]+$ ]]; then
#         models+=("$(basename "$d")")
#     fi
# done

# # 모든 쌍 조합에 대해 병합 시도
# for ((i=0; i<${#models[@]}; i++)); do
#     for ((j=i+1; j<${#models[@]}; j++)); do
#         m1="${models[$i]}"
#         m2="${models[$j]}"
#         OUTPUT_PATH="$MERGE_BASE/${m1}_${m2}"
#         mkdir -p "$OUTPUT_PATH"

#         echo "Merging sparse/$m1 and sparse/$m2 into $OUTPUT_PATH"
#         colmap model_merger \
#             --input_path1 "$BASE_PATH/$m1" \
#             --input_path2 "$BASE_PATH/$m2" \
#             --output_path "$OUTPUT_PATH"

#         echo "Done merging $m1 and $m2"
#         echo
#     done
# done
