import os
from pathlib import Path
from itertools import combinations
import argparse

"""Generate unique image pairs from a directory of images using a sliding window approach.
This script generates pairs of images from a specified directory using a sliding window technique.
It ensures that each pair is unique and sorted, preventing duplicates.
The pairs are saved to a text file in the format "image1 image2"."""

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Generate unique image pairs from a directory of images using a sliding window approach.')
parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images.')
parser.add_argument('--window_size', type=int, default=80, help='Size of the sliding window.')
parser.add_argument('--stride', type=int, default=40, help='Stride for the sliding window.')
parser.add_argument('--output_path', type=str, default='pairs.txt', help='Output file path for the pairs.')
args = parser.parse_args()

# 파라미터
image_dir = Path(args.image_dir)
window_size = args.window_size
stride = args.stride
output_path = args.output_path

# 이미지 파일 리스트 (정렬 보장)
image_list = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

# 중복 제거를 위한 set
pair_set = set()

# 슬라이딩 윈도우 loop
for start in range(0, len(image_list) - window_size + 1, stride):
    window = image_list[start:start+window_size]
    for img1, img2 in combinations(window, 2):
        # 정렬된 order로 저장해서 중복 제거
        pair = tuple(sorted([img1, img2]))
        pair_set.add(pair)

# 결과 저장
with open(output_path, 'w') as f:
    for img1, img2 in sorted(pair_set):
        f.write(f"{img1} {img2}\n")

print(f"Generated {len(pair_set)} unique pairs in {output_path}")
