import os
from pathlib import Path
from itertools import combinations
import argparse
import re

"""
Generate unique image pairs using linear sliding windows plus minimal loop-closure.

Pairs are saved to a text file in the format "image1 image2", where image names are basenames
matching COLMAP's image names in the database (no directory prefix).
"""


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate pairs with linear windows and add minimal loop-closure (retrieval-agnostic)."
        )
    )
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images.')
    parser.add_argument('--window_size', type=int, default=80, help='Size of the sliding window.')
    parser.add_argument('--stride', type=int, default=40, help='Stride for the sliding window.')
    parser.add_argument('--output_path', type=str, default='pairs.txt', help='Output file path for the pairs.')
    parser.add_argument('--print_windows', action='store_true', help='Print window starts and coverage for debugging')
    # Pure circular windows: no NetVLAD dependency here

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    window_size = max(1, int(args.window_size))
    stride = max(1, int(args.stride))
    output_path = args.output_path
    # No retrieval file used in this script

    # List images (basenames only) with natural sorting
    image_list = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'))]
    image_list.sort(key=_natural_key)

    if len(image_list) < 2:
        # Not enough images to form pairs
        with open(output_path, 'w') as f:
            pass
        print(f"No pairs generated (found {len(image_list)} images). Wrote empty file to {output_path}")
        return

    n = len(image_list)
    w = min(window_size, n)

    pair_set = set()

    # Linear windows + minimal loop-closure:
    # 1) generate non-wrapping windows: starts = [0, stride, ..., <= n-w]
    # 2) add wrapped starts only while they fall in [0, stride)
    #    e.g., n=10, stride=4 -> starts: 0,4,8 plus wrapped 2, but not 6
    starts = list(range(0, max(n - w + 1, 1), stride))
    pre_len = len(starts)
    if not starts:
        starts = [0]
    last_pre = starts[-1]

    # Track normalized starts to avoid duplicates across modulo-n wrap.
    norm_seen = set(s % n for s in starts)
    pre_count = len(norm_seen)
    wrapped_added = 0

    # Start-based wrap: continue stepping forward and only add wrapped windows
    # whose start modulo n falls into the first stride segment [0, stride).
    # This yields sequences like: 0, s, 2s, ..., (wrap) r where r < s, avoiding
    # additional starts such as r+s that move beyond the first segment.
    # Two-phase wrap extension to maximize continuity by start index:
    # Phase A) add pre-wrap starts that straddle the end (cur < n and cur + w > n)
    # Phase B) after crossing n, keep adding while the first index stays before
    #          the first window's last (start_mod <= first_last_mod).
    cur = last_pre + stride
    first_last_mod = (w - 1) % n

    # Phase A: straddling windows near the end
    while cur < n and (cur + w) > n:
        mod = cur % n
        if mod not in norm_seen:
            starts.append(cur)
            norm_seen.add(mod)
            wrapped_added += 1
        cur += stride

    # Phase B: wrapped windows whose first is <= first_last_mod (first-based cutoff)
    while True:
        if cur < n:
            cur += stride
            continue
        mod = cur % n
        if mod <= first_last_mod and mod not in norm_seen:
            starts.append(cur)
            norm_seen.add(mod)
            wrapped_added += 1
            cur += stride
            continue
        break

    for start in starts:
        idxs = [(start + k) % n for k in range(w)]
        window = [image_list[i] for i in idxs]
        for img1, img2 in combinations(window, 2):
            pair = tuple(sorted((img1, img2)))
            pair_set.add(pair)

    # Debug: print window layout if requested
    if args.print_windows:
        pre_starts_abs = starts[:pre_len]
        wrap_starts_abs = starts[pre_len:]
        pre_starts_mod = [s % n for s in pre_starts_abs]
        wrap_starts_mod = [s % n for s in wrap_starts_abs]
        print(f"[debug] n={n}, w={w}, stride={stride}, images={len(image_list)}")
        print(f"[debug] cutoff_first_mod={ (w - 1) % n } cutoff_first_name={ image_list[(w - 1) % n] }")
        print(f"[debug] pre_starts({len(pre_starts_abs)}): {pre_starts_mod}")
        print(f"[debug] wrap_starts({len(wrap_starts_abs)}): {wrap_starts_mod}")
        for i, s in enumerate(starts):
            m = s % n
            first = image_list[m]
            last = image_list[(m + w - 1) % n]
            wraps = (m + w) > n
            typ = 'pre' if i < pre_len else 'wrap'
            print(f"[debug] win {i:03d}: start_mod={m:>5} type={typ} wraps={int(wraps)} -> first={first}, last={last}")

    # Retrieval-agnostic: no extra pairs added here

    with open(output_path, 'w') as f:
        for img1, img2 in sorted(pair_set):
            f.write(f"{img1} {img2}\n")

    num_pre = pre_count
    print(f"Generated {len(pair_set)} unique pairs in {output_path} (pre: {num_pre}, wrap: {wrapped_added})")


if __name__ == '__main__':
    main()
