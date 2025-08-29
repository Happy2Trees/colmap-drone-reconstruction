import argparse
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(image_dir: Path) -> List[str]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    names = [f.name for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in exts]
    names.sort(key=_natural_key)
    return names


def parse_retrieval_result(result_path: Path) -> Optional[Tuple[str, str]]:
    if not result_path.exists():
        return None
    ref_path = None
    best_path = None
    with result_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^\[[^\]]+\]\s+(.*)$", line)
            if not m:
                continue
            path = m.group(1)
            if '기준' in line:
                ref_path = path
            elif '유사' in line:
                best_path = path
    if ref_path and best_path:
        return (os.path.basename(ref_path), os.path.basename(best_path))
    return None



def copy_subset(src_dir: Path, dst_dir: Path, names: List[str], overwrite: bool = False):
    if dst_dir.exists():
        if any(dst_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(f"Destination {dst_dir} exists and is not empty. Use --overwrite to replace.")
            # clear directory contents
            for p in dst_dir.iterdir():
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
    else:
        dst_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        src = src_dir / name
        if not src.exists():
            print(f"[warn] missing source file: {src}")
            continue
        shutil.copy2(src, dst_dir / name)


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Build a subset by copying all images from the NetVLAD start to end (inclusive).'
        )
    )
    parser.add_argument('--image_dir', required=True, help='Source image directory')
    parser.add_argument('--retrieval_result_path', default='image_retrival_result.txt', help='Path to NetVLAD result file')
    parser.add_argument('--output_dir', required=True, help='Destination directory to copy the subset images')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite destination directory contents if exists')

    args = parser.parse_args()

    src_dir = Path(args.image_dir)
    dst_dir = Path(args.output_dir)
    retrieval_path = Path(args.retrieval_result_path) if args.retrieval_result_path else None

    names = list_images(src_dir)
    if len(names) == 0:
        raise FileNotFoundError(f"No images found in {src_dir}")

    parsed = parse_retrieval_result(retrieval_path) if retrieval_path else None
    if parsed is None:
        raise FileNotFoundError('Failed to parse retrieval_result_path. Ensure image_retrival_result.txt exists.')
    ref_name, best_name = parsed

    try:
        ref_idx = names.index(ref_name)
        best_idx = names.index(best_name)
    except ValueError as e:
        raise ValueError('Names from retrieval_result not found among images in image_dir') from e

    # Copy contiguous range from start to end (inclusive) in natural-sorted list
    i0, i1 = sorted((ref_idx, best_idx))
    selected = names[i0:i1 + 1]

    copy_subset(src_dir, dst_dir, selected, overwrite=args.overwrite)
    print(f"From {names[i0]} to {names[i1]} -> Copied {len(selected)} images to {dst_dir}")


if __name__ == '__main__':
    main()
