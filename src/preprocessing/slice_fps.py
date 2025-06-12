import os
import shutil
import argparse
import math
from pathlib import Path
from typing import Optional, Union
from natsort import natsorted # Use natsorted for natural sorting of filenames

def slice_fps_core(source_dir: Union[str, Path], 
                   output_dir: Union[str, Path],
                   target_fps: Optional[int] = None, 
                   source_fps: Optional[int] = None,
                   interval: Optional[int] = None) -> int:
    """
    Samples images from source_dir and copies them to output_dir.
    
    Args:
        source_dir: The directory containing the original images.
        output_dir: The directory where sampled images will be copied.
        target_fps: The desired frames per second for sampling.
        source_fps: The frames per second of the source images.
        interval: Direct interval sampling (e.g., 6 = every 6th frame).
                 If provided, overrides fps-based calculation.
    
    Returns:
        Number of frames sampled, or -1 on error.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    if not source_dir.is_dir():
        print(f"오류: 소스 디렉토리 '{source_dir}'를 찾을 수 없습니다.")
        return -1
    
    # Calculate sampling interval
    if interval is not None:
        # Direct interval mode
        if interval <= 0:
            print("오류: interval은 0보다 커야 합니다.")
            return -1
        sampling_interval = float(interval)
    else:
        # FPS-based mode
        if target_fps is None or source_fps is None:
            print("오류: interval이 제공되지 않은 경우 target_fps와 source_fps가 필요합니다.")
            return -1
        if target_fps <= 0 or source_fps <= 0:
            print("오류: fps 값은 0보다 커야 합니다.")
            return -1
        if target_fps > source_fps:
            print("오류: target_fps가 source_fps보다 클 수 없습니다.")
            return -1
        sampling_interval = source_fps / target_fps
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List and sort image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    try:
        all_files = [f for f in source_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in image_extensions]
        # Use natsorted for correct numerical sorting
        image_files = natsorted(all_files, key=lambda x: x.name)
    except OSError as e:
        print(f"오류: '{source_dir}' 디렉토리의 파일 목록을 가져오는 중 오류 발생: {e}")
        return -1
    
    if not image_files:
        print(f"경고: '{source_dir}' 디렉토리에서 이미지 파일을 찾을 수 없습니다.")
        return 0
    
    # Sample and copy files
    copied_count = 0
    last_selected_index = -float('inf')
    
    for i, filepath in enumerate(image_files):
        # Select frame based on interval
        target_index = round(copied_count * sampling_interval)
        
        if i >= target_index and i > last_selected_index:
            # Construct paths
            target_path = output_dir / filepath.name
            
            # Copy the file
            try:
                shutil.copy2(str(filepath), str(target_path))
                copied_count += 1
                last_selected_index = i
            except Exception as e:
                print(f"오류: '{filepath.name}' 파일을 복사하는 중 오류 발생: {e}")
    
    if interval is not None:
        print(f"Interval {interval}: {len(image_files)}개 중 {copied_count}개 이미지 샘플링 완료")
    else:
        print(f"{source_fps}fps → {target_fps}fps: {len(image_files)}개 중 {copied_count}개 이미지 샘플링 완료")
    
    return copied_count

def slice_fps(source_dir, target_fps, source_fps=60):
    """
    Legacy function for backward compatibility.
    Samples images from source_dir at target_fps and copies them to a '_sliced' subdirectory.

    Args:
        source_dir (str): The directory containing the original 60fps images.
        target_fps (int): The desired frames per second for sampling (e.g., 10 or 6).
        source_fps (int): The frames per second of the source images (default: 60).
    """
    if not os.path.isdir(source_dir):
        print(f"오류: 소스 디렉토리 '{source_dir}'를 찾을 수 없습니다.")
        return

    # Create the target directory
    parent_dir = os.path.dirname(source_dir)
    base_name = os.path.basename(source_dir)
    target_dir = os.path.join(parent_dir, base_name + "_sliced")
    
    # Use the new core function
    result = slice_fps_core(source_dir, target_dir, target_fps=target_fps, source_fps=source_fps)
    
    if result > 0:
        print(f"'{target_dir}' 디렉토리에 {result}개의 이미지를 복사했습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample images from a directory at a target FPS or interval")
    parser.add_argument("source_directory", help="Source directory containing images")
    parser.add_argument("--target_fps", type=int, help="Target frames per second")
    parser.add_argument("--source_fps", type=int, default=60, help="Source frames per second (default: 60)")
    parser.add_argument("--interval", type=int, help="Sample every N-th frame (overrides fps settings)")
    parser.add_argument("--output_dir", type=str, help="Output directory (default: source_sliced)")
    
    args = parser.parse_args()
    
    # --- Dependency Check ---
    try:
        import natsort
    except ImportError:
        print("오류: 'natsort' 라이브러리가 필요합니다. 설치해주세요.")
        print("pip install natsort")
        exit(1)
    
    # Validate arguments
    if args.interval is None and args.target_fps is None:
        parser.error("Either --interval or --target_fps must be specified")
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Legacy behavior
        parent_dir = os.path.dirname(args.source_directory)
        base_name = os.path.basename(args.source_directory)
        if args.interval:
            output_dir = os.path.join(parent_dir, f"{base_name}_interval{args.interval}")
        else:
            output_dir = os.path.join(parent_dir, f"{base_name}_sliced")
    
    # Call the core function
    result = slice_fps_core(
        args.source_directory, 
        output_dir,
        target_fps=args.target_fps,
        source_fps=args.source_fps,
        interval=args.interval
    )
    
    if result < 0:
        exit(1)
