import os
import shutil
import argparse
import math
from natsort import natsorted # Use natsorted for natural sorting of filenames

def slice_fps(source_dir, target_fps, source_fps=60):
    """
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
    # target_dir = os.path.join(source_dir, "_sliced") # 기존 방식: 하위 폴더 생성
    parent_dir = os.path.dirname(source_dir)
    base_name = os.path.basename(source_dir)
    target_dir = os.path.join(parent_dir, base_name + "_sliced") # 수정된 방식: 형제 폴더 생성
    os.makedirs(target_dir, exist_ok=True)
    print(f"'{target_dir}' 디렉토리를 생성했거나 이미 존재합니다.")

    # Calculate the sampling interval
    if target_fps <= 0:
        print("오류: target_fps는 0보다 커야 합니다.")
        return
    if source_fps <= 0:
        print("오류: source_fps는 0보다 커야 합니다.")
        return

    interval = source_fps / target_fps
    if interval < 1:
        print("오류: target_fps가 source_fps보다 클 수 없습니다.")
        return

    print(f"소스 FPS: {source_fps}, 타겟 FPS: {target_fps}, 샘플링 간격: {interval:.2f} (매 {int(round(interval))}번째 프레임 선택)")

    # List and sort image files (assuming common image extensions)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    try:
        all_files = [f for f in os.listdir(source_dir)
                     if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(image_extensions)]
        # Use natsorted for correct numerical sorting (e.g., 1, 2, ..., 10, 11, ...)
        image_files = natsorted(all_files)
    except OSError as e:
        print(f"오류: '{source_dir}' 디렉토리의 파일 목록을 가져오는 중 오류 발생: {e}")
        return


    if not image_files:
        print(f"경고: '{source_dir}' 디렉토리에서 이미지 파일을 찾을 수 없습니다.")
        return

    # Sample and copy files
    copied_count = 0
    last_selected_index = -float('inf') # Keep track of the last *actual* index selected

    for i, filename in enumerate(image_files):
        # Calculate the ideal index based on the interval
        current_ideal_frame_num = i + 1 # Frame numbers are 1-based
        target_frame_num_float = current_ideal_frame_num / interval

        # Select the frame if its index is the closest integer to the ideal target frame number
        # and it hasn't been selected yet due to rounding of previous selections.
        # More robustly: select frame 'i' if floor(i / interval) > floor((i-1) / interval)
        # Or simply select the frame corresponding to the desired time step.

        select_this_frame = False
        # We want to select the frame closest to the time instances t = k / target_fps
        # The current frame time is i / source_fps
        # The target time instances are 0, 1/target_fps, 2/target_fps, ...
        # We select frame i if its time is the closest to *some* target time instance k/target_fps
        # compared to other frames.
        # A simpler way: select frame i if floor(i * target_fps / source_fps) > floor((i-1) * target_fps / source_fps)

        # Let's use the simpler approach: select every 'interval'-th frame using floating point comparison
        # Select frame 0, frame 'interval', frame '2*interval', etc.
        target_index = round(copied_count * interval)

        if i >= target_index and i > last_selected_index:
             # Construct full paths
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)

            # Copy the file
            try:
                shutil.copy2(source_path, target_path) # copy2 preserves metadata
                # print(f"Copied: {filename} (Index: {i})")
                copied_count += 1
                last_selected_index = i
            except Exception as e:
                print(f"오류: '{filename}' 파일을 복사하는 중 오류 발생: {e}")


    print(f"\n총 {len(image_files)}개의 이미지 파일 중 {copied_count}개를 '{target_dir}'로 복사했습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample images from a directory at a target FPS")
    parser.add_argument("source_directory", help="Source directory containing images")
    parser.add_argument("--target_fps", type=int, default=6, help="Target frames per second (default: 6)")
    parser.add_argument("--source_fps", type=int, default=60, help="Source frames per second (default: 60)")
    
    args = parser.parse_args()
    
    # --- Dependency Check ---
    try:
        import natsort
    except ImportError:
        print("오류: 'natsort' 라이브러리가 필요합니다. 설치해주세요.")
        print("pip install natsort")
        exit(1) # Use exit(1) for errors

    # 함수 호출
    slice_fps(args.source_directory, args.target_fps, args.source_fps)
