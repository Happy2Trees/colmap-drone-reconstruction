import cv2
import numpy as np
import os
import glob
import argparse
import sys

# --- 캘리브레이션 설정 ---

# 체커보드 설정 (내부 코너 수)
CHECKERBOARD = (7, 10)  # 실제 체커보드의 가로, 세로 내부 코너 수로 변경하세요.
SQUARE_SIZE_MM = 25.0  # 체커보드 사각형의 실제 크기 (mm)cp
FRAME_INTERVAL = 5     # 처리할 프레임 간격 (더 많은 이미지를 샘플링하도록 조정)

# 파일 확장자 (필요에 따라 변경: 'png', 'jpeg' 등)
IMG_EXT = 'jpg'

# 코너 검출 정밀도 향상을 위한 기준
# 반복을 종료할 시기를 결정하는 기준. (정확도, 최대 반복 횟수)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# --- 명령줄 인자 파싱 ---
def parse_args():
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="카메라 캘리브레이션 성능 개선 버전")
    parser.add_argument("image_dir", help="캘리브레이션 이미지가 포함된 디렉토리 경로")
    parser.add_argument("--output_dir", default="outputs/calibration_results", help="캘리브레이션 결과물 저장 디렉토리 (기본값: outputs/calibration_results)")
    parser.add_argument("--no_visualize", action='store_true', help="결과 시각화 이미지 저장을 비활성화합니다.")
    return parser.parse_args()

def find_chessboard_corners(images, checkerboard_size, criteria):
    """
    이미지 리스트에서 체커보드 코너를 찾습니다.

    Returns:
        objpoints: 3D 월드 좌표계의 점
        imgpoints: 2D 이미지 평면의 점
        processed_images_info: 코너 검출에 성공한 이미지의 정보 (경로, 크기)
    """
    print("체커보드 코너 검출을 시작합니다...")

    # 3D 점 준비 (체커보드 좌표)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_MM  # 실제 크기 적용

    objpoints = []  # 3D 점
    imgpoints = []  # 2D 점
    processed_images_info = [] # 성공한 이미지 정보 (경로, 크기)

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"경고: '{fname}' 이미지를 로드할 수 없습니다. 건너뜁니다.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 개선점: 적응형 스레시홀딩을 이용한 이미지 전처리 (조명이 균일하지 않을 때 도움됨)
        # 주석 처리됨. 필요시 활성화하여 테스트.
        # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                               cv2.THRESH_BINARY, 11, 2)

        # 체커보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + # 적응형 스레시홀드 사용
                                               cv2.CALIB_CB_FAST_CHECK +      # 빠른 검사로 코너 존재 여부 확인
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)  # 이미지 정규화

        if ret:
            print(f" - {os.path.basename(fname)}: 코너 검출 성공")
            objpoints.append(objp)
            # 코너 위치 정밀도 향상
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            processed_images_info.append({'path': fname, 'shape': gray.shape[::-1]})
        else:
            print(f" - {os.path.basename(fname)}: 코너 검출 실패")

    print(f"\n총 {len(objpoints)}개의 이미지에서 코너를 성공적으로 검출했습니다.")
    return objpoints, imgpoints, processed_images_info

def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """모든 이미지에 대한 평균 재투영 오차와 각 이미지의 오차를 계산합니다."""
    total_error = 0
    errors_per_image = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        errors_per_image.append(error)
        total_error += error

    mean_error = total_error / len(objpoints) if objpoints else 0
    return mean_error, errors_per_image

def save_calibration_results(output_dir, mtx, dist, mean_error, selected_paths):
    """캘리브레이션 결과를 텍스트 파일로 저장합니다."""
    output_file = os.path.join(output_dir, "calibration_parameters.txt")
    print(f"\n결과를 '{output_file}' 파일에 저장합니다...")
    with open(output_file, 'w') as f:
        f.write("Camera Matrix (mtx):\n")
        np.savetxt(f, mtx, fmt='%10.6f')
        f.write("\nDistortion Coefficients (dist):\n")
        # k1, k2, p1, p2, k3, k4, k5, k6 순서로 저장될 수 있음
        np.savetxt(f, dist, fmt='%10.6f')
        f.write(f"\nMean Reprojection Error: {mean_error:.6f}\n")
        f.write(f"\nSelected Frames for Final Calibration ({len(selected_paths)} frames):\n")
        for img_path in selected_paths:
            f.write(f"- {os.path.basename(img_path)}\n")
    print("결과 저장이 완료되었습니다.")

def visualize_results(output_dir, mtx, dist, all_images_info, selected_imgpoints, selected_objpoints, rvecs, tvecs):
    """캘리브레이션 결과를 시각화하여 이미지로 저장합니다."""
    print("\n결과 시각화를 시작합니다 (왜곡 보정 및 코너 재투영)...")
    vis_dir = os.path.join(output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    # 선택된 이미지들에 대해 왜곡 보정 및 코너 그리기
    for i, info in enumerate(all_images_info):
        if info['path'] in [p['path'] for p in selected_objpoints]: # 선택된 프레임만 시각화
            img = cv2.imread(info['path'])
            
            # 1. 왜곡 보정 이미지 저장
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w] # 유효한 영역만 잘라내기
            undistorted_path = os.path.join(vis_dir, "undistorted_" + os.path.basename(info['path']))
            cv2.imwrite(undistorted_path, dst)
            
            # 2. 원본 이미지에 검출된 코너와 재투영된 코너 그리기
            # 선택된 프레임 목록에서 현재 이미지의 인덱스를 찾아야 함
            selected_idx = [j for j, p_info in enumerate(selected_objpoints) if p_info['path'] == info['path']][0]
            
            imgpoints2, _ = cv2.projectPoints(selected_objpoints[selected_idx]['obj'], rvecs[selected_idx], tvecs[selected_idx], mtx, dist)
            
            # 원본 코너는 초록색 원으로, 재투영된 코너는 빨간색 십자가로 표시
            for j in range(len(selected_imgpoints[selected_idx])):
                cv2.circle(img, tuple(map(int, selected_imgpoints[selected_idx][j][0])), 5, (0, 255, 0), -1) # 원본(녹색)
                cv2.drawMarker(img, tuple(map(int, imgpoints2[j][0])), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1) # 재투영(빨강)

            reprojected_path = os.path.join(vis_dir, "reprojected_" + os.path.basename(info['path']))
            cv2.imwrite(reprojected_path, img)

    print(f"시각화 결과가 '{vis_dir}'에 저장되었습니다.")


def main():
    """메인 캘리브레이션 프로세스"""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. 이미지 로드 및 샘플링 ---
    images = glob.glob(os.path.join(args.image_dir, f'*.{IMG_EXT}'))
    if not images:
        print(f"오류: '{args.image_dir}' 디렉토리에서 '{IMG_EXT}' 확장자의 이미지를 찾을 수 없습니다.")
        sys.exit(1)

    images.sort()
    print(f"총 {len(images)}개의 이미지를 찾았습니다.")
    
    sampled_images = images[::FRAME_INTERVAL]
    print(f"프레임 간격 ({FRAME_INTERVAL})을 적용하여 {len(sampled_images)}개의 이미지를 처리합니다.")

    # --- 2. 코너 검출 ---
    objpoints, imgpoints, processed_images_info = find_chessboard_corners(sampled_images, CHECKERBOARD, SUBPIX_CRITERIA)

    if len(objpoints) < 5: # 안정적인 결과를 위해 최소 5개 이상의 이미지를 권장
        print(f"오류: 코너 검출에 성공한 이미지가 {len(objpoints)}개로 너무 적습니다. 더 많은 이미지를 사용하거나 체커보드 설정을 확인하세요.")
        sys.exit(1)
        
    image_size = processed_images_info[0]['shape']

    # --- 3. 고성능 캘리브레이션 및 프레임 최적화 ---
    print("\n초기 캘리브레이션을 수행합니다...")

    # 개선점: CALIB_RATIONAL_MODEL 플래그 추가하여 왜곡 계수 8개(k1-k6, p1, p2) 사용
    # 이는 복잡한 렌즈 왜곡을 더 잘 모델링할 수 있음
    calibration_flags = cv2.CALIB_RATIONAL_MODEL

    # 모든 유효 프레임으로 캘리브레이션 수행
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None, flags=calibration_flags)

    if not ret:
        print("오류: 카메라 캘리브레이션에 실패했습니다.")
        sys.exit(1)

    print("초기 캘리브레이션 완료. 재투영 오차 기반 프레임 최적화를 시작합니다...")
    
    # 각 이미지의 재투영 오차 계산
    mean_error, errors_per_image = calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    
    print(f"초기 평균 재투영 오차: {mean_error:.6f}")

    # 개선점: 재투영 오차가 큰 프레임을 순차적으로 제거하는 고도화된 프레임 선택
    # 현재 데이터 복사본 생성
    current_objpoints = objpoints[:]
    current_imgpoints = imgpoints[:]
    current_paths = [info['path'] for info in processed_images_info]
    
    # 제거할 프레임이 남아있는 동안 반복
    while len(current_objpoints) > max(5, int(len(objpoints) * 0.7)): # 최소 5개, 또는 초기 프레임의 70%는 남김
        # 현재 프레임셋으로 캘리브레이션
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(current_objpoints, current_imgpoints, image_size, None, None, flags=calibration_flags)
        
        # 재투영 오차 계산
        mean_err, errs = calculate_reprojection_error(current_objpoints, current_imgpoints, rvecs, tvecs, mtx, dist)
        
        # 오차가 가장 큰 프레임의 인덱스 찾기
        worst_frame_idx = np.argmax(errs)
        
        # 임계값(예: 전체 평균의 1.5배)보다 큰 경우에만 제거
        if errs[worst_frame_idx] > mean_err * 1.5:
            print(f" - 오차가 가장 큰 프레임 제거: {os.path.basename(current_paths[worst_frame_idx])} (오차: {errs[worst_frame_idx]:.6f})")
            current_objpoints.pop(worst_frame_idx)
            current_imgpoints.pop(worst_frame_idx)
            current_paths.pop(worst_frame_idx)
        else:
            # 더 이상 제거할 이상치가 없다고 판단
            print("프레임 최적화 완료: 오차가 안정화되었습니다.")
            break
    else:
        print("프레임 최적화 완료: 최소 프레임 수에 도달했습니다.")


    # --- 4. 최종 캘리브레이션 ---
    print(f"\n총 {len(current_objpoints)}개의 최적화된 프레임으로 최종 캘리브레이션을 수행합니다...")
    
    # 개선점: 최종 단계에서 주점(Principal Point)을 고정하여 안정성을 높이는 시도
    # use_intrinsic_guess=True 와 함께 mtx, dist 를 전달
    final_flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_USE_INTRINSIC_GUESS
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(current_objpoints, current_imgpoints, image_size, mtx, dist, flags=final_flags)

    if not ret:
        print("오류: 최종 카메라 캘리브레이션에 실패했습니다.")
        sys.exit(1)
        
    print("최종 캘리브레이션 완료.")
    
    # 최종 재투영 오차 계산
    final_mean_error, _ = calculate_reprojection_error(current_objpoints, current_imgpoints, rvecs, tvecs, mtx, dist)
    print(f"\n최종 평균 재투영 오차: {final_mean_error:.6f}")

    # --- 5. 결과 저장 및 시각화 ---
    save_calibration_results(args.output_dir, mtx, dist, final_mean_error, current_paths)

    if not args.no_visualize:
        # 시각화를 위해 선택된 프레임 정보 재구성
        selected_objpoints_info = [{'path': p, 'obj': o} for p, o in zip(current_paths, current_objpoints)]
        visualize_results(args.output_dir, mtx, dist, processed_images_info, current_imgpoints, selected_objpoints_info, rvecs, tvecs)

    print("\n--- 최종 캘리브레이션 결과 ---")
    print("\n카메라 행렬 (mtx):")
    print(mtx)
    print("\n왜곡 계수 (dist):")
    print(dist)

if __name__ == '__main__':
    main()