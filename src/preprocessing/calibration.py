import cv2
import numpy as np
import os
import glob

import argparse

# 체커보드 설정 (내부 코너 수)
CHECKERBOARD = (7, 10) # 실제 체커보드의 가로, 세로 내부 코너 수로 변경하세요.
SQUARE_SIZE_MM = 25.0 # 체커보드 사각형의 실제 크기 (mm)
FRAME_INTERVAL = 10  # 처리할 프레임 간격 (예: 10은 10프레임마다 1개씩 처리)

# 파일 확장자 (필요에 따라 변경: 'png', 'jpeg' 등)
img_ext = 'jpg'

# Parse command line arguments
parser = argparse.ArgumentParser(description="Perform camera calibration using checkerboard images")
parser.add_argument("image_dir", help="Directory containing calibration images")
parser.add_argument("--output_dir", default="outputs/calibration", help="Output directory for calibration results (default: outputs/calibration)")
args = parser.parse_args()

# 경로 설정
image_dir = args.image_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "intrinsic_parameters.txt")

# 3D 점 준비 (체커보드 좌표)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM # 실제 크기 적용

# 이미지에서 2D 점과 해당 3D 점을 저장할 배열
objpoints = [] # 3D 점 (세계 좌표계)
imgpoints = [] # 2D 점 (이미지 평면)
img_shapes = [] # 코너 검출 성공한 이미지 크기 저장
processed_image_paths = [] # 코너 검출 성공한 이미지 경로 저장

# 이미지 로드 및 코너 찾기
images = glob.glob(os.path.join(image_dir, f'*.{img_ext}'))
if not images:
    print(f"오류: '{image_dir}' 디렉토리에서 '{img_ext}' 확장자의 이미지를 찾을 수 없습니다.")
    exit()

images.sort() # 파일 이름 순으로 정렬하여 일관된 샘플링 보장

print(f"총 {len(images)}개의 이미지를 찾았습니다.")

# 프레임 간격 적용하여 이미지 샘플링
sampled_images = images[::FRAME_INTERVAL]
print(f"프레임 간격 ({FRAME_INTERVAL})을 적용하여 {len(sampled_images)}개의 이미지를 처리합니다. 코너 검출 시작...")

# 샘플링된 이미지에 대해 코너 검출 수행
for fname in sampled_images:
    img = cv2.imread(fname)
    if img is None:
        print(f"경고: '{fname}' 이미지를 로드할 수 없습니다.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_shapes.append(gray.shape[::-1]) # (width, height) 저장

    # 체커보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너가 발견되면, 정밀도 향상 및 저장
    if ret == True:
        print(f" - {os.path.basename(fname)}: 코너 검출 성공")
        objpoints.append(objp)
        # 코너 위치 정밀도 향상
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        processed_image_paths.append(fname) # 성공한 이미지 경로 저장
    else:
        print(f" - {os.path.basename(fname)}: 코너 검출 실패")

if not objpoints:
    print("오류: 샘플링된 이미지 중 어떤 이미지에서도 체커보드 코너를 성공적으로 검출하지 못했습니다.")
    exit()

if not img_shapes:
    print("오류: 코너 검출에 성공한 이미지의 크기 정보를 얻지 못했습니다.")
    exit()

# 모든 이미지가 같은 크기라고 가정하고 첫 번째 이미지 크기 사용
image_size = img_shapes[0]

print(f"총 {len(objpoints)}개의 (샘플링된) 이미지에서 코너를 성공적으로 검출했습니다.")
print("초기 캘리브레이션을 수행합니다...")

# 모든 유효한 프레임으로 초기 캘리브레이션
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

if not ret:
    print("오류: 초기 카메라 캘리브레이션에 실패했습니다.")
    exit()

print("초기 캘리브레이션 완료. 각 이미지의 재투영 오차를 계산합니다...")

# 각 이미지의 재투영 오차 계산
mean_error_per_image = []
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error_per_image.append(error)
    print(f" - 이미지 {os.path.basename(processed_image_paths[i])}: 재투영 오차 = {error:.4f}")

# 재투영 오차 기반 프레임 선택 (예: 평균 오차 + 표준 편차 이하 또는 특정 임계값 사용)
mean_error_all = np.mean(mean_error_per_image)
std_dev_error = np.std(mean_error_per_image)
error_threshold = mean_error_all # 임계값을 평균 오차로 설정 (조정 가능)

print(f"평균 재투영 오차: {mean_error_all:.4f}")
print(f"오차 표준편차: {std_dev_error:.4f}")
print(f"프레임 선택 임계값 (평균 오차 사용): {error_threshold:.4f}")

selected_objpoints = []
selected_imgpoints = []
selected_image_paths = [] # 선택된 이미지 경로 저장

print("재투영 오차 기반 프레임 선택 결과:")
for i, error in enumerate(mean_error_per_image):
    if error <= error_threshold:
        selected_objpoints.append(objpoints[i])
        selected_imgpoints.append(imgpoints[i])
        selected_image_paths.append(processed_image_paths[i]) # 선택된 이미지 경로 저장
        print(f" - 이미지 {os.path.basename(processed_image_paths[i])} 선택됨 (오차: {error:.4f})")
    else:
         print(f" - 이미지 {os.path.basename(processed_image_paths[i])} 제외됨 (오차: {error:.4f})")

if not selected_objpoints:
    print("오류: 선택된 프레임이 없습니다. 임계값을 조정하거나 이미지 품질을 확인하세요.")
    exit()

print(f"총 {len(selected_objpoints)}개의 프레임이 선택되었습니다.")
print("선택된 프레임으로 최종 캘리브레이션을 수행합니다...")

# 선택된 프레임으로 최종 캘리브레이션
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(selected_objpoints, selected_imgpoints, image_size, None, None)

if not ret:
    print("오류: 최종 카메라 캘리브레이션에 실패했습니다.")
    exit()

print("최종 캘리브레이션 완료.")

# 최종 재투영 오차 계산 (두 번째 calibrateCamera에서 반환된 rvecs, tvecs 사용)
mean_error_final = 0
for i in range(len(selected_objpoints)):
    imgpoints2, _ = cv2.projectPoints(selected_objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(selected_imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error_final += error

if selected_objpoints:
    mean_error_final /= len(selected_objpoints)
else:
    mean_error_final = 0 # 또는 오류 처리

print(f"최종 평균 재투영 오차: {mean_error_final:.4f}")

# 결과 저장
print(f"결과를 '{output_file}' 파일에 저장합니다...")
os.makedirs(output_dir, exist_ok=True) # 출력 디렉토리 생성 확인

with open(output_file, 'w') as f:
    f.write("Camera Matrix (mtx):\n")
    np.savetxt(f, mtx, fmt='%10.5f')
    f.write("\nDistortion Coefficients (dist):\n")
    np.savetxt(f, dist, fmt='%10.5f')
    f.write(f"\nMean Reprojection Error: {mean_error_final:.5f}\n")
    f.write(f"\nSelected Frames ({len(selected_image_paths)} frames):\n")
    for img_path in selected_image_paths: # 선택된 이미지 경로 사용
        f.write(f"- {os.path.basename(img_path)}\n")

print("캘리브레이션 및 결과 저장이 완료되었습니다.")
print("\n카메라 행렬 (mtx):")
print(mtx)
print("\n왜곡 계수 (dist):")
print(dist)

