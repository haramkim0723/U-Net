import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.morphology import skeletonize
import pandas as pd

# 이미지 로드
image_path = "D:/New folder.cancelled/HuggingFaceLLM/new_data/data/masks/20241110_150755_stem_0_mask.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 윤곽선 검출 (Contour Detection)
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 🔹 면적 기준 정렬
main_contour = contours[0]  # 🔹 가장 큰 윤곽선 선택

# 윤곽선 내부를 채운 마스크 생성
mask = np.zeros_like(image)
cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)  # 🔹 thickness 인자 올바르게 수정

# 닫힘 연산 (Closing) 적용하여 끊어진 선 연결
kernel = np.ones((5, 5), np.uint8)
closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# 작은 노이즈 제거
denoised_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=2)

# 배경 반전 (Skeletonization 전처리)
corrected_binary_mask = 255 - denoised_mask

# Skeletonization 수행
binary_mask = corrected_binary_mask // 255
skeleton_corrected = skeletonize(binary_mask) * 255

# Skeleton 위에 점 300개 균일 배치
skeleton_points = np.column_stack(np.where(skeleton_corrected > 0))
num_points = min(300, len(skeleton_points))
indices = np.linspace(0, len(skeleton_points) - 1, num=num_points, dtype=int)
selected_points = skeleton_points[indices]

# 윤곽선과 점 사이의 최단 거리 계산
formatted_contours = [contour.astype(np.float32) for contour in contours]
distances = []
for y, x in selected_points:
    min_dist = np.min([cv2.pointPolygonTest(contour, (float(x), float(y)), True) for contour in formatted_contours])
    distances.append(abs(min_dist))  # 🔹 절대값 적용하여 음수 거리 제거

distances = np.array(distances)

# 필터링 (74% 기준: 평균 거리의 74% 이하 제거)
mean_distance = np.mean(distances)
threshold = mean_distance * 0.74
filtered_points = selected_points[distances >= threshold]
filtered_distances = distances[distances >= threshold]
new_mean_distance = np.mean(filtered_distances)

# 스켈레톤 이미지에 필터링 결과 표시
skeleton_with_contours = cv2.cvtColor(skeleton_corrected.astype(np.uint8), cv2.COLOR_GRAY2BGR)
cv2.drawContours(skeleton_with_contours, contours, -1, (0, 255, 0), 1)  # 🔹 초록색 윤곽선 추가
for y, x in filtered_points:
    cv2.circle(skeleton_with_contours, (x, y), 2, (0, 0, 255), -1)  # 🔹 빨간색 점 찍기

#  결과 시각화
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(cv2.cvtColor(corrected_binary_mask, cv2.COLOR_BGR2RGB))
ax[0].set_title("윤곽선 추출")
ax[0].axis("off")

ax[1].imshow(skeleton_corrected, cmap="gray")
ax[1].set_title("2Skeletonization")
ax[1].axis("off")

ax[2].imshow(cv2.cvtColor(skeleton_with_contours, cv2.COLOR_BGR2RGB))
ax[2].set_title("윤곽선 + 필터링된 점들")
ax[2].axis("off")

plt.show()

#  최종 줄기 굵기(Stem Thickness) 계산 및 출력
stem_thickness = 2 * new_mean_distance
print(f"최종 평균 거리: {new_mean_distance:.2f} 픽셀")
print(f"추정된 굵기(Stem Thickness): {stem_thickness:.2f} 픽셀")
