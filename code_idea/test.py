import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# 이미지 로드
image_path = "D:/ss/stem_1.png"
test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 닫힘 연산 적용
kernel = np.ones((5, 5), np.uint8)
closed_mask = cv2.morphologyEx(test_image, cv2.MORPH_CLOSE, kernel, iterations=4)

# 노이즈 제거 (열림 연산)
denoised_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=2)

# 윤곽선 검출 및 내부 채우기
contours, _ = cv2.findContours(denoised_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Detected contours: {len(contours)}")
mask_filled = np.zeros_like(test_image)
if len(contours) > 0:
    main_contour = max(contours, key=cv2.contourArea)  # 가장 큰 윤곽선 선택
    cv2.drawContours(mask_filled, [main_contour], -1, 255, thickness=cv2.FILLED)

# 객체 내부가 흰색(255)인지 확인
if np.mean(mask_filled) < 128:
    mask_filled = cv2.bitwise_not(mask_filled)

# 이진화 및 골격화
binary_mask = (mask_filled > 0).astype(np.uint8) * 255
skeleton_corrected = skeletonize(binary_mask // 255) * 255

# Skeleton 위의 모든 점 좌표 호출
skeleton_points = np.column_stack(np.where(skeleton_corrected > 0))
num_random_points = min(300, len(skeleton_points))
selected_points_random_uniform = skeleton_points[np.random.choice(len(skeleton_points), num_random_points, replace=False)]

# 윤곽선과 점 사이의 거리 계산
distances = [cv2.pointPolygonTest(main_contour.astype(np.float32), (float(x), float(y)), True) for y, x in selected_points_random_uniform]
distances = np.array([abs(d) for d in distances])

# 필터링 (74% 기준)
mean_distance = np.mean(distances)
threshold = mean_distance * 0.74
filtered_points = selected_points_random_uniform[distances >= threshold]

# 시각화
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.ravel()

axes[0].imshow(test_image, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(closed_mask, cmap="gray")
axes[1].set_title("After Closing Operation")
axes[1].axis("off")

axes[2].imshow(denoised_mask, cmap="gray")
axes[2].set_title("After Noise Removal")
axes[2].axis("off")

axes[3].imshow(mask_filled, cmap="gray")
axes[3].set_title("Filled Mask")
axes[3].axis("off")

axes[4].imshow(binary_mask, cmap="gray")
axes[4].set_title("Binary Mask")
axes[4].axis("off")

axes[5].imshow(skeleton_corrected, cmap="gray")
axes[5].set_title("Skeletonized Mask")
axes[5].axis("off")

skeleton_vis = cv2.cvtColor(skeleton_corrected.astype(np.uint8), cv2.COLOR_GRAY2BGR)
for y, x in selected_points_random_uniform:
    cv2.circle(skeleton_vis, (x, y), 4, (255, 0, 0), -1)
axes[6].imshow(skeleton_vis)
axes[6].set_title("Skeleton with Random Points")
axes[6].axis("off")

filtered_vis = cv2.cvtColor(skeleton_corrected.astype(np.uint8), cv2.COLOR_GRAY2BGR)
for y, x in filtered_points:
    cv2.circle(filtered_vis, (x, y), 4, (0, 255, 0), -1)
axes[7].imshow(filtered_vis)
axes[7].set_title("Filtered Points (74% Threshold)")
axes[7].axis("off")

plt.show()

# 최종 줄기 굵기 계산
stem_thickness = 2 * np.mean(distances[distances >= threshold])
print(f"Estimated Stem Thickness: {stem_thickness:.2f} pixels")
