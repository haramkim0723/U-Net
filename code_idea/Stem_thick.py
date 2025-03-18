import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import pandas as pd

# 이미지 로드
image_path = "D:/New folder.cancelled/HuggingFaceLLM/new_data/data/masks/20241110_150755_stem_0_mask.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 1) 윤곽선 추출
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
main_contour = contours[0]  # 가장 큰 윤곽선

# 2) 윤곽선 내부 채우기
mask = np.zeros_like(image)
cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)

# 3) 닫힘 연산 & 노이즈 제거
kernel = np.ones((5, 5), np.uint8)
closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
denoised_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=2)

# 4) 배경 반전 (Skeletonization 전처리)
corrected_binary_mask = 255 - denoised_mask

# 5) Skeletonization 수행
binary_mask = corrected_binary_mask // 255
skeleton_corrected = skeletonize(binary_mask) * 255

# 6) 스켈레톤 위에 점 300개 균일 배치
skeleton_points = np.column_stack(np.where(skeleton_corrected > 0))
num_points = min(300, len(skeleton_points))
indices = np.linspace(0, len(skeleton_points) - 1, num=num_points, dtype=int)
selected_points = skeleton_points[indices]

# 7) 윤곽선과 점 사이 최단 거리 계산
formatted_contours = [cnt.astype(np.float32) for cnt in contours]
distances = []
for y, x in selected_points:
    min_dist = np.min([cv2.pointPolygonTest(c, (float(x), float(y)), True) for c in formatted_contours])
    distances.append(abs(min_dist))  # 음수 값 제거(절대값)

distances = np.array(distances)

# 8) 필터링 전 CSV 저장
df_pre = pd.DataFrame({
    "skeleton_y": selected_points[:, 0],
    "skeleton_x": selected_points[:, 1],
    "distance": distances
})
pre_csv_filename = "skeleton_contour_distances_pre_IQR.csv"
df_pre.to_csv(pre_csv_filename, index=False)
print(f"필터링 전 CSV 파일 저장됨: {pre_csv_filename}")

# 9) IQR 계산
Q1 = np.percentile(distances, 25)  # 25% 지점
Q3 = np.percentile(distances, 75)  # 75% 지점
IQR = Q3 - Q1

lower_bound = Q1 - 0.17 * IQR
upper_bound = Q3 + 0.5 * IQR

# 10) IQR 범위 내 데이터만 필터링
mask_iqr = (distances >= lower_bound) & (distances <= upper_bound)
filtered_points_iqr = selected_points[mask_iqr]
filtered_distances_iqr = distances[mask_iqr]

# 11) 필터링 후 CSV 저장
df_post = pd.DataFrame({
    "skeleton_y": filtered_points_iqr[:, 0],
    "skeleton_x": filtered_points_iqr[:, 1],
    "distance": filtered_distances_iqr
})
post_csv_filename = "skeleton_contour_distances_post_IQR.csv"
df_post.to_csv(post_csv_filename, index=False)
print(f"필터링 후 CSV 파일 저장됨: {post_csv_filename}")

# 12) 필터링 후 점들을 스켈레톤 이미지에 표시
skeleton_with_contours = cv2.cvtColor(skeleton_corrected.astype(np.uint8), cv2.COLOR_GRAY2BGR)
cv2.drawContours(skeleton_with_contours, contours, -1, (0, 255, 0), 1)  # 윤곽선(초록)
for y, x in filtered_points_iqr:
    cv2.circle(skeleton_with_contours, (x, y), 2, (0, 0, 255), -1)  # 필터링된 점(빨강)

# (선택) 필터링된 점들 중 최단 거리 선 표시
if len(filtered_points_iqr) > 0:
    min_distance_val = float("inf")
    min_skel_point = None
    min_contour_point = None

    for y, x in filtered_points_iqr:
        for contour in contours:
            for point in contour:
                contour_x, contour_y = point[0]
                dist_val = np.sqrt((contour_x - x) ** 2 + (contour_y - y) ** 2)
                if dist_val < min_distance_val:
                    min_distance_val = dist_val
                    min_skel_point = (x, y)
                    min_contour_point = (contour_x, contour_y)

    cv2.line(skeleton_with_contours, min_skel_point, min_contour_point, (255, 0, 0), 2)
    cv2.putText(skeleton_with_contours, f"{min_distance_val:.2f}",
                (min_skel_point[0], min_skel_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# 13) 결과 시각화
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(cv2.cvtColor(corrected_binary_mask, cv2.COLOR_BGR2RGB))
ax[0].set_title("윤곽선 추출")
ax[0].axis("off")

ax[1].imshow(skeleton_corrected, cmap="gray")
ax[1].set_title("Skeletonization")
ax[1].axis("off")

ax[2].imshow(cv2.cvtColor(skeleton_with_contours, cv2.COLOR_BGR2RGB))
ax[2].set_title("IQR flitering")
ax[2].axis("off")
plt.show()

# 14) 거리 분포 그래프: 하한·상한 표시
sorted_idx = np.argsort(distances)
sorted_distances = distances[sorted_idx]

plt.figure(figsize=(10, 6))
plt.scatter(range(len(sorted_distances)), sorted_distances, c='blue', s=10, label='All Distances')
plt.axhline(y=lower_bound, color='red', linestyle='--', label=f'Lower Bound={lower_bound:.2f}')
plt.axhline(y=upper_bound, color='green', linestyle='--', label=f'Upper Bound={upper_bound:.2f}')
plt.xlabel("Sorted Skeleton Point Index")
plt.ylabel("Distance to Contour (pixels)")
plt.title("Skeleton Points Distances (IQR flitering)")
plt.legend()
plt.show()

# 15) 최종 줄기 굵기(Stem Thickness) 계산 & 출력
if len(filtered_distances_iqr) > 0:
    new_mean_distance_iqr = np.mean(filtered_distances_iqr)
    stem_thickness_iqr = 2 * new_mean_distance_iqr
    print(f"[IQR] 평균 거리: {new_mean_distance_iqr:.2f} 픽셀")
    print(f"[IQR] 추정된 굵기: {stem_thickness_iqr:.2f} 픽셀")
else:
    print("[IQR] 필터링 후 남은 점이 없습니다.")
