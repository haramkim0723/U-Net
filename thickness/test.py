import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

test_mask_path = "D:/New folder.cancelled/HuggingFaceLLM/new_data/data/masks/20241110_150755_stem_0_mask.png"
test_image = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)

# 닫힘 연산 (Closing) 적용하여 끊어진 윤곽선 채움
kernel = np.ones((5, 5), np.uint8)
closed_mask = cv2.morphologyEx(test_image, cv2.MORPH_CLOSE, kernel, iterations=4)

# 노이즈 제거 (Opening : 열림 연산)
denoised_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=2)

contours, hierarchy = cv2.findContours(denoised_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 윤곽선 내부를 채울 마스크 생성
mask_filled = np.zeros_like(test_image)
if len(contours) > 0:
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:  # 바깥쪽 윤곽선만 유지
            cv2.drawContours(mask_filled, [contours[i]], -1, 255, thickness=cv2.FILLED)

# 객체 내부가 255(흰색), 배경이 0(검은색)인지 확인 후 반전 (골격화 함수 때문)
if np.mean(mask_filled) > 128:  # 배경이 255면 반전
    mask_filled = cv2.bitwise_not(mask_filled)

# 이진화 적용 (객체 내부만 골격화)
binary_mask = (mask_filled > 0).astype(np.uint8)  # 객체가 1, 배경이 0

# Skeletonization 수행
skeleton_corrected = skeletonize(binary_mask) * 255  # 객체 내부만 골격화됨

# Skeleton 위의 모든 점 좌표 호출
skeleton_points = np.column_stack(np.where(skeleton_corrected > 0))

# 균일한 랜덤 샘플링 적용 (스켈레톤 위에서 300개 점 선택)
num_random_points = min(300, len(skeleton_points))
selected_points_random_uniform = skeleton_points[np.random.choice(len(skeleton_points), num_random_points, replace=False)]

# 윤곽선과 점 사이의 최단 거리 계산 (각 점별 참조 윤곽선 확인)
formatted_contours = [contour.astype(np.float32) for contour in contours]
distances_random_uniform = []
point_assigned_contours = []
for y, x in selected_points_random_uniform:
    point_distances = [(cv2.pointPolygonTest(contour, (float(x), float(y)), True), idx)
                        for idx, contour in enumerate(formatted_contours)]
    min_dist, assigned_contour = min(point_distances, key=lambda t: abs(t[0]))  # 최소 거리 값 선택
    distances_random_uniform.append(abs(min_dist))  # 절대값 적용하여 음수 거리 제거
    point_assigned_contours.append(assigned_contour)  # 점이 참조한 윤곽선 인덱스 저장

distances_random_uniform = np.array(distances_random_uniform)

# 필터링 (74% 기준: 평균 거리의 74% 이하 제거)
mean_distance_random_uniform = np.mean(distances_random_uniform)
threshold_random_uniform = mean_distance_random_uniform * 0.74

# 기준보다 작은 거리의 점을 제거
filtered_points_random_uniform = selected_points_random_uniform[distances_random_uniform >= threshold_random_uniform]
removed_points = selected_points_random_uniform[distances_random_uniform < threshold_random_uniform]

# 필터링 후 평균 거리 다시 계산
new_mean_distance = np.mean(distances_random_uniform[distances_random_uniform >= threshold_random_uniform])

# Skeleton 위에 필터링된 점과 제거된 점 시각화 (점마다 참조한 윤곽선 포함)
skeleton_with_removed_points = cv2.cvtColor(skeleton_corrected.astype(np.uint8), cv2.COLOR_GRAY2BGR)
cv2.drawContours(skeleton_with_removed_points, contours, -1, (0, 255, 0), 1)

# 필터링 후 남은 점 (74% 이상 거리 유지된 점)
for i, (y, x) in enumerate(filtered_points_random_uniform):
    color = (255, (point_assigned_contours[i] * 50) % 255, 0)
    cv2.circle(skeleton_with_removed_points, (x, y), 4, color, -1)

# 필터링되어 제거된 점 (74% 이하 거리의 점)
for i, (y, x) in enumerate(removed_points):
    color = (255, (point_assigned_contours[i] * 50) % 255, 165)
    cv2.circle(skeleton_with_removed_points, (x, y), 4, color, -1)

# 모든 선택된 스켈레톤 점에 대해, 해당 윤곽선에서 가장 가까운 점(근사치)을 찾아 선으로 연결
for i, (y, x) in enumerate(selected_points_random_uniform):
    contour_idx = point_assigned_contours[i]
    contour = formatted_contours[contour_idx]
    closest_pt = None
    min_dist = float('inf')
    # 윤곽선상의 각 점과의 거리를 계산
    for pt in contour:
        pt_coords = pt[0]  # contour는 [[x, y]] 형태
        d = np.sqrt((pt_coords[0] - x) ** 2 + (pt_coords[1] - y) ** 2)
        if d < min_dist:
            min_dist = d
            closest_pt = (int(pt_coords[0]), int(pt_coords[1]))
    # 선택된 점의 거리가 필터 기준 이상이면 초록, 이하이면 빨강으로 선 색 지정
    if distances_random_uniform[i] >= threshold_random_uniform:
        line_color = (0, 255, 0)
    else:
        line_color = (0, 0, 255)
    cv2.line(skeleton_with_removed_points, (x, y), closest_pt, line_color, 1)

# 결과 시각화
plt.figure(figsize=(6, 6))
plt.imshow(skeleton_with_removed_points)
plt.title("Skeleton with Removed & Filtered Points (74% Threshold)")
plt.axis("off")
plt.show()

# 최종 줄기 굵기(Stem Thickness)
print(f"필터링 후 평균 거리: {new_mean_distance:.2f} 픽셀")
print(f"추정된 줄기 굵기(Stem Thickness): {2 * new_mean_distance:.2f} 픽셀")
