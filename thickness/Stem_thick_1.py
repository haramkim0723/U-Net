import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def stem_thickness(mask_image):
    """
    주어진 마스크 이미지를 사용하여 줄기의 두께를 분석하는 함수.

    :param mask_image: 마스크 이미지 (numpy array, grayscale)
    :return: 줄기의 추정된 두께(픽셀)
    """

    # 원본 마스크 시각화
    plt.figure(figsize=(5, 5))
    plt.imshow(mask_image, cmap="gray")
    plt.title("Original Mask")
    plt.axis("off")
    plt.show()

    # 윤곽선 검출 (Contours Detection)
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 면적 기준 정렬

    # 윤곽선 결과 확인
    contour_mask = np.zeros_like(mask_image)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=1)

    plt.figure(figsize=(5, 5))
    plt.imshow(contour_mask, cmap="gray")
    plt.title("Contours Detection")
    plt.axis("off")
    plt.show()

    # 닫힘 연산 (Closing) 적용하여 끊어진 선 연결
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    plt.figure(figsize=(5, 5))
    plt.imshow(closed_mask, cmap="gray")
    plt.title("After Closing Operation (Swapped Order)")
    plt.axis("off")
    plt.show()

    # 윤곽선 내부를 채운 마스크 생성
    mask_filled = np.zeros_like(mask_image)
    if len(contours) > 0:
        main_contour = contours[0]  # 가장 큰 윤곽선 선택
        cv2.drawContours(mask_filled, [main_contour], -1, 255, thickness=cv2.FILLED)

    plt.figure(figsize=(5, 5))
    plt.imshow(mask_filled, cmap="gray")
    plt.title("Filled Contour Mask (Swapped Order)")
    plt.axis("off")
    plt.show()

    # 작은 노이즈 제거 (Opening 연산)
    denoised_mask = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel, iterations=2)

    plt.figure(figsize=(5, 5))
    plt.imshow(denoised_mask, cmap="gray")
    plt.title("After Noise Removal")
    plt.axis("off")
    plt.show()

    # 배경 반전 (Skeletonization 전처리)
    corrected_binary_mask = 255 - denoised_mask

    plt.figure(figsize=(5, 5))
    plt.imshow(corrected_binary_mask, cmap="gray")
    plt.title("Background Inverted Mask")
    plt.axis("off")
    plt.show()

    # Skeletonization 수행
    binary_mask = corrected_binary_mask // 255
    skeleton_corrected = skeletonize(binary_mask) * 255

    plt.figure(figsize=(5, 5))
    plt.imshow(skeleton_corrected, cmap="gray")
    plt.title("Skeletonized Mask")
    plt.axis("off")
    plt.show()

    # 스켈레톤 위의 모든 점 좌표 가져오기
    skeleton_points = np.column_stack(np.where(skeleton_corrected > 0))

    # 균일한 랜덤 샘플링 적용 (스켈레톤 위에서 300개 점 선택)
    num_random_points = min(300, len(skeleton_points))
    selected_points_random_uniform = skeleton_points[np.random.choice(len(skeleton_points), num_random_points, replace=False)]

    # 윤곽선과 점 사이의 최단 거리 계산
    formatted_contours = [contour.astype(np.float32) for contour in contours]
    distances_random_uniform = []
    for y, x in selected_points_random_uniform:
        min_dist = np.min([cv2.pointPolygonTest(contour, (float(x), float(y)), True) for contour in formatted_contours])
        distances_random_uniform.append(abs(min_dist))  # 절대값 적용하여 음수 거리 제거

    distances_random_uniform = np.array(distances_random_uniform)

    # 필터링 (74% 기준: 평균 거리의 74% 이하 제거)
    mean_distance_random_uniform = np.mean(distances_random_uniform)
    threshold_random_uniform = mean_distance_random_uniform * 0.74

    # 기준보다 작은 거리의 점을 제거
    filtered_points_random_uniform = selected_points_random_uniform[distances_random_uniform >= threshold_random_uniform]
    filtered_distances_random_uniform = distances_random_uniform[distances_random_uniform >= threshold_random_uniform]

    # 필터링 후 평균 거리 다시 계산
    new_mean_distance = np.mean(filtered_distances_random_uniform)

    # Skeleton 위에 필터링된 점 시각화
    skeleton_with_random_uniform_points = cv2.cvtColor(skeleton_corrected.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(skeleton_with_random_uniform_points, contours, -1, (0, 255, 0), 1)
    for y, x in filtered_points_random_uniform:
        cv2.circle(skeleton_with_random_uniform_points, (x, y), 4, (255, 0, 0), -1)

    plt.figure(figsize=(5, 5))
    plt.imshow(skeleton_with_random_uniform_points)
    plt.title("Skeleton with Filtered Points")
    plt.axis("off")
    plt.show()

    # 최종 줄기 굵기(Stem Thickness) 계산
    stem_thickness = 2 * new_mean_distance
    print(f"필터링 후 평균 거리: {new_mean_distance:.2f} 픽셀")
    print(f"추정된 줄기 굵기(Stem Thickness): {stem_thickness:.2f} 픽셀")

    return stem_thickness

# 이미지 로드
mask_path = "/mnt/data/20241110_150948_stem_14_mask.png"
mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# 줄기 두께 분석 실행
thickness_result = stem_thickness(mask_image)

# 결과 출력
print(f" 최종 줄기 굵기: {thickness_result:.2f} 픽셀")
