import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


def analyze_stem_thickness(segmentation_outputs):
    """
    YOLO + U-Net 결과를 받아 각 클래스별 윤곽선을 분석하고 두께를 계산.

    :param segmentation_outputs: YOLO의 class_id가 포함된 U-Net 마스크 리스트
    :return: 클래스별 두께 정보
    """
    thickness_results = {}

    for output in segmentation_outputs:
        class_id = output["class_id"]
        unet_mask = output["segmentation_mask"].astype(np.uint8)

        contours, _ = cv2.findContours(unet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print(f"클래스 {class_id}: 윤곽선을 찾을 수 없습니다!")
            continue

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        main_contour = contours[0]

        mask = np.zeros_like(unet_mask)
        cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)

        kernel = np.ones((5, 5), np.uint8)
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        denoised_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        corrected_binary_mask = 255 - denoised_mask
        binary_mask = corrected_binary_mask // 255
        skeleton_corrected = skeletonize(binary_mask) * 255

        skeleton_points = np.column_stack(np.where(skeleton_corrected > 0))
        num_points = min(300, len(skeleton_points))
        indices = np.linspace(0, len(skeleton_points) - 1, num=num_points, dtype=int)
        selected_points = skeleton_points[indices]

        formatted_contours = [main_contour.astype(np.float32)]
        distances = []
        for y, x in selected_points:
            min_dist = np.min(
                [cv2.pointPolygonTest(contour, (float(x), float(y)), True) for contour in formatted_contours])
            distances.append(abs(min_dist))

        distances = np.array(distances)

        mean_distance = np.mean(distances)
        threshold = mean_distance * 0.74
        filtered_points = selected_points[distances >= threshold]
        filtered_distances = distances[distances >= threshold]
        new_mean_distance = np.mean(filtered_distances)

        stem_thickness = 2 * new_mean_distance
        thickness_results[class_id] = stem_thickness

    return thickness_results
