import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import csv

class ThickLast:
    def __init__(self, image_path, class_label):
        self.image_path = image_path
        self.class_label = class_label
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise FileNotFoundError(f"{class_label} 이미지 경로 오류.")
        self.processed_image = self.image.copy()
        self.skeleton = None
        self.contours = None
        self.distances = None
        self.filtered_points = None
        self.removed_points = None
        self.mean_distance = None

    def preprocess_image(self):
        """ 노이즈 제거 수행 후, 강제 반전 적용 """
        kernel = np.ones((5, 5), np.uint8)
        closed_mask = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel, iterations=4)
        self.processed_image = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # **무조건 반전 적용** (객체가 흰색, 배경이 검은색이 되도록 강제 변환)
        self.processed_image = cv2.bitwise_not(self.processed_image)

    def extract_contours(self):
        """ 윤곽선 검출 시, 전처리된 이미지를 사용 """
        self.contours, _ = cv2.findContours(self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 내부 채우기 추가 (골격화를 위해 객체가 흰색이 되도록 보장)
        mask_filled = np.zeros_like(self.processed_image)
        cv2.drawContours(mask_filled, self.contours, -1, 255, thickness=cv2.FILLED)

        # 다시 노이즈 제거 (경계선을 부드럽게 만들기 위해)
        kernel = np.ones((3, 3), np.uint8)
        mask_filled = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel, iterations=1)

        self.processed_image = mask_filled

    def skeletonize_image(self):
        """ 골격화 수행 """
        binary_mask = (self.processed_image > 0).astype(np.uint8)
        self.skeleton = skeletonize(binary_mask) * 255

    def sample_skeleton_points(self, num_points=300):
        """ 스켈레톤 점 샘플링 """
        skeleton_points = np.column_stack(np.where(self.skeleton > 0))
        num_random_points = min(num_points, len(skeleton_points))
        return skeleton_points[np.random.choice(len(skeleton_points), num_random_points, replace=False)]

    def compute_distances(self, sampled_points, save_csv=True, csv_base_filename="distances.csv"):
        """ 윤곽선과 스켈레톤 점 사이 거리 계산 및 가장 가까운 윤곽선 좌표 저장 """
        if self.contours is None or len(self.contours) == 0:
            print(f"[Warning] No contours found for {self.class_label}. Skipping distance computation.")
            self.distances = np.zeros(sampled_points.shape[0])  # 거리값 0으로 설정
            return

        distances = []
        closest_contour_points = []  # 가장 가까운 윤곽선 점 저장

        for y, x in sampled_points:
            min_distance = float("inf")
            closest_point = (0, 0)  # 가장 가까운 윤곽선 좌표 초기화

            for contour in self.contours:
                for contour_point in contour:
                    contour_x, contour_y = contour_point[0]
                    distance = np.sqrt((contour_x - x) ** 2 + (contour_y - y) ** 2)

                    if distance < min_distance:
                        min_distance = distance
                        closest_point = (contour_x, contour_y)

            distances.append(min_distance)
            closest_contour_points.append((x, y, closest_point[0], closest_point[1], min_distance))

        self.distances = np.array(distances)
        if save_csv:
            # 이미지 파일명에서 확장자 제거 후 CSV 이름 생성
            image_name = os.path.splitext(os.path.basename(self.image_path))[0]
            csv_filename = f"{csv_base_filename}_{image_name}.csv"

            with open(csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["skeleton_x", "skeleton_y", "contour_x", "contour_y", "distance"])
                writer.writerows(closest_contour_points)

            print(f"[INFO] 거리 데이터를 '{csv_filename}' 파일로 저장 완료!")

    def filter_points(self, sampled_points, threshold_ratio=0.74):
        """ 스켈레톤 점 필터링 """
        mean_distance = np.mean(self.distances)
        threshold = mean_distance * threshold_ratio
        self.filtered_points = sampled_points[self.distances >= threshold]
        self.removed_points = sampled_points[self.distances < threshold]
        self.mean_distance = np.mean(self.distances[self.distances >= threshold])

    def compute_stem_thickness(self):
        """ 최종 줄기 굵기 계산 """
        return 2 * self.mean_distance

    def visualize_results(self):
        """ 최종 결과 시각화 """
        img = cv2.cvtColor(self.skeleton.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for (y, x) in self.filtered_points:
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
        for (y, x) in self.removed_points:
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        return img
