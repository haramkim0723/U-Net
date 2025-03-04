import os
import cv2
import numpy as np
import logging
import pandas as pd
from skimage.morphology import skeletonize

# 🔹 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ThickLast 클래스 정의 (재사용)
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
        kernel = np.ones((5, 5), np.uint8)
        closed_mask = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel, iterations=4)
        self.processed_image = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        self.processed_image = cv2.bitwise_not(self.processed_image)

    def extract_contours(self):
        self.contours, _ = cv2.findContours(self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_filled = np.zeros_like(self.processed_image)
        cv2.drawContours(mask_filled, self.contours, -1, 255, thickness=cv2.FILLED)
        kernel = np.ones((3, 3), np.uint8)
        mask_filled = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel, iterations=1)
        self.processed_image = mask_filled

    def skeletonize_image(self):
        binary_mask = (self.processed_image > 0).astype(np.uint8)
        self.skeleton = skeletonize(binary_mask) * 255

    def sample_skeleton_points(self, num_points=300):
        skeleton_points = np.column_stack(np.where(self.skeleton > 0))
        num_random_points = min(num_points, len(skeleton_points))
        return skeleton_points[np.random.choice(len(skeleton_points), num_random_points, replace=False)]

    def compute_distances(self, sampled_points):
        if self.contours is None or len(self.contours) == 0:
            self.distances = np.zeros(sampled_points.shape[0])
            return
        distances = []
        for y, x in sampled_points:
            min_distance = float("inf")
            for contour in self.contours:
                for contour_point in contour:
                    contour_x, contour_y = contour_point[0]
                    distance = np.sqrt((contour_x - x) ** 2 + (contour_y - y) ** 2)
                    min_distance = min(min_distance, distance)
            distances.append(min_distance)
        self.distances = np.array(distances)

    def filter_points(self, sampled_points, threshold_ratio=0.74):
        mean_distance = np.mean(self.distances)
        threshold = mean_distance * threshold_ratio
        self.filtered_points = sampled_points[self.distances >= threshold]
        self.removed_points = sampled_points[self.distances < threshold]
        self.mean_distance = np.mean(self.distances[self.distances >= threshold])

    def compute_stem_thickness(self):
        return 2 * self.mean_distance


def PexelToCm(x, real_cm=0.5, index=1):
    if isinstance(x, list):
        x = x[index]
    elif isinstance(x, dict):
        values = list(x.values())
        x = values[index]
    if x == 0:
        raise ZeroDivisionError("[ERROR] PexelToCm 함수에서 0으로 나눌 수 없음")
    return real_cm / x


# 이미지 세트 정의
image_sets = [
    {"stem": "/mnt/data/stem_1.png", "bar": "/mnt/data/bar_1.png"},
    {"stem": "/mnt/data/stem_2.png", "bar": "/mnt/data/bar_2.png"}
]

# 50회 반복 수행
num_trials = 50
trial_results = []

for trial in range(num_trials):
    logging.info(f"🔁 {trial+1}번째 반복 수행 중...")

    for set_index, image_set in enumerate(image_sets, start=1):
        logging.info(f"🔹 세트 {set_index} 처리 시작!")

        # 기준 막대 처리
        bar_thickness = ThickLast(image_set["bar"], class_label=4)
        bar_thickness.preprocess_image()
        bar_thickness.extract_contours()
        bar_thickness.skeletonize_image()

        bar_points = bar_thickness.sample_skeleton_points()
        bar_thickness.compute_distances(bar_points)
        bar_thickness.filter_points(bar_points)

        bar_pixel_thickness = bar_thickness.compute_stem_thickness()
        real_pexel = PexelToCm(bar_pixel_thickness, real_cm=0.5)

        # 줄기 처리
        stem_thickness = ThickLast(image_set["stem"], class_label=3)
        stem_thickness.preprocess_image()
        stem_thickness.extract_contours()
        stem_thickness.skeletonize_image()

        stem_points = stem_thickness.sample_skeleton_points()
        stem_thickness.compute_distances(stem_points)
        stem_thickness.filter_points(stem_points)

        stem_pixel_thickness = stem_thickness.compute_stem_thickness()
        stem_real_cm = stem_pixel_thickness * real_pexel

        trial_results.append({
            "trial": trial + 1,
            "set_index": set_index,
            "stem_real_cm": stem_real_cm,
            "real_pexel": real_pexel
        })

# 데이터프레임으로 정리 및 저장
df_trials = pd.DataFrame(trial_results)
df_trials.to_csv("/mnt/data/stem_thickness_trials.csv", index=False)

logging.info("✅ 50회 반복 결과를 'stem_thickness_trials.csv'에 저장 완료!")
