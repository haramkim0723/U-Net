import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


class ThickLast:
    def __init__(self, image_path, class_label):
        self.image_path = image_path
        self.class_label = class_label
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise FileNotFoundError(f"{class_label} 이미지 경로 오류.")
        self.processed_image = None
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

    def extract_contours(self):
        self.contours, _ = cv2.findContours(self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 윤곽선 내부를 채울 마스크 생성
        mask_filled = np.zeros_like(self.processed_image)
        if len(self.contours) > 0:
            for i in range(len(self.contours)):
                if _[0][i][3] == -1:  # 바깥쪽 윤곽선만 유지
                    cv2.drawContours(mask_filled, [self.contours[i]], -1, 255, thickness=cv2.FILLED)

        # 객체 내부가 255(흰색), 배경이 0(검은색)인지 확인 후 반전 (골격화 함수 때문)
        if np.mean(mask_filled) > 128:  # 배경이 255면 반전
            mask_filled = cv2.bitwise_not(mask_filled)
        self.processed_image = mask_filled

    def skeletonize_image(self):
        binary_mask = (self.processed_image > 0).astype(np.uint8)
        self.skeleton = skeletonize(binary_mask) * 255

    def sample_skeleton_points(self, num_points=300):
        skeleton_points = np.column_stack(np.where(self.skeleton > 0))
        num_random_points = min(num_points, len(skeleton_points))
        return skeleton_points[np.random.choice(len(skeleton_points), num_random_points, replace=False)]

    def compute_distances(self, sampled_points):
        formatted_contours = [contour.astype(np.float32) for contour in self.contours]
        distances = []
        for y, x in sampled_points:
            point_distances = [cv2.pointPolygonTest(contour, (float(x), float(y)), True) for contour in
                               formatted_contours]
            distances.append(abs(min(point_distances)))
        self.distances = np.array(distances)

    def filter_points(self, sampled_points, threshold_ratio=0.74):
        mean_distance = np.mean(self.distances)
        threshold = mean_distance * threshold_ratio
        self.filtered_points = sampled_points[self.distances >= threshold]
        self.removed_points = sampled_points[self.distances < threshold]
        self.mean_distance = np.mean(self.distances[self.distances >= threshold])

    def compute_stem_thickness(self):
        return 2 * self.mean_distance

    def visualize_results(self):
        img = cv2.cvtColor(self.skeleton.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for (y, x) in self.filtered_points:
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
        for (y, x) in self.removed_points:
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        plt.imshow(img)
        plt.title(f"Skeleton ({self.class_label})")
        plt.axis("off")
        plt.show()