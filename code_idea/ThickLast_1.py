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
        """이진화 후 모폴로지 연산 적용하여 이미지 전처리"""
        # 1️⃣ 명확한 이진화 적용 (객체를 흰색, 배경을 검은색으로 설정)
        _, binary_image = cv2.threshold(self.image, 128, 255, cv2.THRESH_BINARY)

        # 2️⃣ 닫힘 연산 (Closing) → 끊어진 윤곽선 연결
        kernel = np.ones((5, 5), np.uint8)
        closed_mask = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=4)

        # 3️⃣ 열림 연산 (Opening) → 작은 노이즈 제거
        self.processed_image = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    def extract_contours(self):
        """윤곽선을 추출하고 내부를 채운 마스크 생성"""
        self.contours, _ = cv2.findContours(self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 윤곽선 내부를 채울 마스크 생성
        mask_filled = np.zeros_like(self.processed_image)
        if len(self.contours) > 0:
            largest_contour = max(self.contours, key=cv2.contourArea)  # 가장 큰 윤곽선 선택
            cv2.drawContours(mask_filled, [largest_contour], -1, 255, thickness=cv2.FILLED)

        self.processed_image = mask_filled

    def skeletonize_image(self):
        """골격화(Skeletonization) 수행"""
        binary_mask = (self.processed_image > 0).astype(np.uint8)  # 확실한 이진화
        self.skeleton = skeletonize(binary_mask) * 255

    def sample_skeleton_points(self, num_points=300):
        """스켈레톤 상에서 일정 개수의 랜덤 점 샘플링"""
        skeleton_points = np.column_stack(np.where(self.skeleton > 0))
        num_random_points = min(num_points, len(skeleton_points))
        return skeleton_points[np.random.choice(len(skeleton_points), num_random_points, replace=False)]

    def compute_distances(self, sampled_points):
        """샘플링된 점과 윤곽선 사이의 거리 계산"""
        formatted_contours = [contour.astype(np.float32) for contour in self.contours]
        distances = []
        for y, x in sampled_points:
            point_distances = [cv2.pointPolygonTest(contour, (float(x), float(y)), True) for contour in
                               formatted_contours]
            distances.append(abs(min(point_distances)))
        self.distances = np.array(distances)

    def filter_points(self, sampled_points, threshold_ratio=0.74):
        """74% 기준으로 너무 가까운 점을 필터링"""
        mean_distance = np.mean(self.distances)
        threshold = mean_distance * threshold_ratio
        self.filtered_points = sampled_points[self.distances >= threshold]
        self.removed_points = sampled_points[self.distances < threshold]
        self.mean_distance = np.mean(self.distances[self.distances >= threshold])

    def compute_stem_thickness(self):
        """최종 줄기 두께 계산"""
        return 2 * self.mean_distance

    def visualize_results(self):
        """골격화된 이미지와 필터링된 점을 시각화"""
        img = cv2.cvtColor(self.skeleton.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # 필터링된 점(유지된 점) - 초록색
        for (y, x) in self.filtered_points:
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)

        # 제거된 점(너무 가까운 점) - 빨간색
        for (y, x) in self.removed_points:
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

        plt.imshow(img)
        plt.title(f"Skeleton ({self.class_label})")
        plt.axis("off")
        plt.show()
