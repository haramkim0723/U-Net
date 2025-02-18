import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# 이미지
image_path = "D:/New folder.cancelled/HuggingFaceLLM/new_data/data/masks/20241110_150755_stem_0_mask.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 윤곽선 검출 (Contour Detection)
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Contour Filler 적용 (윤곽선 내부를 흰색으로 채움)
mask = np.zeros_like(image)
cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

# Distance Transform 적용
dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

# 동적 Threshold 값 적용하여 중심선 후보 영역 설정
threshold_value = 0.4 * dist_transform.max()  # 거리값의 40% 이상인 부분을 중심선 후보로 설정
binary_skeleton = (dist_transform >= threshold_value).astype(np.uint8) * 255  # 0 또는 255로 변환

# Morphological Thinning (침식 연산) 적용하여 중심선 강조
kernel = np.ones((3, 3), np.uint8)
binary_skeleton = cv2.erode(binary_skeleton, kernel, iterations=2)  # 중심선을 더 강조

# 골격화
skeleton = skeletonize(binary_skeleton // 255) * 255  # `skimage.skeletonize()` 적용 후 255 스케일로 변환
skeleton = skeleton.astype(np.uint8)  # OpenCV 연산을 위해 uint8 형식 변환

# 중심선 연결 - 작은 끊어진 선 제거 및 연결
skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel, iterations=2)

# 결과 시각화
fig, ax = plt.subplots(1, 4, figsize=(24, 6))

ax[0].imshow(image, cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Contour Filler Applied")
ax[1].axis("off")

ax[2].imshow(dist_transform, cmap="jet")
ax[2].set_title("Distance Transform")
ax[2].axis("off")

ax[3].imshow(skeleton, cmap="gray")
ax[3].set_title("Ridge Detection + Thinning + Skeletonization + Post-Processing")
ax[3].axis("off")

plt.show()
