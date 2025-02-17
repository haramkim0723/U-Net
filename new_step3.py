import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# 이미지 로드
image_path = "D:/New folder.cancelled/HuggingFaceLLM/new_data/data/masks/20241110_150755_stem_0_mask.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 윤곽선 검출 (Contour Detection)
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 면적 기준 정렬
main_contour = contours[0]  # 가장 큰 윤곽선 선택

# 윤곽선 내부를 채운 마스크 생성 (Contour Filler)
mask = np.zeros_like(image)
cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)

# 닫힘 연산 (Closing) 적용하여 끊어진 선 연결
kernel = np.ones((5, 5), np.uint8)
closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# 작은 노이즈 제거 (Opening 적용)
denoised_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=2)

# 배경 반전 (Skeletonization 전처리)
corrected_binary_mask = 255 - denoised_mask  # 배경을 반전하여 줄기 부분 강조

# 원본 이미지에 윤곽선 추가
original_with_contours = cv2.cvtColor(corrected_binary_mask, cv2.COLOR_GRAY2BGR)
cv2.drawContours(original_with_contours, contours, -1, (0, 255, 0), 1)  # 초록색 윤곽선 추가

# Skeletonization 수행
binary_mask = corrected_binary_mask // 255  # 0-1 이진화 변환
skeleton_corrected = skeletonize(binary_mask) * 255  # Skeletonization 적용

# 골격화한 이미지에 윤곽선 추가
skeleton_with_contours = cv2.cvtColor(skeleton_corrected.astype(np.uint8), cv2.COLOR_GRAY2BGR)
cv2.drawContours(skeleton_with_contours, contours, -1, (0, 255, 0), 1)  # 초록색 윤곽선 추가

# 결과 시각화 (원본 + Skeletonization + 윤곽선)
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(original_with_contours, cmap="gray")
ax[0].set_title("Original 윤곽선")
ax[0].axis("off")

ax[1].imshow(skeleton_corrected, cmap="gray")
ax[1].set_title("Skeletonization")
ax[1].axis("off")

ax[2].imshow(skeleton_with_contours, cmap="gray")
ax[2].set_title("Skeletonization 윤곽선")
ax[2].axis("off")

plt.show()
