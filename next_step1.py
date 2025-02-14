import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드

image_path = "D:/New folder.cancelled/HuggingFaceLLM/new_data/data/masks/20241110_150755_stem_0_mask.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#윤곽선(Contour) 검출
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Contour Filler 적용
mask = np.zeros_like(image)  # 배경을 검은색으로 초기화
cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

# 4️⃣ 작은 노이즈 제거 (Morphological Opening 적용)
kernel = np.ones((3, 3), np.uint8)
denoised_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

#시각화
fig, ax = plt.subplots(1, 2, figsize=(18, 12))

ax[0].imshow(mask, cmap="gray")
ax[0].set_title("1️⃣ Contour Filler Applied")
ax[0].axis("off")

ax[1].imshow(denoised_mask, cmap="gray")
ax[1].set_title("2️⃣ Noise Removal Applied")
ax[1].axis("off")

plt.show()
