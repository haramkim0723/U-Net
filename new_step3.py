import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# 1️⃣ 이미지 로드
image_path = "D:/New folder.cancelled/HuggingFaceLLM/new_data/data/masks/20241110_150755_stem_0_mask.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 2️⃣ 윤곽선 검출 (Contour Detection)
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 면적 기준 정렬
main_contour = contours[0]  # 가장 큰 윤곽선 선택

# 3️⃣ 윤곽선 내부를 채운 마스크 생성 (Contour Filler)
mask = np.zeros_like(image)
cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)

# 4️⃣ 닫힘 연산 (Closing) 적용하여 끊어진 선 연결
kernel = np.ones((5, 5), np.uint8)
closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# 5️⃣ 작은 노이즈 제거 (Opening 적용)
denoised_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=2)

# 6️⃣ 배경 반전 (Skeletonization 전처리)
corrected_binary_mask = 255 - denoised_mask  # 배경을 반전하여 줄기 부분 강조

# 7️⃣ Skeletonization 수행
binary_mask = corrected_binary_mask // 255  # 0-1 이진화 변환
skeleton_corrected = skeletonize(binary_mask) * 255  # Skeletonization 적용

# 8️⃣ 추가적인 작은 선 제거 (불필요한 잔가지 제거)
kernel_small = np.ones((3, 3), np.uint8)
skeleton_final = cv2.morphologyEx(skeleton_corrected.astype(np.uint8), cv2.MORPH_OPEN, kernel_small, iterations=1)

# ✅ 결과 시각화
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(corrected_binary_mask, cmap="gray")
ax[0].set_title("1️⃣ Corrected Input (Background Inverted)")
ax[0].axis("off")

ax[1].imshow(skeleton_corrected, cmap="gray")
ax[1].set_title("2️⃣ Skeletonization Applied (Corrected)")
ax[1].axis("off")

ax[2].imshow(skeleton_final, cmap="gray")
ax[2].set_title("3️⃣ Final Skeleton (Filtered)")
ax[2].axis("off")

plt.show()
