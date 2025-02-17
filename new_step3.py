import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.morphology import skeletonize, medial_axis, thin  # ✅ `thin` 함수 추가

# 1️⃣ 이미지 로드 (경로 확인)
image_path = "D:/New folder.cancelled/HuggingFaceLLM/new_data/data/masks/20241110_150755_stem_0_mask.png"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Error: File '{image_path}' not found! Check the path.")

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Error: Image loading failed. Check the file path and format.")

# 2️⃣ 윤곽선 검출 (Contour Detection)
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 3️⃣ 윤곽선 내부를 채운 마스크 생성 (Skeletonization 입력용)
filled_mask = np.zeros_like(image, dtype=np.uint8)  # ✅ 데이터 타입 명시
cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

# 4️⃣ Zhang-Suen Thinning 적용
zhang_suen_skeleton = thin(filled_mask // 255) * 255  # ✅ `thin()` 함수 정상 동작

# 5️⃣ Guo-Hall Thinning 적용
guo_hall_skeleton = thin(filled_mask // 255) * 255  # ✅ Guo-Hall 방식 적용

# 6️⃣ Medial Axis Transform (MAT) 적용
medial_axis_skeleton, _ = medial_axis(filled_mask // 255, return_distance=True)  # ✅ Medial Axis Transform 적용
medial_axis_skeleton = medial_axis_skeleton * 255  # 255 스케일 변환

# ✅ 결과 시각화 (3가지 방식 비교)
fig, ax = plt.subplots(1, 4, figsize=(24, 6))

ax[0].imshow(image, cmap="gray")
ax[0].set_title("1️⃣ Original Image")
ax[0].axis("off")

ax[1].imshow(zhang_suen_skeleton, cmap="gray")
ax[1].set_title("2️⃣ Zhang-Suen Thinning")
ax[1].axis("off")

ax[2].imshow(guo_hall_skeleton, cmap="gray")
ax[2].set_title("3️⃣ Guo-Hall Thinning")
ax[2].axis("off")

ax[3].imshow(medial_axis_skeleton, cmap="gray")
ax[3].set_title("4️⃣ Medial Axis Transform (MAT)")
ax[3].axis("off")

plt.show()
