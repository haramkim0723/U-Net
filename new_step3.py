import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

# 1️⃣ 이미지 로드
image_path = "D:/New folder.cancelled/HuggingFaceLLM/new_data/data/masks/20241110_150755_stem_0_mask.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 2️⃣ 윤곽선 검출 (Contour Detection)
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 3️⃣ 윤곽선 내부를 채운 마스크 생성 (Distance Transform 입력용)
filled_mask = np.zeros_like(image)
cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

# 4️⃣ Distance Transform 적용
dist_transform = cv2.distanceTransform(filled_mask, cv2.DIST_L2, 5)

# 5️⃣ Distance Transform에서 지역 최대값(Local Maxima) 찾기
local_max = peak_local_max(dist_transform, min_distance=3)  # 최소 거리 3px 이상 유지

# 6️⃣ 중심선을 직접 그리기 위한 빈 이미지 생성
centerline = np.zeros_like(image)

# 7️⃣ Local Maxima를 선으로 연결 (cv2.fitLine() 사용)
if len(local_max) > 1:
    vx, vy, x0, y0 = cv2.fitLine(local_max.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)

    for i in range(len(local_max) - 1):
        pt1 = (local_max[i][1], local_max[i][0])  # (x, y) 좌표
        pt2 = (local_max[i + 1][1], local_max[i + 1][0])  # 다음 점
        cv2.line(centerline, pt1, pt2, 255, 1)  # 중심선 그리기

# ✅ 결과 시각화
fig, ax = plt.subplots(1, 4, figsize=(24, 6))

ax[0].imshow(image, cmap="gray")
ax[0].set_title("1️⃣ Original Image")
ax[0].axis("off")

ax[1].imshow(filled_mask, cmap="gray")
ax[1].set_title("2️⃣ Contour Filler Applied")
ax[1].axis("off")

ax[2].imshow(dist_transform, cmap="jet")
ax[2].set_title("3️⃣ Distance Transform")
ax[2].axis("off")

ax[3].imshow(centerline, cmap="gray")
ax[3].set_title("4️⃣ Centerline from Distance Transform Peaks")
ax[3].axis("off")

plt.show()
