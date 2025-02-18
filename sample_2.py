import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skan.csr import skeleton_to_csgraph

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

# 8️⃣ Skeleton을 그래프로 변환 (Graph-based Filtering)
graph, coordinates = skeleton_to_csgraph(skeleton_corrected.astype(bool))

# 9️⃣ 데이터 유형 확인 및 변환 (디버깅)
print(f"Type of coordinates: {type(coordinates)}")  # 🎯 유형 확인

if isinstance(coordinates, tuple):
    coordinates = coordinates[0]  # 🎯 튜플이라면 첫 번째 요소 사용
print(f"Coordinates shape: {coordinates.shape}")  # 🎯 올바르게 변환되었는지 확인

# 🔹 10️⃣ 'degrees' 값을 별도로 계산하여 추가 (각 노드의 연결 개수 계산)
degrees = np.diff(graph.indptr)  # 각 노드가 연결된 개수 확인

# 🔹 11️⃣ 끝점(leaf nodes) 찾아 제거 (작은 가지 정리)
leaf_nodes = np.where(degrees == 1)[0]  # 노드 중에서 끝점(1개의 연결만 있는 노드) 찾기

# 🔹 12️⃣ 주요 중심선만 유지 (작은 가지 제거)
pruned_graph = skeleton_corrected.copy()
for node in leaf_nodes:
    # 🎯 데이터 구조에 따라 적절한 x, y 값 추출
    if coordinates.shape[1] == 2:  # (N, 2) 구조일 경우
        x, y = coordinates[node]
    elif coordinates.shape[1] == 3:  # (N, 3) 구조일 경우 (index, x, y)
        _, x, y = coordinates[node]  # 🎯 첫 번째 값(index) 무시
    else:
        print("⚠️ Unexpected coordinates shape:", coordinates.shape)
        continue  # 예상치 못한 데이터 형식이면 건너뛰기

    pruned_graph[int(y), int(x)] = 0  # 해당 픽셀 삭제

# ✅ 결과 시각화 (Skeletonization + Graph-based Filtering)
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(skeleton_corrected, cmap="gray")
ax[0].set_title("1️⃣ Original Skeletonization")
ax[0].axis("off")

ax[1].imshow(pruned_graph, cmap="gray")
ax[1].set_title("2️⃣ Graph-based Filtering Applied (Small Branches Removed)")
ax[1].axis("off")

ax[2].imshow(corrected_binary_mask, cmap="gray")
ax[2].set_title("3️⃣ Input Mask (For Reference)")
ax[2].axis("off")

plt.show()
