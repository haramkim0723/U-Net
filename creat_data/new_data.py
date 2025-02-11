import cv2
import numpy as np
import os
import glob
import itertools

# 원본 이미지 & 마스크 폴더
image_folder = r"D:\code\Pytorch-UNet-master\creat_data\imgs"
mask_folder = r"D:\code\Pytorch-UNet-master\creat_data\masks"

# 저장 폴더
output_image_folder = r"D:\code\Pytorch-UNet-master\data\imgs"
output_mask_folder = r"D:\code\Pytorch-UNet-master\data\masks"

# 폴더가 없으면 생성
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# 채도 & 대비 조절 값
saturation_values = [20, -20]  # 채도 증가 & 감소
contrast_values = [5, 10, 15, 20]  # 대비 증가 (5 단위)

# 이미지 & 마스크 쌍으로 처리
image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))  # 이미지 파일 확장자 JPG

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f" [ERROR] 이미지 로드 실패: {img_path}")
        continue

    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]  # 확장자 제거

    # 마스크 이미지 로드 (파일명이 동일한 PNG 확장자로 존재해야 함)
    mask_path = os.path.join(mask_folder, f"{base_name}_mask.png")
    if not os.path.exists(mask_path):
        print(f"⚠ [SKIP] 마스크 없음: {mask_path}")
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f" [ERROR] 마스크 로드 실패: {mask_path}")
        continue

    # **원본 저장 (JPG 이미지 & PNG 마스크)**
    cv2.imwrite(os.path.join(output_image_folder, f"{base_name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(output_mask_folder, f"{base_name}.png"), mask)
    print(f" [ORIGINAL] {filename}")

    # **좌우 반전 (이미지 & 마스크)**
    img_flip_lr = cv2.flip(img, 1)
    mask_flip_lr = cv2.flip(mask, 1)
    cv2.imwrite(os.path.join(output_image_folder, f"{base_name}_flip_lr.jpg"), img_flip_lr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(output_mask_folder, f"{base_name}_flip_lr.png"), mask_flip_lr)
    print(f" [FLIP_LR] {filename}")

    # **상하 반전 (이미지 & 마스크)**
    img_flip_ud = cv2.flip(img, 0)
    mask_flip_ud = cv2.flip(mask, 0)
    cv2.imwrite(os.path.join(output_image_folder, f"{base_name}_flip_ud.jpg"), img_flip_ud, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(output_mask_folder, f"{base_name}_flip_ud.png"), mask_flip_ud)
    print(f" [FLIP_UD] {filename}")

    # **채도 & 대비 조합 (이미지 & 마스크 동일 적용)**
    for sat, contrast in itertools.product(saturation_values, contrast_values):
        # 채도 조정 (이미지만 적용)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat, 0, 255)
        img_mod = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 대비 조정 (이미지만 적용)
        img_mod = np.clip(img_mod + contrast, 0, 255).astype(np.uint8)

        # 저장 (JPG 이미지 & PNG 마스크)
        new_filename = f"{base_name}_s{sat}_c{contrast}"
        cv2.imwrite(os.path.join(output_image_folder, f"{new_filename}.jpg"), img_mod, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(output_mask_folder, f"{new_filename}.png"), mask)  # **마스크 원본 유지**
        print(f" [COMBO] {new_filename} (Saturation: {sat}, Contrast: {contrast})")

        # 좌우 반전 저장 (JPG 이미지 & PNG 마스크)
        cv2.imwrite(os.path.join(output_image_folder, f"{new_filename}_flip_lr.jpg"), cv2.flip(img_mod, 1), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(output_mask_folder, f"{new_filename}_flip_lr.png"), cv2.flip(mask, 1))
        print(f" [COMBO_FLIP_LR] {new_filename}")

        # 상하 반전 저장 (JPG 이미지 & PNG 마스크)
        cv2.imwrite(os.path.join(output_image_folder, f"{new_filename}_flip_ud.jpg"), cv2.flip(img_mod, 0), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(output_mask_folder, f"{new_filename}_flip_ud.png"), cv2.flip(mask, 0))
        print(f" [COMBO_FLIP_UD] {new_filename}")

    print(f"✅ [DONE] {filename}")

print(" 데이터 증강 완료!")
