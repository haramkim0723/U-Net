import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from thickness.ThickLast import ThickLast


def PexelToCm(x, real_cm=0.5):
    """
    클래스 4의 픽셀 두께를 기준으로 다른 두께를 실제 거리로 변환하는 함수.
    :param x: 클래스 4의 픽셀 단위 두께
    :param real_cm: 클래스 4의 실제 두께 (기본값: 0.5cm)
    :return: 픽셀당 cm 변환 비율
    """
    return real_cm / x

##("D:/code/Pytorch-UNet-master/data/masks/20241110_150259_stem_5_s20_c15_flip_ud.png", 3),  # 첫 번째 이미지 경로 및 클래스
if __name__ == "__main__":
    image_paths = [
        ("D:/New folder.cancelled/HuggingFaceLLM/new_data/data/masks/20241110_150755_stem_0_mask.png", 3),
        ("D:/code/Pytorch-UNet-master/data/masks/20241110_160723_refbar_7.png", 4)  # 두 번째 이미지 경로 및 클래스
    ]

    real_pexel = None
    x = {}

    for i, (image_path, class_label) in enumerate(image_paths):
        try:
            thickness = ThickLast(image_path, class_label)
            thickness.preprocess_image()
            plt.figure(figsize=(6, 6))
            plt.imshow(thickness.image, cmap='gray')
            plt.title(f"Preprocessed Image - Class {class_label}")
            plt.axis("off")
            plt.show()

            thickness.extract_contours()
            thickness.skeletonize_image()
            plt.figure(figsize=(6, 6))
            plt.imshow(thickness.skeleton, cmap='gray')
            plt.title(f"Skeletonized Image - Class {class_label}")
            plt.axis("off")
            plt.show()

            sampled_points = thickness.sample_skeleton_points()
            thickness.compute_distances(sampled_points)
            thickness.filter_points(sampled_points)
            thickness.visualize_results()
            x[i] = thickness.compute_stem_thickness()
            print(f"[{class_label}] 추정된 줄기 굵기: {x[i]:.2f} 픽셀")

            if class_label == 4:
                real_pexel = PexelToCm(x[i])
                print(f"픽셀당 cm 변환 비율: {real_pexel:.5f} cm/px")

        except FileNotFoundError as e:
            print(e)

    if real_pexel:
        for i, pixel in x.items():
            real_thickness = pixel * real_pexel
            print(f"[{image_paths[i][1]}] 실제 줄기 굵기: {real_thickness:.3f} cm")
