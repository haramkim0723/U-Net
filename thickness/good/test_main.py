import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from r_btest import ThickLast  # ThickLast 클래스 사용

# 🔹 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def PexelToCm(x, real_cm=0.5, index=1):
    """
    리스트, 딕셔너리를 지원하는 픽셀-실제 변환 함수 (특정 인덱스 값 사용)
    """
    logging.debug(f"PexelToCm 호출됨: x={x}, 타입: {type(x)}, 사용 인덱스: {index}")

    try:
        if isinstance(x, list):
            if len(x) <= index:
                raise IndexError(f"[ERROR] 리스트 길이가 {index}보다 작음: {x}")
            x = x[index]
            logging.info(f"리스트에서 선택된 값: {x}")

        elif isinstance(x, dict):
            values = list(x.values())
            if len(values) <= index:
                raise IndexError(f"[ERROR] 딕셔너리 값 개수가 {index}보다 작음: {x}")
            x = values[index]
            logging.info(f"딕셔너리에서 선택된 값: {x}")

        elif not isinstance(x, (int, float)):
            raise TypeError(f"[ERROR] 숫자가 아닌 값이 전달됨: {x} (타입: {type(x)})")

        if x == 0:
            raise ZeroDivisionError("[ERROR] PexelToCm 함수에서 0으로 나눌 수 없음")

        return real_cm / x

    except (TypeError, ValueError, IndexError, ZeroDivisionError) as e:
        logging.error(f"변환 실패: {e}")
        return None  # 오류 발생 시 None 반환


def main():
    """ 주 실행 함수 """
    logging.info("프로그램 시작")

    image_paths = [
        ("D:/New folder.cancelled/HuggingFaceLLM/new_data/data/masks/20241110_150755_stem_0_mask.png", 3),
        ("D:/code/Pytorch-UNet-master/data/masks/20241110_160723_refbar_7.png", 4)  # 기준 막대
    ]

    real_pexel = None  # 픽셀-실제 변환 비율 저장
    x = {}  # 각 클래스별 줄기 굵기 저장
    contour_visualizations = []

    # 🔹 Class 4(기준 막대)를 먼저 실행하여 변환 비율 계산 (한 번만 실행)
    for i, (image_path, class_label) in enumerate(image_paths):
        if class_label == 4 and real_pexel is None:
            logging.info(f"기준 막대 처리 중: {image_path}")
            thickness = ThickLast(image_path, class_label)

            thickness.preprocess_image()
            thickness.extract_contours()
            thickness.skeletonize_image()

            sampled_points = thickness.sample_skeleton_points()
            thickness.compute_distances(sampled_points)
            thickness.filter_points(sampled_points)

            if not isinstance(x, dict):
                logging.error(f"x가 딕셔너리가 아님! 현재 타입: {type(x)}, 값: {x}")
                x = {}  # 다시 딕셔너리로 초기화

            x["Class_4"] = thickness.compute_stem_thickness()
            logging.debug(f"기준 막대 (Class 4)의 두께: {x['Class_4']}, 타입: {type(x['Class_4'])}")

            real_pexel = PexelToCm(x["Class_4"], index=1)
            if real_pexel is None:
                logging.error("기준 막대 변환 비율 계산 실패!")
                raise ValueError("[ERROR] 기준 막대 변환 비율 계산 실패!")
            logging.info(f"기준 막대 변환 비율: {real_pexel:.5f} cm/px")

    # 🔹 Class 3(줄기) 실행 - 기준 막대 변환 비율을 적용
    for i, (image_path, class_label) in enumerate(image_paths):
        if class_label == 3:
            logging.info(f"Processing Image: {image_path}, Class {class_label}")

            thickness = ThickLast(image_path, class_label)

            thickness.preprocess_image()
            thickness.extract_contours()
            thickness.skeletonize_image()

            sampled_points = thickness.sample_skeleton_points()
            thickness.compute_distances(sampled_points)
            thickness.filter_points(sampled_points)

            if not isinstance(x, dict):
                logging.error(f"x가 딕셔너리가 아님! 현재 타입: {type(x)}, 값: {x}")
                x = {}  # 다시 딕셔너리로 초기화

            x[f"Class_{i}"] = thickness.compute_stem_thickness()
            logging.debug(f"x[f'Class_{i}'] = {x[f'Class_{i}']}, 타입: {type(x[f'Class_{i}'])}")

            if real_pexel is None:
                logging.warning("기준 막대 변환 비율이 없어서 줄기 변환을 수행하지 못함.")
            else:
                logging.info("줄기 두께를 실제 크기로 변환 중...")
                stem_real_cm = x[f"Class_{i}"] * real_pexel
                logging.info(f"변환된 줄기 두께: {stem_real_cm:.3f} cm")

    logging.info("최종 변환 완료!")


if __name__ == "__main__":
    main()
