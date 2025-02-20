import cv2
import numpy as np


def process_yolo_unet(yolo_detections, image, unet_model, target_classes=[4, 5]):
    """
    YOLO 검출 결과를 바탕으로 이미지를 잘라낸 후 U-Net에 입력하여 세그멘테이션 수행

    :param yolo_detections: YOLO 결과 (class_id, x_min, y_min, x_max, y_max, confidence)
    :param image: 원본 이미지 (numpy 배열)
    :param unet_model: U-Net 모델
    :param target_classes: 처리할 클래스 리스트 (기본값: 줄기(4), 기준 막대(5))
    :return: YOLO 클래스 정보가 포함된 U-Net 세그멘테이션 결과 리스트
    """
    segmented_results = []

    for detection in yolo_detections:
        class_id, x_min, y_min, x_max, y_max, confidence = detection

        if class_id in target_classes:
            cropped_img = image[y_min:y_max, x_min:x_max]

            # U-Net 입력 데이터 전처리
            cropped_resized = cv2.resize(cropped_img, (256, 256))
            cropped_resized = cropped_resized / 255.0
            cropped_resized = np.expand_dims(cropped_resized, axis=0)

            # U-Net 예측 수행
            unet_mask = unet_model.predict(cropped_resized)[0]

            # YOLO의 class_id 정보 포함
            segmented_results.append({
                "class_id": class_id,
                "segmentation_mask": unet_mask
            })

    return segmented_results
