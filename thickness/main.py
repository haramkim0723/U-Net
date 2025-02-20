import cv2
from yolo_unet_processing import process_yolo_unet
from skeleton_analysis import analyze_stem_thickness
from data_management import separate_stem_and_ruler
from real_thick import real_thick
# 실제 기준 막대 길이 = 19.5cm, 굵기 = 0.5cm

# 실제 기준 막대 크기 (길이 & 굵기)
RULER_ACTUAL_LENGTH_CM = 19.5  # 기준 막대 길이 (사용하지 않지만 참고 가능)
RULER_ACTUAL_THICKNESS_CM = 0.5  # 기준 막대 굵기

# YOLO 검출 결과 예제 (줄기: 4, 기준 막대: 5)
yolo_results = [
    [4, 50, 50, 200, 200, 0.9],
    [5, 220, 60, 350, 180, 0.8]
]

# 원본 이미지 로드
image = cv2.imread("sample.jpg")

# YOLO + U-Net 결합 처리
segmentation_outputs = process_yolo_unet(yolo_results, image, unet_model)

# 두께 분석 수행
thickness_results = analyze_stem_thickness(segmentation_outputs)

# 줄기 & 기준 막대 데이터 분리
stem_mask, ruler_mask, stem_thickness, ruler_thickness = separate_stem_and_ruler(segmentation_outputs,
                                                                                 thickness_results)

# 실제 줄기 두께(cm) 계산
actual_stem_thickness = real_thick(stem_thickness, ruler_thickness, RULER_ACTUAL_THICKNESS_CM)

# 결과 출력
print(f" 기준 막대 픽셀 두께: {ruler_thickness:.2f} px")
print(f" 줄기 픽셀 두께: {stem_thickness:.2f} px")
print(f" 줄기의 실제 두께: {actual_stem_thickness:.2f} cm")
