def separate_stem_and_ruler(segmentation_outputs, thickness_results):
    """
    YOLO + U-Net 분석 결과에서 줄기(Stem)와 기준 막대를 변수로 분리

    :param segmentation_outputs: YOLO + U-Net 결과 리스트
    :param thickness_results: 각 클래스별 두께 계산 결과
    :return: 줄기와 기준 막대의 마스크 및 두께 정보를 각각 반환
    """
    stem_mask = None
    ruler_mask = None
    stem_thickness = None
    ruler_thickness = None

    for output in segmentation_outputs:
        class_id = output["class_id"]
        mask = output["segmentation_mask"]

        if class_id == 4:
            stem_mask = mask
            stem_thickness = thickness_results.get(4, None)
        elif class_id == 5:
            ruler_mask = mask
            ruler_thickness = thickness_results.get(5, None)

    return stem_mask, ruler_mask, stem_thickness, ruler_thickness
