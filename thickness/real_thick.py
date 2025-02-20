
def real_thick(stem_thickness_px, ruler_thickness_px, ruler_actual_thickness_cm=0.5):
    """
    기준 막대의 실제 크기를 이용하여 줄기의 실제 두께(cm)를 계산.

    :param stem_thickness_px: 줄기의 픽셀 단위 두께
    :param ruler_thickness_px: 기준 막대의 픽셀 단위 두께
    :param ruler_actual_thickness_cm: 기준 막대의 실제 굵기(cm) (기본값 0.5cm)
    :return: 줄기의 실제 두께(cm)
    """
    if not ruler_thickness_px or not stem_thickness_px:
        return None

        # 픽셀당 실제 길이 (cm/px)
    cm_per_pixel = ruler_actual_thickness_cm / ruler_thickness_px

    # 줄기의 실제 두께 계산
    actual_stem_thickness_cm = stem_thickness_px * cm_per_pixel

    return actual_stem_thickness_cm
