# object_info.py

class BaseObjectInfo:
    def __init__(self, class_id, class_name, bbox, cropped_image, mask=None):
        self.class_id = class_id
        self.class_name = class_name
        self.bbox = bbox
        self.cropped_image = cropped_image
        self.mask = mask
        self.thickness_px = None


class ReferenceBarInfo(BaseObjectInfo):
    def __init__(self, bbox, cropped_image, mask=None):
        super().__init__(class_id=4, class_name='reference bar', bbox=bbox, cropped_image=cropped_image, mask=mask)
        self.reference_thickness = None


class StemInfo(BaseObjectInfo):
    def __init__(self, bbox, cropped_image, mask=None):
        super().__init__(class_id=3, class_name='stem', bbox=bbox, cropped_image=cropped_image, mask=mask)
        self.thickness_cm = None
