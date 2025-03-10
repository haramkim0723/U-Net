import os
import cv2
import numpy as np
import logging
from PIL import Image
import torch
from unet import UNet
from utils.data_loading import BasicDataset
from thickness.ThickLast import ThickLast

# 🔹 로그 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_class_map(dataset_path):
    class_map = {}
    with open(os.path.join(dataset_path, "classes.txt"), "r") as f:
        for idx, class_name in enumerate(f.readlines()):
            class_map[class_name.strip()] = idx
    logging.info(f"클래스 매핑 로드 완료: {class_map}")
    return class_map


def crop_object(image, bbox, w, h):
    x_center, y_center, bbox_width, bbox_height = bbox
    x_center, y_center = x_center * w, y_center * h
    bbox_width, bbox_height = bbox_width * w, bbox_height * h

    x1, y1 = max(0, int(x_center - bbox_width / 2)), max(0, int(y_center - bbox_height / 2))
    x2, y2 = min(w, int(x_center + bbox_width / 2)), min(h, int(y_center + bbox_height / 2))

    logging.debug(f"Cropped bbox: ({x1}, {y1}) ~ ({x2}, {y2})")
    return image[y1:y2, x1:x2]


def predict_mask(net, img, device):
    try:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = torch.from_numpy(BasicDataset.preprocess(None, img, scale=1.0, is_mask=False))
        img = img.unsqueeze(0).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = net(img).cpu()
            mask = torch.sigmoid(output).squeeze().numpy()
            mask = (mask > 0.5).astype(np.uint8)

        logging.debug(f"U-Net mask prediction 성공")
        return mask

    except Exception as e:
        logging.error(f"U-Net mask prediction 실패: {e}")
        raise


def process_image(image_path, label_path, class_map, net, device):
    logging.info(f"이미지 처리 시작: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"이미지 로드 실패: {image_path}")
        return

    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        lines = [line.strip().split() for line in f.readlines()]

    reference_thickness = None
    stem_thickness = None

    for idx, data in enumerate(lines):
        class_id = int(data[0])
        bbox = list(map(float, data[1:]))

        cropped = crop_object(image, bbox, w, h)
        mask = predict_mask(net, cropped, device)

        thickness_calculator = ThickLast(None, class_id)
        thickness_calculator.image = (mask * 255).astype(np.uint8)

        thickness_calculator.preprocess_image()
        thickness_calculator.extract_contours()
        thickness_calculator.skeletonize_image()

        sampled_points = thickness_calculator.sample_skeleton_points()
        thickness_calculator.compute_distances(sampled_points, save_csv=False)
        thickness_calculator.filter_points(sampled_points)

        thickness = thickness_calculator.compute_stem_thickness()
        logging.info(f"Class {class_id} 두께 측정 완료: {thickness:.3f}px")

        if class_id == class_map['reference bar']:
            reference_thickness = thickness
        elif class_id == class_map['stem']:
            stem_thickness = thickness

    if reference_thickness and stem_thickness:
        real_pexel = 0.5 / reference_thickness
        stem_real_cm = stem_thickness * real_pexel
        logging.info(f" 최종 실제 줄기 두께: {stem_real_cm:.3f} cm")


def process_all_images(dataset_path, model_path):
    logging.info(f"전체 이미지 처리 시작 (데이터셋 경로: {dataset_path})")

    class_map = load_class_map(dataset_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=1)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    for file in os.listdir(dataset_path):
        if file.endswith(".txt"):
            label_path = os.path.join(dataset_path, file)
            image_path = label_path.replace(".txt", ".jpg")
            if not os.path.exists(image_path):
                image_path = label_path.replace(".txt", ".png")

            if not os.path.exists(image_path):
                logging.warning(f"이미지 파일 없음: {image_path}")
                continue

            process_image(image_path, label_path, class_map, net, device)

    logging.info(" 전체 이미지 처리 완료")
