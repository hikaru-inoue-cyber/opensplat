import cv2
import numpy as np
import os
import logging
import shutil
from ultralytics import YOLO
import glob

model = YOLO("yolov8m.pt")  # (n: nano, s: small, m: medium)
model.verbose = False
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def detect_main_object(image):
    results = model(image)
    height, width = image.shape[:2]

    center_x, center_y = width // 2, height // 2
    min_dist, main_obj = float("inf"), None

    for r in results:
        for box in r.boxes.xyxy:  # x1, y1, x2, y2
            x1, y1, x2, y2 = map(int, box[:4])
            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2
            dist = (center_x - obj_center_x) ** 2 + (center_y - obj_center_y) ** 2

            if dist < min_dist:
                min_dist, main_obj = dist, (x1, y1, x2, y2)

    return main_obj  # (x1, y1, x2, y2)

def detect_blur(image, bbox, threshold=50):
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian < threshold, laplacian

def process_directory(directory_path, ng_directory):
    image_files = glob.glob(os.path.join(directory_path, "*.jpg"))
    image_files += glob.glob(os.path.join(directory_path, "*.png"))

    if not os.path.exists(ng_directory):
        os.makedirs(ng_directory)

    for image_path in image_files:
        image = cv2.imread(image_path)

        bbox = detect_main_object(image)

        if bbox:
            is_blurred, score = detect_blur(image, bbox)
            print(f"{image_path} : {'NG' if is_blurred else 'OK'} Score : {score:.2f}")

            if is_blurred:
                new_path = os.path.join(ng_directory, os.path.basename(image_path))
                shutil.move(image_path, new_path)
        else:
            print(f"{image_path} : Can not detect object")

process_directory("test", "test/NG")
