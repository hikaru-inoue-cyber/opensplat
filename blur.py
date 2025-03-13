import cv2
import glob
import os
import shutil
import argparse

THRESHOLD=30

def detect_blur(image_path, threshold=THRESHOLD):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian < threshold, laplacian

def move_blurred_images(src_dir, ng_dir, threshold=30):
    os.makedirs(ng_dir, exist_ok=True)
    image_files = glob.glob(os.path.join(src_dir, "*.png"))

    for img in image_files:
        is_blurred, score = detect_blur(img, threshold)
        if is_blurred:
            print(f"NG : {img} (Score : {score:.2f} ) - Moving to {ng_dir}")
            shutil.move(img, os.path.join(ng_dir, os.path.basename(img)))
        else:
            print(f"OK : {img} (Score : {score:.2f} )")

parser = argparse.ArgumentParser(description="Detect and move blurred images.")
parser.add_argument("src_directory", type=str, help="Source directory containing images")
args = parser.parse_args()

src_directory = args.src_directory
ng_directory = os.path.join(src_directory, "NG")

move_blurred_images(src_directory, ng_directory)
