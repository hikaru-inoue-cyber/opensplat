import cv2
import glob

def detect_blur(image_path, threshold=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian < threshold, laplacian  # Trueならブレている

# 画像フォルダ内のすべての画像をチェック
image_files = glob.glob("path/to/images/*.jpg")
for img in image_files:
    is_blurred, score = detect_blur(img)
    if is_blurred:
        print(f"ブレ: {img} (スコア: {score:.2f})")
    else:
        print(f"クリア: {img} (スコア: {score:.2f})")
