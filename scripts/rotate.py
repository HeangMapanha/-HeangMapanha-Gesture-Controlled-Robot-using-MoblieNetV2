import cv2
import os

folder = r"C:\Users\USER\Desktop\CNNmodel\rawdata\dataset\valid\one_finger"
angle = cv2.ROTATE_90_COUNTERCLOCKWISE  # ← choose angle here

for root, dirs, files in os.walk(folder):
    for file in files:
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(root, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        rotated = cv2.rotate(img, angle)
        cv2.imwrite(img_path, rotated)

print("✔ All images rotated!")
