
import cv2
import os
import mediapipe as mp
import numpy as np
import random

# --------------------
# Configurations
# --------------------
videos_path = r"C:\Users\USER\Desktop\CNNmodel\rawdata\video\valid"
output_path = r"C:\Users\USER\Desktop\CNNmodel\rawdata\dataset"
fps_extract = 5
split = "valid"

apply_flip = True
apply_brightness = True
brightness_range = (0.7, 1.3)  # brightness multiplier

# --------------------
# Setup MediaPipe
# --------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

# --------------------
# Helper functions
# --------------------
def crop_hand(image, hand_landmarks, expand_ratio=0.15):
    h, w, _ = image.shape

    x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
    x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
    y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
    y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

    x_min = int(x_min)
    x_max = int(x_max)
    y_min = int(y_min)
    y_max = int(y_max)

    box_w = x_max - x_min
    box_h = y_max - y_min

    pad_w = int(box_w * expand_ratio)
    pad_h = int(box_h * expand_ratio)

    x_min = max(0, x_min - pad_w)
    y_min = max(0, y_min - pad_h)
    x_max = min(w, x_max + pad_w)
    y_max = min(h, y_max + pad_h)

    crop_w = x_max - x_min
    crop_h = y_max - y_min

    if crop_w > crop_h:
        diff = crop_w - crop_h
        y_min = max(0, y_min - diff // 2)
        y_max = min(h, y_max + diff // 2)
    elif crop_h > crop_w:
        diff = crop_h - crop_w
        x_min = max(0, x_min - diff // 2)
        x_max = min(w, x_max + diff // 2)

    return image[y_min:y_max, x_min:x_max]

def augment_image(image):
    if apply_flip and random.random() < 0.5:
        image = cv2.flip(image, 1)
    if apply_brightness:
        factor = random.uniform(*brightness_range)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
    return image

# --------------------
# Process videos
# --------------------
for video_file in os.listdir(videos_path):
    if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
        continue

    class_name = os.path.splitext(video_file)[0]
    class_output = os.path.join(output_path, split, class_name)
    os.makedirs(class_output, exist_ok=True)

    cap = cv2.VideoCapture(os.path.join(videos_path, video_file))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps // fps_extract))

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Fix for portrait mode if needed
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if frame_count % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]

                # Crop
                cropped = crop_hand(frame, hand_landmarks)
                cropped = cv2.resize(cropped, (224, 224))

                # Augment
                final_img = augment_image(cropped)

                save_path = os.path.join(class_output, f"{saved_count}.jpg")
                cv2.imwrite(save_path, final_img)
                saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} images for class '{class_name}'")

hands.close()
print("All videos processed!")
