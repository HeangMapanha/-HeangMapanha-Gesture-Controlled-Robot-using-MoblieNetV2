import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import time

# -----------------------------
# Load ONNX model
# -----------------------------
onnx_model_path = r"C:\Users\User\Desktop\scripts\mobilenetv2_gesture2.onnx"
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# -----------------------------
# MediaPipe Hands setup
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.6
)

# -----------------------------
# Labels and preprocessing
# -----------------------------
labels = ['fist', 'one', 'palm', 'thumb', 'two']

def preprocess_for_model(bgr_img):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    if arr.shape[:2] != (224, 224):
        arr = cv2.resize(arr, (224, 224))
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def make_square(img):
    h, w = img.shape[:2]
    size = max(h, w)
    square = np.ones((size, size, 3), dtype=np.uint8) * 255
    y = (size - h) // 2
    x = (size - w) // 2
    square[y:y+h, x:x+w] = img
    return square

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

# -----------------------------
# Webcam capture
# -----------------------------
cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]

            x_min_raw = int(min(xs) * w)
            y_min_raw = int(min(ys) * h)
            x_max_raw = int(max(xs) * w)
            y_max_raw = int(max(ys) * h)

            box_w = x_max_raw - x_min_raw
            box_h = y_max_raw - y_min_raw
            pad = int(0.2 * max(box_w, box_h))
            x_min = max(0, x_min_raw - pad)
            y_min = max(0, y_min_raw - pad)
            x_max = min(w, x_max_raw + pad)
            y_max = min(h, y_max_raw + pad)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            hand_img = make_square(hand_img)
            inp = preprocess_for_model(hand_img)

            logits = session.run([output_name], {input_name: inp})[0]
            logits = logits.reshape(1, -1)

            if np.allclose(logits.sum(), 1.0, atol=1e-3):
                probs = logits
            else:
                probs = softmax(logits)

            pred_idx = int(np.argmax(probs))
            pred_label = labels[pred_idx]
            confidence = float(probs[0, pred_idx])

            display_text = f"{pred_label} {confidence:.2f}"
            cv2.putText(frame, display_text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # -----------------------------
    # Calculate FPS
    # -----------------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Gesture Recognition (ONNX)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
