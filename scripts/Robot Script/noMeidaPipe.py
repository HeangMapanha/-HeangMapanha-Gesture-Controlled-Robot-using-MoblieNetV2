import cv2
import time
import numpy as np
import onnxruntime as ort
from collections import deque
from threading import Thread
from queue import Queue
from flask import Flask, Response
from auppbot import AUPPBot

# ================= CONFIG =================
MODEL_PATH = "/home/aupp/Documents/ponita/mobilenetv2_gesture_new.onnx"
CLASS_NAMES = ["fist", "thumb", "one", "two", "palm"]
CAM_INDEX = 0
W, H = 640, 480

BUFFER_SIZE = 5            # Number of consecutive predictions to confirm
CONFIRM_THRESHOLD = 4      # Minimum times a gesture must appear in buffer to confirm
CONFIDENCE_THRESHOLD = 0.7 # Minimum probability to consider a prediction
NO_DETECTION_TIMEOUT = 1.0

# Robot config
PORT = "/dev/ttyUSB0"
BAUD = 115200
MOVE_SPEED = 15
TURN_SPEED = 20
TURN_DURATION = 0.5

# Shared frame buffer for web streaming
frame_queue = Queue(maxsize=1)

# ================= ONNX =================
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)
    return img

# ================= MOTOR HELPERS =================
def clamp(x, a=-99, b=99): return int(max(a, min(b, x)))
def set_tank(bot, left, right):
    left = clamp(left); right = clamp(right)
    try:
        bot.motor1.speed(left); bot.motor2.speed(left)
        bot.motor3.speed(right); bot.motor4.speed(right)
    except: pass

def execute_gesture(bot, gesture, last_gesture="palm"):
    if gesture == "fist": set_tank(bot, MOVE_SPEED, MOVE_SPEED)       # forward
    elif gesture == "thumb": set_tank(bot, -MOVE_SPEED, -MOVE_SPEED) # backward
    elif gesture == "one":                                           # left
        set_tank(bot, -TURN_SPEED, TURN_SPEED)
        time.sleep(TURN_DURATION)
        set_tank(bot, 0, 0)
    elif gesture == "two":                                           # right
        set_tank(bot, TURN_SPEED, -TURN_SPEED)
        time.sleep(TURN_DURATION)
        set_tank(bot, 0, 0)
    elif gesture == "palm": set_tank(bot, 0, 0)                      # stop
    else:
        # continue last movement if unknown
        if last_gesture == "fist": set_tank(bot, MOVE_SPEED, MOVE_SPEED)
        elif last_gesture == "thumb": set_tank(bot, -MOVE_SPEED, -MOVE_SPEED)

# ==================== ROBOT LOOP ====================
def robot_loop():
    bot = None
    try: bot = AUPPBot(PORT, BAUD, auto_safe=True)
    except: print("⚠ Robot dry run mode")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(3, W); cap.set(4, H)

    gesture_buffer = deque(maxlen=BUFFER_SIZE)
    last_confirmed = "palm"
    last_detect_time = time.time()
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)

        # Preprocess full frame
        inp = preprocess(frame)
        pred = session.run(None, {input_name: inp})[0][0]
        idx = int(np.argmax(pred))
        conf = float(np.max(pred))
        gesture = CLASS_NAMES[idx]

        # Only add gesture to buffer if confidence is high enough
        if conf >= CONFIDENCE_THRESHOLD:
            gesture_buffer.append(gesture)

        # Count gestures in buffer
        counts = {cls:0 for cls in CLASS_NAMES}
        for g in gesture_buffer: counts[g] += 1

        # Confirm gesture if count >= CONFIRM_THRESHOLD
        confirmed_gesture = None
        for g, c in counts.items():
            if c >= CONFIRM_THRESHOLD:
                confirmed_gesture = g
                last_confirmed = g
                break

        # Handle timeout: no confident detection for a while
        if time.time() - last_detect_time > NO_DETECTION_TIMEOUT:
            confirmed_gesture = "palm"
            last_confirmed = "palm"
        else:
            if not confirmed_gesture:
                confirmed_gesture = last_confirmed

        # Execute gesture
        if confirmed_gesture:
            execute_gesture(bot, confirmed_gesture, last_confirmed)

        # Draw label
        label = f"{confirmed_gesture} ({conf:.2f})" if confirmed_gesture else "Detecting..."
        cv2.putText(frame, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # FPS
        curr_time = time.time()
        fps = 1/(curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        # Update shared frame for streaming
        if not frame_queue.full(): frame_queue.put(frame.copy())

# ==================== WEB SERVER ====================
app = Flask(__name__)

def generate_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame_bytes+b'\r\n')

@app.route('/video_feed')
def video_feed(): 
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index(): 
    return "<h1>Robot Camera Feed</h1><img src='/video_feed' style='width:100%;height:100%;object-fit:contain'>"

# ==================== MAIN ====================
if __name__ == "__main__":
    t = Thread(target=robot_loop)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False)
