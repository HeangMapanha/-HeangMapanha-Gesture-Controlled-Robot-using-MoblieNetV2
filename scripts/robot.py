import cv2
import time
import numpy as np
import onnxruntime as ort
import mediapipe as mp
from collections import deque
from threading import Thread
from queue import Queue
from flask import Flask, Response
from auppbot import AUPPBot

# ================= CONFIG =================
MODEL_PATH = "hand_model.onnx"
CLASS_NAMES = ["fist", "thumb", "one", "two", "palm"]
CAM_INDEX = 0
W, H = 640, 480
BUFFER_SIZE = 5
CONFIRM_THRESHOLD = 4

# Robot config
PORT = "/dev/ttyUSB0"
BAUD = 115200
MOVE_SPEED = 20
TURN_SPEED = 15
TURN_DURATION = 0.5

# Timeout after losing hand detection before stopping (seconds)
NO_DETECTION_TIMEOUT = 1.0

# Shared frame buffer for web streaming
frame_queue = Queue(maxsize=1)

# ================= ONNX & MEDIAPIPE =================
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)/255.0
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
    if gesture == "fist":  # forward
        set_tank(bot, MOVE_SPEED, MOVE_SPEED)
    elif gesture == "thumb":  # reverse
        set_tank(bot, -MOVE_SPEED, -MOVE_SPEED)
    elif gesture == "one":  # left
        set_tank(bot, -TURN_SPEED, TURN_SPEED)
        time.sleep(TURN_DURATION)
        set_tank(bot, 0, 0)
    elif gesture == "two":  # right
        set_tank(bot, TURN_SPEED, -TURN_SPEED)
        time.sleep(TURN_DURATION)
        set_tank(bot, 0, 0)
    elif gesture == "palm":  # stop
        set_tank(bot, 0, 0)
    else:
        # continue last movement if unknown gesture
        if last_gesture == "fist":
            set_tank(bot, MOVE_SPEED, MOVE_SPEED)
        elif last_gesture == "thumb":
            set_tank(bot, -MOVE_SPEED, -MOVE_SPEED)

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

    fps = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        confirmed_gesture = None

        if result.multi_hand_landmarks:
            last_detect_time = time.time()
            lm = result.multi_hand_landmarks[0]
            xs = [p.x for p in lm.landmark]; ys = [p.y for p in lm.landmark]
            x1, y1 = int(min(xs)*W), int(min(ys)*H)
            x2, y2 = int(max(xs)*W), int(max(ys)*H)
            pad = 20; x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(W,x2+pad); y2=min(H,y2+pad)
            hand_crop = frame[y1:y2, x1:x2]

            if hand_crop.size > 0:
                inp = preprocess(hand_crop)
                pred = session.run(None, {input_name: inp})[0][0]
                idx = int(np.argmax(pred)); conf = float(np.max(pred))
                gesture = CLASS_NAMES[idx]

                gesture_buffer.append(gesture)
                counts = {cls:0 for cls in CLASS_NAMES}
                for g in gesture_buffer: counts[g]+=1
                for g,c in counts.items():
                    if c>=CONFIRM_THRESHOLD:
                        confirmed_gesture = g
                        last_confirmed = g
                        break

                # draw box + label
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                if confirmed_gesture:
                    cv2.putText(frame,f"{confirmed_gesture} ({conf:.2f})",(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                else:
                    cv2.putText(frame,"Detecting...",(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        else:
            # No hand detected → check timeout
            if time.time() - last_detect_time > NO_DETECTION_TIMEOUT:
                confirmed_gesture = "palm"  # Stop
                last_confirmed = "palm"
            else:
                # keep last gesture temporarily
                confirmed_gesture = last_confirmed

        # Execute gesture
        if confirmed_gesture: execute_gesture(bot, confirmed_gesture, last_confirmed)

        # ------------------- FPS calculation -------------------
        curr_time = time.time()
        fps = 1/(curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

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
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index(): return "<h1>Robot Camera Feed</h1><img src='/video_feed'>"

# ==================== MAIN ====================
if __name__ == "__main__":
    import threading
    t = threading.Thread(target=robot_loop)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False)
