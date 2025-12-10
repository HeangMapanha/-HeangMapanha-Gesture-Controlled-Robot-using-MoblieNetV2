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
MODEL_PATH = "hand_model.onnx"
CLASS_NAMES = ["fist", "thumb", "one", "two", "palm"]
CAM_INDEX = 0
W, H = 224, 224
BUFFER_SIZE = 3
CONFIRM_THRESHOLD = 2
CONFIDENCE_THRESHOLD = 0.8  # Only accept predictions above this
NO_DETECTION_TIMEOUT = 1  # Short timeout to immediately stop when no hand

# Robot config
PORT = "/dev/ttyUSB0"
BAUD = 115200
MOVE_SPEED = 15
TURN_SPEED = 18
TURN_DURATION = 0.5

# ================= QUEUES =================
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
    left = clamp(left)
    right = clamp(right)
    try:
        bot.motor1.speed(left); bot.motor2.speed(left)
        bot.motor3.speed(right); bot.motor4.speed(right)
    except: pass

def execute_gesture(bot, gesture):
    action = "Unknown"
    if gesture == "fist":
        set_tank(bot, MOVE_SPEED, MOVE_SPEED)
        action = "Moving Forward"
    elif gesture == "thumb":
        set_tank(bot, -MOVE_SPEED, -MOVE_SPEED)
        action = "Moving Backward"
    elif gesture == "one":
        set_tank(bot, -TURN_SPEED, TURN_SPEED)
        time.sleep(TURN_DURATION)
        set_tank(bot, 0, 0)
        action = "Turning Left"
    elif gesture == "two":
        set_tank(bot, TURN_SPEED, -TURN_SPEED)
        time.sleep(TURN_DURATION)
        set_tank(bot, 0, 0)
        action = "Turning Right"
    elif gesture == "palm":
        set_tank(bot, 0, 0)
        action = "Stopped"
    print(f"[GESTURE] Detected: {gesture}, Action: {action}")
    return action

# ================= ROBOT LOOP =================
def robot_loop():
    bot = None
    try:
        bot = AUPPBot(PORT, BAUD, auto_safe=True)
    except:
        print("⚠ Robot dry run mode")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(3, W); cap.set(4, H)
    gesture_buffer = deque(maxlen=BUFFER_SIZE)
    last_detect_time = time.time()
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Skin mask
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.GaussianBlur(mask, (5,5), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        confirmed_gesture = None

        if contours:
            hand = max(contours, key=cv2.contourArea)
            if cv2.contourArea(hand) > 1000:
                x, y, w, h = cv2.boundingRect(hand)
                hand_crop = frame[y:y+h, x:x+w]
                if hand_crop.size > 0:
                    inp = preprocess(hand_crop)
                    pred = session.run(None, {input_name: inp})[0][0]
                    idx = int(np.argmax(pred))
                    conf = float(np.max(pred))

                    if conf >= CONFIDENCE_THRESHOLD:
                        gesture = CLASS_NAMES[idx]
                        gesture_buffer.append(gesture)
                        counts = {cls:0 for cls in CLASS_NAMES}
                        for g in gesture_buffer: counts[g] += 1
                        for g, c in counts.items():
                            if c >= CONFIRM_THRESHOLD:
                                confirmed_gesture = g
                                break
                    # Draw box & label
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                    label = f"{confirmed_gesture} ({conf:.2f})" if confirmed_gesture else "Detecting..."
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                last_detect_time = time.time()
        else:
            gesture_buffer.clear()
            confirmed_gesture = "palm"

        # Execute gesture only if detected
        if confirmed_gesture:
            execute_gesture(bot, confirmed_gesture)

        # FPS display
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        if not frame_queue.full():
            frame_queue.put(frame.copy())

# ================= WEB SERVER =================
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
    return """
    <html>
    <head>
        <title>Robot Camera Feed</title>
        <style>
            body, html { margin:0; padding:0; height:100%; width:100%; background:black; display:flex; justify-content:center; align-items:center; }
            #videoFeed { width:100%; height:100%; object-fit:contain; }
        </style>
    </head>
    <body>
        <img id="videoFeed" src="/video_feed">
    </body>
    </html>
    """

# ================= MAIN =================
if __name__ == "__main__":
    t_robot = Thread(target=robot_loop, daemon=True)
    t_robot.start()
    app.run(host='0.0.0.0', port=5000, debug=False)
