from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import threading
import pygame
import time
import requests

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("D:\\sheraz\\yolo_web_app\\best (1).pt")  # Replace with your model path

# Initialize Pygame for sound
pygame.mixer.init()
beep_path = "beep.mp3"

# Globals for streaming
last_beep_time = 0
beep_interval = 2  # seconds
receiver_ip = None
is_streaming = False

# Lock to prevent multiple camera access
camera_lock = threading.Lock()

def play_beep():
    try:
        pygame.mixer.music.load(beep_path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"❌ Beep Error: {e}")

def send_frame_to_receiver(frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        requests.post(
            f'http://{receiver_ip}:5000/stream',
            data=buffer.tobytes(),
            headers={'Content-Type': 'image/jpeg'},
            timeout=0.5
        )
    except Exception as e:
        print(f"❌ Stream Error: {e}")

def open_camera_with_retry(retries=5, delay=1):
    for attempt in range(retries):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print(f"✅ Camera opened on attempt {attempt + 1}")
            return cap
        print(f"🔁 Retry {attempt + 1}: Failed to open camera")
        time.sleep(delay)
    return None

def gen_frames():
    global last_beep_time, is_streaming, receiver_ip
    with camera_lock:
        cap = open_camera_with_retry()
        if cap is None:
            print("❌ Camera could not be opened after retries.")
            return

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print("❌ Failed to read frame.")
                    break

                results = model(frame, imgsz=640, conf=0.4)[0]
                detected = False

                for box in results.boxes:
                    cls_id = int(box.cls)
                    label = model.names[cls_id]
                    conf = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    color = (0, 255, 0)
                    if "corrosion" in label:
                        color = (0, 0, 255)
                        detected = True
                    elif "fracture" in label:
                        color = (0, 165, 255)
                        detected = True

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if detected and (time.time() - last_beep_time > beep_interval):
                    threading.Thread(target=play_beep).start()
                    last_beep_time = time.time()

                if is_streaming and receiver_ip:
                    threading.Thread(target=send_frame_to_receiver, args=(frame.copy(),)).start()

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            cap.release()
            print("📷 Camera released.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_streaming/<ip>')
def start_streaming(ip):
    global receiver_ip, is_streaming
    receiver_ip = ip
    is_streaming = True
    return f"🔴 Streaming started to {ip}"

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
