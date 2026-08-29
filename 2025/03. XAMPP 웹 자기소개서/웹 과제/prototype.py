from flask import Flask, render_template, Response, jsonify, request
from camera import Camera 
import threading
import time
import serial
import psutil
from collections import deque
import requests
import cv2
import numpy as np
from ultralytics import YOLO 

app = Flask(__name__)

# --- 설정 ---
NOTEBOOK_SERVER_URL = "http://172.16.11.200:5000"

# --- YOLOv8-seg ONNX 모델 로드 ---
try:
    model = YOLO('yolov5/best.onnx', task='segment')
    print(f"✅ YOLOv8-seg 모델 로드 성공. 클래스: {model.names}")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    model = None

# --- 색상 팔레트 ---
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(model.names), 3)).astype(int)

# --- 데이터 저장소 ---
current_stats = { "cpu": 0, "memory": 0, "fps": 0 }
detection_log = deque(maxlen=20)
action_log = deque(maxlen=20)
system_log = deque(maxlen=20)
latest_results = [] 

data_lock = threading.Lock()
cam = Camera()
ser = None

# --- 백그라운드 스레드: 시스템 상태 ---
def update_system_data():
    while True:
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        fps = cam.fps
        timestamp = time.strftime('%H:%M:%S')
        
        with data_lock:
            current_stats.update({"cpu": cpu, "memory": mem, "fps": fps})
            system_log.append({"time": timestamp, "cpu": cpu, "mem": mem, "fps": f"{fps:.2f}"})
        
        try:
            payload = {"cpu_temp": cpu, "memory_usage": mem}
            requests.post(f"{NOTEBOOK_SERVER_URL}/api/log/system", json=payload, timeout=1)
        except: pass
        time.sleep(1)

# --- 백그라운드 스레드: 아두이노 연결 ---
def read_from_arduino():
    global ser
    while True:
        try:
            if ser is None:
                ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1) 
                ser.flush()
                print("아두이노 연결 성공!")
        except:
            ser = None
            time.sleep(5)
        time.sleep(1)

# --- ★★★ YOLOv8 AI 분석 스레드 (수정됨) ★★★ ---
def run_yolo_detection():
    global latest_results
    while True:
        if model is None or cam is None:
            time.sleep(1)
            continue
            
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        # YOLOv8 추론
        results = model(frame, imgsz=640, conf=0.5, verbose=False)
        
        temp_results = []
        timestamp = time.strftime('%H:%M:%S')

        for r in results:
            if r.masks is not None:
                masks = r.masks.xy 
                boxes = r.boxes
                
                for mask, box in zip(masks, boxes):
                    points = np.int32([mask])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls_id]
                    
                    # 박스 좌표
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # 중심 좌표 계산
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # 라벨 텍스트
                    label = f"{class_name} ({cx}, {cy})"
                    
                    # 화면 그리기용 데이터 저장
                    temp_results.append({
                        "type": "poly", 
                        "points": points, 
                        "label": label, 
                        "txt_pos": (x1, y1),
                        "cls_id": cls_id,
                        "center": (cx, cy)
                    })

                    # 1. 원격 전송용 (딕셔너리 사용)
                    coords_dict = {"x": cx, "y": cy}
                    
                    # 2. 로컬 HTML 로그용 (튜플 사용 - 이게 수정된 부분입니다!)
                    coords_tuple = (cx, cy)

                    # 로그 기록
                    with data_lock:
                        # HTML에서는 인덱스[0], [1]로 접근하므로 튜플로 저장해야 함
                        detection_log.append({
                            "time": timestamp, 
                            "class": class_name, 
                            "conf": round(conf, 2), 
                            "coords": coords_tuple  # ✅ 수정됨: 딕셔너리 -> 튜플
                        })
                        action_log.append({
                            "time": timestamp, 
                            "action": f"AI {class_name} 탐지", 
                            "result": "SUCCESS"
                        })
                        
                    # 원격 전송 (딕셔너리 유지)
                    try:
                        detection_payload = {"box_coordinates": coords_dict, "class_name": class_name, "confidence": conf}
                        requests.post(f"{NOTEBOOK_SERVER_URL}/api/log/detection", json=detection_payload, timeout=0.5)
                        
                        action_payload = {"action": f"AI {class_name} 탐지", "status": "SUCCESS"}
                        requests.post(f"{NOTEBOOK_SERVER_URL}/api/log/robot", json=action_payload, timeout=0.5)
                    except: pass

        with data_lock:
            latest_results = temp_results
            
        time.sleep(0.1) 

# --- 실시간 스트리밍 스레드 ---
def gen(camera):
    global latest_results
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
            
        with data_lock:
            current_draw_data = latest_results

        overlay = frame.copy()
        
        for item in current_draw_data:
            if item["type"] == "poly":
                points = item["points"]
                label = item["label"]
                x, y = item["txt_pos"]
                cls_id = item["cls_id"]
                cx, cy = item["center"]
                
                color = colors[cls_id % len(colors)].tolist()
                
                cv2.fillPoly(overlay, [points], color)
                cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
                
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x, y - 20), (x + w, y), color, -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, f"FPS: {camera.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- Flask 라우팅 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    with data_lock:
        return jsonify({ 
            "stats": current_stats, 
            "detection_log": list(detection_log), 
            "action_log": list(action_log), 
            "system_log": list(system_log) 
        })

@app.route('/control', methods=['POST'])
def control():
    command = request.json.get('command')
    if ser and ser.is_open:
        try:
            ser.write(f"{command}\n".encode('utf-8'))
            return jsonify({"status": "success", "command": command})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    else:
        return jsonify({"status": "error", "message": "Arduino not connected"})

if __name__ == '__main__':
    threading.Thread(target=run_yolo_detection, daemon=True).start()
    threading.Thread(target=read_from_arduino, daemon=True).start()
    threading.Thread(target=update_system_data, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)