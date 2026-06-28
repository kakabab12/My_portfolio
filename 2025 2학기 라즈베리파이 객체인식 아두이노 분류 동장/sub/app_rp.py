from flask import Flask, render_template, Response, jsonify, request
from camera import Camera 
import threading
import time
import random
import serial
import psutil
from collections import deque
import requests # <- 원격 전송을 위해 추가
import json     # <- 원격 전송을 위해 추가

app = Flask(__name__)

# --- TODO: 노트북 서버의 IP 주소를 여기에 입력하세요 ---
# 예: "http://192.168.0.10:5000"
NOTEBOOK_SERVER_URL = "http://<YOUR_NOTEBOOK_IP>:5000"
# ---------------------------------------------------

# --- 데이터 저장소 및 하드웨어 객체 ---
current_stats = { "cpu": 0, "memory": 0, "fps": 0 }
# (deque는 기존 index.html의 실시간 차트를 위해 유지)
detection_log = deque(maxlen=20) 
action_log = deque(maxlen=20)
system_log = deque(maxlen=20)

data_lock = threading.Lock()
cam = Camera()
ser = None # 아두이노 시리얼 객체

# --- 백그라운드 스레드들 ---

def update_system_data():
    """ 1초마다 시스템 상태를 기록하고 원격 서버로 전송 """
    while True:
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        fps = cam.fps
        timestamp = time.strftime('%H:%M:%S')
        
        # 1. (기존) 실시간 대시보드를 위해 deque에 저장
        with data_lock:
            current_stats["cpu"] = cpu
            current_stats["memory"] = mem
            current_stats["fps"] = fps
            system_log.append({"time": timestamp, "cpu": cpu, "mem": mem, "fps": f"{fps:.2f}"})
        
        # 2. (추가) 원격 노트북 서버로 데이터 전송
        try:
            payload = {
                "cpu_temp": cpu,       # (참고) psutil.sensors_temperatures()를 사용하면 더 정확한 온도 전송 가능
                "memory_usage": mem
            }
            requests.post(f"{NOTEBOOK_SERVER_URL}/api/log/system", json=payload, timeout=2)
            # print("System log sent to remote server.") # (디버깅용)
        except requests.exceptions.RequestException as e:
            print(f"Failed to send system log: {e}") # (디버깅용)

        time.sleep(1)

def read_from_arduino():
    global ser
    while True:
        try:
            if ser is None:
                ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1) 
                ser.flush()
                print("아두이노 연결 성공!")
        except serial.SerialException:
            if ser: ser.close()
            ser = None
            print("아두이노 연결 실패. 5초 후 재시도...")
            time.sleep(5)
        except Exception:
            time.sleep(1)

def simulate_robot_data():
    """ (시뮬레이션) 데이터를 생성하고 원격 서버로 전송 """
    total_count = 0
    while True:
        time.sleep(2) # (참고) 실제로는 카메라 탐지 로직이 이 함수를 대체해야 함
        timestamp = time.strftime('%H:%M:%S')
        total_count += 1
        is_defect = random.random() < 0.1
        detected_class = "불량박스" if is_defect else "정상박스"
        
        # (참고) API 스키마에 맞게 키 이름을 변경
        coords_dict = {"x": random.randint(100, 500), "y": random.randint(100, 400), "w": 50, "h": 50}
        
        detection_entry = {"time": timestamp, "class": detected_class, "conf": round(random.uniform(0.85, 0.99), 2), "coords": coords_dict}
        action_entry = {"time": timestamp, "action": f"#{total_count} {detected_class} PICK", "result": "FAIL" if is_defect else "SUCCESS"}
        
        # 1. (기존) 실시간 대시보드를 위해 deque에 저장
        with data_lock:
            detection_log.append(detection_entry)
            action_log.append(action_entry)
            
        # 2. (추가) 원격 노트북 서버로 데이터 전송
        try:
            # 2a. detection_log 전송
            detection_payload = {
                "box_coordinates": coords_dict,
                "class_name": detected_class,
                "confidence": detection_entry["conf"]
            }
            requests.post(f"{NOTEBOOK_SERVER_URL}/api/log/detection", json=detection_payload, timeout=2)
            
            # 2b. robot_log 전송
            action_payload = {
                "action": action_entry["action"],
                "status": action_entry["result"]
            }
            requests.post(f"{NOTEBOOK_SERVER_URL}/api/log/robot", json=action_payload, timeout=2)
            # print("Detection and Robot logs sent to remote server.") # (디버깅용)

        except requests.exceptions.RequestException as e:
            print(f"Failed to send robot/detection log: {e}") # (디버깅용)


# --- Flask 라우팅 (기존과 동일) ---
@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    # (이 API는 실시간 대시보드를 위해 deque의 데이터를 반환 - DB 데이터가 아님)
    with data_lock:
        return jsonify({ "stats": current_stats, "detection_log": list(detection_log), "action_log": list(action_log), "system_log": list(system_log) })

@app.route('/control', methods=['POST'])
def control():
    command = request.json.get('command')
    if ser and ser.is_open:
        try:
            ser.write(f"{command}\n".encode('utf-8'))
            print(f"아두이노에 명령어 전송: {command}")
            return jsonify({"status": "success", "command": command})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    else:
        return jsonify({"status": "error", "message": "Arduino not connected"})

if __name__ == '__main__':
    threading.Thread(target=simulate_robot_data, daemon=True).start()
    threading.Thread(target=read_from_arduino, daemon=True).start()
    threading.Thread(target=update_system_data, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)