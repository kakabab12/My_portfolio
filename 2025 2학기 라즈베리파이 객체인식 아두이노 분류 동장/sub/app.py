from flask import Flask, render_template, Response, jsonify, request
from camera import Camera 
import threading
import time
import random
import serial
import psutil
from collections import deque

app = Flask(__name__)

# --- 데이터 저장소 및 하드웨어 객체 ---
current_stats = { "cpu": 0, "memory": 0, "fps": 0 }
detection_log = deque(maxlen=20)
action_log = deque(maxlen=20)
system_log = deque(maxlen=20)

data_lock = threading.Lock()
cam = Camera()
ser = None # 아두이노 시리얼 객체

# --- 백그라운드 스레드들 ---
def update_system_data():
    """ 1초마다 시스템 상태를 기록 """
    while True:
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        fps = cam.fps
        timestamp = time.strftime('%H:%M:%S')
        with data_lock:
            current_stats["cpu"] = cpu
            current_stats["memory"] = mem
            current_stats["fps"] = fps
            system_log.append({"time": timestamp, "cpu": cpu, "mem": mem, "fps": f"{fps:.2f}"})
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
    total_count = 0
    while True:
        time.sleep(2)
        timestamp = time.strftime('%H:%M:%S')
        total_count += 1
        is_defect = random.random() < 0.1
        detected_class = "불량박스" if is_defect else "정상박스"
        
        detection_entry = {"time": timestamp, "class": detected_class, "conf": round(random.uniform(0.85, 0.99), 2), "coords": (random.randint(100, 500), random.randint(100, 400))}
        action_entry = {"time": timestamp, "action": f"#{total_count} {detected_class} PICK", "result": "FAIL" if is_defect else "SUCCESS"}
        
        with data_lock:
            detection_log.append(detection_entry)
            action_log.append(action_entry)

# --- Flask 라우팅 ---
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
    
    # ### 여기가 수정되었습니다! (use_reloader=False 추가, host 명시) ###
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    # ###############################################################