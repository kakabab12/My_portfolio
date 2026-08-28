# from flask import Flask, render_template, Response, jsonify, request
# from camera import Camera 
# import threading
# import time
# import random
# import serial
# import psutil
# from collections import deque
# import requests # <- 원격 전송을 위해 추가
# import json

# app = Flask(__name__)

# # --- TODO: 노트북 서버의 IP 주소---
# NOTEBOOK_SERVER_URL = "http://172.16.11.200:5000"
# # ---------------------------------------------------

# # --- 데이터 저장소 및 하드웨어 객체 ---
# current_stats = { "cpu": 0, "memory": 0, "fps": 0 }
# # (deque는 기존 index.html의 실시간 차트를 위해 유지)
# detection_log = deque(maxlen=20) 
# action_log = deque(maxlen=20)
# system_log = deque(maxlen=20)

# data_lock = threading.Lock()
# cam = Camera()
# ser = None # 아두이노 시리얼 객체

# # --- 백그라운드 스레드들 ---

# def update_system_data():
#     """ 1초마다 시스템 상태를 기록하고 원격 서버로 전송 """
#     while True:
#         cpu = psutil.cpu_percent()
#         mem = psutil.virtual_memory().percent
#         fps = cam.fps
#         timestamp = time.strftime('%H:%M:%S')
        
#         # 1. (기존) 실시간 대시보드를 위해 deque에 저장
#         with data_lock:
#             current_stats["cpu"] = cpu
#             current_stats["memory"] = mem
#             current_stats["fps"] = fps
#             system_log.append({"time": timestamp, "cpu": cpu, "mem": mem, "fps": f"{fps:.2f}"})
        
#         # 2. (추가) 원격 노트북 서버로 데이터 전송
#         try:
#             payload = {
#                 "cpu_temp": cpu,       # (참고) psutil.sensors_temperatures()를 사용하면 더 정확한 온도 전송 가능
#                 "memory_usage": mem
#             }
#             requests.post(f"{NOTEBOOK_SERVER_URL}/api/log/system", json=payload, timeout=2)
#             # print("System log sent to remote server.") # (디버깅용)
#         except requests.exceptions.RequestException as e:
#             print(f"Failed to send system log: {e}") # (디버깅용)

#         time.sleep(1)

# def read_from_arduino():
#     global ser
#     while True:
#         try:
#             if ser is None:
#                 ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1) 
#                 ser.flush()
#                 print("아두이노 연결 성공!")
#         except serial.SerialException:
#             if ser: ser.close()
#             ser = None
#             print("아두이노 연결 실패. 5초 후 재시도...")
#             time.sleep(5)
#         except Exception:
#             time.sleep(1)

# def simulate_robot_data():
#     """ (시뮬레이션) 데이터를 생성하고 원격 서버로 전송 """
#     total_count = 0
#     while True:
#         time.sleep(2) # (참고) 실제로는 카메라 탐지 로직이 이 함수를 대체해야 함
#         timestamp = time.strftime('%H:%M:%S')
#         total_count += 1
#         is_defect = random.random() < 0.1
#         detected_class = "불량박스" if is_defect else "정상박스"
        
#         # (참고) API 스키마에 맞게 키 이름을 변경
#         coords_dict = {"x": random.randint(100, 500), "y": random.randint(100, 400), "w": 50, "h": 50}
        
#         detection_entry = {"time": timestamp, "class": detected_class, "conf": round(random.uniform(0.85, 0.99), 2), "coords": coords_dict}
#         action_entry = {"time": timestamp, "action": f"#{total_count} {detected_class} PICK", "result": "FAIL" if is_defect else "SUCCESS"}
        
#         # 1. (기존) 실시간 대시보드를 위해 deque에 저장
#         with data_lock:
#             detection_log.append(detection_entry)
#             action_log.append(action_entry)
            
#         # 2. (추가) 원격 노트북 서버로 데이터 전송
#         try:
#             # 2a. detection_log 전송
#             detection_payload = {
#                 "box_coordinates": coords_dict,
#                 "class_name": detected_class,
#                 "confidence": detection_entry["conf"]
#             }
#             requests.post(f"{NOTEBOOK_SERVER_URL}/api/log/detection", json=detection_payload, timeout=2)
            
#             # 2b. robot_log 전송
#             action_payload = {
#                 "action": action_entry["action"],
#                 "status": action_entry["result"]
#             }
#             requests.post(f"{NOTEBOOK_SERVER_URL}/api/log/robot", json=action_payload, timeout=2)
#             # print("Detection and Robot logs sent to remote server.") # (디버깅용)

#         except requests.exceptions.RequestException as e:
#             print(f"Failed to send robot/detection log: {e}") # (디버깅용)


# # --- Flask 라우팅 (기존과 동일) ---
# @app.route('/')
# def index():
#     return render_template('index.html')

# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         if frame: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/data')
# def data():
#     # (이 API는 실시간 대시보드를 위해 deque의 데이터를 반환 - DB 데이터가 아님)
#     with data_lock:
#         return jsonify({ "stats": current_stats, "detection_log": list(detection_log), "action_log": list(action_log), "system_log": list(system_log) })

# @app.route('/control', methods=['POST'])
# def control():
#     command = request.json.get('command')
#     if ser and ser.is_open:
#         try:
#             ser.write(f"{command}\n".encode('utf-8'))
#             print(f"아두이노에 명령어 전송: {command}")
#             return jsonify({"status": "success", "command": command})
#         except Exception as e:
#             return jsonify({"status": "error", "message": str(e)})
#     else:
#         return jsonify({"status": "error", "message": "Arduino not connected"})

# if __name__ == '__main__':
#     threading.Thread(target=simulate_robot_data, daemon=True).start()
#     threading.Thread(target=read_from_arduino, daemon=True).start()
#     threading.Thread(target=update_system_data, daemon=True).start()
    
#     app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)










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
from ultralytics import YOLO # YOLOv8 공식 라이브러리

app = Flask(__name__)

# --- 설정 ---
NOTEBOOK_SERVER_URL = "http://172.16.11.200:5000"

# --- YOLOv8-seg 모델 로드 ---
try:
    # PC에서 변환해온 best.onnx (또는 best.pt) 파일을 로드합니다.
    # task='segment'는 세그멘테이션 모드로 동작하라는 뜻입니다.
    model = YOLO('yolov5/best.onnx', task='segment') 
    print("✅ YOLOv8-seg 모델 로드 성공")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    model = None

# --- 데이터 저장소 ---
current_stats = { "cpu": 0, "memory": 0, "fps": 0 }
detection_log = deque(maxlen=20)
action_log = deque(maxlen=20)
system_log = deque(maxlen=20)

# 스트리밍 스레드와 공유할 최신 탐지 결과 (폴리곤 점들 포함)
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
            current_stats["cpu"] = cpu
            current_stats["memory"] = mem
            current_stats["fps"] = fps
            system_log.append({"time": timestamp, "cpu": cpu, "mem": mem, "fps": f"{fps:.2f}"})
        
        # 원격 서버 전송
        try:
            payload = {"cpu_temp": cpu, "memory_usage": mem}
            requests.post(f"{NOTEBOOK_SERVER_URL}/api/log/system", json=payload, timeout=2)
        except requests.exceptions.RequestException:
            pass # 로그 전송 실패 시 무시 (시스템 부하 방지)
            
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
        except serial.SerialException:
            if ser: ser.close()
            ser = None
            time.sleep(5)
        except Exception:
            time.sleep(1)

# --- ★★★ YOLOv8 AI 분석 스레드 ★★★ ---
def run_yolo_detection():
    global latest_results
    
    while True:
        if model is None or cam is None:
            time.sleep(1)
            continue
            
        # 1. 카메라 프레임 가져오기
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        # 2. YOLOv8 추론 실행
        # imgsz=640: 이미지 크기, conf=0.5: 정확도 50% 이상만 탐지
        # verbose=False: 터미널에 자잘한 로그 출력 안 함
        results = model(frame, imgsz=640, conf=0.5, verbose=False)
        
        temp_results = []
        timestamp = time.strftime('%H:%M:%S')

        # 3. 결과 처리 (Segmentation)
        for r in results:
            # 마스크(폴리곤)가 탐지된 경우에만 처리
            if r.masks is not None:
                # 마스크 좌표들 (이미지 크기에 맞게 조정된 좌표)
                masks = r.masks.xy 
                # 박스 정보 (클래스, 정확도)
                boxes = r.boxes
                
                for mask, box in zip(masks, boxes):
                    # 폴리곤 좌표를 정수형(int)으로 변환 (cv2.polylines용)
                    points = np.int32([mask])
                    
                    # 클래스 정보 추출
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls_id]
                    
                    # 라벨 텍스트 생성
                    label = f"{class_name} {conf:.2f}"
                    
                    # 박스 좌표 (텍스트 표시용)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 스트리밍 스레드로 넘길 데이터 구성
                    temp_results.append({
                        "type": "poly", 
                        "points": points, 
                        "label": label, 
                        "txt_pos": (x1, y1 - 10)
                    })

                    # --- 로그 기록 및 원격 전송 ---
                    # 폴리곤의 중심점 계산 (로그에 좌표로 남기기 위해)
                    M = cv2.moments(points)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                    else:
                        cx, cy = 0, 0
                        
                    coords_dict = {"x": cx, "y": cy}

                    with data_lock:
                        detection_log.append({
                            "time": timestamp, 
                            "class": class_name, 
                            "conf": round(conf, 2), 
                            "coords": coords_dict
                        })
                        action_log.append({
                            "time": timestamp, 
                            "action": f"AI {class_name} 탐지", 
                            "result": "SUCCESS"
                        })
                    
                    # 원격 서버 전송
                    try:
                        detection_payload = {"box_coordinates": coords_dict, "class_name": class_name, "confidence": conf}
                        requests.post(f"{NOTEBOOK_SERVER_URL}/api/log/detection", json=detection_payload, timeout=1)
                        
                        action_payload = {"action": f"AI {class_name} 탐지", "status": "SUCCESS"}
                        requests.post(f"{NOTEBOOK_SERVER_URL}/api/log/robot", json=action_payload, timeout=1)
                    except Exception:
                        pass

        # 4. 최신 결과 업데이트 (화면 그리기용)
        with data_lock:
            latest_results = temp_results
            
        # CPU 부하 조절 (너무 자주 돌면 웹 스트리밍이 끊길 수 있음)
        time.sleep(0.1) 

# --- 실시간 스트리밍 스레드 ---
def gen(camera):
    global latest_results
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
            
        # AI가 분석한 최신 결과 가져오기
        with data_lock:
            current_draw_data = latest_results

        # --- 그리기 작업 (폴리곤) ---
        for item in current_draw_data:
            if item["type"] == "poly":
                # 1. 폴리곤 외곽선 그리기 (초록색, 두께 2)
                cv2.polylines(frame, [item["points"]], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # 2. 폴리곤 내부 색칠 (반투명 효과)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [item["points"]], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                # 3. 텍스트 그리기
                cv2.putText(frame, item["label"], item["txt_pos"], 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # FPS 표시
        cv2.putText(frame, f"FPS: {camera.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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
        return jsonify({ "stats": current_stats, "detection_log": list(detection_log), "action_log": list(action_log), "system_log": list(system_log) })

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