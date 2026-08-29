from flask import Flask, render_template, Response, jsonify, request
from camera import Camera
import threading
import time
import serial
import serial.tools.list_ports
import psutil
from collections import deque, Counter
import requests
import cv2
import numpy as np
import math 
from ultralytics import YOLO

app = Flask(__name__)

# -----------------------------
#  설정 및 상수
# -----------------------------
NOTEBOOK_SERVER_URL = "http://172.16.11.200:5000"
PICKUP_Z_LEVEL = 0  

# [튜닝 포인트 1: 면적(Size) 기준]
AREA_LIMIT_SMALL = 52100
AREA_LIMIT_BIG = 260000

# [튜닝 포인트 2: 형상(Shape) 기준]
# 이 값보다 크면 무조건 normal, 작으면 무조건 abnormal로 분류됩니다.
RECTANGULARITY_THRESHOLD = 0.83 # (이전 대화에서 0.70으로 낮춘 값 적용)

# [속도 최적화]
VOTE_BUFFER_SIZE = 7        
CONSISTENCY_THRESHOLD = 0.5 

# --- 모델 로드 ---
try:
    model = YOLO('yolov5/best.onnx', task='segment')
    print(f" YOLOv8-seg 모델 로드 성공")
except Exception as e:
    print(f" 모델 로드 실패: {e}")
    model = None

np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(model.names), 3)).astype(int)

# --- 데이터 저장소 ---
current_stats = { "cpu": 0, "memory": 0, "fps": 0 }
detection_log = deque(maxlen=20)
action_log = deque(maxlen=20)
system_log = deque(maxlen=20)

class_vote_buffer = deque(maxlen=VOTE_BUFFER_SIZE)
latest_results = []
locked_target = None

data_lock = threading.Lock()
cam = Camera()
ser = None

# -----------------------------
# 🛠️ 유틸리티 함수
# -----------------------------

def release_target():
    global locked_target
    with data_lock:
        locked_target = None
        class_vote_buffer.clear()
        print("\n🔓 [시스템] 타겟 고정 해제 -> 재탐색 시작\n")

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
            requests.post(f"{NOTEBOOK_SERVER_URL}/api/log/system", json=payload, timeout=0.5)
        except: pass
        time.sleep(1)

def read_from_arduino():
    global ser
    while True:
        try:
            if ser is None:
                ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
                ser.flush()
                print(" 아두이노 연결 성공!")
            
            if ser.in_waiting > 0:
                 ser.readline() 
        except:
            ser = None
            time.sleep(5)
        time.sleep(1)

# -----------------------------
#  YOLO 분석 (R값 절대 기준 적용)
# -----------------------------
def run_yolo_detection():
    global latest_results, locked_target
    
    print(f" YOLO 시작: R값 {RECTANGULARITY_THRESHOLD} 기준으로 정상/불량을 나눕니다.")
    
    while True:
        if model is None or cam is None:
            time.sleep(1)
            continue
        
        if locked_target is not None:
            time.sleep(0.05) 
            continue

        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        # 속도 최적화: imgsz=640 (느리면 320으로 변경)
        results = model(frame, imgsz=640, conf=0.5, iou=0.4, agnostic_nms=True, verbose=False)
        
        temp_results = []
        detected_objects_in_frame = []
        timestamp = time.strftime('%H:%M:%S')
        best_candidate_in_frame = None

        for r in results:
            if r.masks is not None:
                masks = r.masks.xy
                boxes = r.boxes
                
                for mask, box in zip(masks, boxes):
                    if len(mask) == 0: continue

                    points = np.int32([mask])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    poly_area = cv2.contourArea(points)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_area = (x2 - x1) * (y2 - y1)
                    
                    # R값 계산
                    rectangularity = poly_area / box_area if box_area > 0 else 0
                    
                    # 1. [크기 분류]
                    size_prefix = "medium"
                    if poly_area < AREA_LIMIT_SMALL: size_prefix = "small"
                    elif poly_area > AREA_LIMIT_BIG: size_prefix = "big"
                    
                    # 2. [상태 분류] - ★★★ R값 절대 기준 적용 ★★★
                    if rectangularity < RECTANGULARITY_THRESHOLD:
                        state_suffix = "abnormal_box"
                    else:
                        state_suffix = "normal_box"
                    
                    # 최종 이름 조합
                    final_class_name = f"{size_prefix}_{state_suffix}"
                    
                    # 무게 중심(Centroid) 정밀 계산
                    M = cv2.moments(points)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                    else:
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                    
                    cz = PICKUP_Z_LEVEL

                    # 화면 중앙과의 거리 계산
                    dist = math.sqrt((cx - 320)**2 + (cy - 480)**2)

                    # 좌표 계산 (유지)
                    real_x = round((cx - 320) * 0.05, 1)
                    real_y = round((480 - cy) * 0.05 + 5.0, 1)
                    
                    label = f"{final_class_name} A:{int(poly_area)} R:{rectangularity:.2f}"
                    
                    obj_data = {
                        "type": "poly", "points": points, "label": label,
                        "txt_pos": (x1, y1), "cls_id": cls_id, "center": (cx, cy),
                        "class": final_class_name, "conf": conf,
                        "real_coords": (real_x, real_y, cz),
                        "dist": dist
                    }
                    temp_results.append(obj_data)
                    detected_objects_in_frame.append(obj_data)

        # 화면 중앙에 가장 가까운 객체 우선 선택
        if len(detected_objects_in_frame) > 0:
            best_candidate_in_frame = min(detected_objects_in_frame, key=lambda x: x['dist'])

        with data_lock:
            if best_candidate_in_frame:
                class_vote_buffer.append(best_candidate_in_frame['class'])
            else:
                if len(class_vote_buffer) > 0: class_vote_buffer.popleft()

            # --- 타겟 확정 로직 ---
            if locked_target is None and len(class_vote_buffer) >= VOTE_BUFFER_SIZE:
                vote_counts = Counter(class_vote_buffer)
                most_common_class, count = vote_counts.most_common(1)[0]
                vote_ratio = count / len(class_vote_buffer)
                
                if vote_ratio >= CONSISTENCY_THRESHOLD and best_candidate_in_frame and best_candidate_in_frame['class'] == most_common_class:
                    tx, ty, tz = best_candidate_in_frame['real_coords']
                    px, py = best_candidate_in_frame['center']
                    t_class = most_common_class

                    # LOCK 좌표를 정확히 무게 중심(px, py)으로 설정
                    locked_target = { "x": px, "y": py, "z": tz, "class": t_class }
                    print(f"⚡ [타겟 확정] {t_class} (중심: {px},{py})")

                    # 번호 매핑 (1~6번)
                    sort_type = 0
                    if   t_class == "small_normal_box":    sort_type = 1
                    elif t_class == "medium_normal_box":   sort_type = 2
                    elif t_class == "big_normal_box":      sort_type = 3
                    elif t_class == "small_abnormal_box":  sort_type = 4
                    elif t_class == "medium_abnormal_box": sort_type = 5
                    elif t_class == "big_abnormal_box":    sort_type = 6

                    detection_log.append({
                        "time": timestamp, "class": t_class,
                        "conf": round(best_candidate_in_frame['conf'], 2),
                        "coords": (tx, ty, tz)
                    })
                    action_log.append({
                        "time": timestamp,
                        "action": f"SENDING TYPE: {sort_type}",
                        "result": "SENT"
                    })
                    
                    if ser and ser.is_open:
                        cmd = f"SORT:{sort_type}\n" 
                        ser.write(cmd.encode())
                        print(f"📤 아두이노 전송: {cmd.strip()}")

                    try:
                        detection_payload = {"box_coordinates": {"x":tx, "y":ty, "z":tz},
                                             "class_name": t_class, "confidence": best_candidate_in_frame['conf']}
                        requests.post(f"{NOTEBOOK_SERVER_URL}/api/log/detection", json=detection_payload, timeout=0.01)
                    except: pass

                    # 타겟 유지 시간 3.0초
                    threading.Timer(3.0, release_target).start()

            latest_results = temp_results
            
        time.sleep(0.01)

# -----------------------------
#  웹 스트리밍
# -----------------------------
def gen(camera):
    global latest_results, locked_target
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05); continue
            
        with data_lock:
            current_draw_data = latest_results
            current_locked = locked_target

        overlay = frame.copy()
        
        for item in current_draw_data:
            if item["type"] == "poly":
                points = item["points"]
                label = item["label"]
                x, y = item["txt_pos"]
                cls_id = item["cls_id"]
                color = colors[cls_id % len(colors)].tolist()
                
                cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
                
                # 인식 중일 때도 중심점(흰 점) 표시
                cx, cy = item['center']
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 🔒 LOCK이 걸렸을 때 (빨간 십자선) - 정확한 무게중심에 표시
        if current_locked is not None:
            lx, ly = current_locked['x'], current_locked['y']
            lx, ly = max(0, min(lx, 639)), max(0, min(ly, 479))
            
            # 십자선 그리기
            cv2.line(frame, (lx-30, ly), (lx+30, ly), (0, 0, 255), 3)
            cv2.line(frame, (lx, ly-30), (lx, ly+30), (0, 0, 255), 3)
            cv2.circle(frame, (lx, ly), 20, (0, 0, 255), 2)
            
            msg = f"LOCKED: {current_locked['class']}"
            cv2.putText(frame, msg, (lx+20, ly-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# -----------------------------
#  Flask 라우트
# -----------------------------
@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(gen(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

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
        ser.write(f"{command}\n".encode())
        return jsonify({"status": "success", "command": command})
    return jsonify({"status": "error", "message": "Arduino not connected"})

if __name__ == '__main__':
    threading.Thread(target=run_yolo_detection, daemon=True).start()
    threading.Thread(target=read_from_arduino, daemon=True).start()
    threading.Thread(target=update_system_data, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)