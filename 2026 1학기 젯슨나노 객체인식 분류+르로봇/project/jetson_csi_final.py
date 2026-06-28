import cv2
import serial
import threading
import uvicorn
import psutil
import time
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO
import anthropic
import torch

# ==========================================
# 설정값
# ==========================================
INDEX_HTML     = "/home/user/project/index.html"

MODEL_CAM1     = "/home/user/project/number.engine"     
MODEL_CAM2     = "/home/user/project/box.pt"   

DIGIT_CLASS_MAP = {
    2: "0", 3: "1", 4: "2", 5: "3", 6: "4", 
    7: "5", 8: "6", 9: "7", 10: "8", 11: "9"
}

BOX_LABEL_MAP = {
    0: "small_box", 
    1: "samll_box"
}

CAM1_INDEX     =  2    
CAM2_INDEX     =  0    
CAM2_USE_GST   = True  

CAM2_GST = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=2 ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

ARDUINO_PORT   = "/dev/ttyACM0"
LEROBOT_PORT   = "/dev/ttyUSB0"

CLAUDE_API_KEY = "YOUR_CLAUDE_API_KEY_HERE"
CLAUDE_MODEL   = "claude-sonnet-4-20250514"

STREAM_FPS     = 15
DEVICE         = 'cuda'  

CONF_CAM1      = 0.20
CONF_CAM2      = 0.60  

WEIGHT_THRESHOLD = 98  
COMMAND_COOLDOWN = 5.0   
MAX_DIGITS       = 3     
SCREEN_PADDING   = 15
# ==========================================


# ==========================================
# 1. Pydantic Models 
# ==========================================
class RobotRequest(BaseModel):
    target_item: str
    action: str

class ConveyorRequest(BaseModel):
    command: str

class ControlRequest(BaseModel):
    command: str

class ClaudeRequest(BaseModel):
    prompt: str
    context: dict

class TestRequest(BaseModel):
    command: str


app = FastAPI(title="Logistics Robot Control Server")


# ==========================================
# 2. 시리얼 통신 연결
# ==========================================
try:
    arduino = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
    print(f"[INFO] 아두이노 연결 성공! ({ARDUINO_PORT})")
except Exception as e: 
    print(f"[WARN] 아두이노 연결 실패!: {e}")
    arduino = None

try:
    lerobot = serial.Serial(LEROBOT_PORT, 9600, timeout=1)
    print(f"[INFO] 르로봇 연결 성공! ({LEROBOT_PORT})")
except Exception as e: 
    print(f"[WARN] 르로봇 연결 실패!: {e}")
    lerobot = None


# ==========================================
# 3. YOLO 모델 로드
# ==========================================
def load_model(path, label):
    try:
        m = YOLO(path, task='detect')
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        m.predict(dummy, verbose=False, device=DEVICE)
        print(f"[INFO] {label} 로딩 완료: {path}")
        return m
    except Exception as e:
        print(f"[WARN] {label} 로딩 실패!: {e}")
        return None

model1 = load_model(MODEL_CAM1, "CAM1(RealSense)")
model2 = load_model(MODEL_CAM2, "CAM2(IMX219)")


# ==========================================
# 4. 공유 상태 변수
# ==========================================
system_running = False 

raw1_frame = None; raw1_lock = threading.Lock(); latest1_frame = None; frame1_lock = threading.Lock()
cam1_fps = 0.0; _c1_cnt = 0; _c1_t = time.time()

raw2_frame = None; raw2_lock = threading.Lock(); latest2_frame = None; frame2_lock = threading.Lock()
cam2_fps = 0.0; _c2_cnt = 0; _c2_t = time.time()

detection_log_1 = []; action_log_1 = []
detection_log_2 = []; action_log_2 = []

last_cmd_t1 = 0.0 
last_cmd_t2 = 0.0


# ==========================================
# 5. 숫자/박스 후처리 유틸
# ==========================================
def get_top_digits(dets, max_digits=3):
    if not dets: return []
    dets.sort(key=lambda x: x['conf'], reverse=True)
    top_dets = dets[:max_digits]
    top_dets.sort(key=lambda x: x['cx'])
    return top_dets

def combine_digits(dets):
    if not dets: return None
    try: return int(''.join(d['digit'] for d in dets))
    except: return None


# ==========================================
# 6. 추론 로직 (하드웨어 제어 연동)
# ==========================================
def run_inference(model, frame, conf_thresh, max_det=10):
    if model is None: return [], [], []
    
    results = model.predict(frame, conf=conf_thresh, iou=0.60, agnostic_nms=True, max_det=max_det, verbose=False, device=DEVICE)
    
    boxes, scores, class_ids = [], [], []
    for box in results[0].boxes:
        score = float(box.conf)
        if score < conf_thresh: continue 
            
        bx, by, bw, bh = box.xywh[0]
        boxes.append([float(bx), float(by), float(bw), float(bh)])
        scores.append(score)
        class_ids.append(int(box.cls))
    return boxes, scores, class_ids

def get_screen_crop(frame, boxes, scores, class_ids, fw, fh):
    best_box = None; best_conf = 0.0
    for (cx,cy,w,h), score, cls_id in zip(boxes, scores, class_ids):
        if cls_id == 12 and score > best_conf:
            best_conf = score; best_box = (cx,cy,w,h)
            
    if best_box is None: return None, None
    cx,cy,w,h = best_box
    x1=max(0,  int(cx-w/2) - SCREEN_PADDING)
    y1=max(0,  int(cy-h/2) - SCREEN_PADDING)
    x2=min(fw, int(cx+w/2) + SCREEN_PADDING)
    y2=min(fh, int(cy+h/2) + SCREEN_PADDING)
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0: return None, None
    
    zoomed = cv2.resize(cropped, (640, 640))
    offset = (x1, y1, (x2-x1)/640, (y2-y1)/640)
    return zoomed, offset

def extract_digits_from(boxes, scores, class_ids, offset=None):
    raw = []
    for (cx,cy,w,h), score, cls_id in zip(boxes, scores, class_ids):
        digit = DIGIT_CLASS_MAP.get(cls_id)
        if not digit: continue
        
        if offset:
            ox,oy,csx,csy = offset
            bx1=int((cx-w/2)*640*csx)+ox; by1=int((cy-h/2)*640*csy)+oy
            bx2=int((cx+w/2)*640*csx)+ox; by2=int((cy+h/2)*640*csy)+oy
            bcx=int(cx*640*csx)+ox
        else:
            bx1=int(cx-w/2); by1=int(cy-h/2)
            bx2=int(cx+w/2); by2=int(cy+h/2)
            bcx=int(cx)
        raw.append({"digit":digit,"cx":bcx,"conf":score,"box":(bx1,by1,bx2,by2)})
    return raw

def process_frame(frame, model, cam_id=1):
    global system_running, last_cmd_t1, last_cmd_t2
    
    FW, FH = frame.shape[1], frame.shape[0]
    annotated = frame.copy()

    if cam_id == 1:
        boxes1, scores1, cls1 = run_inference(model, frame, CONF_CAM1, max_det=10)
        zoomed, offset = get_screen_crop(frame, boxes1, scores1, cls1, FW, FH)
        
        if zoomed is not None:
            boxes2, scores2, cls2 = run_inference(model, zoomed, CONF_CAM1, max_det=10)
            raw_dets = extract_digits_from(boxes2, scores2, cls2, offset)
        else:
            raw_dets = extract_digits_from(boxes1, scores1, cls1)

        clustered = get_top_digits(raw_dets, MAX_DIGITS)
        weight    = combine_digits(clustered)

        for d in clustered:
            x1,y1,x2,y2 = d['box']
            over  = weight is not None and weight >= WEIGHT_THRESHOLD
            color = (0,255,0) if over else (0,180,255)
            cv2.rectangle(annotated, (x1,y1),(x2,y2), color, 2)
            cv2.putText(annotated, f"{d['digit']} {d['conf']:.2f}",
                        (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            detection_log_1.append({"class": d['digit'], "conf": f"{d['conf']:.2f}"})
            if len(detection_log_1) > 200: detection_log_1.pop(0)

        if weight is not None:
            over  = weight >= WEIGHT_THRESHOLD
            color = (0,255,0) if over else (0,180,255)
            status_text = "[START]" if system_running else "[STOP]"
            cv2.putText(annotated, f"{status_text} {weight}g (Limit:{WEIGHT_THRESHOLD}g)", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            now = time.time()
            if over and system_running and (now - last_cmd_t1 > COMMAND_COOLDOWN):
                if arduino:
                    try: arduino.write(b"BUZZER_ON\n")
                    except: pass
                
                action_log_1.append({"action": f"부저 울림 ({weight}g)", "result": "전송됨"})
                if len(action_log_1) > 200: action_log_1.pop(0)
                last_cmd_t1 = now

    else:
        boxes1, scores1, cls1 = run_inference(model, frame, CONF_CAM2, max_det=5)

        for (cx, cy, w, h), score, cls_id in zip(boxes1, scores1, cls1):
            if cls_id not in BOX_LABEL_MAP:
                continue

            cls_name = BOX_LABEL_MAP[cls_id]
            x1 = int(cx - w/2); y1 = int(cy - h/2)
            x2 = int(cx + w/2); y2 = int(cy + h/2)
            
            color = (255, 0, 0) if cls_id == 0 else (255, 0, 255) 
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{cls_name} {score:.2f}",
                        (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            detection_log_2.append({"class": cls_name, "conf": f"{score:.2f}"})
            if len(detection_log_2) > 200: detection_log_2.pop(0)
            
            now = time.time()
            if system_running and (now - last_cmd_t2 > COMMAND_COOLDOWN):
                cmd_str = ""
                if cls_id == 0: 
                    cmd_str = "MOVE_BIG\n"
                    log_msg = "대형 박스 픽업"
                elif cls_id == 1: 
                    cmd_str = "MOVE_SMALL\n"
                    log_msg = "소형 박스 픽업"
                
                if cmd_str:
                    if lerobot:
                        try: lerobot.write(cmd_str.encode('utf-8'))
                        except: pass
                        
                    action_log_2.append({"action": log_msg, "result": "전송됨"})
                    if len(action_log_2) > 200: action_log_2.pop(0)
                    last_cmd_t2 = now

        status_color = (0,255,0) if system_running else (0,0,255)
        cv2.putText(annotated, f"SYS: {'RUNNING' if system_running else 'STOPPED'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    return annotated


# ==========================================
# 7. FastAPI 웹훅 
# ==========================================
@app.post("/control")
async def control(req: ControlRequest):
    global system_running
    cmd = req.command.upper()
    
    if cmd == "START":
        system_running = True
        print("[SYSTEM] 공정 제어 시작 (START)")
        
    elif cmd == "STOP":
        system_running = False
        print("[SYSTEM] 공정 제어 정지 (STOP)")
        
    elif cmd == "CALIB":
        print("[SYSTEM] 캘리브레이션 명령 수신")
        if lerobot:
            try: lerobot.write(b"CALIB\n")
            except: pass
        if arduino:
            try: arduino.write(b"CALIB\n")
            except: pass
            
    return {"status": "success", "command": cmd}


# ==========================================
# 8. FastAPI 영상 및 데이터 스트리밍
# ==========================================
@app.get("/")
async def serve_index(): return FileResponse(INDEX_HTML)

async def _stream(get_fn):
    interval = 1.0 / STREAM_FPS
    while True:
        frame = get_fn()
        if frame is None: await asyncio.sleep(0.05); continue
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret: await asyncio.sleep(0.05); continue
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        await asyncio.sleep(interval)

@app.get("/video_feed/1")
async def feed1():
    def g():
        with frame1_lock: return latest1_frame.copy() if latest1_frame is not None else None
    return StreamingResponse(_stream(g), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed/2")
async def feed2():
    def g():
        with frame2_lock: return latest2_frame.copy() if latest2_frame is not None else None
    return StreamingResponse(_stream(g), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/data")
async def get_data():
    return {
        "stats": {"cpu": psutil.cpu_percent(interval=None), "memory": psutil.virtual_memory().percent, "fps1": round(cam1_fps, 2), "fps2": round(cam2_fps, 2)},
        "detection_log_1": detection_log_1[-50:], "action_log_1": action_log_1[-50:],
        "detection_log_2": detection_log_2[-50:], "action_log_2": action_log_2[-50:]
    }


# ==========================================
# 9. 스레드 구동부
# ==========================================
def capture1_loop():
    global raw1_frame, cam1_fps, _c1_cnt, _c1_t
    import pyrealsense2 as rs
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
    except: return
        
    while True:
        try:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            frame = np.asanyarray(color_frame.get_data())
            with raw1_lock: raw1_frame = frame
            _c1_cnt += 1; now = time.time()
            if now - _c1_t >= 1.0: cam1_fps = _c1_cnt/(now-_c1_t); _c1_cnt=0; _c1_t=now
        except: time.sleep(0.1)

def capture2_loop():
    global raw2_frame, cam2_fps, _c2_cnt, _c2_t
    cap = cv2.VideoCapture(CAM2_GST, cv2.CAP_GSTREAMER)
    if not cap.isOpened(): return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(0.2); continue
        frame = cv2.resize(frame, (640, 480))
        with raw2_lock: raw2_frame = frame
        _c2_cnt += 1; now = time.time()
        if now - _c2_t >= 1.0: cam2_fps = _c2_cnt/(now-_c2_t); _c2_cnt=0; _c2_t=now

def inference1_loop():
    global latest1_frame
    while True:
        with raw1_lock: frame = raw1_frame.copy() if raw1_frame is not None else None
        if frame is None: time.sleep(0.01); continue
        try:
            frame = cv2.flip(frame, -1) 
            
            annotated = process_frame(frame, model1, cam_id=1)
            with frame1_lock: latest1_frame = annotated
        except Exception as e:
            print(f"[ERROR] CAM1(RealSense) 에러: {e}")
        time.sleep(0.01)

def inference2_loop():
    global latest2_frame
    while True:
        with raw2_lock: frame = raw2_frame.copy() if raw2_frame is not None else None
        if frame is None: time.sleep(0.01); continue
        try:
            annotated = process_frame(frame, model2, cam_id=2)
            with frame2_lock: latest2_frame = annotated
        except Exception as e:
            print(f"[ERROR] CAM2(IMX) 에러: {e}")
        time.sleep(0.01)


if __name__ == "__main__":
    latest1_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    latest2_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    threading.Thread(target=capture1_loop,   daemon=True).start()
    threading.Thread(target=capture2_loop,   daemon=True).start()
    threading.Thread(target=inference1_loop, daemon=True).start()
    threading.Thread(target=inference2_loop, daemon=True).start()

    print(f"[INFO] FastAPI 서버 및 실시간 가동 완료")
    uvicorn.run(app, host="0.0.0.0", port=5000)