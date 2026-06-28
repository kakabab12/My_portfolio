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
import pyrealsense2 as rs  # [추가됨] 리얼센스 공식 라이브러리

# ==========================================
# ★ 설정값 — 여기만 수정하세요
# ==========================================
INDEX_HTML     = "/home/user/project/index.html"

MODEL_CAM1     = "/home/user/project/best.pt"     
MODEL_CAM2     = "/home/user/project/best.pt"     

# 1. IMX219 (CSI) 카메라 설정
CAM2_INDEX     =  0    
CAM2_USE_GST   = True  

CAM2_GST = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

# 시리얼
ARDUINO_PORT   = "/dev/ttyACM0"
LEROBOT_PORT   = "/dev/ttyUSB0"

# Claude API
CLAUDE_API_KEY = "YOUR_CLAUDE_API_KEY_HERE"
CLAUDE_MODEL   = "claude-sonnet-4-20250514"

# 추론
STREAM_FPS     = 15
CONF_THRESHOLD = 0.5
INFER_INTERVAL = 5.0
DEVICE         = 'cuda'

WEIGHT_THRESHOLD = 118
LEROBOT_COMMAND  = "PICK"
COMMAND_COOLDOWN = 10.0
MAX_DIGITS       = 3
SCREEN_PADDING   = 15

DIGIT_MAP = {
    "0":0,"1":1,"2":2,"3":3,"4":4,
    "5":5,"6":6,"7":7,"8":8,"9":9,
    "zero":0,"one":1,"two":2,"three":3,"four":4,
    "five":5,"six":6,"seven":7,"eight":8,"nine":9,
}
# ==========================================

app = FastAPI(title="Logistics Robot Control Server")


# ==========================================
# 1. 시리얼 통신
# ==========================================
try:
    arduino = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
    print(f"[INFO] 아두이노 연결 성공 ({ARDUINO_PORT})")
except Exception as e:
    print(f"[WARN] 아두이노 미연결: {e}")
    arduino = None

try:
    lerobot = serial.Serial(LEROBOT_PORT, 9600, timeout=1)
    print(f"[INFO] 르로봇 연결 성공 ({LEROBOT_PORT})")
except Exception as e:
    print(f"[WARN] 르로봇 미연결: {e}")
    lerobot = None


# ==========================================
# 2. YOLO 모델 로드
# ==========================================
import torch
print(f"[INFO] CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

def load_model(path, label):
    try:
        m = YOLO(path, task='detect')
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        m.predict(dummy, verbose=False, device=DEVICE)
        print(f"[INFO] {label} 모델 로딩 완료: {path} | 장치: {DEVICE}")
        return m
    except Exception as e:
        print(f"[WARN] {label} 모델 로딩 실패: {e}")
        return None

model1 = load_model(MODEL_CAM1, "CAM1")
model2 = load_model(MODEL_CAM2, "CAM2")


# ==========================================
# 3. Claude AI
# ==========================================
try:
    claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    print("[INFO] Claude 클라이언트 초기화 완료")
except Exception as e:
    print(f"[WARN] Claude 초기화 실패: {e}")
    claude_client = None


# ==========================================
# 4. 공유 상태 변수
# ==========================================
raw1_frame    = None;  raw1_lock  = threading.Lock()
latest1_frame = None;  frame1_lock = threading.Lock()
cam1_fps = 0.0;  _c1_cnt = 0;  _c1_t = time.time()

raw2_frame    = None;  raw2_lock  = threading.Lock()
latest2_frame = None;  frame2_lock = threading.Lock()
cam2_fps = 0.0;  _c2_cnt = 0;  _c2_t = time.time()

system_running    = False
detection_log_1   = [];  action_log_1 = []
detection_log_2   = [];  action_log_2 = []
last_command_time = 0.0


# ==========================================
# 5. Pydantic Models
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


# ==========================================
# 7. 숫자 처리 함수 (그대로 유지)
# ==========================================
def cls_to_digit(cls_name):
    if cls_name in DIGIT_MAP:         return str(DIGIT_MAP[cls_name])
    if cls_name.lower() in DIGIT_MAP: return str(DIGIT_MAP[cls_name.lower()])
    digits = ''.join(filter(str.isdigit, cls_name))
    return digits if digits else None

def calc_iou(b1, b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
    x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter/u if u > 0 else 0

def apply_nms(dets, iou_thresh=0.4):
    if not dets: return []
    dets.sort(key=lambda d: d['conf'], reverse=True)
    kept = []
    for d in dets:
        if not any(calc_iou(d['box'], k['box']) > iou_thresh for k in kept):
            kept.append(d)
    return kept

def cluster_by_sections(dets):
    if not dets: return []
    if len(dets) <= MAX_DIGITS: return sorted(dets, key=lambda d: d['cx'])
    dets.sort(key=lambda d: d['cx'])
    min_x = dets[0]['cx']; max_x = dets[-1]['cx']
    sec_w = max(max_x-min_x, 1) / MAX_DIGITS
    secs  = [[] for _ in range(MAX_DIGITS)]
    for d in dets:
        secs[min(int((d['cx']-min_x)/sec_w), MAX_DIGITS-1)].append(d)
    return [max(s, key=lambda x: x['conf']) for s in secs if s]

def check_double_digit(dets, raw):
    if len(dets) >= MAX_DIGITS or not raw: return dets
    left   = dets[0]
    nearby = [r for r in raw if abs(r['cx']-left['cx']) < 80 and r['digit'] == left['digit']]
    needed = MAX_DIGITS - len(dets)
    if len(nearby) >= 4 and needed > 0:
        result = [{**left} for _ in range(needed)] + dets
        return result
    return dets

def combine_digits(dets):
    if not dets: return None
    try: return int(''.join(d['digit'] for d in dets))
    except: return None

def send_command_to_lerobot(command, weight):
    global last_command_time
    now = time.time()
    if (now - last_command_time) < COMMAND_COOLDOWN:
        return
    msg = f"{command}:{weight}\n"
    if lerobot:
        try:
            lerobot.write(msg.encode('utf-8'))
            last_command_time = now
            print(f"[LEROBOT] 전송: {msg.strip()}")
            action_log_1.append({"action": f"{command} ({weight}g)", "result": "OK"})
            if len(action_log_1) > 200: action_log_1.pop(0)
        except Exception as e:
            print(f"[ERROR] 르로봇 전송 실패: {e}")
    else:
        last_command_time = now
        action_log_1.append({"action": f"{command} ({weight}g)", "result": "미연결"})


# ==========================================
# 8. 추론 & 후처리
# ==========================================
def run_inference(model, frame):
    if model is None: return [], [], []
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False, device=DEVICE)
    boxes, scores, class_ids = [], [], []
    for box in results[0].boxes:
        bx, by, bw, bh = box.xywh[0]
        boxes.append([float(bx), float(by), float(bw), float(bh)])
        scores.append(float(box.conf))
        class_ids.append(int(box.cls))
    return boxes, scores, class_ids

def get_screen_crop(frame, boxes, scores, class_ids, model, fw, fh):
    names = model.names if model else {}
    best_box = None; best_conf = 0.0
    for (cx,cy,w,h), score, cls_id in zip(boxes, scores, class_ids):
        if names.get(cls_id,'') == 'screen' and score > best_conf:
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

def extract_digits_from(boxes, scores, class_ids, model, fw, fh, offset=None):
    names = model.names if model else {}
    raw = []
    for (cx,cy,w,h), score, cls_id in zip(boxes, scores, class_ids):
        cls_name = names.get(cls_id, str(cls_id))
        if cls_name == 'screen': continue
        digit = cls_to_digit(cls_name)
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
    FW, FH   = frame.shape[1], frame.shape[0]
    det_log  = detection_log_1 if cam_id == 1 else detection_log_2

    boxes1, scores1, cls1 = run_inference(model, frame)
    zoomed, offset = get_screen_crop(frame, boxes1, scores1, cls1, model, FW, FH)
    
    if zoomed is not None:
        boxes2, scores2, cls2 = run_inference(model, zoomed)
        raw_dets = extract_digits_from(boxes2, scores2, cls2, model, FW, FH, offset)
    else:
        raw_dets = extract_digits_from(boxes1, scores1, cls1, model, FW, FH)

    after_nms = apply_nms(raw_dets)
    clustered = cluster_by_sections(after_nms)
    clustered = check_double_digit(clustered, raw_dets)
    weight    = combine_digits(clustered)

    annotated = frame.copy()
    for d in clustered:
        x1,y1,x2,y2 = d['box']
        over  = weight is not None and weight >= WEIGHT_THRESHOLD
        color = (0,255,0) if over else (0,180,255)
        cv2.rectangle(annotated, (x1,y1),(x2,y2), color, 2)
        cv2.putText(annotated, f"{d['digit']} {d['conf']:.2f}",
                    (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        det_log.append({"class": d['digit'], "conf": f"{d['conf']:.2f}"})
        if len(det_log) > 200: det_log.pop(0)

    if weight is not None:
        over  = weight >= WEIGHT_THRESHOLD
        color = (0,255,0) if over else (0,180,255)
        cv2.putText(annotated, f"{weight}g ({'초과!' if over else '미달'})",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        if over and cam_id == 1:
            send_command_to_lerobot(LEROBOT_COMMAND, weight)

    return annotated


# ==========================================
# 9. FastAPI 엔드포인트
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

@app.post("/control")
async def control(req: ControlRequest):
    global system_running
    cmd = req.command.upper()
    if cmd == "START":  system_running = True
    elif cmd == "STOP": system_running = False
    if arduino:
        try: arduino.write(f"{cmd}\n".encode('utf-8'))
        except: pass
    return {"status": "success", "command": cmd}

@app.get("/data")
async def get_data():
    return {
        "stats": {
            "cpu":    psutil.cpu_percent(interval=None),
            "memory": psutil.virtual_memory().percent,
            "fps1":   round(cam1_fps, 2),
            "fps2":   round(cam2_fps, 2),
        },
        "detection_log_1": detection_log_1[-50:],
        "action_log_1":    action_log_1[-50:],
        "detection_log_2": detection_log_2[-50:],
        "action_log_2":    action_log_2[-50:],
    }

@app.post("/ask_claude")
async def ask_claude(req: ClaudeRequest):
    if not claude_client: return {"answer": "Claude 미초기화"}
    try:
        ctx = req.context
        msg = claude_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=512,
            messages=[{"role":"user","content": f"질문: {req.prompt}"}])
        return {"answer": msg.content[0].text}
    except Exception as e: return {"answer": str(e)}

@app.post("/api/robot/move")
async def move_robot(req: RobotRequest): return {"status":"success"}

@app.post("/api/conveyor")
async def control_conveyor(req: ConveyorRequest): return {"status":"success"}

@app.post("/api/robot/test")
async def test_robot(req: TestRequest): return {"status":"success"}


# ==========================================
# 10. 캡처 스레드 1 (RealSense - pyrealsense2 적용)
# ==========================================
def capture1_loop():
    global raw1_frame, cam1_fps, _c1_cnt, _c1_t
    
    print("[INFO] CAM1(RealSense) 공식 라이브러리로 연결 시도 중...")
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        # 리얼센스의 "컬러 렌즈"만 콕 집어서 가져옵니다. (회색 화면 방지)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        print("[INFO] CAM1(RealSense) 컬러 스트림 연결 완벽 성공!")
    except Exception as e:
        print(f"[ERROR] RealSense 연결 실패 (USB를 뺐다 꽂아주세요): {e}")
        return
        
    while True:
        try:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: 
                continue
                
            frame = np.asanyarray(color_frame.get_data())
            with raw1_lock: raw1_frame = frame
            
            _c1_cnt += 1
            now = time.time()
            if now - _c1_t >= 1.0:
                cam1_fps = _c1_cnt/(now-_c1_t); _c1_cnt=0; _c1_t=now
        except Exception as e:
            time.sleep(0.1)


# ==========================================
# 11. 캡처 스레드 2 (IMX219)
# ==========================================
def capture2_loop():
    global raw2_frame, cam2_fps, _c2_cnt, _c2_t
    
    print(f"[INFO] CAM2(IMX219) GStreamer 파이프라인 오픈 시도 중...")
    cap = cv2.VideoCapture(CAM2_GST, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("[ERROR] CAM2 GStreamer 오픈 실패! sudo systemctl restart nvargus-daemon 실행 필수!")
        return
        
    print("[INFO] CAM2(IMX219) CSI GStreamer 열기 성공! (초록 화면 방지됨)")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.2)
            continue
            
        frame = cv2.resize(frame, (640, 480))
        with raw2_lock: 
            raw2_frame = frame
            
        _c2_cnt += 1
        now = time.time()
        if now - _c2_t >= 1.0:
            cam2_fps = _c2_cnt/(now-_c2_t); _c2_cnt=0; _c2_t=now


# ==========================================
# 12. 추론 스레드 1 (RealSense)
# ==========================================
def inference1_loop():
    global latest1_frame
    last_t = 0.0
    while True:
        with raw1_lock: frame = raw1_frame.copy() if raw1_frame is not None else None
        if frame is None: time.sleep(0.01); continue
        now = time.time()
        if (now - last_t) >= INFER_INTERVAL:
            last_t = now
            annotated = process_frame(frame, model1, cam_id=1)
            with frame1_lock: latest1_frame = annotated
        else:
            with frame1_lock: latest1_frame = frame.copy()
        time.sleep(0.005)


# ==========================================
# 13. 추론 스레드 2 (IMX219)
# ==========================================
def inference2_loop():
    global latest2_frame
    last_t = 0.0
    while True:
        with raw2_lock: frame = raw2_frame.copy() if raw2_frame is not None else None
        if frame is None: time.sleep(0.01); continue
        now = time.time()
        if (now - last_t) >= INFER_INTERVAL:
            last_t = now
            annotated = process_frame(frame, model2, cam_id=2)
            with frame2_lock: latest2_frame = annotated
        else:
            with frame2_lock: latest2_frame = frame.copy()
        time.sleep(0.005)


if __name__ == "__main__":
    latest1_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    latest2_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    threading.Thread(target=capture1_loop,   daemon=True).start()
    threading.Thread(target=capture2_loop,   daemon=True).start()
    threading.Thread(target=inference1_loop, daemon=True).start()
    threading.Thread(target=inference2_loop, daemon=True).start()

    print(f"[INFO] FastAPI 서버 시작 완료")
    uvicorn.run(app, host="0.0.0.0", port=5000)