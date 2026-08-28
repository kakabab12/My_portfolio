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

# ==========================================
# ★ 설정값 — 여기만 수정하세요
# ==========================================
INDEX_HTML     = "/home/user/project/index.html"

# 모델 경로
# best.pt → GPU 추론 (현재)
# best.engine → TensorRT 추론 (변환 후)
MODEL_CAM1     = "/home/user/project/best.pt"
MODEL_CAM2     = "/home/user/project/best.pt"

# 카메라
CAM1_INDEX     = 4      # RealSense D435 컬러 스트림 (video4)
CAM2_USE_GST   = True   # IMX219 CSI → GStreamer
CAM2_GST = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
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

# 무게 분류
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
# 2. YOLO 모델 로드 (GPU)
# ==========================================
import torch
print(f"[INFO] CUDA: {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

def load_model(path, label):
    try:
        m = YOLO(path, task='detect')
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        m.predict(dummy, verbose=False, device=DEVICE)
        print(f"[INFO] {label} 모델 로딩 완료: {path}")
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
raw1_frame    = None;  raw1_lock   = threading.Lock()
latest1_frame = None;  frame1_lock = threading.Lock()
cam1_fps = 0.0;  _c1_cnt = 0;  _c1_t = time.time()
cam1_connected = False

raw2_frame    = None;  raw2_lock   = threading.Lock()
latest2_frame = None;  frame2_lock = threading.Lock()
cam2_fps = 0.0;  _c2_cnt = 0;  _c2_t = time.time()
cam2_connected = False

system_running    = False
detection_log_1   = [];  action_log_1 = []
detection_log_2   = [];  action_log_2 = []
last_command_time = 0.0

# 카메라 미연결 시 표시할 플레이스홀더 프레임
def make_placeholder(label, color=(40, 40, 40)):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = color
    cv2.putText(frame, f"{label}", (160, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (120,120,120), 2)
    cv2.putText(frame, "Camera Not Connected", (100, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,80), 1)
    return frame

placeholder1 = make_placeholder("RealSense D435")
placeholder2 = make_placeholder("IMX219 CSI")


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
# 6. 숫자 처리 함수
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
        print(f"[보완] '{left['digit']}' x{needed+1}: {[d['digit'] for d in result]}")
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
        print(f"[LEROBOT] 쿨다운 ({COMMAND_COOLDOWN-(now-last_command_time):.1f}초)")
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
        print(f"[LEROBOT] (미연결) 시도: {msg.strip()}")
        action_log_1.append({"action": f"{command} ({weight}g)", "result": "미연결"})


# ==========================================
# 7. 추론 & 후처리
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
    x1=max(0,  int(cx-w/2)-SCREEN_PADDING)
    y1=max(0,  int(cy-h/2)-SCREEN_PADDING)
    x2=min(fw, int(cx+w/2)+SCREEN_PADDING)
    y2=min(fh, int(cy+h/2)+SCREEN_PADDING)
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0: return None, None
    zoomed = cv2.resize(cropped, (640, 640))
    return zoomed, (x1, y1, (x2-x1)/640, (y2-y1)/640)

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
    FW, FH  = frame.shape[1], frame.shape[0]
    det_log = detection_log_1 if cam_id == 1 else detection_log_2

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
    print(f"[캠{cam_id}] {[d['digit'] for d in clustered]} → {weight}g")

    annotated = frame.copy()
    for d in clustered:
        x1,y1,x2,y2 = d['box']
        over  = weight is not None and weight >= WEIGHT_THRESHOLD
        color = (0,255,0) if over else (0,180,255)
        cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
        cv2.putText(annotated,f"{d['digit']} {d['conf']:.2f}",
                    (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        det_log.append({"class":d['digit'],"conf":f"{d['conf']:.2f}"})
        if len(det_log) > 200: det_log.pop(0)

    if weight is not None:
        over  = weight >= WEIGHT_THRESHOLD
        color = (0,255,0) if over else (0,180,255)
        cv2.putText(annotated,f"{weight}g ({'초과!' if over else '미달'})",
                    (20,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,color,3)
        if over and cam_id == 1:
            send_command_to_lerobot(LEROBOT_COMMAND, weight)
    return annotated


# ==========================================
# 8. FastAPI 엔드포인트
# ==========================================
@app.get("/")
async def serve_index():
    return FileResponse(INDEX_HTML)

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
        with frame1_lock:
            return latest1_frame.copy() if latest1_frame is not None else placeholder1.copy()
    return StreamingResponse(_stream(g), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed/2")
async def feed2():
    def g():
        with frame2_lock:
            return latest2_frame.copy() if latest2_frame is not None else placeholder2.copy()
    return StreamingResponse(_stream(g), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/control")
async def control(req: ControlRequest):
    global system_running
    cmd = req.command.upper()
    if cmd == "START":  system_running = True
    elif cmd == "STOP": system_running = False
    if arduino:
        try: arduino.write(f"{cmd}\n".encode('utf-8'))
        except Exception as e: print(f"[ERROR] 아두이노: {e}")
    print(f"[CONTROL] {cmd}")
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
            messages=[{"role":"user","content":
                f"스마트 팩토리 물류 로봇 공정 분석 전문가로서 답변해주세요.\n"
                f"시스템 — CPU:{ctx.get('cpu','N/A')}, 1캠FPS:{ctx.get('fps1','N/A')}, 2캠FPS:{ctx.get('fps2','N/A')}\n"
                f"질문: {req.prompt}\n한국어로 3~5문장으로 답변해 주세요."}])
        return {"answer": msg.content[0].text}
    except Exception as e:
        return {"answer": f"Claude 오류: {str(e)}"}

@app.post("/api/robot/move")
async def move_robot(req: RobotRequest):
    if not arduino: return {"status":"warn","message":"아두이노 미연결"}
    try:
        arduino.write(f"{req.target_item}\n".encode('utf-8'))
        action_log_1.append({"action":f"MOVE:{req.target_item}","result":"OK"})
        if len(action_log_1) > 200: action_log_1.pop(0)
        return {"status":"success"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conveyor")
async def control_conveyor(req: ConveyorRequest):
    if not arduino: return {"status":"warn","message":"아두이노 미연결"}
    try:
        arduino.write(f"CONV:{req.command.upper()}\n".encode('utf-8'))
        return {"status":"success"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/robot/test")
async def test_robot(req: TestRequest):
    if not arduino: return {"status":"warn","message":"아두이노 미연결"}
    try:
        arduino.write(f"MOVE:{req.command.upper()}\n".encode('utf-8'))
        return {"status":"success"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 9. 캡처 스레드 1 — RealSense (video4 컬러)
# ==========================================
def capture1_loop():
    global raw1_frame, cam1_fps, _c1_cnt, _c1_t, cam1_connected
    print(f"[INFO] 캡처1 시작 (RealSense, video{CAM1_INDEX})")
    while True:
        cap = cv2.VideoCapture(CAM1_INDEX)
        if not cap.isOpened():
            print(f"[WARN] CAM1 열기 실패, 5초 후 재시도...")
            cam1_connected = False
            time.sleep(5); continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cam1_connected = True
        print("[INFO] CAM1 (RealSense) 연결 완료")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] CAM1 프레임 읽기 실패, 재연결...")
                cam1_connected = False
                cap.release(); break
            frame = cv2.resize(frame, (640, 480))
            with raw1_lock: raw1_frame = frame
            _c1_cnt += 1
            now = time.time()
            if now - _c1_t >= 1.0:
                cam1_fps = _c1_cnt/(now-_c1_t); _c1_cnt=0; _c1_t=now


# ==========================================
# 10. 캡처 스레드 2 — IMX219 (CSI GStreamer)
# ==========================================
def capture2_loop():
    global raw2_frame, cam2_fps, _c2_cnt, _c2_t, cam2_connected
    print("[INFO] 캡처2 시작 (IMX219 CSI)")
    while True:
        if CAM2_USE_GST:
            cap = cv2.VideoCapture(CAM2_GST, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            print("[WARN] CAM2 열기 실패 (IMX219 미연결?), 10초 후 재시도...")
            cam2_connected = False
            time.sleep(10); continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam2_connected = True
        print("[INFO] CAM2 (IMX219) 연결 완료")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] CAM2 프레임 읽기 실패, 재연결...")
                cam2_connected = False
                cap.release(); break
            frame = cv2.resize(frame, (640, 480))
            with raw2_lock: raw2_frame = frame
            _c2_cnt += 1
            now = time.time()
            if now - _c2_t >= 1.0:
                cam2_fps = _c2_cnt/(now-_c2_t); _c2_cnt=0; _c2_t=now


# ==========================================
# 11. 추론 스레드 1 (RealSense)
# ==========================================
def inference1_loop():
    global latest1_frame
    last_t = 0.0
    while True:
        with raw1_lock: frame = raw1_frame.copy() if raw1_frame is not None else None
        if frame is None:
            with frame1_lock: latest1_frame = placeholder1.copy()
            time.sleep(0.1); continue
        now = time.time()
        if (now - last_t) >= INFER_INTERVAL:
            last_t = now
            print(f"\n[추론1] 시작 (FPS:{cam1_fps:.1f})")
            annotated = process_frame(frame, model1, cam_id=1)
            with frame1_lock: latest1_frame = annotated
        else:
            with frame1_lock: latest1_frame = frame.copy()
        time.sleep(0.005)


# ==========================================
# 12. 추론 스레드 2 (IMX219)
# ==========================================
def inference2_loop():
    global latest2_frame
    last_t = 0.0
    while True:
        with raw2_lock: frame = raw2_frame.copy() if raw2_frame is not None else None
        if frame is None:
            with frame2_lock: latest2_frame = placeholder2.copy()
            time.sleep(0.1); continue
        now = time.time()
        if (now - last_t) >= INFER_INTERVAL:
            last_t = now
            print(f"\n[추론2] 시작 (FPS:{cam2_fps:.1f})")
            annotated = process_frame(frame, model2, cam_id=2)
            with frame2_lock: latest2_frame = annotated
        else:
            with frame2_lock: latest2_frame = frame.copy()
        time.sleep(0.005)


# ==========================================
# 13. TensorRT 변환 유틸
# ==========================================
def export_engine(pt_path):
    print(f"[EXPORT] {pt_path} → engine 변환 중 (FP16)...")
    m = YOLO(pt_path)
    m.export(format='engine', device=0, half=True, imgsz=640)
    print(f"[EXPORT] 완료! {pt_path.replace('.pt','.engine')} 생성됨")


# ==========================================
# 14. 메인 실행
# ==========================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--export':
        export_engine("/home/user/project/best.pt")
        sys.exit(0)

    threading.Thread(target=capture1_loop,   daemon=True).start()
    threading.Thread(target=capture2_loop,   daemon=True).start()
    threading.Thread(target=inference1_loop, daemon=True).start()
    threading.Thread(target=inference2_loop, daemon=True).start()

    print(f"[INFO] FastAPI 서버 → http://0.0.0.0:5000")
    print(f"[INFO] RealSense: video{CAM1_INDEX} | IMX219: CSI(GStreamer)")
    print(f"[INFO] 기준 무게: {WEIGHT_THRESHOLD}g → {LEROBOT_COMMAND}")
    uvicorn.run(app, host="0.0.0.0", port=5000)
