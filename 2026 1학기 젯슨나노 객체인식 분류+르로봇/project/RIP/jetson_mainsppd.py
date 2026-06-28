import cv2
import serial
import threading
import uvicorn
import psutil
import time
import asyncio
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO
from google import genai

# ==========================================
# ★ 설정값 — 여기만 수정하세요
# ==========================================
INDEX_HTML      = "/home/user/project/index.html"
MODEL_PATH      = "/home/user/project/best.onnx"
ARDUINO_PORT    = "/dev/ttyACM0"
LEROBOT_PORT    = "/dev/ttyUSB0"
GEMINI_API_KEY  = "YOUR_GEMINI_API_KEY_HERE"
CAMERA_INDEX    = 1

STREAM_FPS      = 15      # 스트림 FPS
CONF_THRESHOLD  = 0.8     # 신뢰도 임계값
INFER_INTERVAL  = 5.0     # 추론 주기 (초)
CPU_CORES       = 6       # CPU 코어 수
# ==========================================

app = FastAPI(title="Logistics Robot Control Server")


# ==========================================
# 1. 아두이노 / 르로봇 시리얼
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
# 2. ONNX 모델 로드 — 입력 크기 자동 감지 + 6코어
# ==========================================
print(f"[INFO] ONNX 모델 로딩 중 (CPU {CPU_CORES}코어)... ({MODEL_PATH})")

MODEL_W, MODEL_H = 640, 640   # 기본값, 아래에서 자동 감지로 덮어씀
_ort_session     = None
_class_names     = {}
USE_DIRECT_ORT   = False
model            = None

try:
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads     = CPU_CORES
    sess_options.inter_op_num_threads     = CPU_CORES
    sess_options.execution_mode           = ort.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    _ort_session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )

    # ✅ 모델 입력 크기 자동 감지
    input_shape = _ort_session.get_inputs()[0].shape
    # shape 예시: [1, 3, 640, 640] 또는 ['batch', 3, 'height', 'width']
    h = input_shape[2]
    w = input_shape[3]
    MODEL_H = h if isinstance(h, int) else 640
    MODEL_W = w if isinstance(w, int) else 640
    print(f"[INFO] ONNX 세션 로딩 완료! 입력 크기: {MODEL_W}x{MODEL_H}")

    # 클래스명은 YOLO로 읽어오기
    model = YOLO(MODEL_PATH, task='detect')
    _class_names = model.names
    print(f"[INFO] 클래스: {_class_names}")
    USE_DIRECT_ORT = True

except Exception as e:
    print(f"[WARN] ONNX 직접 로딩 실패, YOLO 폴백: {e}")
    try:
        model = YOLO(MODEL_PATH, task='detect')
        _class_names = model.names
        USE_DIRECT_ORT = False
        print("[INFO] YOLO 폴백 로딩 완료")
    except Exception as e2:
        print(f"[WARN] 모델 전체 로딩 실패: {e2}")


# ==========================================
# 3. Gemini AI
# ==========================================
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("[INFO] Gemini 초기화 완료")
except Exception as e:
    print(f"[WARN] Gemini 초기화 실패: {e}")
    gemini_client = None


# ==========================================
# 4. 공유 상태 변수
# ==========================================
raw_frame      = None
raw_lock       = threading.Lock()

latest_frame   = None
frame_lock     = threading.Lock()

system_running = False
detection_log  = []
action_log     = []

last_sent_value = None
last_sent_time  = 0.0

cam_fps  = 0.0
_cam_cnt = 0
_cam_t   = time.time()


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

class GeminiRequest(BaseModel):
    prompt: str
    context: dict

class TestRequest(BaseModel):
    command: str


# ==========================================
# 6. 숫자 필터 & 르로봇 전송
# ==========================================
def is_digit_class(cls_name: str) -> bool:
    return bool(''.join(filter(str.isdigit, cls_name)))

def extract_segment_value(cls_name: str):
    digits = ''.join(filter(str.isdigit, cls_name))
    return digits if digits else None

def send_to_lerobot(value: str):
    global last_sent_value, last_sent_time
    now = time.time()
    if value == last_sent_value and (now - last_sent_time) < 1.0:
        return
    if lerobot:
        try:
            lerobot.write(f"SEG:{value}\n".encode('utf-8'))
            last_sent_value = value
            last_sent_time  = now
            print(f"[LEROBOT] 전송: SEG:{value}")
            action_log.append({"action": f"SEG 전송: {value}", "result": "OK"})
            if len(action_log) > 200:
                action_log.pop(0)
        except Exception as e:
            print(f"[ERROR] 르로봇 전송 실패: {e}")
    else:
        print(f"[LEROBOT] (미연결) 시도값: SEG:{value}")


# ==========================================
# 7. ONNX 추론 함수 (모델 크기에 맞게 전처리)
# ==========================================
def run_inference(frame):
    """6코어 병렬 ONNX 추론. 모델 입력 크기(MODEL_W x MODEL_H) 자동 적용."""

    if USE_DIRECT_ORT and _ort_session is not None:
        # 전처리: 모델이 요구하는 크기로 리사이즈
        inp = cv2.resize(frame, (MODEL_W, MODEL_H))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis]  # → [1, 3, H, W]

        input_name = _ort_session.get_inputs()[0].name
        outputs    = _ort_session.run(None, {input_name: inp})

        # 후처리: YOLO output [1, num_classes+4, num_anchors] → boxes
        preds = outputs[0][0].T  # (num_anchors, 4 + num_classes)
        boxes, scores, class_ids = [], [], []

        for pred in preds:
            cls_scores = pred[4:]
            cls_id     = int(np.argmax(cls_scores))
            score      = float(cls_scores[cls_id])
            if score < CONF_THRESHOLD:
                continue
            cx, cy, w, h = pred[:4]
            boxes.append([float(cx), float(cy), float(w), float(h)])
            scores.append(score)
            class_ids.append(cls_id)

        return boxes, scores, class_ids

    elif model is not None:
        # YOLO 폴백 (내부에서 리사이즈 자동 처리)
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        boxes, scores, class_ids = [], [], []
        for box in results[0].boxes:
            bx, by, bw, bh = box.xywh[0]
            # YOLO 결과는 원본 frame 기준 좌표
            boxes.append([float(bx), float(by), float(bw), float(bh)])
            scores.append(float(box.conf))
            class_ids.append(int(box.cls))
        return boxes, scores, class_ids

    return [], [], []


# ==========================================
# 8. FastAPI 엔드포인트
# ==========================================

@app.get("/")
async def serve_index():
    return FileResponse(INDEX_HTML)


async def generate_frames():
    interval = 1.0 / STREAM_FPS
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            await asyncio.sleep(0.05)
            continue
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            await asyncio.sleep(0.05)
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')
        await asyncio.sleep(interval)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/control")
async def control(req: ControlRequest):
    global system_running
    cmd = req.command.upper()
    if cmd == "START":  system_running = True
    elif cmd == "STOP": system_running = False
    if arduino:
        try: arduino.write(f"{cmd}\n".encode('utf-8'))
        except Exception as e: print(f"[ERROR] 아두이노 전송 실패: {e}")
    return {"status": "success", "command": cmd}


@app.get("/data")
async def get_data():
    return {
        "stats": {
            "cpu":    psutil.cpu_percent(interval=None),
            "memory": psutil.virtual_memory().percent,
            "fps":    round(cam_fps, 2)
        },
        "detection_log": detection_log[-50:],
        "action_log":    action_log[-50:]
    }


@app.post("/ask_gemini")
async def ask_gemini(req: GeminiRequest):
    if not gemini_client:
        return {"answer": "Gemini 미초기화"}
    try:
        ctx = req.context
        prompt = f"""
물류 로봇 공정 분석 전문가로서 답변해주세요.
시스템 상태 — CPU: {ctx.get('cpu','N/A')}, FPS: {ctx.get('fps','N/A')}
질문: {req.prompt}
한국어로 3~5문장으로 답변해 주세요.
"""
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash', contents=prompt)
        return {"answer": response.text}
    except Exception as e:
        return {"answer": f"Gemini 오류: {str(e)}"}


@app.post("/api/robot/move")
async def move_robot(req: RobotRequest):
    if not arduino: return {"status": "warn", "message": "아두이노 미연결"}
    try:
        arduino.write(f"{req.target_item}\n".encode('utf-8'))
        action_log.append({"action": f"MOVE:{req.target_item}", "result": "OK"})
        if len(action_log) > 200: action_log.pop(0)
        return {"status": "success", "message": f"{req.target_item} 전송 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conveyor")
async def control_conveyor(req: ConveyorRequest):
    if not arduino: return {"status": "warn", "message": "아두이노 미연결"}
    try:
        arduino.write(f"CONV:{req.command.upper()}\n".encode('utf-8'))
        return {"status": "success", "message": f"컨베이어 {req.command} 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/robot/test")
async def test_robot(req: TestRequest):
    if not arduino: return {"status": "warn", "message": "아두이노 미연결"}
    try:
        arduino.write(f"MOVE:{req.command.upper()}\n".encode('utf-8'))
        return {"status": "success", "message": f"테스트 {req.command} 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 9. 스레드 ① 캡처 (최대 속도)
# ==========================================
def capture_loop():
    global raw_frame, cam_fps, _cam_cnt, _cam_t

    print(f"[INFO] 캡처 스레드 시작 (index={CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("[INFO] 카메라 캡처 스레드 시작 완료")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, (640, 480))
        with raw_lock:
            raw_frame = frame

        _cam_cnt += 1
        now = time.time()
        if now - _cam_t >= 1.0:
            cam_fps  = _cam_cnt / (now - _cam_t)
            _cam_cnt = 0
            _cam_t   = now


# ==========================================
# 10. 스레드 ② 추론 (5초 간격, 6코어 병렬)
# ==========================================
def inference_loop():
    global latest_frame
    last_infer_time = 0.0

    print(f"[INFO] 추론 스레드 시작 (주기: {INFER_INTERVAL}초, {CPU_CORES}코어, 모델: {MODEL_W}x{MODEL_H})")

    # 좌표 스케일: 모델 출력(MODEL_W x MODEL_H 기준) → 화면(640x480) 기준
    scale_x = 640 / MODEL_W
    scale_y = 480 / MODEL_H

    while True:
        with raw_lock:
            frame = raw_frame.copy() if raw_frame is not None else None

        if frame is None:
            time.sleep(0.01)
            continue

        now = time.time()

        if (now - last_infer_time) >= INFER_INTERVAL:
            last_infer_time = now
            print(f"[추론] 시작... (캡처 FPS: {cam_fps:.1f})")

            boxes, scores, class_ids = run_inference(frame)

            annotated = frame.copy()
            detected  = 0

            for (cx, cy, w, h), score, cls_id in zip(boxes, scores, class_ids):
                cls_name = _class_names.get(cls_id, str(cls_id))
                if not is_digit_class(cls_name):
                    continue

                x1 = int((cx - w / 2) * scale_x)
                y1 = int((cy - h / 2) * scale_y)
                x2 = int((cx + w / 2) * scale_x)
                y2 = int((cy + h / 2) * scale_y)

                seg_value = extract_segment_value(cls_name)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{seg_value} {score:.2f}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

                detection_log.append({"class": cls_name, "conf": f"{score:.2f}"})
                if len(detection_log) > 200:
                    detection_log.pop(0)

                if seg_value:
                    send_to_lerobot(seg_value)
                detected += 1

            print(f"[추론] 완료 — {detected}개 탐지, 다음까지 {INFER_INTERVAL}초")

            with frame_lock:
                latest_frame = annotated

        else:
            with frame_lock:
                latest_frame = frame.copy()

        time.sleep(0.005)


# ==========================================
# 11. 메인 실행
# ==========================================
if __name__ == "__main__":
    threading.Thread(target=capture_loop,   daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()

    print(f"[INFO] FastAPI 서버 구동 → http://0.0.0.0:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)
