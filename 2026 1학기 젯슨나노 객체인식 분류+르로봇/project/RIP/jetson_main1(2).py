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
INDEX_HTML        = "/home/user/project/index.html"
MODEL_PATH        = "/home/user/project/best.onnx"
ARDUINO_PORT      = "/dev/ttyACM0"
LEROBOT_PORT      = "/dev/ttyUSB0"
GEMINI_API_KEY    = "YOUR_GEMINI_API_KEY_HERE"
CAMERA_INDEX      = 1

STREAM_FPS        = 15
CONF_THRESHOLD    = 0.8
INFER_INTERVAL    = 5.0
CPU_CORES         = 6

WEIGHT_THRESHOLD  = 118     # 기준 무게 (g)
LEROBOT_COMMAND   = "PICK"  # 기준 초과 시 르로봇 명령
COMMAND_COOLDOWN  = 10.0    # 명령 반복 방지 간격 (초)

# ✅ 클래스명 → 숫자 매핑 테이블
# 모델 클래스명이 확인되면 여기에 맞게 수정하세요
# 예) 영어: {"zero":0,"one":1,...,"nine":9}
# 예) 숫자: {"0":0,"1":1,...,"9":9}  ← 현재 설정
DIGIT_MAP = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    # 영어 클래스명 대비 추가
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
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

MODEL_W, MODEL_H = 640, 640
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

    input_shape = _ort_session.get_inputs()[0].shape
    h = input_shape[2]; w = input_shape[3]
    MODEL_H = h if isinstance(h, int) else 640
    MODEL_W = w if isinstance(w, int) else 640
    print(f"[INFO] ONNX 세션 로딩 완료! 입력 크기: {MODEL_W}x{MODEL_H}")

    model = YOLO(MODEL_PATH, task='detect')
    _class_names   = model.names
    USE_DIRECT_ORT = True

    # ✅ 클래스명 전체 출력 — 실제 클래스명 확인용
    print(f"[INFO] 모델 클래스 목록: {_class_names}")

except Exception as e:
    print(f"[WARN] ONNX 직접 로딩 실패, YOLO 폴백: {e}")
    try:
        model = YOLO(MODEL_PATH, task='detect')
        _class_names   = model.names
        USE_DIRECT_ORT = False
        print(f"[INFO] YOLO 폴백 로딩 완료. 클래스: {_class_names}")
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

system_running    = False
detection_log     = []
action_log        = []
last_command_time = 0.0

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
# 6. 숫자 매핑 & 무게 판정 & 르로봇 전송
# ==========================================
def cls_to_digit(cls_name: str):
    """
    클래스명 → 숫자(0~9) 변환.
    DIGIT_MAP 우선 적용, 없으면 클래스명에서 숫자 문자 추출.
    """
    # 1) 정확한 매핑 테이블 확인
    if cls_name in DIGIT_MAP:
        return str(DIGIT_MAP[cls_name])
    # 2) 소문자로 재시도
    if cls_name.lower() in DIGIT_MAP:
        return str(DIGIT_MAP[cls_name.lower()])
    # 3) 클래스명에서 숫자만 추출 (fallback)
    digits = ''.join(filter(str.isdigit, cls_name))
    return digits if digits else None

def combine_digits(detections: list):
    """x좌표 왼→오 정렬 후 숫자 조합. 예) ['2','9','1'] → 291"""
    if not detections:
        return None
    detections.sort(key=lambda d: d['cx'])
    number_str = ''.join(d['digit'] for d in detections)
    try:
        return int(number_str)
    except ValueError:
        return None

def send_command_to_lerobot(command: str, weight: int):
    global last_command_time
    now = time.time()
    if (now - last_command_time) < COMMAND_COOLDOWN:
        remaining = COMMAND_COOLDOWN - (now - last_command_time)
        print(f"[LEROBOT] 쿨다운 중 ({remaining:.1f}초 남음)")
        return
    msg = f"{command}:{weight}\n"
    if lerobot:
        try:
            lerobot.write(msg.encode('utf-8'))
            last_command_time = now
            print(f"[LEROBOT] 명령 전송: {msg.strip()}")
            action_log.append({"action": f"{command} ({weight}g)", "result": "OK"})
            if len(action_log) > 200: action_log.pop(0)
        except Exception as e:
            print(f"[ERROR] 르로봇 전송 실패: {e}")
    else:
        last_command_time = now
        print(f"[LEROBOT] (미연결) 시도: {msg.strip()}")
        action_log.append({"action": f"{command} ({weight}g)", "result": "미연결"})


# ==========================================
# 7. ONNX 추론 함수
# ==========================================
def run_inference(frame):
    if USE_DIRECT_ORT and _ort_session is not None:
        inp = cv2.resize(frame, (MODEL_W, MODEL_H))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis]
        input_name = _ort_session.get_inputs()[0].name
        outputs    = _ort_session.run(None, {input_name: inp})
        preds = outputs[0][0].T
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
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        boxes, scores, class_ids = [], [], []
        for box in results[0].boxes:
            bx, by, bw, bh = box.xywh[0]
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
# 9. 스레드 ① 캡처
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
# 10. 스레드 ② 추론 + 무게 판정
# ==========================================
def inference_loop():
    global latest_frame
    last_infer_time = 0.0
    scale_x = 640 / MODEL_W
    scale_y = 480 / MODEL_H

    print(f"[INFO] 추론 스레드 시작 (주기: {INFER_INTERVAL}초, 기준: {WEIGHT_THRESHOLD}g)")

    while True:
        with raw_lock:
            frame = raw_frame.copy() if raw_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue

        now = time.time()

        if (now - last_infer_time) >= INFER_INTERVAL:
            last_infer_time = now
            print(f"\n[추론] 시작 (캡처 FPS: {cam_fps:.1f})")

            boxes, scores, class_ids = run_inference(frame)

            # ✅ 탐지된 원시 클래스명 전부 출력 (디버그)
            print(f"[DEBUG] 탐지 수: {len(boxes)}개")
            for i, (_, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
                cls_name = _class_names.get(cls_id, str(cls_id))
                print(f"[DEBUG]  [{i}] cls_id={cls_id}, cls_name='{cls_name}', conf={score:.2f}")

            annotated  = frame.copy()
            detections = []

            for (cx, cy, w, h), score, cls_id in zip(boxes, scores, class_ids):
                cls_name = _class_names.get(cls_id, str(cls_id))
                digit    = cls_to_digit(cls_name)
                if digit is None:
                    continue

                x1 = int((cx - w / 2) * scale_x)
                y1 = int((cy - h / 2) * scale_y)
                x2 = int((cx + w / 2) * scale_x)
                y2 = int((cy + h / 2) * scale_y)
                screen_cx = int(cx * scale_x)

                detections.append({
                    "digit":    digit,
                    "cx":       screen_cx,
                    "conf":     score,
                    "box":      (x1, y1, x2, y2),
                    "cls_name": cls_name
                })

            # 숫자 조합
            weight = combine_digits(detections)
            sorted_digits = [d['digit'] for d in sorted(detections, key=lambda d: d['cx'])]
            print(f"[추론] 정렬된 숫자: {sorted_digits} → 조합: {weight}g")

            # 바운딩박스 그리기
            for d in detections:
                x1, y1, x2, y2 = d['box']
                color = (0, 255, 0) if (weight and weight >= WEIGHT_THRESHOLD) else (0, 180, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{d['digit']} {d['conf']:.2f}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 무게 + 판정 화면 표시
            if weight is not None:
                over = weight >= WEIGHT_THRESHOLD
                label_color = (0, 255, 0) if over else (0, 180, 255)
                status = f"{weight}g  ({'기준 초과!' if over else '기준 미달'})"
                cv2.putText(annotated, status, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)

                # 탐지 로그
                detection_log.append({
                    "class": f"{weight}g",
                    "conf":  "초과" if over else "미달"
                })
                if len(detection_log) > 200: detection_log.pop(0)

                # 기준 초과 → 르로봇 명령
                if over:
                    print(f"[판정] {weight}g >= {WEIGHT_THRESHOLD}g → {LEROBOT_COMMAND}")
                    send_command_to_lerobot(LEROBOT_COMMAND, weight)
                else:
                    print(f"[판정] {weight}g < {WEIGHT_THRESHOLD}g → 미달")

            print(f"[추론] 완료 — 다음까지 {INFER_INTERVAL}초")

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
    print(f"[INFO] 기준 무게: {WEIGHT_THRESHOLD}g 이상 → {LEROBOT_COMMAND} 명령 전송")
    uvicorn.run(app, host="0.0.0.0", port=5000)
