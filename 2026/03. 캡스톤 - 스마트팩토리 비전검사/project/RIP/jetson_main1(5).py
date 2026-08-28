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
CONF_THRESHOLD    = 0.4     # 2차 추론 시 약간 낮춰서 잘 잡히게
INFER_INTERVAL    = 5.0
CPU_CORES         = 6

WEIGHT_THRESHOLD  = 118
LEROBOT_COMMAND   = "PICK"
COMMAND_COOLDOWN  = 10.0
MAX_DIGITS        = 3

SCREEN_PADDING    = 15      # screen 크롭 시 여백 (px)

DIGIT_MAP = {
    "0":0,"1":1,"2":2,"3":3,"4":4,
    "5":5,"6":6,"7":7,"8":8,"9":9,
    "zero":0,"one":1,"two":2,"three":3,"four":4,
    "five":5,"six":6,"seven":7,"eight":8,"nine":9,
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
# 2. ONNX 모델 로드
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
    print(f"[INFO] 모델 클래스: {_class_names}")

except Exception as e:
    print(f"[WARN] ONNX 직접 로딩 실패, YOLO 폴백: {e}")
    try:
        model = YOLO(MODEL_PATH, task='detect')
        _class_names   = model.names
        USE_DIRECT_ORT = False
        print(f"[INFO] YOLO 폴백. 클래스: {_class_names}")
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
# 6. 숫자 처리 함수
# ==========================================
def cls_to_digit(cls_name: str):
    if cls_name in DIGIT_MAP:
        return str(DIGIT_MAP[cls_name])
    if cls_name.lower() in DIGIT_MAP:
        return str(DIGIT_MAP[cls_name.lower()])
    digits = ''.join(filter(str.isdigit, cls_name))
    return digits if digits else None


def calc_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def apply_nms(detections: list, iou_thresh=0.4) -> list:
    if not detections:
        return []
    detections.sort(key=lambda d: d['conf'], reverse=True)
    kept = []
    for d in detections:
        if not any(calc_iou(d['box'], k['box']) > iou_thresh for k in kept):
            kept.append(d)
    return kept


def cluster_by_sections(detections: list) -> list:
    if not detections:
        return []
    if len(detections) <= MAX_DIGITS:
        return sorted(detections, key=lambda d: d['cx'])
    detections.sort(key=lambda d: d['cx'])
    min_x = detections[0]['cx']
    max_x = detections[-1]['cx']
    span  = max(max_x - min_x, 1)
    sec_w = span / MAX_DIGITS
    sections = [[] for _ in range(MAX_DIGITS)]
    for d in detections:
        idx = min(int((d['cx'] - min_x) / sec_w), MAX_DIGITS - 1)
        sections[idx].append(d)
    return [max(s, key=lambda x: x['conf']) for s in sections if s]


def check_double_digit(detections: list, raw_before_nms: list) -> list:
    if len(detections) >= MAX_DIGITS or not raw_before_nms:
        return detections
    leftmost = detections[0]
    nearby   = [r for r in raw_before_nms
                if abs(r['cx'] - leftmost['cx']) < 80
                and r['digit'] == leftmost['digit']]
    needed = MAX_DIGITS - len(detections)
    if len(nearby) >= 4 and needed > 0:
        extras = [{**leftmost} for _ in range(needed)]
        result = extras + detections
        print(f"[보완] '{leftmost['digit']}' 중복 → x{needed+1}: {[d['digit'] for d in result]}")
        return result
    return detections


def combine_digits(detections: list):
    if not detections:
        return None
    try:
        return int(''.join(d['digit'] for d in detections))
    except ValueError:
        return None


def send_command_to_lerobot(command: str, weight: int):
    global last_command_time
    now = time.time()
    if (now - last_command_time) < COMMAND_COOLDOWN:
        print(f"[LEROBOT] 쿨다운 ({COMMAND_COOLDOWN-(now-last_command_time):.1f}초 남음)")
        return
    msg = f"{command}:{weight}\n"
    if lerobot:
        try:
            lerobot.write(msg.encode('utf-8'))
            last_command_time = now
            print(f"[LEROBOT] 전송: {msg.strip()}")
            action_log.append({"action": f"{command} ({weight}g)", "result": "OK"})
            if len(action_log) > 200: action_log.pop(0)
        except Exception as e:
            print(f"[ERROR] 르로봇 전송 실패: {e}")
    else:
        last_command_time = now
        print(f"[LEROBOT] (미연결) 시도: {msg.strip()}")
        action_log.append({"action": f"{command} ({weight}g)", "result": "미연결"})


# ==========================================
# 7. 추론 함수 (프레임 직접 받아서 처리)
# ==========================================
def run_inference_on(frame):
    """주어진 프레임에 대해 ONNX/YOLO 추론 실행"""
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


def get_screen_crop(frame, boxes, scores, class_ids, frame_w, frame_h):
    """
    ✅ 1차 추론에서 'screen' 클래스 탐지 시 해당 영역 크롭 후 확대 반환.
    없으면 None 반환 (원본 프레임 사용).
    """
    best_box  = None
    best_conf = 0.0

    for (cx, cy, w, h), score, cls_id in zip(boxes, scores, class_ids):
        cls_name = _class_names.get(cls_id, str(cls_id))
        if cls_name == 'screen' and score > best_conf:
            best_conf = score
            best_box  = (cx, cy, w, h)

    if best_box is None:
        return None, None

    cx, cy, w, h = best_box
    sx = frame_w / MODEL_W
    sy = frame_h / MODEL_H

    x1 = max(0,        int((cx - w/2) * sx) - SCREEN_PADDING)
    y1 = max(0,        int((cy - h/2) * sy) - SCREEN_PADDING)
    x2 = min(frame_w,  int((cx + w/2) * sx) + SCREEN_PADDING)
    y2 = min(frame_h,  int((cy + h/2) * sy) + SCREEN_PADDING)

    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return None, None

    # ✅ 640x640으로 확대 (디지털 줌 — 거리 영향 제거)
    zoomed = cv2.resize(cropped, (MODEL_W, MODEL_H))
    print(f"[줌] screen 감지 → 크롭 ({x1},{y1})~({x2},{y2}) → {MODEL_W}x{MODEL_H} 확대")

    # 좌표 복원용 오프셋 반환
    offset = (x1, y1, (x2-x1)/MODEL_W, (y2-y1)/MODEL_H)
    return zoomed, offset


def extract_digits(boxes, scores, class_ids, frame_w, frame_h,
                   offset=None):
    """
    탐지 결과에서 숫자 박스 추출.
    offset=(ox, oy, sx, sy) 이 있으면 크롭 좌표 → 원본 좌표 변환.
    """
    sx = frame_w / MODEL_W
    sy = frame_h / MODEL_H

    raw = []
    for (cx, cy, w, h), score, cls_id in zip(boxes, scores, class_ids):
        cls_name = _class_names.get(cls_id, str(cls_id))
        if cls_name == 'screen':
            continue
        digit = cls_to_digit(cls_name)
        if digit is None:
            continue

        if offset:
            # 크롭된 프레임 좌표 → 원본 프레임 좌표
            ox, oy, crop_sx, crop_sy = offset
            bx1 = int((cx - w/2) * MODEL_W * crop_sx) + ox
            by1 = int((cy - h/2) * MODEL_H * crop_sy) + oy
            bx2 = int((cx + w/2) * MODEL_W * crop_sx) + ox
            by2 = int((cy + h/2) * MODEL_H * crop_sy) + oy
            bcx = int(cx * MODEL_W * crop_sx) + ox
        else:
            bx1 = int((cx - w/2) * sx)
            by1 = int((cy - h/2) * sy)
            bx2 = int((cx + w/2) * sx)
            by2 = int((cy + h/2) * sy)
            bcx = int(cx * sx)

        raw.append({
            "digit": digit,
            "cx":    bcx,
            "conf":  score,
            "box":   (bx1, by1, bx2, by2),
        })

    return raw


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
    FW, FH = 640, 480

    print(f"[INFO] 추론 스레드 시작 (주기:{INFER_INTERVAL}초, 기준:{WEIGHT_THRESHOLD}g)")

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

            # ── 1차 추론: 전체 프레임 ──
            boxes1, scores1, cls1 = run_inference_on(frame)

            # ── screen 크롭 시도 ──
            zoomed, offset = get_screen_crop(frame, boxes1, scores1, cls1, FW, FH)

            if zoomed is not None:
                # ── 2차 추론: 확대된 screen 영역 ──
                print("[줌] 2차 추론 실행 (확대 이미지)")
                boxes2, scores2, cls2 = run_inference_on(zoomed)
                raw_dets = extract_digits(boxes2, scores2, cls2, FW, FH, offset)
            else:
                # screen 미감지 → 1차 결과 그대로 사용
                print("[줌] screen 미감지 → 1차 추론 결과 사용")
                raw_dets = extract_digits(boxes1, scores1, cls1, FW, FH, None)

            print(f"[DEBUG] 원시: {len(raw_dets)}개 "
                  f"{[(d['digit'], d['cx']) for d in sorted(raw_dets, key=lambda x: x['cx'])]}")

            # NMS → 구역분할 → 자리 보완
            after_nms = apply_nms(raw_dets, iou_thresh=0.4)
            clustered = cluster_by_sections(after_nms)
            clustered = check_double_digit(clustered, raw_dets)

            print(f"[DEBUG] 최종 자리: {[d['digit'] for d in clustered]}")

            weight = combine_digits(clustered)
            print(f"[추론] 최종 무게: {weight}g")

            # 화면 그리기
            annotated = frame.copy()

            # screen 박스 표시
            for (cx, cy, w, h), score, cls_id in zip(boxes1, scores1, cls1):
                cls_name = _class_names.get(cls_id, str(cls_id))
                if cls_name == 'screen':
                    sx = FW / MODEL_W; sy = FH / MODEL_H
                    x1 = int((cx-w/2)*sx); y1 = int((cy-h/2)*sy)
                    x2 = int((cx+w/2)*sx); y2 = int((cy+h/2)*sy)
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (255,200,0), 1)

            # 숫자 박스 표시
            for d in clustered:
                x1, y1, x2, y2 = d['box']
                over  = weight is not None and weight >= WEIGHT_THRESHOLD
                color = (0, 255, 0) if over else (0, 180, 255)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                cv2.putText(annotated, f"{d['digit']} {d['conf']:.2f}",
                            (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if weight is not None:
                over   = weight >= WEIGHT_THRESHOLD
                color  = (0, 255, 0) if over else (0, 180, 255)
                status = f"{weight}g  ({'기준 초과!' if over else '기준 미달'})"
                cv2.putText(annotated, status, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                detection_log.append({
                    "class": f"{weight}g",
                    "conf":  "초과" if over else "미달"
                })
                if len(detection_log) > 200: detection_log.pop(0)

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
