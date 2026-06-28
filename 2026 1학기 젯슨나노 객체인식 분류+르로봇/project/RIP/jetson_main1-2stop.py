import cv2
import serial
import threading
import uvicorn
import psutil
import time
import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO
from google import genai

# ==========================================
# ★ 설정값 — 여기만 수정하세요
# ==========================================
INDEX_HTML     = "/home/user/project/index.html"
MODEL_PATH     = "/home/user/project/best.onnx"   # ONNX 사용
ARDUINO_PORT   = "/dev/ttyACM0"
LEROBOT_PORT   = "/dev/ttyUSB0"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
CAMERA_INDEX   = 1

TARGET_FPS     = 15          # 목표 최소 FPS
INFER_SIZE     = 320         # YOLO 입력 해상도 (작을수록 빠름, 기본 640→320)
CONF_THRESHOLD = 0.5         # 신뢰도 임계값 (낮추면 더 많이 인식)
# ==========================================

app = FastAPI(title="Logistics Robot Control Server")


# ==========================================
# 1. 아두이노 시리얼
# ==========================================
try:
    arduino = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
    print(f"[INFO] 아두이노 연결 성공 ({ARDUINO_PORT})")
except Exception as e:
    print(f"[WARN] 아두이노 미연결: {e}")
    arduino = None


# ==========================================
# 2. 르로봇 시리얼
# ==========================================
try:
    lerobot = serial.Serial(LEROBOT_PORT, 9600, timeout=1)
    print(f"[INFO] 르로봇 연결 성공 ({LEROBOT_PORT})")
except Exception as e:
    print(f"[WARN] 르로봇 미연결: {e}")
    lerobot = None


# ==========================================
# 3. AI 비전 모델 로드 (ONNX)
# ==========================================
print(f"[INFO] ONNX 모델 로딩 중... ({MODEL_PATH})")
try:
    model = YOLO(MODEL_PATH, task='detect')
    print(f"[INFO] 모델 로딩 완료! 클래스: {model.names}")
except Exception as e:
    print(f"[WARN] 모델 로딩 실패: {e}")
    model = None


# ==========================================
# 4. Gemini AI 설정
# ==========================================
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("[INFO] Gemini 초기화 완료")
except Exception as e:
    print(f"[WARN] Gemini 초기화 실패: {e}")
    gemini_client = None


# ==========================================
# 5. 공유 상태 변수
# ==========================================
latest_frame    = None
frame_lock      = threading.Lock()
system_running  = False

detection_log   = []
action_log      = []

last_sent_value = None
last_sent_time  = 0.0

fps_value   = 0.0
fps_counter = 0
fps_timer   = time.time()


# ==========================================
# 6. Pydantic Models
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
# 7. 숫자 전용 필터 & 르로봇 전송
# ==========================================
def is_digit_class(cls_name: str) -> bool:
    """클래스명이 숫자(0~9)인지 판별"""
    digits = ''.join(filter(str.isdigit, cls_name))
    return bool(digits)


def extract_segment_value(cls_name: str) -> str | None:
    """클래스명에서 숫자만 추출. 'seg_3'→'3', '7'→'7'"""
    digits = ''.join(filter(str.isdigit, cls_name))
    return digits if digits else None


def send_to_lerobot(value: str):
    """7-segment 숫자를 르로봇 시리얼로 전송 (1초 중복 방지)"""
    global last_sent_value, last_sent_time

    now = time.time()
    if value == last_sent_value and (now - last_sent_time) < 1.0:
        return

    if lerobot:
        try:
            command = f"SEG:{value}\n"
            lerobot.write(command.encode('utf-8'))
            last_sent_value = value
            last_sent_time  = now
            print(f"[LEROBOT] 전송: {command.strip()}")

            action_log.append({
                "time":   datetime.now().strftime("%H:%M:%S"),
                "action": f"SEG 전송: {value}",
                "result": "OK"
            })
            if len(action_log) > 200:
                action_log.pop(0)
        except Exception as e:
            print(f"[ERROR] 르로봇 전송 실패: {e}")
    else:
        print(f"[LEROBOT] (미연결) 시도값: SEG:{value}")


# ==========================================
# 8. FastAPI 엔드포인트
# ==========================================

@app.get("/")
async def serve_index():
    return FileResponse(INDEX_HTML)


async def generate_frames():
    frame_interval = 1.0 / TARGET_FPS
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            await asyncio.sleep(0.05)
            continue

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            await asyncio.sleep(0.05)
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buffer.tobytes()
            + b'\r\n'
        )
        await asyncio.sleep(frame_interval)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/control")
async def control(req: ControlRequest):
    global system_running
    cmd = req.command.upper()
    if cmd == "START":
        system_running = True
    elif cmd == "STOP":
        system_running = False
    if arduino:
        try:
            arduino.write(f"{cmd}\n".encode('utf-8'))
        except Exception as e:
            print(f"[ERROR] 아두이노 전송 실패: {e}")
    return {"status": "success", "command": cmd}


@app.get("/data")
async def get_data():
    return {
        "stats": {
            "cpu":    psutil.cpu_percent(interval=None),
            "memory": psutil.virtual_memory().percent,
            "fps":    round(fps_value, 2)
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
당신은 물류 로봇 공정 분석 전문가입니다.
현재 시스템 상태:
- CPU: {ctx.get('cpu', 'N/A')}  FPS: {ctx.get('fps', 'N/A')}
- 정상: {ctx.get('good', 0)}개  불량: {ctx.get('defect', 0)}개
질문: {req.prompt}
한국어로 3~5문장으로 답변해 주세요.
"""
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash', contents=prompt
        )
        return {"answer": response.text}
    except Exception as e:
        return {"answer": f"Gemini 오류: {str(e)}"}


@app.post("/api/robot/move")
async def move_robot(req: RobotRequest):
    if not arduino:
        return {"status": "warn", "message": "아두이노 미연결"}
    try:
        arduino.write(f"{req.target_item}\n".encode('utf-8'))
        action_log.append({"time": datetime.now().strftime("%H:%M:%S"),
                            "action": f"MOVE:{req.target_item}", "result": "OK"})
        if len(action_log) > 200: action_log.pop(0)
        return {"status": "success", "message": f"{req.target_item} 전송 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/conveyor")
async def control_conveyor(req: ConveyorRequest):
    if not arduino:
        return {"status": "warn", "message": "아두이노 미연결"}
    try:
        arduino.write(f"CONV:{req.command.upper()}\n".encode('utf-8'))
        return {"status": "success", "message": f"컨베이어 {req.command} 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/robot/test")
async def test_robot(req: TestRequest):
    if not arduino:
        return {"status": "warn", "message": "아두이노 미연결"}
    try:
        arduino.write(f"MOVE:{req.command.upper()}\n".encode('utf-8'))
        return {"status": "success", "message": f"테스트 {req.command} 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 9. 카메라 & 비전 처리 쓰레드
# ==========================================
def camera_loop():
    global latest_frame, fps_value, fps_counter, fps_timer

    print(f"[INFO] 카메라 시작 (index={CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    # 카메라 자체 FPS를 TARGET_FPS로 제한
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    print(f"[INFO] 카메라 열기 성공 — 목표 FPS: {TARGET_FPS}")

    frame_time   = 1.0 / TARGET_FPS  # 한 프레임당 최대 허용 시간
    last_infer   = 0.0               # 마지막 추론 시각

    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임 읽기 실패, 재시도...")
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (640, 480))

        # FPS 계산
        fps_counter += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps_value   = fps_counter / (now - fps_timer)
            fps_counter = 0
            fps_timer   = now

        # ✅ YOLO 추론 — TARGET_FPS 유지를 위해 추론 간격 조절
        if model is not None:
            now = time.time()
            # 추론이 너무 느리면 프레임 스킵 (FPS 유지 우선)
            if (now - last_infer) >= frame_time:
                # INFER_SIZE로 축소해 추론 속도 향상
                small = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))
                results = model(small, conf=CONF_THRESHOLD, verbose=False)
                last_infer = now

                # 좌표를 원본(640x480) 비율로 복원
                scale_x = 640 / INFER_SIZE
                scale_y = 480 / INFER_SIZE

                annotated = frame.copy()

                for box in results[0].boxes:
                    cls_name = model.names[int(box.cls)]

                    # ✅ 숫자 클래스만 처리 (숫자 아닌 클래스 완전 무시)
                    if not is_digit_class(cls_name):
                        continue

                    conf = float(box.conf)
                    bx, by, bw, bh = box.xywh[0]
                    x1 = int((bx - bw / 2) * scale_x)
                    y1 = int((by - bh / 2) * scale_y)
                    x2 = int((bx + bw / 2) * scale_x)
                    y2 = int((by + bh / 2) * scale_y)
                    cx = int(bx * scale_x)
                    cy = int(by * scale_y)

                    seg_value = extract_segment_value(cls_name)

                    # 바운딩 박스 직접 그리기
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{seg_value} {conf:.2f}"
                    cv2.putText(annotated, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    detection_log.append({
                        "time":   datetime.now().strftime("%H:%M:%S"),
                        "class":  cls_name,
                        "conf":   f"{conf:.2f}",
                        "coords": f"({cx}, {cy}, -)"
                    })
                    if len(detection_log) > 200:
                        detection_log.pop(0)

                    # 르로봇 전송
                    if seg_value:
                        send_to_lerobot(seg_value)

                with frame_lock:
                    latest_frame = annotated
            else:
                # 추론 스킵된 프레임도 스트림엔 표시
                with frame_lock:
                    latest_frame = frame.copy()
        else:
            with frame_lock:
                latest_frame = frame.copy()

        # FPS 유지: 남은 시간 sleep
        elapsed = time.time() - loop_start
        sleep_t = frame_time - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


# ==========================================
# 10. 메인 실행
# ==========================================
if __name__ == "__main__":
    vision_thread = threading.Thread(target=camera_loop, daemon=True)
    vision_thread.start()

    print("[INFO] FastAPI 서버 구동 → http://0.0.0.0:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)
