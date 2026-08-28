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
INDEX_HTML     = "/home/user/project/index.html"   # index.html 절대 경로
MODEL_PATH     = "/home/user/project/best.onnx"  # best.engine 절대 경로
ARDUINO_PORT   = "/dev/ttyACM0"                    # 아두이노 포트
LEROBOT_PORT   = "/dev/ttyUSB0"                    # 르로봇 포트
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"        # Gemini API 키
CAMERA_INDEX   = 1                                 # 카메라 인덱스 (1로 확인됨)
# ==========================================

app = FastAPI(title="Logistics Robot Control Server")


# ==========================================
# 1. 아두이노 시리얼 (미연결이면 None으로 계속 실행)
# ==========================================
try:
    arduino = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
    print(f"[INFO] 아두이노 연결 성공 ({ARDUINO_PORT})")
except Exception as e:
    print(f"[WARN] 아두이노 미연결 — 연결 없이 계속 실행: {e}")
    arduino = None


# ==========================================
# 2. 르로봇 시리얼 (미연결이면 None으로 계속 실행)
# ==========================================
try:
    lerobot = serial.Serial(LEROBOT_PORT, 9600, timeout=1)
    print(f"[INFO] 르로봇 연결 성공 ({LEROBOT_PORT})")
except Exception as e:
    print(f"[WARN] 르로봇 미연결 — 연결 없이 계속 실행: {e}")
    lerobot = None


# ==========================================
# 3. AI 비전 모델 로드
# ==========================================
print(f"[INFO] AI 엔진 로딩 중... ({MODEL_PATH})")
try:
    model = YOLO(MODEL_PATH, task='detect')
    print(f"[INFO] AI 엔진 로딩 완료! 클래스: {model.names}")
except Exception as e:
    print(f"[WARN] 모델 로딩 실패 — 카메라 원본만 스트리밍됩니다: {e}")
    model = None


# ==========================================
# 4. Gemini AI 설정
# ==========================================
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("[INFO] Gemini 클라이언트 초기화 완료")
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
    command: str  # START | STOP | CALIB

class GeminiRequest(BaseModel):
    prompt: str
    context: dict

class TestRequest(BaseModel):
    command: str


# ==========================================
# 7. 7-segment 값 추출 & 르로봇 전송
# ==========================================
def extract_segment_value(cls_name: str):
    """클래스명에서 숫자만 추출. 예) 'seg_3' → '3', '7' → '7'"""
    digits = ''.join(filter(str.isdigit, cls_name))
    return digits if digits else None


def send_to_lerobot(value: str):
    """7-segment 인식값을 르로봇으로 시리얼 전송 (1초 내 중복 방지)"""
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
        print(f"[LEROBOT] (미연결) 전송 시도값: SEG:{value}")


# ==========================================
# 8. FastAPI 엔드포인트
# ==========================================

@app.get("/")
async def serve_index():
    return FileResponse(INDEX_HTML)  # 절대 경로 사용


# --- MJPEG 스트림 ---
async def generate_frames():
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
        await asyncio.sleep(0.033)  # ~30fps

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# --- 시스템 제어 ---
@app.post("/control")
async def control(req: ControlRequest):
    global system_running
    cmd = req.command.upper()

    if cmd == "START":
        system_running = True
        print("[CONTROL] 시스템 시작")
    elif cmd == "STOP":
        system_running = False
        print("[CONTROL] 시스템 정지")
    elif cmd == "CALIB":
        print("[CONTROL] 캘리브레이션")

    if arduino:
        try:
            arduino.write(f"{cmd}\n".encode('utf-8'))
        except Exception as e:
            print(f"[ERROR] 아두이노 전송 실패: {e}")

    return {"status": "success", "command": cmd}


# --- 실시간 데이터 ---
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


# --- Gemini AI ---
@app.post("/ask_gemini")
async def ask_gemini(req: GeminiRequest):
    if not gemini_client:
        return {"answer": "Gemini 클라이언트가 초기화되지 않았습니다."}
    try:
        ctx = req.context
        prompt = f"""
당신은 물류 로봇 공정 분석 전문가입니다.
현재 시스템 상태:
- CPU 사용률: {ctx.get('cpu', 'N/A')}
- 카메라 FPS: {ctx.get('fps', 'N/A')}
- 정상 처리: {ctx.get('good', 0)}개
- 불량 감지: {ctx.get('defect', 0)}개

사용자 질문: {req.prompt}

간결하고 실용적인 한국어로 3~5문장으로 답변해 주세요.
"""
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        return {"answer": response.text}
    except Exception as e:
        return {"answer": f"Gemini 오류: {str(e)}"}


# --- 로봇팔 이동 ---
@app.post("/api/robot/move")
async def move_robot(req: RobotRequest):
    if not arduino:
        return {"status": "warn", "message": "아두이노 미연결 — 명령 무시됨"}
    try:
        arduino.write(f"{req.target_item}\n".encode('utf-8'))
        print(f"[API] 로봇팔 명령: {req.target_item}")
        action_log.append({
            "time":   datetime.now().strftime("%H:%M:%S"),
            "action": f"MOVE:{req.target_item}",
            "result": "OK"
        })
        if len(action_log) > 200:
            action_log.pop(0)
        return {"status": "success", "message": f"{req.target_item} 전송 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 컨베이어 ---
@app.post("/api/conveyor")
async def control_conveyor(req: ConveyorRequest):
    if not arduino:
        return {"status": "warn", "message": "아두이노 미연결 — 명령 무시됨"}
    try:
        cmd = f"CONV:{req.command.upper()}\n"
        arduino.write(cmd.encode('utf-8'))
        print(f"[API] 컨베이어: {cmd.strip()}")
        return {"status": "success", "message": f"컨베이어 {req.command} 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 테스트 ---
@app.post("/api/robot/test")
async def test_robot(req: TestRequest):
    if not arduino:
        return {"status": "warn", "message": "아두이노 미연결 — 명령 무시됨"}
    try:
        cmd = f"MOVE:{req.command.upper()}\n"
        arduino.write(cmd.encode('utf-8'))
        print(f"[API] 테스트: {cmd.strip()}")
        return {"status": "success", "message": f"테스트 {req.command} 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 9. 카메라 & 비전 처리 쓰레드
# ==========================================
def camera_loop():
    global latest_frame, fps_value, fps_counter, fps_timer

    print(f"[INFO] 카메라 워커 시작 (index={CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다. CAMERA_INDEX 확인 필요.")
        return

    print(f"[INFO] 카메라 열기 성공 (index={CAMERA_INDEX})")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임 읽기 실패, 재시도...")
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (640, 480))

        # FPS 계산
        fps_counter += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps_value   = fps_counter / (now - fps_timer)
            fps_counter = 0
            fps_timer   = now

        # ✅ YOLO 항상 실행 (START 버튼 무관)
        if model is not None:
            results         = model(frame)
            annotated_frame = results[0].plot()

            for box in results[0].boxes:
                cls_name = model.names[int(box.cls)]
                conf     = float(box.conf)
                x        = int(box.xywh[0][0])
                y        = int(box.xywh[0][1])

                detection_log.append({
                    "time":   datetime.now().strftime("%H:%M:%S"),
                    "class":  cls_name,
                    "conf":   f"{conf:.2f}",
                    "coords": f"({x}, {y}, -)"
                })
                if len(detection_log) > 200:
                    detection_log.pop(0)

                # ✅ 7-segment 숫자 → 르로봇 전송
                seg_value = extract_segment_value(cls_name)
                if seg_value:
                    send_to_lerobot(seg_value)

            with frame_lock:
                latest_frame = annotated_frame
        else:
            # 모델 없으면 원본 프레임 스트리밍
            with frame_lock:
                latest_frame = frame.copy()


# ==========================================
# 10. 메인 실행
# ==========================================
if __name__ == "__main__":
    vision_thread = threading.Thread(target=camera_loop, daemon=True)
    vision_thread.start()

    print("[INFO] FastAPI 서버 구동 → http://0.0.0.0:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)
