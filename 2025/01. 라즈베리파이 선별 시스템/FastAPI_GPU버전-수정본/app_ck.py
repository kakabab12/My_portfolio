import os
import time
import threading
import psutil
import cv2
import numpy as np
import asyncio
from collections import deque, Counter
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import google.genai as genai

# ✅ [1] 하이브리드 카메라 로드
try:
    from camera import Camera
    cam = Camera()
    print("✅ 카메라 모듈 연결 성공")
except Exception as e:
    print(f"⚠️ 카메라 로드 실패({e}). 더미 카메라 생성.")
    class DummyCamera:
        def __init__(self): self.fps = 30
        def get_frame(self): return np.zeros((480, 640, 3), dtype=np.uint8)
    cam = DummyCamera()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ [2] Gemini 설정
GEMINI_API_KEY = "AIzaSyC6xJAAPG0dN7hrjsyxswU6-quK8mqaVbE"
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-pro" 

# --- [전역 변수] ---
RECTANGULARITY_THRESHOLD = 0.83 
current_stats = { "cpu": 0, "memory": 0, "fps": 0, "good": 0, "defect": 0 }
detection_log = deque(maxlen=20)
action_log = deque(maxlen=20)
data_lock = threading.Lock()

# ✅ [3] YOLO TensorRT 모델 로드 (경로 및 GPU 설정)
# ONNX 대신 아까 만든 .engine 파일을 사용합니다.
model_path = 'best.engine' 
if os.path.exists(model_path):
    try:
        # task='segment'는 학습시킨 모델 종류에 맞춰 'detect'로 바뀔 수 있습니다.
        model = YOLO(model_path) 
        print(f"✅ TensorRT 엔진({model_path}) GPU 모드 로드 완료")
    except Exception as e:
        model = None
        print(f"❌ 모델 로드 오류: {e}")
else:
    model = None
    print(f"⚠️ {model_path} 없음. 탐지 없이 진행.")

# -----------------------------
# 🛠️ 백그라운드 로직
# -----------------------------
def update_system_stats():
    while True:
        with data_lock:
            current_stats["cpu"] = psutil.cpu_percent()
            current_stats["memory"] = psutil.virtual_memory().percent
            current_stats["fps"] = getattr(cam, 'fps', 0)
        time.sleep(1)

def run_yolo_detection():
    """실제로 GPU를 써서 물체를 찾는 핵심 루프"""
    global model
    while True:
        if model is None: 
            time.sleep(1); continue
        
        frame = cam.get_frame()
        if frame is None: 
            time.sleep(0.01); continue
        
        # ✅ [핵심] GPU(device=0)와 FP16 가속(half=True) 사용
        results = model.predict(source=frame, device=0, half=True, verbose=False)
        
        # 탐지 결과 분석 및 로그 기록
        for r in results:
            if len(r.boxes) > 0:
                conf = float(r.boxes.conf[0])
                label = "Box" # 클래스 이름에 맞게 수정 가능
                
                with data_lock:
                    detection_log.append({"time": time.strftime('%H:%M:%S'), "label": label, "conf": conf})
                    # 간단한 판별 로직 (예시: 정확도 0.8 이상이면 정상 카운트)
                    if conf > 0.8:
                        current_stats["good"] += 1
                    else:
                        current_stats["defect"] += 1
        
        time.sleep(0.01)

# -----------------------------
# 🌐 FastAPI 엔드포인트
# -----------------------------
@app.get("/data")
async def get_data():
    with data_lock:
        return {
            "stats": current_stats,
            "detection_log": list(detection_log),
            "action_log": list(action_log)
        }

@app.post("/ask_gemini")
async def ask_gemini(request: Request):
    try:
        data = await request.json()
        user_prompt = data.get("prompt")
        ctx = data.get("context", {}) # 프론트에서 보낸 데이터

        prompt_with_data = f"""
        너는 스마트 팩토리 '스카이넷' 프로젝트의 AI 분석가야.
        [실시간 데이터] CPU: {current_stats['cpu']}%, 정상: {current_stats['good']}개, 불량: {current_stats['defect']}개
        사용자 질문: {user_prompt}
        위 데이터를 기반으로 한글로 전문적인 답변을 해줘.
        """
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt_with_data)
        return {"answer": response.text}
    except Exception as e:
        return {"answer": f"분석 오류: {str(e)}"}

async def video_streamer():
    while True:
        frame = cam.get_frame()
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        await asyncio.sleep(0.04)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(video_streamer(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    threading.Thread(target=update_system_stats, daemon=True).start()
    threading.Thread(target=run_yolo_detection, daemon=True).start()
    
    import uvicorn
    # ✅ [중요] 0.0.0.0으로 설정해야 외부(노트북 등)에서 접속 가능
    uvicorn.run(app, host="0.0.0.0", port=5000)