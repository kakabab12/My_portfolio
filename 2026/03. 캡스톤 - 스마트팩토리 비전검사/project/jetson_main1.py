import cv2
import serial
import threading
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# FastAPI 앱 초기화
app = FastAPI(title="Logistics Robot Control Server")

# ==========================================
# 1. 아두이노 시리얼 통신 설정
# ==========================================
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600)
    print("[INFO] 아두이노 연결 성공 (/dev/ttyACM0)")
except Exception as e:
    print(f"[ERROR] 아두이노 연결 실패: {e}")
    arduino = None

# ==========================================
# 2. AI 비전 모델 로드 (.engine 적용 완료)
# ==========================================
print("[INFO] TensorRT AI 엔진 로딩 중... (best.engine)")
try:
    # 🚨 여기에 변환하신 엔진 파일 이름을 정확히 넣어주세요!
    model = YOLO("best.engine", task='detect')
    print("[INFO] AI 엔진 로딩 완료!")
except Exception as e:
    print(f"[ERROR] 모델 로딩 실패: {e}")


# ==========================================
# 3. 데이터 통신 규격 (Pydantic Models)
# ==========================================
class RobotRequest(BaseModel):
    target_item: str
    action: str

class ConveyorRequest(BaseModel):
    command: str


# ==========================================
# 4. FastAPI 엔드포인트 (명령 수신부)
# ==========================================
@app.post("/api/robot/move")
async def move_robot(req: RobotRequest):
    if not arduino:
        raise HTTPException(status_code=500, detail="아두이노가 연결되지 않았습니다.")
    
    try:
        # LLM(노트북)이 보낸 물건 이름을 아두이노로 전송
        # 예: 'apple\n', 'box1\n'
        command_str = f"{req.target_item}\n"
        arduino.write(command_str.encode('utf-8'))
        print(f"[API] 로봇팔 이동 명령 전송: {req.target_item}")
        return {"status": "success", "message": f"{req.target_item} 픽업 명령 전송 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conveyor")
async def control_conveyor(req: ConveyorRequest):
    if not arduino:
        raise HTTPException(status_code=500, detail="아두이노가 연결되지 않았습니다.")
    
    try:
        # 컨베이어 제어 명령 전송 (예: CONV:START\n)
        command_str = f"CONV:{req.command.upper()}\n"
        arduino.write(command_str.encode('utf-8'))
        print(f"[API] 컨베이어 명령 전송: {command_str.strip()}")
        return {"status": "success", "message": f"컨베이어 {req.command} 명령 전송 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 5. 카메라 및 비전 처리 쓰레드 (독립 실행)
# ==========================================
def camera_loop():
    print("[INFO] 카메라 워커 시작됨...")
    cap = cv2.VideoCapture(0) # 카메라 포트 확인 필요 시 0, 1, 2 등으로 변경
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # 프레임 크기 줄이기 (연산 속도 향상용)
        frame = cv2.resize(frame, (640, 480))
        
        # YOLO 엔진으로 객체 인식
        results = model(frame)
        
        # 인식 결과 화면에 그리기
        annotated_frame = results[0].plot()
        cv2.imshow("Logistics Vision", annotated_frame)
        
        # 'q' 키를 누르면 카메라 창 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

    # jetson_main.py 에 추가할 코드

class TestRequest(BaseModel):
    command: str

@app.post("/api/robot/test")
async def test_robot(req: TestRequest):
    if not arduino:
        raise HTTPException(status_code=500, detail="아두이노가 연결되지 않았습니다.")
    
    try:
        # 클로드가 보낸 명령을 아두이노 포맷으로 변경 (예: MOVE:RIGHT\n, MOVE:STOP\n)
        command_str = f"MOVE:{req.command.upper()}\n"
        arduino.write(command_str.encode('utf-8'))
        print(f"[API] 클로드로부터 테스트 명령 수신: {command_str.strip()}")
        return {"status": "success", "message": f"테스트 명령 {req.command} 전송 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 6. 메인 실행부
# ==========================================
if __name__ == "__main__":
    # 1) 카메라 쓰레드를 뒤에서 먼저 돌립니다. (서버와 겹치지 않게 방지)
    vision_thread = threading.Thread(target=camera_loop, daemon=True)
    vision_thread.start()
    
    # 2) FastAPI 웹 서버를 시작합니다.
    print("[INFO] FastAPI 서버 구동 시작...")
    uvicorn.run(app, host="0.0.0.0", port=5000)
