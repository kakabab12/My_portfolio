import cv2
from ultralytics import YOLO

class VideoCamera:
    def __init__(self):
        # 1. TensorRT 엔진 로드 (기능 절대 유지!)
        self.model = YOLO("best.engine", task="detect")
        # 2. 카메라 연결 (USB캠 0번)
        self.cap = cv2.VideoCapture(0)
        
    def __del__(self):
        self.cap.release()

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return None

        # 3. TensorRT GPU 가속 추론 (XYZ 계산 없이 박스만 그리기)
        # device=0 옵션으로 64% 점유율 유지!
        results = self.model.predict(frame, device=0, conf=0.5, verbose=False)
        
        # 4. 인식 결과가 그려진 화면 가져오기
        annotated_frame = results[0].plot()
        
        # 5. JPEG로 인코딩하여 전송 가능한 형태로 변환
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        return jpeg.tobytes()

# FastAPI에서 호출할 generator 함수
def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')