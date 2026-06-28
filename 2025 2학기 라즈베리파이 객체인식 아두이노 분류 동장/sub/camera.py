# camera.py (라즈베리파이 카메라 모듈 3 전용)
import cv2
from picamera2 import Picamera2
import time

class Camera:
    def __init__(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": (640, 480)})
        self.picam2.configure(config)
        self.picam2.start()
        
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        time.sleep(2)
        print("카메라 모듈 3가 성공적으로 연결되었습니다.")

    def __del__(self):
        self.picam2.stop()

    def get_frame(self):
        frame = self.picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv.COLOR_RGBA2BGR)
        
        # 좌우 반전 적용
        frame_bgr = cv2.flip(frame_bgr, 1)
        
        # FPS 계산 및 표시
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        fps_text = f"FPS: {self.fps:.2f}"
        cv2.putText(frame_bgr, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        return buffer.tobytes()