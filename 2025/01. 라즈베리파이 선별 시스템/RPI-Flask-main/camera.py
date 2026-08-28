from picamera2 import Picamera2
import cv2
import time
import numpy as np
from libcamera import controls

class Camera:
    def __init__(self):
        try:
            self.picam2 = Picamera2()
            
            # ✅ 해상도 설정: 1920x1080 (FHD)
            # 속도가 느리다면 (640, 480)으로 변경하세요.
            config = self.picam2.create_preview_configuration(
                main={"format": 'XRGB8888', "size": (1920, 1080)}
            )
            self.picam2.configure(config)
            
            # ✅ 제어 설정: 오토 포커스(AF) 및 화이트밸런스(AWB) 활성화
            # FrameRate는 30으로 설정하여 안정성 확보
            self.picam2.set_controls({
                "AwbMode": controls.AwbModeEnum.Auto,       
                "AfMode": controls.AfModeEnum.Continuous,   
                "FrameRate": 30                            
            })

            self.picam2.start()
            
            # [에러 해결] fps 및 시간 변수 초기화
            self.fps = 0
            self._prev_time = time.time()
            self.frame_count = 0
            
            # 카메라 워밍업
            time.sleep(2.0) 
            print("✅ 카메라 초기화 성공 (1920x1080 @ 30FPS, AF On)")
            
        except Exception as e:
            print(f"❌ 카메라 초기화 실패: {e}")
            self.picam2 = None

    def get_frame(self):
        if not self.picam2:
            return None
            
        try:
            # 1. 프레임 캡처 (Numpy 배열)
            frame = self.picam2.capture_array()
            
            # 2. 색상 변환
            # OpenCV는 기본적으로 BGR을 사용하므로 RGB로 변환하면 색이 반전(파란색<->빨간색)될 수 있습니다.
            # 색상이 이상하면 cv2.COLOR_RGBA2RGB 대신 cv2.COLOR_RGBA2BGR 를 사용하세요.
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            # 3. 화면 반전 (카메라가 뒤집힌 경우 주석 해제)
            # frame_bgr = cv2.flip(frame_bgr, -1) 

            # 4. FPS 계산 (app.py에서 호출할 때 에러 방지)
            self.frame_count += 1
            curr_time = time.time()
            elapsed = curr_time - self._prev_time
            
            if elapsed > 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self._prev_time = curr_time

            return frame_bgr
            
        except Exception as e:
            print(f"프레임 캡처 오류: {e}")
            return None

    def __del__(self):
        if self.picam2:
            try:
                self.picam2.stop()
            except:
                pass