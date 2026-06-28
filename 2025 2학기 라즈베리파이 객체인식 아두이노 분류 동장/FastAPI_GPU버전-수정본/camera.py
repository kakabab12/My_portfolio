import cv2
import time
import numpy as np
import threading

# ✅ 인텔 리얼센스 라이브러리 로드
HAS_REALSENSE = False
try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
    print("✅ Intel RealSense SDK 로드 성공")
except ImportError:
    HAS_REALSENSE = False
    print("⚠️ pyrealsense2 라이브러리가 없습니다. 일반 웹캠 모드로 전환합니다.")

class Camera:
    def __init__(self):
        self.fps = 0
        self._prev_time = time.time()
        self.frame_count = 0
        self.frame = None
        self.stopped = False
        
        # RealSense 관련 설정
        self.pipeline = None
        self.cap = None

        if HAS_REALSENSE:
            try:
                # 1. 리얼센스 파이프라인 및 설정 초기화
                self.pipeline = rs.pipeline()
                config = rs.config()
                
                # 컬러 스트림 설정 (640x480, 30fps)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                
                # 파이프라인 시작
                self.pipeline.start(config)
                print("🚀 RealSense 카메라 스트리밍 시작")
                time.sleep(1.0)
            except Exception as e:
                print(f"❌ RealSense 초기화 실패: {e}. 웹캠으로 전환합니다.")
                self.pipeline = None
                self._init_webcam()
        else:
            self._init_webcam()

        # 프레임 업데이트 스레드 시작
        threading.Thread(target=self._update, daemon=True).start()

    def _init_webcam(self):
        """리얼센스 실패 시 일반 USB 웹캠으로 작동"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _update(self):
        while not self.stopped:
            if self.pipeline:
                # 2. 리얼센스 프레임 가져오기
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                # numpy 배열로 변환
                self.frame = np.asanyarray(color_frame.get_data())
            
            elif self.cap:
                # 일반 웹캠 프레임 가져오기
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
            
            # --- [유지] FPS 계산 로직 ---
            self.frame_count += 1
            curr = time.time()
            if curr - self._prev_time > 1.0:
                self.fps = self.frame_count / (curr - self._prev_time)
                self.frame_count = 0
                self._prev_time = curr
            
            # CPU 점유율 과다 방지를 위해 아주 잠깐 휴식
            time.sleep(0.01)

    def get_frame(self):
        return self.frame

    def release(self):
        self.stopped = True
        if self.pipeline:
            self.pipeline.stop()
        if self.cap:
            self.cap.release()
        print("👋 카메라 연결이 종료되었습니다.")