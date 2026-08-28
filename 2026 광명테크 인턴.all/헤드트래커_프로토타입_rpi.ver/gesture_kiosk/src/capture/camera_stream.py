"""capture 모듈 — 카메라에서 프레임을 실시간으로 읽어 온다 (기획서 2.2).

캡처는 전용 스레드에서 돌리고(기획서 3.2 멀티스레딩), capture_frame()은
항상 가장 최신 프레임을 돌려준다. 추론이 느려도 오래된 프레임이 쌓이지 않는다.

2026-07-22 라즈베리파이5 이식(rpi.ver): 백엔드 2종 지원 —
- v4l2  : USB 웹캠(win.ver와 동일 장치를 그대로 재사용할 때). cv2.VideoCapture(CAP_V4L2).
- picamera2 : 공식 카메라 모듈(CSI). libcamera 하드웨어 ISP가 리사이즈·BGR 변환을
              직접 수행해 CPU를 전혀 안 쓴다 — proc_width_px/proc_height_px 그대로
              센서에서 뽑아내면 Preprocessor의 크롭/리사이즈 단계가 통째로 스킵돼
              추론에 쓸 CPU 여유가 그만큼 늘어난다(이 이식에서 "추론 성능 최대한"의
              핵심 레버). config의 camera.backend로 선택 — 기본은 v4l2(USB 웹캠 가정,
              하드웨어 구성 확정 전 가장 안전한 기본값).

⚠ picamera2 경로는 실제 라즈베리파이5 + 카메라 모듈에서 검증되지 않았다(개발 PC가
윈도우라 이 세션에서 실기 테스트가 불가능했다) — picamera2 API 문서 기준으로 작성.
실기에서 문제가 있으면 이 파일의 _capture_loop_picamera2/_open_picamera2만 고치면 된다.
"""
import threading
import time

import cv2

from src.utils.logger import get_logger
from src.utils.metrics import FpsMeter

logger = get_logger("capture")

FIRST_FRAME_TIMEOUT_SEC = 5.0


def _open_v4l2(config):
    """USB 웹캠 등 V4L2 장치를 연다 (라즈베리파이OS 기본 백엔드)."""
    cam = config["camera"]
    cap = cv2.VideoCapture(cam["device_id"], cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"카메라(device_id={cam['device_id']})를 열 수 없습니다 (V4L2)")
    # 무압축(YUY2) 1280x720은 USB 대역폭 한계로 캡처가 ~5 FPS에 묶인다 — MJPG 기본
    fourcc = cam.get("fourcc", "mjpg")
    if fourcc and fourcc != "auto":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc.upper()))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width_px"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height_px"])
    return cap


def _open_picamera2(config):
    """공식 카메라 모듈(CSI)을 libcamera 하드웨어 ISP로 연다.

    센서에서 곧바로 proc_width_px/proc_height_px·BGR888로 뽑아낸다 — Preprocessor가
    받는 시점에 이미 목표 해상도라 크롭/리사이즈가 스킵된다(pass-through 경로,
    test_passes_through_when_already_proc_size와 동일 조건).
    """
    from picamera2 import Picamera2  # 지연 임포트 — RPi OS(libcamera) 전용 의존성

    cam = config["camera"]
    proc_w, proc_h = cam["proc_width_px"], cam["proc_height_px"]
    picam2 = Picamera2(camera_num=cam["device_id"])
    video_config = picam2.create_video_configuration(
        main={"size": (proc_w, proc_h), "format": "BGR888"}
    )
    picam2.configure(video_config)
    picam2.start()
    return picam2


class CameraStream:
    """카메라 캡처 스레드. capture_frame()으로 최신 프레임(BGR, np.ndarray)을 얻는다."""

    def __init__(self, config):
        self._config = config
        self._backend = config["camera"].get("backend", "v4l2")
        self._cap = None          # v4l2: cv2.VideoCapture, picamera2: Picamera2 인스턴스
        self._frame = None
        self._frame_lock = threading.Lock()
        self._thread = None
        self.is_running = False
        self.fps_meter = FpsMeter()

    def start(self):
        if self._backend == "picamera2":
            self._cap = _open_picamera2(self._config)
            loop = self._capture_loop_picamera2
        else:
            self._cap = _open_v4l2(self._config)
            loop = self._capture_loop_v4l2
        self.is_running = True
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        logger.info(
            "카메라 캡처 스레드 시작 (backend=%s, device_id=%s)",
            self._backend, self._config["camera"]["device_id"],
        )
        return self

    def _capture_loop_v4l2(self):
        while self.is_running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            with self._frame_lock:
                self._frame = frame
            self.fps_meter.update()

    def _capture_loop_picamera2(self):
        while self.is_running:
            frame = self._cap.capture_array("main")  # BGR888 요청 — 변환 없이 바로 사용
            with self._frame_lock:
                self._frame = frame
            self.fps_meter.update()

    def capture_frame(self):
        """최신 프레임(np.ndarray, BGR)을 돌려준다. 첫 프레임은 잠시 대기한다."""
        deadline_sec = time.monotonic() + FIRST_FRAME_TIMEOUT_SEC
        while True:
            with self._frame_lock:
                if self._frame is not None:
                    return self._frame.copy()
            if time.monotonic() > deadline_sec:
                raise RuntimeError("카메라에서 프레임을 받지 못했습니다 (연결/장치 번호 확인)")
            time.sleep(0.01)

    def stop(self):
        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is None:
            return
        if self._backend == "picamera2":
            self._cap.stop()
            self._cap.close()
        else:
            self._cap.release()
