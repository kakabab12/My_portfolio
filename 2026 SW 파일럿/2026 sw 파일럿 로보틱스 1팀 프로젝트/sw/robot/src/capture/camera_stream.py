"""capture 모듈 — USB 카메라에서 프레임을 실시간으로 읽어 온다.

캡처는 전용 스레드에서 돌리고, capture_frame()은 항상 가장 최신 프레임을
돌려준다. 추론이 느려도 오래된 프레임이 쌓이지 않는다.

런타임 자동 복구: USB 탈락·드라이버 멈춤으로 프레임이 recovery_timeout_sec
동안 안 오면 장치를 닫고 재연결될 때까지 계속 다시 연다 — 무인 주행 중
카메라 하나가 조용히 멈추는 사고를 방지한다.
"""
import sys
import threading
import time

import cv2

from src.utils.logger import get_logger
from src.utils.metrics import FpsMeter

logger = get_logger("capture")

FIRST_FRAME_TIMEOUT_SEC = 5.0
NEW_FRAME_TIMEOUT_SEC = 1.0


def init_camera(config, device_id):
    """config 기준으로 카메라 장치를 열어 cv2.VideoCapture를 돌려준다.

    device_id가 문자열(URL)이면 다른 컴퓨터(예: 노트북 camera_server.py)가
    스트리밍하는 네트워크 카메라로 취급한다 — 로컬 백엔드 지정(V4L2 등)·
    해상도/포맷 요청은 로컬 장치 전용이라 URL에는 의미가 없으므로 건너뛴다
    (송신 측이 이미 인코딩해 보내는 그대로 받는다).
    """
    if isinstance(device_id, str):
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            raise RuntimeError(f"카메라 스트림({device_id})을 열 수 없습니다")
        _log_camera_negotiation(cap, device_id)
        return cap

    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    elif sys.platform.startswith("win"):
        backend_name = config["camera"].get("windows_backend", "auto")
        windows_backends = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
        if backend_name in windows_backends:
            cap = cv2.VideoCapture(device_id, windows_backends[backend_name])
        else:
            cap = cv2.VideoCapture(device_id)
    else:
        cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError(f"카메라(device_id={device_id})를 열 수 없습니다")
    fourcc = config["camera"].get("fourcc", "mjpg")
    if fourcc and fourcc != "auto":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc.upper()))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width_px"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height_px"])
    requested_fps = config["camera"].get("fps")
    if requested_fps is not None:
        cap.set(cv2.CAP_PROP_FPS, requested_fps)
    _log_camera_negotiation(cap, device_id)
    return cap


def _log_camera_negotiation(cap, device_id):
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_text = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)).strip() or "?"
    try:
        backend_label = cap.getBackendName()
    except cv2.error:
        backend_label = "?"
    logger.info(
        "카메라 협상 결과: device_id=%s · 백엔드=%s · %dx%d @ %.0f FPS · 포맷=%s",
        device_id, backend_label,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        cap.get(cv2.CAP_PROP_FPS), fourcc_text,
    )


class CameraStream:
    """카메라 캡처 스레드. capture_frame()으로 최신 프레임(BGR)을 얻는다."""

    def __init__(self, config, device_id, cap=None):
        self._config = config
        self._device_id = device_id
        self._recovery_timeout_sec = config["camera"].get("recovery_timeout_sec", 3.0)
        self._recovery_retry_sec = config["camera"].get("recovery_retry_sec", 2.0)
        self._preopened_cap = cap
        self._cap = None
        self._frame = None
        self._frame_seq = 0
        self._frame_lock = threading.Lock()
        self._new_frame_condition = threading.Condition(self._frame_lock)
        self._thread = None
        self.is_running = False
        self.fps_meter = FpsMeter()

    def start(self):
        self._cap = (self._preopened_cap if self._preopened_cap is not None
                     else init_camera(self._config, self._device_id))
        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("카메라 캡처 스레드 시작 (device_id=%s)", self._device_id)
        return self

    def _capture_loop(self):
        last_frame_sec = time.monotonic()
        while self.is_running:
            ret, frame = self._cap.read()
            if ret:
                self._publish_frame(frame)
                self.fps_meter.update()
                last_frame_sec = time.monotonic()
                continue
            if time.monotonic() - last_frame_sec < self._recovery_timeout_sec:
                time.sleep(0.1)
                continue
            self._recover_camera()
            last_frame_sec = time.monotonic()

    def _recover_camera(self):
        logger.warning("카메라 응답 없음 %.0f초 — 자동 복구 시작 (device_id=%s)",
                       self._recovery_timeout_sec, self._device_id)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        attempt_count = 0
        while self.is_running:
            deadline_sec = time.monotonic() + self._recovery_retry_sec
            while self.is_running and time.monotonic() < deadline_sec:
                time.sleep(0.1)
            if not self.is_running:
                return
            attempt_count += 1
            try:
                self._cap = init_camera(self._config, self._device_id)
            except RuntimeError:
                if attempt_count % 10 == 1:
                    logger.warning("카메라 재연결 실패 %d회 — %.0f초 간격 재시도 중 (device_id=%s)",
                                   attempt_count, self._recovery_retry_sec, self._device_id)
                continue
            logger.info("카메라 자동 복구 성공 (device_id=%s, 재시도 %d회)",
                        self._device_id, attempt_count)
            return

    def _publish_frame(self, frame):
        with self._new_frame_condition:
            self._frame = frame
            self._frame_seq += 1
            self._new_frame_condition.notify_all()

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

    def try_get_frame(self):
        """최신 (프레임, seq)를 블로킹 없이 즉시 돌려준다 — 아직 없으면 (None, seq).

        capture_frame()과 달리 대기·예외가 없다 — 카메라 2대 중 하나가 죽어도
        나머지 한 대로 루프가 계속 돌아야 하는 dual_camera에서 쓴다. seq는
        직전 호출과 비교해 "진짜 새 프레임인지"(같은 프레임 중복 추론 방지)
        판단하는 데 쓴다.
        """
        with self._frame_lock:
            frame = None if self._frame is None else self._frame.copy()
            return frame, self._frame_seq

    def capture_new_frame(self, last_seq):
        """last_seq 이후의 새 프레임을 기다려 (frame, seq)로 돌려준다.

        NEW_FRAME_TIMEOUT_SEC 안에 새 프레임이 없으면(카메라 멈칫) 기존 프레임을
        그대로 돌려줘 파이프라인이 죽지 않게 한다.
        """
        deadline_sec = time.monotonic() + NEW_FRAME_TIMEOUT_SEC
        with self._new_frame_condition:
            while self._frame_seq <= last_seq or self._frame is None:
                remaining_sec = deadline_sec - time.monotonic()
                if remaining_sec <= 0:
                    break
                self._new_frame_condition.wait(timeout=remaining_sec)
            if self._frame is None:
                raise RuntimeError("카메라에서 프레임을 받지 못했습니다 (연결/장치 번호 확인)")
            return self._frame.copy(), self._frame_seq

    def stop(self):
        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
