"""capture 모듈 — USB 웹캠에서 프레임을 실시간으로 읽어 온다 (기획서 2.2).

캡처는 전용 스레드에서 돌리고(기획서 3.2 멀티스레딩), capture_frame()은
항상 가장 최신 프레임을 돌려준다. 추론이 느려도 오래된 프레임이 쌓이지 않는다.

런타임 자동 복구(2026-07-28): 무인 키오스크에서 USB 탈락·드라이버 멈춤이
조용한 정지가 되면 사고다 — 프레임이 recovery_timeout_sec 동안 안 오면 장치를
닫고 재연결될 때까지 계속 다시 연다. 복구 중에도 파이프라인은 마지막 프레임으로
생존한다 (capture_new_frame의 멈칫 폴백).
"""
import sys
import threading
import time

import cv2

from src.utils.logger import get_logger
from src.utils.metrics import FpsMeter

logger = get_logger("capture")

FIRST_FRAME_TIMEOUT_SEC = 5.0
NEW_FRAME_TIMEOUT_SEC = 1.0   # 새 프레임 대기 한도 — 카메라 멈칫 시 기존 프레임으로 진행(파이프라인 생존)
# 같은 읽기 오류가 계속될 때 다시 기록하기까지의 간격(초) — _capture_loop 참고
ERROR_REPEAT_LOG_SEC = 30.0


def _open_raw(config, device_id):
    """config의 현재 설정 그대로 딱 한 번 열어 본다 — 성공/실패만 돌려주고
    협상은 모른다(init_camera가 그 판단을 한다). 리눅스는 V4L2가 안정적.
    윈도우는 MSMF가 열리는 데 수십 초 걸리는 장치가 있어 기본을 DSHOW로 두고
    config(camera.windows_backend)로 바꿀 수 있게 한다."""
    if sys.platform.startswith("linux"):
        return cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    if sys.platform.startswith("win"):
        backend_name = config["camera"].get("windows_backend", "auto")
        windows_backends = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
        if backend_name in windows_backends:
            return cv2.VideoCapture(device_id, windows_backends[backend_name])
        return cv2.VideoCapture(device_id)
    return cv2.VideoCapture(device_id)


def _finish_open(cap, config, device_id):
    """연 카메라에 나머지 설정(포맷·해상도)을 적용하고 로그를 남긴다."""
    # 무압축(YUY2) 1280x720은 USB 대역폭 한계로 캡처가 ~5 FPS에 묶인다 — MJPG 기본
    fourcc = config["camera"].get("fourcc", "mjpg")
    if fourcc and fourcc != "auto":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc.upper()))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width_px"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height_px"])
    _log_camera_negotiation(cap, config, device_id)
    return cap


def init_camera(config, device_id=None, config_path=None):
    """config 기준으로 카메라 장치를 열어 cv2.VideoCapture를 돌려준다.

    device_id: 지정 시 config 값 대신 이 장치를 연다 — 자동 선별(camera_probe,
    A안 2026-07-28)이 고른 장치·프로브 후보를 열 때 쓴다.

    config_path(2026-08-27 신설, camera_negotiate 모듈 독스트링 참고):
    이 경로를 넘기면, 설정된 조합으로 안 열릴 때 **자동으로 다른 조합을
    찾아본다**(MSMF+MJPG+1280x720 하나에 고정하지 않는다). 안 넘기면(None)
    예전과 완전히 동일하게 한 번만 시도하고 실패하면 바로 예외를 던진다 —
    호출부를 안 바꾼 곳(camera_probe 등)은 동작이 그대로다.

    ★비용 — 설정대로 잘 열리는 절대다수의 경우엔 딱 한 번만 연다(예전과
    동일 비용). 협상은 그게 실패했을 때만 시작되고, 새 조합 시험은 전부
    자식 프로세스에서 이뤄진다(MSMF 크래시로부터 이 프로세스를 지킨다 —
    camera_negotiate.find_working_combo 참고).
    """
    if device_id is None:
        device_id = config["camera"]["device_id"]
    cap = _open_raw(config, device_id)
    if cap.isOpened():
        return _finish_open(cap, config, device_id)

    if not config_path:
        raise RuntimeError(f"카메라(device_id={device_id})를 열 수 없습니다")

    logger.warning("카메라(device_id=%s) 기본 설정으로 열기 실패 — 자동 협상 시작"
                   " (다른 조합을 순서대로 시험합니다, 몇 초 걸릴 수 있습니다)",
                   device_id)
    from src.capture.camera_negotiate import find_working_combo   # 지연 임포트
    winner = find_working_combo(config_path, device_id)
    if winner is None:
        logger.error("카메라(device_id=%s) — 어떤 조합으로도 열리지 않았습니다. "
                     "py scripts/camera_check.py --diagnose %s 로 직접 확인하세요",
                     device_id, device_id)
        raise RuntimeError(f"카메라(device_id={device_id})를 열 수 없습니다"
                          f" (자동 협상도 실패)")

    config["camera"].update(winner)
    logger.warning("카메라(device_id=%s) 자동 협상 성공 — %s로 다시 엽니다",
                   device_id, winner)
    cap = _open_raw(config, device_id)
    if not cap.isOpened():
        # 방금 자식 프로세스에서는 열렸는데 본 프로세스에서 다시 열 때
        # 실패하는 극단적 상황(장치 점유 등) — 실측된 적은 없지만 방어
        raise RuntimeError(f"카메라(device_id={device_id})를 열 수 없습니다"
                          f" (협상된 조합 재오픈 실패)")
    return _finish_open(cap, config, device_id)


def _log_camera_negotiation(cap, config, device_id):
    """어떤 카메라가 어떤 조건으로 열렸는지 기록 — 기기별 실기 로그의 증거용 (2026-07-16).

    OpenCV는 장치 이름을 못 주므로 기종은 config(camera.model — 사람이 기록)를 싣고,
    협상 결과(실제 해상도·FPS·픽셀포맷·백엔드)는 장치에서 읽어 함께 남긴다 —
    요청값과 협상값이 다르면(예: YUY2 5 FPS 함정) 여기서 바로 드러난다.
    """
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_text = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)).strip() or "?"
    try:
        backend_label = cap.getBackendName()
    except cv2.error:
        backend_label = "?"
    logger.info(
        "카메라 협상 결과: device_id=%s · 기종(config)=%s · 백엔드=%s · %dx%d @ %.0f FPS · 포맷=%s",
        device_id,
        config["camera"].get("model", "미기록"),
        backend_label,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        cap.get(cv2.CAP_PROP_FPS),
        fourcc_text,
    )


class CameraStream:
    """카메라 캡처 스레드. capture_frame()으로 최신 프레임(BGR)을 얻는다."""

    def __init__(self, config, device_id=None, cap=None, config_path=None):
        self._config = config
        # 자동 선별(A안)·보조 카메라가 config 밖의 장치를 열 수 있게 오버라이드 허용
        self._device_id = (device_id if device_id is not None
                           else config["camera"]["device_id"])
        # 자동 협상용(2026-08-27, camera_negotiate 모듈 참고) — 안 넘기면
        # init_camera가 예전과 동일하게 협상 없이 한 번만 시도한다
        self._config_path = config_path
        # 런타임 자동 복구(2026-07-28) — 키가 없어도 기본값으로 켠다: 무인 운영에서
        # 조용한 정지보다 나쁜 기본값은 없다 (모듈 독스트링 참고)
        self._recovery_timeout_sec = config["camera"].get("recovery_timeout_sec", 3.0)
        self._recovery_retry_sec = config["camera"].get("recovery_retry_sec", 2.0)
        # 프로브가 이미 연 핸들 재사용(A안 2026-07-28) — MSMF는 release 직후
        # 같은 장치 재오픈 시 프레임을 주지 않는다 (camera_probe.rank_cameras 주석)
        self._preopened_cap = cap
        self._cap = None
        self._frame = None
        self._frame_seq = 0                # 프레임 일련번호 — 새 프레임 동기화(2026-07-20)
        self._frame_lock = threading.Lock()
        self._new_frame_condition = threading.Condition(self._frame_lock)
        self._thread = None
        self.is_running = False
        self.fps_meter = FpsMeter()
        # 복구 시작 알림을 마지막으로 남긴 시각 — _recover_camera 참고
        self._last_recover_log_sec = 0.0

    def start(self):
        self._cap = (self._preopened_cap if self._preopened_cap is not None
                     else init_camera(self._config, self._device_id, self._config_path))
        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("카메라 캡처 스레드 시작 (device_id=%s)", self._device_id)
        return self

    def _capture_loop(self):
        last_frame_sec = time.monotonic()
        last_error_log_sec = 0.0
        while self.is_running:
            try:
                ret, frame = self._cap.read()
                if ret:
                    self._publish_frame(frame)
                    self.fps_meter.update()
                    last_frame_sec = time.monotonic()
                    continue
            except Exception:   # noqa: 방어적 — 아래 설명 참고
                # ★2026-08-25 신설. read()가 False를 돌려주는 게 아니라 **예외를
                # 던지는** 경우가 있다(드라이버 충돌, 핸들 무효화 등). 예전엔
                # 이걸 안 막아서 캡처 스레드가 통째로 죽었는데, 자동 복구는
                # 애초에 이 루프가 돌고 있어야 작동하므로 영영 돌지 않았다.
                # 그러면 capture_new_frame은 마지막 프레임을 계속 돌려주고
                # 예외도 안 난다 — 화면상으로는 "얼굴이 얼어붙은 채 커서만
                # 안 움직이는" 조용한 정지가 된다. 무인 키오스크에서 가장
                # 나쁜 실패 방식이라, 예외도 끊김과 똑같이 복구 경로로 보낸다.
                #
                # 실측(가짜 카메라로 재현): 고치기 전에는 스레드가 죽고 프레임
                # 번호가 20에서 멈춘 채 그대로였다.
                now_sec = time.monotonic()
                if now_sec - last_error_log_sec >= ERROR_REPEAT_LOG_SEC:
                    # 매번 다 남기면 로그가 폭주한다 — 복구가 곧바로 성공했다가
                    # 다시 실패하는 상황이 이어질 수 있다(_recover_camera의
                    # attempt_count 로그와 같은 취지)
                    logger.exception("카메라 읽기 중 오류 - 복구를 시도합니다")
                    last_error_log_sec = now_sec
                self._recover_camera()
                last_frame_sec = time.monotonic()
                continue
            # 읽기 실패 — 일시 멈칫(자동 노출 조정 등)과 끊김(USB 탈락)을 시간으로 구분
            if time.monotonic() - last_frame_sec < self._recovery_timeout_sec:
                time.sleep(0.1)   # 짧은 멈칫 — 종전대로 그냥 기다린다
                continue
            self._recover_camera()
            last_frame_sec = time.monotonic()   # 복구 후 끊김 판정 시계 재시작

    def _recover_camera(self):
        """카메라 자동 복구 — 핸들을 버리고 같은 장치를 재연결될 때까지 다시 연다.

        같은 device_id만 다시 여는 이유: 키오스크에서 카메라를 교체해 꽂아도
        보통 같은 번호로 잡힌다. 다른 번호로 옮겨 잡히는 경우는 재시작(프로브
        자동 선별)이 담당한다. 복구 대기 중에도 stop()이 즉시 먹히도록 재시도
        간격을 잘게 쪼개 잔다.
        """
        # ★복구 시작 알림도 간격을 둔다 (2026-08-25). 고장이 계속되면 이
        # 함수가 재시도 간격마다 다시 불리는데, 그때마다 한 줄씩 남기면
        # 하루에 수만 줄이 된다. 첫 줄은 반드시 남기고 그 뒤로는 드문드문만
        # — "복구 중"이라는 사실은 한 번만 알면 충분하고, 재시도 횟수는
        # 아래 실패 로그가 이미 알려준다.
        #
        # ★"복구에 성공하면 시계를 되돌린다"를 넣었다가 뺐다 — 장치를 다시
        # 여는 데는 성공했는데 곧바로 또 읽기가 실패하는 깜빡임(flapping)이
        # 바로 이 폭주의 정체라, 성공할 때마다 되돌리면 제한이 통째로
        # 무력해진다(실측: 초당 2줄 그대로였다). 되돌리지 않아도 30초 넘게
        # 멀쩡했다가 새로 난 고장은 어차피 바로 기록된다
        now_sec = time.monotonic()
        if now_sec - self._last_recover_log_sec >= ERROR_REPEAT_LOG_SEC:
            logger.warning("카메라 응답 없음 %.0f초 — 자동 복구 시작 (device_id=%s)",
                           self._recovery_timeout_sec, self._device_id)
            self._last_recover_log_sec = now_sec
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
                if attempt_count % 10 == 1:   # 로그 폭주 방지 — 첫 실패 후 드문드문만
                    logger.warning("카메라 재연결 실패 %d회 — %.0f초 간격 재시도 중 (device_id=%s)",
                                   attempt_count, self._recovery_retry_sec, self._device_id)
                continue
            logger.info("카메라 자동 복구 성공 (device_id=%s, 재시도 %d회)",
                        self._device_id, attempt_count)
            return

    def _publish_frame(self, frame):
        """새 프레임 게시 — 일련번호를 올리고 대기 중인 소비자를 깨운다 (테스트 접점)."""
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

    def capture_new_frame(self, last_seq):
        """last_seq **이후의 새 프레임**을 기다려 (frame, seq)로 돌려준다 (2026-07-20).

        카메라(30 FPS)보다 추론 루프가 빠르면 같은 프레임을 두 번 추론하는
        낭비가 생긴다 — 같은 입력은 같은 판정이라 정보 이득이 0이므로, 새
        프레임이 올 때까지 재운다(추론 속도가 카메라 속도에 자동 동기화).
        NEW_FRAME_TIMEOUT_SEC 안에 새 프레임이 없으면(카메라 멈칫) 기존
        프레임을 그대로 돌려줘 파이프라인이 죽지 않게 한다 — 이때 seq가
        그대로라 호출자는 다음 호출에서 다시 새 프레임을 기다린다.
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
