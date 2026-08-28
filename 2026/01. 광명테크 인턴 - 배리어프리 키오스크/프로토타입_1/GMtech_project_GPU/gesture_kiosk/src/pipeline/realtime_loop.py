"""pipeline 모듈 — 캡처·추론·판정·안내를 연결해 실시간 루프를 구동한다 (기획서 2.2, 3.2).

프레임 흐름 (윈도우 + NVIDIA GPU 기준):
  카메라(스레드) → 거울 반전 → 제스처 검출(MediaPipe, CPU) → 사용자 잠금 귀속(person_lock)
  → 손 귀속(HandObservation) → 동작 판정(gesture_filter) → 이벤트 전송 + 음성 안내

사람 포즈(RTMPose, GPU)와 주민등록증 OCR(EasyOCR, GPU)은 각각 별도 워커 스레드에서
돈다. 포즈는 CPU 실측 63~93ms/회(2026-07-15)로 메인 추론 루프(60 FPS 목표) 안에
있으면 그때마다 루프가 막힌다 — GPU로 옮겨도 이 스레드 분리 자체는 유지한다.
포즈 워커는 person_lock을 자체 주기로 갱신만 하고, 메인 루프는 항상 마지막
갱신 상태를 읽는다(person_lock.attach_detections). OCR은 UI가 요청할 때만
(state.start_ocr_mode) 원본(반전 없는) 프레임으로 동작한다.

PipelineState가 예시 UI 서버와 공유되는 유일한 상태 저장소다.
"""
import threading
import time

from src.announce.announcer import Announcer
from src.capture.camera_stream import CameraStream
from src.inference.detector import create_gesture_detector
from src.inference.pose_estimator import PoseEstimator
from src.inference.preprocessor import Preprocessor
from src.pipeline.event_sender import create_event_sender
from src.postprocess.gesture_filter import GestureEvent, GestureFilter
from src.postprocess.person_lock import PersonLock
from src.utils.logger import get_logger
from src.utils.metrics import FpsMeter
from src.utils.visualize import (
    draw_bbox,
    draw_ocr_mode,
    draw_person_lock,
    draw_status,
    draw_two_palm_hold,
)

logger = get_logger("pipeline")

EVENT_LOG_MAX_COUNT = 200
EVENT_OVERLAY_HOLD_SEC = 1.5
OCR_IDLE_POLL_SEC = 0.2
ASSUMED_CAMERA_FPS = 30.0  # OCR·포즈 워커의 "N프레임마다" 설정을 폴링 주기(초)로 환산하는 기준
                            #   2026-07-15 실측: 이 웹캠은 요청 해상도·FPS와 무관하게 실제로
                            #   정확히 30 FPS로 협상·유지된다(MSMF·DSHOW 공통, 워밍업 후 29.9 FPS
                            #   측정) — 이 상수를 올릴 이유가 없다는 뜻


class PipelineState:
    """추론 결과·성능 수치를 스레드 안전하게 공유한다."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_frame = None
        self.capture_fps = 0.0
        self.infer_fps = 0.0
        self.last_event = None
        self.event_log = []
        self.is_running = False
        self.is_user_locked = False
        self.two_palm_hold_ratio = 0.0
        self.announcer = None          # demo_server의 POST /announce가 사용한다
        self._ocr_deadline_sec = None  # None이면 OCR 모드 꺼짐

    def update_frame(self, frame):
        with self._lock:
            self._latest_frame = frame

    def get_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def append_event(self, gesture_event):
        with self._lock:
            self.last_event = gesture_event
            self.event_log.append(gesture_event)
            if len(self.event_log) > EVENT_LOG_MAX_COUNT:
                self.event_log.pop(0)

    # ----- OCR 모드 제어 (UI -> 엔진) -----

    def start_ocr_mode(self, timeout_sec):
        with self._lock:
            self._ocr_deadline_sec = time.monotonic() + timeout_sec
        logger.info("OCR 모드 시작 (timeout=%.0f초)", timeout_sec)

    def stop_ocr_mode(self):
        with self._lock:
            self._ocr_deadline_sec = None

    def is_ocr_mode_active(self):
        with self._lock:
            if self._ocr_deadline_sec is None:
                return False
            if time.monotonic() > self._ocr_deadline_sec:
                self._ocr_deadline_sec = None
                return False
            return True


def _start_ocr_worker(state, config, camera, event_sender, announcer):
    """주민등록증 OCR 워커 — OCR 모드일 때만 원본 프레임을 주기적으로 판독한다."""
    from src.ocr.idcard_reader import IdCardReader  # easyocr 의존 — 켠 경우에만 임포트

    reader = IdCardReader(config)
    poll_interval_sec = config["ocr"]["interval_frames"] / ASSUMED_CAMERA_FPS

    def _ocr_loop():
        while state.is_running:
            if not state.is_ocr_mode_active():
                time.sleep(OCR_IDLE_POLL_SEC)
                continue
            frame = camera.capture_frame()  # 원본(반전 없음) — 글자를 읽어야 한다
            try:
                fields = reader.read(frame)
            except Exception:
                logger.exception("OCR 판독 오류 — 모드를 종료합니다")
                state.stop_ocr_mode()
                continue
            if fields is None:
                time.sleep(poll_interval_sec)
                continue
            event = GestureEvent(
                class_name="fill_id_fields",
                conf=fields["conf"],
                ts_sec=time.monotonic(),
                data={"name": fields["name"], "rrn": fields["rrn"]},
            )
            event_sender.send(event)
            state.append_event(event)
            announcer.on_event(event)
            state.stop_ocr_mode()  # 1회 인식이 목적 — 성공 즉시 종료

    threading.Thread(target=_ocr_loop, daemon=True).start()
    logger.info("OCR 워커 시작 (poll=%.2f초)", poll_interval_sec)


def _start_pose_worker(state, config, camera, preprocessor, person_lock, pose_estimator):
    """포즈(RTMPose) 워커 — person_lock을 자체 주기로 갱신한다. 메인 루프와 완전히 분리.

    2026-07-15 GPU화: RTMPose 1회가 CPU 실측 63~93ms로, 메인 추론 루프(60 FPS 목표)
    안에 있으면 그때마다 루프가 막혀 있었다. GPU로 옮겨 비용 자체는 줄어도 "메인
    루프 타이밍에 절대 끼어들지 않는다"는 이 구조는 그대로 유지한다 — person_lock은
    사람이 몇 백 ms 안에 크게 움직이지 않는다는 전제(release_sec=2.0)라 매 프레임
    갱신할 필요가 없다. 폴링 주기는 새 config 없이 기존 pose_infer_every_n_frames를
    재해석(ASSUMED_CAMERA_FPS 기준 초 단위)해서 쓴다 — OCR 워커의 interval_frames와
    동일 패턴.
    """
    poll_interval_sec = config["model"]["pose_infer_every_n_frames"] / ASSUMED_CAMERA_FPS

    def _pose_loop():
        while state.is_running:
            loop_start_sec = time.monotonic()
            try:
                frame = camera.capture_frame()
                # OCR과 달리 반전(mirror) 프레임이 필요하다 — person_lock·제스처 검출이
                # 전부 preprocessor를 거친 좌표계를 기준으로 동작한다
                input_tensor = preprocessor.preprocess_frame(frame)
                persons = pose_estimator.infer(input_tensor)
                person_lock.update(input_tensor, persons)
            except Exception:
                logger.exception("포즈 추론 오류 — 다음 주기에 재시도합니다")
            elapsed_sec = time.monotonic() - loop_start_sec
            if elapsed_sec < poll_interval_sec:
                time.sleep(poll_interval_sec - elapsed_sec)

    threading.Thread(target=_pose_loop, daemon=True).start()
    logger.info("포즈 워커 시작 (poll=%.2f초)", poll_interval_sec)


def run_pipeline(config):
    """파이프라인 전체를 조립해 시작하고 PipelineState를 돌려준다 (기획서 4.6 계약)."""
    state = PipelineState()
    camera = CameraStream(config).start()
    preprocessor = Preprocessor(config)
    detector = create_gesture_detector(config)
    pose_estimator = PoseEstimator(config) if config["person_lock"]["enabled"] else None

    first_frame = camera.capture_frame()
    frame_width_px = first_frame.shape[1]
    person_lock = PersonLock(config, frame_width_px)
    gesture_filter = GestureFilter(config)
    event_sender = create_event_sender(config)
    announcer = Announcer(config)
    state.announcer = announcer

    class_map = config["model"]["class_map"]
    min_loop_interval_sec = 1.0 / config["model"]["max_infer_fps"]
    ocr_guide_region = config["ocr"]["guide_region_ratio"]

    state.is_running = True
    if config["ocr"]["enabled"]:
        _start_ocr_worker(state, config, camera, event_sender, announcer)
    if pose_estimator is not None:
        _start_pose_worker(state, config, camera, preprocessor, person_lock, pose_estimator)

    def _inference_loop():
        infer_fps_meter = FpsMeter()
        while state.is_running:
            loop_start_sec = time.monotonic()

            frame = camera.capture_frame()
            input_tensor = preprocessor.preprocess_frame(frame)
            detections = detector.infer(input_tensor)

            # 포즈(person_lock 갱신)는 _start_pose_worker가 별도 스레드에서 돈다 — 여기서는
            # 그 워커가 마지막으로 남긴 잠금 상태를 읽기만 한다(항상 최신일 필요 없음)
            observations = person_lock.attach_detections(detections, class_map)
            state.is_user_locked = person_lock.locked_person is not None

            gesture_event = gesture_filter.filter_observations(observations)
            state.two_palm_hold_ratio = gesture_filter.two_palm_hold_ratio

            if gesture_event is not None:
                event_sender.send(gesture_event)
                state.append_event(gesture_event)
                announcer.on_event(gesture_event)

            infer_fps_meter.update()
            state.capture_fps = camera.fps_meter.avg_fps
            state.infer_fps = infer_fps_meter.avg_fps

            annotated = draw_bbox(input_tensor, detections)
            annotated = draw_person_lock(annotated, person_lock)
            annotated = draw_two_palm_hold(annotated, state.two_palm_hold_ratio)
            if state.is_ocr_mode_active():
                annotated = draw_ocr_mode(annotated, ocr_guide_region)
            overlay_event = state.last_event
            if overlay_event is not None and (
                time.monotonic() - overlay_event.ts_sec > EVENT_OVERLAY_HOLD_SEC
            ):
                overlay_event = None
            annotated = draw_status(annotated, state.infer_fps, overlay_event)
            state.update_frame(annotated)

            # FPS 상한 — 개발 PC에서 200+ FPS로 도는 낭비를 막는다
            elapsed_sec = time.monotonic() - loop_start_sec
            if elapsed_sec < min_loop_interval_sec:
                time.sleep(min_loop_interval_sec - elapsed_sec)

    threading.Thread(target=_inference_loop, daemon=True).start()
    logger.info("실시간 파이프라인 시작 (frame_width_px=%d)", frame_width_px)
    return state
