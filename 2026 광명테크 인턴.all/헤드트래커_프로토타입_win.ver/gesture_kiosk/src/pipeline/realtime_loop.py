"""pipeline 모듈 — 캡처·추론·판정·안내를 연결해 실시간 루프를 구동한다 (기획서 2.2, 3.2).

프레임 흐름 (2026-07-18 헤드트래커 전환 — 얼굴 랜드마크 단일 엔진):
  카메라(스레드) → 거울 반전 → 얼굴 랜드마크(FaceLandmarker) → 사용자 잠금(person_lock)
  → 동작 판정(head_tracker: 코끝/시선 커서(모드 전환 가능) + 입벌리기/응시 클릭 + 눈감기 뒤로가기)
  → 이벤트 전송 + 음성 안내

2026-07-16: 주민등록증 OCR 기능 제거 — 제스처 집중(사용자 결정). 개인정보
(주민등록번호) 처리 이슈가 함께 소멸했다. 백업: _before_ocr_removal/.

PipelineState가 예시 UI 서버와 공유되는 유일한 상태 저장소다.
"""
import sys
import threading
import time

from src.announce.announcer import Announcer
from src.capture.camera_stream import CameraStream
from src.inference.face_estimator import FaceEstimator
from src.inference.preprocessor import Preprocessor
from src.pipeline.event_sender import create_event_sender
from src.postprocess.head_tracker import HeadTracker
from src.postprocess.person_lock import PersonLock
from src.utils.logger import get_logger
from src.utils.metrics import FpsMeter
from src.utils.visualize import draw_debug_panel, draw_person_lock, draw_status

logger = get_logger("pipeline")

EVENT_LOG_MAX_COUNT = 200
EVENT_OVERLAY_HOLD_SEC = 1.5


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
        self.cursor_x_ratio = None     # 헤드트래커 커서 위치 — 추적 끊기면 None(유령 커서 방지)
        self.cursor_y_ratio = None
        self.cursor_mode = None        # head_tracker.cursor_mode 미러 — /data status에 노출(2026-07-22)
        self.debug = {}                # 판정 계기판(head_tracker.debug) — 실기 튜닝용
        self.announcer = None          # demo_server의 POST /announce가 사용한다
        self.head_tracker = None       # demo_server의 POST /set_cursor_mode가 사용한다(announcer와 동일 패턴)

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


def _raise_process_priority():
    """프로세스 우선순위를 한 단계 올린다 — 키오스크 전용 PC 가정.

    같은 PC의 브라우저(UI 표시)·TTS 합성이 CPU를 뺏으면 추론 FPS가 순간 10대까지
    떨어진다(2026-07-20 실측) — 우선순위를 올려 추론 루프가 경합에서 밀리지 않게 한다.
    HIGH가 아닌 ABOVE_NORMAL인 이유: UI 브라우저를 굶기면 화면이 끊겨 본말전도.
    실패해도 무해하다(기본 우선순위로 계속).
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import psutil

        psutil.Process().nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        logger.info("프로세스 우선순위 상향 (ABOVE_NORMAL)")
    except Exception:
        logger.info("프로세스 우선순위 조정 실패 — 기본 우선순위로 계속")


def run_pipeline(config):
    """파이프라인 전체를 조립해 시작하고 PipelineState를 돌려준다 (기획서 4.6 계약)."""
    _raise_process_priority()
    state = PipelineState()
    camera = CameraStream(config).start()
    preprocessor = Preprocessor(config)
    face_estimator = FaceEstimator(config)   # 유일한 추론 모델 — 모든 판정의 입력

    # 처리 해상도(proc_*) 기준으로 잰다 — 캡처 해상도 기준이면 person_lock의
    # 추적 반경(follow_radius_ratio × 프레임폭)이 실제 프레임보다 크게 잡힌다
    first_frame = preprocessor.preprocess_frame(camera.capture_frame())
    frame_height_px, frame_width_px = first_frame.shape[:2]
    person_lock = PersonLock(config, frame_width_px, frame_height_px)
    head_tracker = HeadTracker(config)
    state.head_tracker = head_tracker
    event_sender = create_event_sender(config)
    announcer = Announcer(config)
    state.announcer = announcer

    min_loop_interval_sec = 1.0 / config["model"]["max_infer_fps"]

    state.is_running = True

    def _inference_loop():
        infer_fps_meter = FpsMeter()
        while state.is_running:
            loop_start_sec = time.monotonic()

            frame = camera.capture_frame()
            input_tensor = preprocessor.preprocess_frame(frame)

            faces = face_estimator.infer(input_tensor)
            person_lock.update(input_tensor, faces)
            state.is_user_locked = (
                person_lock.enabled and person_lock.locked_face is not None
            )

            result = head_tracker.update(person_lock.locked_face, person_lock.lock_generation)
            state.cursor_x_ratio = result.cursor_x_ratio
            state.cursor_y_ratio = result.cursor_y_ratio
            state.cursor_mode = head_tracker.cursor_mode
            state.debug = head_tracker.debug

            for gesture_event in result.events:
                event_sender.send(gesture_event)
                state.append_event(gesture_event)
                announcer.on_event(gesture_event)

            infer_fps_meter.update()
            state.capture_fps = camera.fps_meter.avg_fps
            state.infer_fps = infer_fps_meter.avg_fps

            annotated = draw_person_lock(
                input_tensor, person_lock, result.cursor_x_ratio, result.cursor_y_ratio
            )
            annotated = draw_debug_panel(annotated, state.debug)
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

        face_estimator.close()

    threading.Thread(target=_inference_loop, daemon=True).start()
    logger.info("실시간 파이프라인 시작 (frame_width_px=%d)", frame_width_px)
    return state
