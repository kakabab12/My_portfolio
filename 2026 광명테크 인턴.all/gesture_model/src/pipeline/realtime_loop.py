"""pipeline 모듈 — 캡처·추론·판정을 연결해 실시간 루프를 조립한다.

프레임 흐름:
  카메라(스레드) → 거울 반전 → 손 제스처 검출(MediaPipe Hand Landmarker + 기하 규칙)
  + 사람 포즈(rtmlib RTMPose) → 사용자 잠금(person_lock) → 손 귀속(HandObservation)
  → 동작 판정(gesture_filter) → 이벤트 전송 + on_event 콜백

gesture_kiosk의 FastAPI 데모서버·PipelineState 스트리밍(웹 UI에 프레임을 흘려보내는
부분)은 이 프로젝트 범위 밖이라 제외했다. 대신 cv2.imshow는 항상 메인 스레드에서
호출해야 하므로, 검출→잠금→판정→시각화는 PipelineRunner.step()에 담아 두고
scripts/run_demo.py가 자신의 메인 루프에서 매 프레임 직접 호출한다 — 카메라 캡처만
CameraStream이 자체 스레드로 돈다.
"""
from src.capture.camera_stream import CameraStream
from src.inference.detector import create_gesture_detector
from src.inference.pose_estimator import PoseEstimator
from src.inference.preprocessor import Preprocessor
from src.pipeline.event_sender import create_event_sender
from src.postprocess.gesture_filter import GestureFilter
from src.postprocess.person_lock import PersonLock
from src.utils.logger import get_logger
from src.utils.metrics import FpsMeter
from src.utils.visualize import draw_bbox, draw_person_lock, draw_status

logger = get_logger("pipeline")


class GesturePipeline:
    """프레임 1장을 검출→잠금→판정까지 처리해 (annotated_frame, gesture_event)를 돌려준다."""

    def __init__(self, config):
        self.preprocessor = Preprocessor(config)
        self.detector = create_gesture_detector(config)
        self.pose_estimator = PoseEstimator(config) if config["person_lock"]["enabled"] else None
        self.gesture_filter = GestureFilter(config)
        self.class_map = config["model"]["class_map"]
        self.infer_fps_meter = FpsMeter()
        self._config = config
        self._person_lock = None  # 첫 프레임의 폭을 알아야 만들 수 있다

    def _ensure_person_lock(self, frame_width_px):
        if self._person_lock is None:
            self._person_lock = PersonLock(self._config, frame_width_px)
        return self._person_lock

    def process(self, frame):
        """frame(BGR, 원본) -> (annotated_frame, gesture_event | None)."""
        frame = self.preprocessor.preprocess_frame(frame)
        person_lock = self._ensure_person_lock(frame.shape[1])

        detections = self.detector.infer(frame)
        if self.pose_estimator is not None:
            persons = self.pose_estimator.infer(frame)
            person_lock.update(frame, persons)
        observations = person_lock.attach_detections(detections, self.class_map)

        gesture_event = self.gesture_filter.filter_observations(observations)
        self.infer_fps_meter.update()

        annotated = draw_bbox(frame.copy(), detections)
        annotated = draw_person_lock(annotated, person_lock)
        annotated = draw_status(annotated, self.infer_fps_meter.avg_fps, gesture_event)
        return annotated, gesture_event


class PipelineRunner:
    """run_pipeline()이 돌려주는 핸들 — step()을 메인 루프에서 반복 호출한다."""

    def __init__(self, camera, pipeline, event_sender, on_event):
        self.camera = camera
        self._pipeline = pipeline
        self._event_sender = event_sender
        self._on_event = on_event

    def step(self):
        """프레임 1장을 처리한다 -> (annotated_frame, gesture_event | None).

        이벤트가 확정되면 event_sender.send() 다음에 on_event 콜백을 호출한다.
        """
        frame = self.camera.capture_frame()
        annotated, event = self._pipeline.process(frame)
        if event is not None:
            self._event_sender.send(event)
            if self._on_event is not None:
                self._on_event(event)
        return annotated, event

    def stop(self):
        self.camera.stop()


def run_pipeline(config, on_event=None):
    """카메라·검출·판정·이벤트 전송을 조립해 PipelineRunner를 돌려준다.

    on_event(gesture_event)는 제스처가 확정될 때마다 호출된다 — 팀원 키오스크
    프레임워크와 연동하는 지점 (scripts/run_demo.py의 on_gesture_detected() 참고).
    """
    camera = CameraStream(config).start()
    pipeline = GesturePipeline(config)
    event_sender = create_event_sender(config)
    logger.info("실시간 파이프라인 조립 완료")
    return PipelineRunner(camera, pipeline, event_sender, on_event)
