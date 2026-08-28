"""inference 모듈 — 손 랜드마크 추적 (MediaPipe HandLandmarker, Apache-2.0).

랜드마크 번호(MediaPipe 21점): 0=손목뿌리, 1~4=엄지, 5~8=검지, 9~12=중지,
13~16=약지, 17~20=새끼 — 각 손가락은 (MCP, PIP, DIP, TIP) 순서.
"""
import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("inference")

HAND_KPT_COUNT = 21
DUPLICATE_CENTER_SPAN_RATIO = 0.5   # 두 검출의 중심 거리가 손 크기의 이 비율 이내면
                                    # 같은 물리적 손(서로 다른 두 손은 겹칠 수 없다)
VALID_DELEGATES = frozenset(("cpu", "gpu"))


def normalize_delegate(value):
    """설정의 MediaPipe 실행 대상을 검증해 소문자 이름으로 돌려준다."""
    delegate = str(value).strip().lower()
    if delegate not in VALID_DELEGATES:
        choices = ", ".join(sorted(VALID_DELEGATES))
        raise ValueError(
            f"hand_tracker.delegate는 {choices} 중 하나여야 합니다: {value!r}")
    return delegate


def suppress_duplicate_hands(hands):
    """같은 물리적 손이 좌/우 라벨로 중복 검출되는 경우를 억제한다."""
    kept = []
    for hand in sorted(hands, key=lambda entry: -entry.conf):
        center_x = float(hand.landmarks[:, 0].mean())
        center_y = float(hand.landmarks[:, 1].mean())
        span_px = max(
            float(hand.landmarks[:, 0].max() - hand.landmarks[:, 0].min()),
            float(hand.landmarks[:, 1].max() - hand.landmarks[:, 1].min()),
        )
        is_duplicate = False
        for other_center_x, other_center_y, other_span_px, _ in kept:
            dist_px = ((center_x - other_center_x) ** 2
                       + (center_y - other_center_y) ** 2) ** 0.5
            if dist_px < DUPLICATE_CENTER_SPAN_RATIO * max(span_px, other_span_px):
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append((center_x, center_y, span_px, hand))
    return [entry[3] for entry in kept]


@dataclass
class HandDetection:
    """손 1개의 추적 결과.

    user_side: 사용자 기준 "left"/"right" — HandLandmarker의 handedness는 반전
    없는 원본 영상 기준이라, 거울 모드(camera.mirror=true) 프레임에서는 라벨을
    뒤집어 사용자 기준으로 맞춘다.
    """

    user_side: str
    landmarks: np.ndarray        # shape (21, 3) — (x_px, y_px, z_px) 화면 좌표
    world_landmarks: np.ndarray  # shape (21, 3) — 미터 단위 월드 좌표(손 중심 원점)
    conf: float                  # handedness 신뢰도


class HandTracker:
    """MediaPipe HandLandmarker 래퍼. infer(frame) -> list[HandDetection]."""

    def __init__(self, config):
        tracker_cfg = config["hand_tracker"]
        self._model_path = tracker_cfg["model_path"]
        self._is_mirror = config["camera"]["mirror"]
        self._requested_delegate = normalize_delegate(tracker_cfg.get("delegate", "gpu"))
        self._active_delegate = None
        self._fallback_reason = None
        self._gpu_fallback_to_cpu = bool(tracker_cfg.get("gpu_fallback_to_cpu", True))
        self._inference_scale = min(
            1.0, max(0.1, float(tracker_cfg.get("inference_scale", 1.0))))
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        # 한글 경로 대응: model_asset_path 대신 바이트로 직접 읽어 넘긴다
        with open(self._model_path, "rb") as model_file:
            model_bytes = model_file.read()

        try:
            self._landmarker = self._create_landmarker(
                vision, mp_python, model_bytes, tracker_cfg, self._requested_delegate)
            self._active_delegate = self._requested_delegate
        except Exception as exc:  # noqa: BLE001 - GPU/EGL 초기화 오류는 환경마다 다르다
            if self._requested_delegate != "gpu" or not self._gpu_fallback_to_cpu:
                raise RuntimeError(
                    f"MediaPipe {self._requested_delegate.upper()} delegate 초기화에 "
                    "실패했습니다. GPU 드라이버·EGL 설정을 확인하세요.") from exc
            self._fallback_reason = str(exc)
            logger.warning(
                "MediaPipe GPU delegate 초기화 실패 — CPU로 자동 전환합니다: %s", exc)
            self._landmarker = self._create_landmarker(
                vision, mp_python, model_bytes, tracker_cfg, "cpu")
            self._active_delegate = "cpu"

        self._start_sec = time.monotonic()
        self._last_timestamp_ms = -1
        logger.info(
            "손 모델 로딩 완료: MediaPipe HandLandmarker "
            "(delegate=%s, requested=%s, max_num_hands=%d, inference_scale=%.2f, %s)",
            self._active_delegate, self._requested_delegate,
            tracker_cfg["max_num_hands"], self._inference_scale, self._model_path,
        )

    @staticmethod
    def _create_landmarker(vision, mp_python, model_bytes, tracker_cfg, delegate_name):
        """지정한 CPU/GPU delegate로 HandLandmarker 한 개를 생성한다."""
        delegate = getattr(mp_python.BaseOptions.Delegate, delegate_name.upper())
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_buffer=model_bytes,
                delegate=delegate,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=tracker_cfg["max_num_hands"],
            min_hand_detection_confidence=tracker_cfg["min_detection_conf"],
            min_hand_presence_confidence=tracker_cfg["min_presence_conf"],
            min_tracking_confidence=tracker_cfg["min_tracking_conf"],
        )
        return vision.HandLandmarker.create_from_options(options)

    def inference_status(self):
        """HTTP 진단용 실행 delegate 상태. 실제 생성에 성공한 대상을 보고한다."""
        return {
            "requested_delegate": self._requested_delegate,
            "active_delegate": self._active_delegate,
            "gpu_accelerated": self._active_delegate == "gpu",
            "fallback_reason": self._fallback_reason,
        }

    def infer(self, frame):
        """프레임(BGR)에서 보이는 손을 추적한다 -> list[HandDetection].

        frame은 이미 거울 반전이 적용된 상태로 넘어와야 한다(호출부 책임).
        """
        h_px, w_px = frame.shape[:2]
        inference_frame = frame
        if self._inference_scale < 1.0:
            inference_frame = cv2.resize(
                frame, None, fx=self._inference_scale, fy=self._inference_scale,
                interpolation=cv2.INTER_AREA)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=inference_frame[:, :, ::-1].copy(),
        )
        timestamp_ms = int((time.monotonic() - self._start_sec) * 1000.0)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands = []
        for hand_landmarks, world_landmarks, handedness in zip(
                result.hand_landmarks, result.hand_world_landmarks, result.handedness):
            category = handedness[0]
            side = category.category_name.lower()
            if self._is_mirror:
                side = "right" if side == "left" else "left"
            landmarks = np.array(
                [(lm.x * w_px, lm.y * h_px, lm.z * w_px) for lm in hand_landmarks],
                dtype=np.float32,
            )
            world = np.array(
                [(lm.x, lm.y, lm.z) for lm in world_landmarks], dtype=np.float32,
            )
            hands.append(
                HandDetection(user_side=side, landmarks=landmarks,
                              world_landmarks=world, conf=float(category.score))
            )
        return suppress_duplicate_hands(hands)
