"""inference 모듈 — MediaPipe 손 랜드마크로 "손등이 카메라를 향하는지"만 판정한다.

동작 하나만 쓰기로 단순화(2026-07 결정): 손가락을 굽혔는지(주먹 모양)를 정밀하게
보는 대신, **손목→검지MCP, 손목→새끼MCP 두 벡터의 외적(cross product) 부호**로
손의 회전 방향(손등이 보이는지 손바닥이 보이는지)만 본다. 주먹을 쥐면 손가락
마디가 접혀 랜드마크가 흔들리기 쉬운데, "손을 뒤집었는지" 방향 자체는 훨씬
안정적으로 바뀌어서 더 견고하다.

출력 계약은 detector.Detection 과 동일: infer(frame) -> list[Detection].
class_name은 항상 "손등팔등"(손등이 보임) 하나뿐이다.

랜드마크 번호(MediaPipe Hands 규격): 0=손목, 5=검지 MCP, 17=새끼 MCP.
"""
import time

from src.inference.detector import Detection
from src.utils.logger import get_logger

logger = get_logger("inference")

LM_WRIST = 0
LM_INDEX_MCP = 5
LM_PINKY_MCP = 17

CLASS_IDS = {"손등팔등": 0}

OPPOSITE_SIDE = {"left": "right", "right": "left"}


def user_side_from_label(label, is_mirror, flip_handedness):
    """MediaPipe handedness 라벨 -> 사용자 기준 좌/우 (없거나 모르면 None).

    공식 문서는 "거울(셀피) 입력 가정" 라벨이라고 하지만, Tasks API에서 라벨이
    반대로 나오는 사례가 보고되어 있다(google-ai-edge/mediapipe#4724). 그래서
    문서 기준 매핑 위에 flip_handedness(config)를 두었다 — 데모 화면 손 박스의
    L/R 표시가 실제와 뒤집혀 보이면 이 값을 반전하면 된다.
    """
    if label not in OPPOSITE_SIDE:
        return None
    side = label if is_mirror else OPPOSITE_SIDE[label]  # 문서 기준 매핑
    return OPPOSITE_SIDE[side] if flip_handedness else side


def is_back_of_hand(landmarks, handedness_label, flip_orientation=False):
    """21개 랜드마크 [(x, y), ...] + MediaPipe 원본 handedness 라벨('left'|'right')
    -> 손등이 카메라를 향하면 True.

    순수 함수 — mediapipe 없이 단위 테스트 가능.

    원리: 손목->검지MCP 벡터와 손목->새끼MCP 벡터의 외적 부호가 손바닥/손등에 따라
    반대가 된다(손을 180도 뒤집으면 두 벡터의 좌우 배치가 뒤바뀌므로). 오른손·왼손은
    서로 거울상이라 부호 판정 기준도 반대다.

    부호가 실제와 반대로 나오면(오른손 손등을 보였는데 False가 나오는 등)
    config의 model.mediapipe.flip_orientation을 true로 바꾸면 된다 — 실측 전
    이론적으로 유도한 부호라 실제 카메라로 검증 필요.
    """
    wrist = landmarks[LM_WRIST]
    index_mcp = landmarks[LM_INDEX_MCP]
    pinky_mcp = landmarks[LM_PINKY_MCP]
    v1x, v1y = index_mcp[0] - wrist[0], index_mcp[1] - wrist[1]
    v2x, v2y = pinky_mcp[0] - wrist[0], pinky_mcp[1] - wrist[1]
    cross = v1x * v2y - v1y * v2x

    if handedness_label == "right":
        is_palm_facing = cross > 0
    elif handedness_label == "left":
        is_palm_facing = cross < 0
    else:
        return False

    if flip_orientation:
        is_palm_facing = not is_palm_facing
    return not is_palm_facing


def _bbox_from_landmarks(landmarks, frame_shape, pad_ratio):
    """랜드마크를 감싸는 픽셀 박스 — person_lock의 손목 귀속 거리 계산에 쓰인다."""
    h_px, w_px = frame_shape[:2]
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    pad = max(max(xs) - min(xs), max(ys) - min(ys)) * pad_ratio
    return (
        max(0.0, min(xs) - pad), max(0.0, min(ys) - pad),
        min(w_px - 1.0, max(xs) + pad), min(h_px - 1.0, max(ys) + pad),
    )


class MediaPipeGestureDetector:
    """MediaPipe 손 랜드마크 기반 검출기. infer(frame) -> list[Detection] (fist만)."""

    def __init__(self, config):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        mp_cfg = config["model"]["mediapipe"]
        self._bbox_pad_ratio = mp_cfg["bbox_pad_ratio"]
        self._flip_orientation = mp_cfg.get("flip_orientation", False)
        self._conf_threshold = config["detect"]["conf_threshold"]
        self._is_mirror = config["camera"]["mirror"]
        self._flip_handedness = mp_cfg["flip_handedness"]  # user_side_from_label 참고
        self._mp = mp
        self._clock = time.monotonic
        self._last_timestamp_ms = -1

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=mp_cfg["hand_landmarker_path"]),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=mp_cfg["num_hands"],
            min_hand_detection_confidence=self._conf_threshold,
            min_tracking_confidence=mp_cfg["min_tracking_confidence"],
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        logger.info("제스처 모델 로딩 완료: MediaPipe HandLandmarker (%s)",
                    mp_cfg["hand_landmarker_path"])

    def infer(self, frame):
        """프레임(BGR)에서 손을 찾아 손등이 보이는지만 판정한다."""
        h_px, w_px = frame.shape[:2]
        rgb = frame[:, :, ::-1]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                  data=rgb.copy(order="C"))
        # VIDEO 모드는 단조 증가 타임스탬프(ms)를 요구한다 — 추론 루프 단일 스레드 전제
        timestamp_ms = max(int(self._clock() * 1000.0), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        detections = []
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            conf = float(handedness[0].score) if handedness else 1.0
            if conf < self._conf_threshold:
                continue
            raw_label = handedness[0].category_name.lower() if handedness else None
            points_px = [(lm.x * w_px, lm.y * h_px) for lm in hand_landmarks]
            if not is_back_of_hand(points_px, raw_label, self._flip_orientation):
                continue
            hand_side = user_side_from_label(raw_label, self._is_mirror, self._flip_handedness)
            detections.append(
                Detection(
                    class_id=CLASS_IDS["손등팔등"],
                    class_name="손등팔등",
                    conf=conf,
                    bbox=_bbox_from_landmarks(points_px, frame.shape, self._bbox_pad_ratio),
                    hand_side=hand_side,
                )
            )
        return detections
