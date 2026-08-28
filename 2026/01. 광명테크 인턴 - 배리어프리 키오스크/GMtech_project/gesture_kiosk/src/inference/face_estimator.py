"""inference 모듈 — 얼굴 랜드마크(MediaPipe FaceLandmarker, Apache-2.0)를 추론한다.

2026-07-30 헤드트래커 병합 도입: 손 제스처(hand_tracker)는 그대로 두고, **머리 흔들기로
전환되는 헤드트래커 모드**의 입력으로 얼굴 랜드마크를 추가한다 — 코끝(커서 포인터)·
입 벌림/눈 감김/입 오므림(블렌드셰이프)이 전부 이 모델 하나에서 나온다.

★ face_anchor(FaceDetector, 얼굴 박스만)와는 다른 모델이다 — 그쪽은 손 인식용
"가장 가까운 사람" 앵커(bbox만 필요, 저FPS)이고, 이 모듈은 헤드트래커 모드의
커서·블렌드셰이프 판정에 랜드마크 478점이 필요해 별도로 둔다. 두 모델이 함께
도는 대가로 프레임당 비용이 늘어난다(hand 모드 중에도 머리 흔들기 감지를 위해
이 모듈은 상시 실행 — realtime_loop.py 참고).

구 프로토타입(헤드트래커_프로토타입_win.ver)에서 이식하며 **아이트래커(시선) 경로는
제외**했다(사용자 결정 2026-07-30) — 홍채·눈꺼풀 랜드마크 상수와 시선 비율 계산이
전부 그쪽 전용이라 함께 제거. 커서는 코끝 기준 하나만 남는다.

라이선스: MediaPipe(Apache-2.0) — hand_tracker와 동일 엔진·동일 기준이라 의존성이
새로 늘지 않는다 (2026-07-11 라이선스 B안 유지: 상업 허용·카피레프트 없음).

모델 파일: models/weights/face_landmarker.task — download_weights.py가 받는다.
"""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("inference")

# FaceLandmarker 랜드마크 인덱스 (478점 — 468 얼굴 + 10 홍채. 홍채는 아이트래커 전용이라 미사용)
LMK_NOSE_TIP = 1
LMK_LEFT_EYE_OUTER = 33     # 안구간 거리 = 카메라 거리 정규화 자(尺)
LMK_RIGHT_EYE_OUTER = 263

BBOX_PAD_RATIO = 0.10  # 랜드마크 묶음 -> 얼굴 박스로 넓히는 패딩


@dataclass
class FaceLandmarks:
    """얼굴 1개의 추정 결과 (기획서 4.6 공통 데이터 구조 스타일)."""

    bbox: tuple                       # (x1, y1, x2, y2) 픽셀 좌표
    conf: float
    landmarks_px: np.ndarray          # shape (478, 2) — (x_px, y_px)
    blendshapes: dict = field(default_factory=dict)   # category_name -> score(0~1)

    def landmark_px(self, index):
        """랜드마크 픽셀 좌표 (x, y). 인덱스는 항상 존재 — 신뢰도 게이트가 없다."""
        x, y = self.landmarks_px[index]
        return float(x), float(y)

    def blendshape(self, category_name, default=0.0):
        return self.blendshapes.get(category_name, default)

    @property
    def area_px(self):
        """얼굴 박스 면적 — 가까운 사용자 선별 기준 (select_user_face)."""
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_user_face(faces):
    """여러 얼굴 중 사용자 1명 -> FaceLandmarks | None.

    가장 큰 얼굴 = 카메라에 가장 가까운 사람. hand_select의 face_anchor가
    "가장 큰 얼굴 = 사용자"로 앵커를 고르는 것과 같은 기준이라 두 모드가 같은
    사람을 가리킨다 — 별도 얼굴 잠금을 새로 만들지 않는다.
    """
    if not faces:
        return None
    return max(faces, key=lambda face: face.area_px)


def _landmarks_to_bbox_px(landmarks_px, frame_shape):
    """랜드마크 전체를 감싸는 박스 + 패딩."""
    xs = landmarks_px[:, 0]
    ys = landmarks_px[:, 1]
    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()
    pad = max(x2 - x1, y2 - y1, 20.0) * BBOX_PAD_RATIO
    h_px, w_px = frame_shape[:2]
    return (
        max(0.0, float(x1 - pad)), max(0.0, float(y1 - pad)),
        min(w_px - 1.0, float(x2 + pad)), min(h_px - 1.0, float(y2 + pad)),
    )


class FaceEstimator:
    """MediaPipe FaceLandmarker 래퍼. infer(frame) -> list[FaceLandmarks]."""

    def __init__(self, config):
        face_cfg = config["face_tracker"]
        # mediapipe는 무거운 의존이라 사용 시점에 임포트한다 (단위 테스트가 가벼워지게)
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker, FaceLandmarkerOptions, RunningMode,
        )

        # 2026-07-31 실기 — 한글 경로 대응 (hand_tracker.py와 동일 사유·동일 처방:
        # mediapipe 0.10.14가 model_asset_path의 한글 경로를 못 연다)
        with open(face_cfg["model_path"], "rb") as model_file:
            model_bytes = model_file.read()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=RunningMode.VIDEO,
            num_faces=face_cfg["max_num_faces"],
            min_face_detection_confidence=face_cfg["min_detection_conf"],
            min_face_presence_confidence=face_cfg["min_presence_conf"],
            min_tracking_confidence=face_cfg["min_tracking_conf"],
            output_face_blendshapes=True,   # 입 벌림·눈 감김·입 오므림 판정의 입력
        )
        self._mp = mp
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._infer_scale_ratio = face_cfg["infer_scale_ratio"]
        # VIDEO 모드는 단조 증가 타임스탬프(ms)가 필수 — 프레임 간 추적에 쓰인다
        self._start_sec = time.monotonic()
        self._last_timestamp_ms = -1
        logger.info(
            "얼굴 랜드마크 모델 로딩 완료: MediaPipe FaceLandmarker (max_num_faces=%d, infer_scale=%.2f, %s)",
            face_cfg["max_num_faces"], self._infer_scale_ratio, face_cfg["model_path"],
        )

    def infer(self, frame):
        """프레임(BGR·거울 반전 후)에서 얼굴 랜드마크를 추정한다 -> list[FaceLandmarks].

        infer_scale_ratio < 1.0이면 추론 입력만 축소한다 — MediaPipe는 정규화(0~1) 좌표를
        돌려주므로 원본 프레임 크기로 곱하면 좌표는 자동으로 원본 기준이 된다
        (시각화·판정 코드는 축소 여부를 몰라도 된다).
        """
        infer_frame = frame
        if self._infer_scale_ratio < 1.0:
            infer_frame = cv2.resize(
                frame, None, fx=self._infer_scale_ratio, fy=self._infer_scale_ratio,
                interpolation=cv2.INTER_AREA,
            )
        # cvtColor는 SIMD 최적화 + 연속 메모리 출력 — numpy 역순 슬라이스+복사보다 빠르다
        rgb = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.monotonic() - self._start_sec) * 1000.0)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1   # 단조 증가 보장 (고FPS 보호)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        h_px, w_px = frame.shape[:2]
        faces = []
        for face_idx, landmarks in enumerate(result.face_landmarks):
            landmarks_px = np.array(
                [(pt.x * w_px, pt.y * h_px) for pt in landmarks], dtype=np.float32
            )
            blendshapes = {}
            if result.face_blendshapes:
                blendshapes = {c.category_name: c.score for c in result.face_blendshapes[face_idx]}
            faces.append(FaceLandmarks(
                bbox=_landmarks_to_bbox_px(landmarks_px, frame.shape), conf=1.0,
                landmarks_px=landmarks_px, blendshapes=blendshapes,
            ))
        return faces

    def close(self):
        self._landmarker.close()
