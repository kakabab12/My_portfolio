"""inference 모듈 — 얼굴 랜드마크(MediaPipe FaceLandmarker)를 추론한다.

2026-07-18 전면 개편: 팔 쓸기(포즈) → 헤드트래커(얼굴)로 입력 방식을 교체하며
포즈 추정기를 대체하는 **유일한 추론 모델**이 됐다. 코끝(포인터)·입 벌림·입 오므림
(블렌드셰이프) 전부 이 얼굴 랜드마크 하나로 판정된다.

라이선스: MediaPipe(Apache-2.0). rtmlib(RTMPose)·onnxruntime-gpu·torch(CUDA DLL
등록용) 전부 제거 — MediaPipe FaceLandmarker는 작은 모델이라 CPU만으로 ~30 FPS가
나와(2026-07-20 실측, 처리 해상도 640x360) GPU 의존 자체가 불필요해졌다. 이전
RTMPose는 CPU 0.5 FPS로 실사용이 불가능해 onnxruntime-gpu·CUDA 버전 고정에 상당한
공수가 들었는데, 이번 전환으로 그 복잡도 전체가 사라진다.

모델 파일(face_landmarker.task)은 빌드 타임 1회 다운로드해 models/weights/에 둔다
(scripts/download_weights.py) — 런타임 네트워크 접근이 없어 내부망 배포와 호환된다.
"""
import time

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("inference")

# FaceLandmarker 랜드마크 인덱스 (478점 — 468 얼굴 + 10 홍채, 레거시 face_mesh와 동일 토폴로지)
LMK_NOSE_TIP = 1
LMK_LEFT_EYE_OUTER = 33     # 안구간 거리 정규화 자(尺)
LMK_RIGHT_EYE_OUTER = 263

BBOX_PAD_RATIO = 0.10  # 랜드마크 묶음 -> 얼굴 박스로 넓히는 패딩 (추적용)


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
    """MediaPipe FaceLandmarker 추정기. infer(frame, ts_ms) -> list[FaceLandmarks]."""

    def __init__(self, config):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

        model = config["model"]
        base_options = BaseOptions(model_asset_path=model["face_landmarker_path"])
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_faces=model["num_faces"],   # PersonLock이 후보 중 고를 수 있게 1보다 크게
            min_face_detection_confidence=model["min_face_detection_conf"],
            min_face_presence_confidence=model["min_face_presence_conf"],
            min_tracking_confidence=model["min_tracking_conf"],
            output_face_blendshapes=True,
        )
        self._mp = mp
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._infer_scale_ratio = model["infer_scale_ratio"]
        self._last_ts_ms = -1
        logger.info(
            "얼굴 랜드마크 모델 로딩 완료: FaceLandmarker(num_faces=%d, infer_scale=%.2f)",
            model["num_faces"], self._infer_scale_ratio,
        )

    def infer(self, frame):
        """프레임에서 얼굴 랜드마크를 추정한다. RunningMode.VIDEO라 ts_ms는 내부에서 단조 증가시킨다.

        infer_scale_ratio < 1.0이면 추론 입력만 축소한다 — MediaPipe는 정규화(0~1) 좌표를
        돌려주므로 원본 프레임 크기로 곱하면 좌표는 자동으로 원본 기준이 된다
        (시각화·판정 코드는 축소 여부를 몰라도 된다).
        """
        ts_ms = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        infer_frame = frame
        if self._infer_scale_ratio < 1.0:
            infer_frame = cv2.resize(
                frame, None, fx=self._infer_scale_ratio, fy=self._infer_scale_ratio,
                interpolation=cv2.INTER_AREA,
            )
        # cvtColor는 SIMD 최적화 + 연속 메모리 출력 — numpy 역순 슬라이스+복사보다 빠르다 (2026-07-20)
        rgb = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        h_px, w_px = frame.shape[:2]
        faces = []
        for face_idx, landmarks in enumerate(result.face_landmarks):
            landmarks_px = np.array(
                [(pt.x * w_px, pt.y * h_px) for pt in landmarks], dtype=np.float32
            )
            blendshapes = {}
            if result.face_blendshapes:
                blendshapes = {c.category_name: c.score for c in result.face_blendshapes[face_idx]}
            bbox = _landmarks_to_bbox_px(landmarks_px, frame.shape)
            faces.append(FaceLandmarks(
                bbox=bbox, conf=1.0, landmarks_px=landmarks_px, blendshapes=blendshapes,
            ))
        return faces

    def close(self):
        self._landmarker.close()
