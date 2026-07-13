"""inference 모듈 — 제스처 검출 공통 데이터 구조 + 검출기 생성.

gesture_kiosk의 ONNX(HaGRID YOLOv10) 엔진은 AGPL-3.0 라이선스 리스크로 상업 납품이
금지되어 있어 이식하지 않았다. 이 프로젝트는 MediaPipe Hand Landmarker + 기하 규칙
판정(detector_mediapipe.py) 단일 엔진만 쓴다 — 학습 0회, Apache-2.0.
"""
from dataclasses import dataclass


@dataclass
class Detection:
    """검출 결과 1건 (공통 데이터 구조)."""

    class_id: int
    class_name: str
    conf: float
    bbox: tuple  # (x1, y1, x2, y2) 좌상단·우하단 픽셀 좌표 (원본 프레임 기준)
    hand_side: str = None  # "left"|"right" — MediaPipe handedness. person_lock이 좌/우 판정에 쓴다


def create_gesture_detector(config):
    """MediaPipe Hand Landmarker 기반 제스처 검출기를 만든다."""
    from src.inference.detector_mediapipe import MediaPipeGestureDetector  # 순환 임포트 방지

    return MediaPipeGestureDetector(config)
