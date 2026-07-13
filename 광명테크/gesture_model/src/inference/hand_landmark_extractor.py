"""MediaPipe HandLandmarker 래퍼 — 프레임에서 손 21랜드마크(픽셀 좌표)를 뽑는다.

MediaPipeGestureDetector(실시간 추론)와 scripts/collect_landmarks.py(학습 데이터
녹화)가 이 클래스를 공유해서 쓴다 — 그래야 "학습 데이터를 뽑을 때"와 "실전에서
추론할 때"가 완전히 같은 방식으로 랜드마크를 추출한다는 게 보장된다.
"""
import time


class HandLandmarkExtractor:
    """VIDEO 모드 HandLandmarker 래퍼. detect(frame) -> (hands_px, handedness)."""

    def __init__(self, model_path, num_hands=2,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._clock = time.monotonic
        self._last_timestamp_ms = -1

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def detect(self, frame_bgr):
        """frame(BGR) -> (hands_px, handedness_list).

        hands_px: 손마다 [(x_px, y_px), ...] 21개 랜드마크 리스트.
        handedness_list: mediapipe HandLandmarkerResult.handedness 그대로 (손과 같은 순서).
        """
        h_px, w_px = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb.copy(order="C"))
        # VIDEO 모드는 단조 증가 타임스탬프(ms)를 요구한다 — 호출 스레드 단일 전제
        timestamp_ms = max(int(self._clock() * 1000.0), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands_px = [
            [(lm.x * w_px, lm.y * h_px) for lm in hand_landmarks]
            for hand_landmarks in result.hand_landmarks
        ]
        return hands_px, result.handedness
