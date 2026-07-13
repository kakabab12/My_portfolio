"""inference 전처리 — 추론 입력용으로 프레임을 가공한다. 현재는 거울 반전만 담당한다."""
import cv2


class Preprocessor:
    def __init__(self, config):
        self._is_mirror = config["camera"]["mirror"]

    def preprocess_frame(self, frame):
        """frame(BGR) -> input_tensor. 거울 모드면 좌우 반전한다."""
        if self._is_mirror:
            frame = cv2.flip(frame, 1)
        return frame
