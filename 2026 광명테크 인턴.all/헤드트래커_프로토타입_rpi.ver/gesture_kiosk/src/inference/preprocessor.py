"""inference 전처리 — 추론 입력용으로 프레임을 가공한다 (기획서 4.6 계약).

2026-07-20: 캡처 해상도(720p)를 파이프라인 처리 해상도(camera.proc_*)로 낮추는 책임 추가.
720p 프레임을 그대로 흘리면 추론이 빨라져도 프레임 배관(복사·시각화·JPEG 인코딩)
비용이 3배로 늘어 추론 FPS가 오히려 무너진다(2026-07-20 실측: 27→10 FPS).
캡처만 크게 받고 여기서 축소하면 카메라가 어떤 해상도로 협상되든 파이프라인은
항상 같은 16:9 캔버스를 쓴다 — 종횡비가 다르면(4:3 카메라 등) 중앙 크롭으로
왜곡 없이 맞춘다(데모 UI 스테이지의 16:9 cover 표시와 기준 일치).
"""
import cv2


class Preprocessor:
    def __init__(self, config):
        camera = config["camera"]
        self._is_mirror = camera["mirror"]
        self._proc_width_px = camera["proc_width_px"]
        self._proc_height_px = camera["proc_height_px"]

    def preprocess_frame(self, frame):
        """frame(BGR) -> 처리 해상도의 input_tensor. 중앙 크롭·축소 후 거울 반전."""
        frame = self._crop_to_proc_aspect(frame)
        h_px, w_px = frame.shape[:2]
        if (w_px, h_px) != (self._proc_width_px, self._proc_height_px):
            frame = cv2.resize(
                frame, (self._proc_width_px, self._proc_height_px), interpolation=cv2.INTER_AREA
            )
        if self._is_mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def _crop_to_proc_aspect(self, frame):
        """처리 종횡비와 다르면 중앙 크롭 — 그냥 리사이즈하면 얼굴이 눌려 왜곡된다."""
        h_px, w_px = frame.shape[:2]
        target_ratio = self._proc_width_px / self._proc_height_px
        current_ratio = w_px / h_px
        if abs(current_ratio - target_ratio) < 1e-3:
            return frame
        if current_ratio > target_ratio:   # 가로가 남는다 — 좌우를 잘라낸다
            crop_width_px = int(h_px * target_ratio)
            x_start = (w_px - crop_width_px) // 2
            return frame[:, x_start:x_start + crop_width_px]
        crop_height_px = int(w_px / target_ratio)   # 세로가 남는다 — 위아래를 잘라낸다
        y_start = (h_px - crop_height_px) // 2
        return frame[y_start:y_start + crop_height_px]
