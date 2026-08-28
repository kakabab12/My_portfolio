"""inference 전처리 — 추론 입력용으로 프레임을 가공한다 (기획서 4.6 계약).

거울 반전을 담당한다. 세로 크롭(2026-07-30 사용자 결정 — 헤드트래커 모드용):
카메라는 회전 없이 그대로 두고(웹캠 센서는 가로 고정이라 세로 해상도 자체를
협상할 수 없다), 캡처된 가로 프레임 중앙에서 좌우만 잘라 세로 9:16 화면 비율을
만든다. apply_crop 인자를 안 넘기면(camera_probe 등 모드 개념이 없는 호출부)
config portrait_crop.enabled를 그대로 따른다(하위 호환).
"""
import cv2


class Preprocessor:
    def __init__(self, config):
        self._is_mirror = config["camera"]["mirror"]
        crop = config["camera"].get("portrait_crop") or {}
        self._crop_enabled = crop.get("enabled", False)
        self._crop_aspect_ratio = crop.get("aspect_ratio")

    def preprocess_frame(self, frame, apply_crop=None):
        """frame(BGR) -> input_tensor. 거울 모드면 좌우 반전, 세로 크롭 여부는
        apply_crop으로 매 호출 재정의할 수 있다(None이면 config 기본값 사용)."""
        if self._is_mirror:
            frame = cv2.flip(frame, 1)
        is_crop_applied = self._crop_enabled if apply_crop is None else apply_crop
        if is_crop_applied:
            frame = _crop_to_portrait(frame, self._crop_aspect_ratio)
        return frame


def _crop_to_portrait(frame, aspect_ratio):
    """가로 프레임을 중앙에서 세로 비율(aspect_ratio = 폭/높이)로 자른다.

    높이는 그대로 두고 폭만 줄인다 — 카메라 물리 방향은 바꾸지 않는다는
    전제라 회전은 하지 않는다(2026-07-30). 화각이 좁아지는 만큼 손 쓸기가
    화면 가장자리에 가까워지므로, 스윙이 잘리면 aspect_ratio를 올릴 것.
    """
    height_px, width_px = frame.shape[:2]
    target_width_px = min(width_px, round(height_px * aspect_ratio))
    x1_px = (width_px - target_width_px) // 2
    return frame[:, x1_px:x1_px + target_width_px]
