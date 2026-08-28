"""inference 전처리 — 추론 입력용으로 프레임을 가공한다 (기획서 4.6 계약).

현재 백엔드(rtmlib)는 리사이즈·정규화를 내부에서 처리하므로 여기서는
거울 반전과 밝기 보정만 담당한다.

밝기 보정(2026-07-24 — 어둡거나 밝은 장소에서 인식 저하 신고 대응): 프레임
평균 밝기가 목표치에서 많이 벗어나면 감마 보정으로 목표 근처로 되돌린다.
CLAHE 등 영역별 보정 대신 전역 감마를 쓴 이유 — 이미 6 FPS대인 CPU 추론
루프에 프레임당 비용을 더 얹지 않기 위해서다(감마 LUT 적용은 단순 픽셀
치환이라 매우 싸다, CLAHE는 영역별 히스토그램 계산이 상대적으로 무겁다).
"""
import cv2
import numpy as np

GAMMA_MIN = 0.4   # 극단 보정 방지 하한(너무 밝아지지 않게)
GAMMA_MAX = 2.5   # 극단 보정 방지 상한(너무 어두워지지 않게)


def _build_gamma_lut(gamma):
    """감마 보정 LUT(0~255) — cv2.LUT로 프레임 전체에 한 번에 적용한다."""
    table = ((np.arange(256, dtype=np.float32) / 255.0) ** gamma) * 255.0
    return np.clip(table, 0, 255).astype(np.uint8)


class Preprocessor:
    def __init__(self, config):
        self._is_mirror = config["camera"]["mirror"]
        brightness_cfg = config["camera"].get("auto_brightness") or {}
        self._brightness_enabled = brightness_cfg.get("enabled", False)
        self._target_mean = brightness_cfg.get("target_mean", 128.0)
        self._deadband = brightness_cfg.get("deadband", 25.0)
        self._lut_cache = {}   # 반올림한 감마값 -> LUT (재계산 방지)

    def preprocess_frame(self, frame):
        """frame(BGR) -> input_tensor. 거울 모드면 좌우 반전, 밝기 보정이 켜져
        있고 평균 밝기가 목표 범위를 벗어나면 감마 보정도 적용한다."""
        if self._is_mirror:
            frame = cv2.flip(frame, 1)
        if self._brightness_enabled:
            frame = self._apply_auto_brightness(frame)
        return frame

    def _apply_auto_brightness(self, frame):
        mean = float(frame.mean())
        if mean <= 1.0 or abs(mean - self._target_mean) <= self._deadband:
            return frame   # 이미 목표 범위 안 — 매 프레임 보정하는 낭비·잔떨림 방지
        # 감마 공식: target = 255*(mean/255)^gamma 를 gamma로 풀면 아래와 같다
        gamma = np.log(self._target_mean / 255.0) / np.log(mean / 255.0)
        gamma = min(max(gamma, GAMMA_MIN), GAMMA_MAX)
        gamma_key = round(gamma, 2)
        lut = self._lut_cache.get(gamma_key)
        if lut is None:
            lut = _build_gamma_lut(gamma_key)
            self._lut_cache[gamma_key] = lut
        return cv2.LUT(frame, lut)
