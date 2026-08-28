"""Preprocessor 단위 테스트 — 거울 반전 + 세로 크롭(2026-07-30 병합)을 카메라 없이 검증한다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.inference.preprocessor import Preprocessor

FRAME_HEIGHT_PX = 100
FRAME_WIDTH_PX = 200


def make_frame():
    """폭 방향으로 열 번호가 그대로 픽셀값이 되는 프레임 — 크롭 중심이 맞는지 열 값으로 확인."""
    columns = np.arange(FRAME_WIDTH_PX, dtype=np.uint8)
    return np.tile(columns.reshape(1, -1, 1), (FRAME_HEIGHT_PX, 1, 3))


def make_config(mirror=False, crop_enabled=False, aspect_ratio=0.5625):
    return {
        "camera": {
            "mirror": mirror,
            "portrait_crop": {"enabled": crop_enabled, "aspect_ratio": aspect_ratio},
        }
    }


class MirrorTest(unittest.TestCase):
    def test_mirror_flips_left_right(self):
        preprocessor = Preprocessor(make_config(mirror=True))
        result = preprocessor.preprocess_frame(make_frame())
        self.assertEqual(result[0, 0, 0], FRAME_WIDTH_PX - 1)
        self.assertEqual(result[0, -1, 0], 0)

    def test_no_mirror_keeps_order(self):
        preprocessor = Preprocessor(make_config(mirror=False))
        result = preprocessor.preprocess_frame(make_frame())
        self.assertEqual(result[0, 0, 0], 0)
        self.assertEqual(result[0, -1, 0], FRAME_WIDTH_PX - 1)


class PortraitCropTest(unittest.TestCase):
    def test_disabled_keeps_original_size(self):
        preprocessor = Preprocessor(make_config(crop_enabled=False))
        result = preprocessor.preprocess_frame(make_frame())
        self.assertEqual(result.shape[:2], (FRAME_HEIGHT_PX, FRAME_WIDTH_PX))

    def test_enabled_crops_to_target_aspect_ratio_centered(self):
        preprocessor = Preprocessor(make_config(crop_enabled=True, aspect_ratio=0.5625))
        result = preprocessor.preprocess_frame(make_frame())
        target_width_px = round(FRAME_HEIGHT_PX * 0.5625)   # 56
        self.assertEqual(result.shape[:2], (FRAME_HEIGHT_PX, target_width_px))
        x1_px = (FRAME_WIDTH_PX - target_width_px) // 2      # 72
        self.assertEqual(result[0, 0, 0], x1_px)
        self.assertEqual(result[0, -1, 0], x1_px + target_width_px - 1)

    def test_crop_applies_after_mirror(self):
        # 거울 반전 후 크롭이라 중심 열 값은 반전된 좌표계 기준이어야 한다
        preprocessor = Preprocessor(make_config(mirror=True, crop_enabled=True, aspect_ratio=0.5625))
        result = preprocessor.preprocess_frame(make_frame())
        target_width_px = round(FRAME_HEIGHT_PX * 0.5625)
        x1_px = (FRAME_WIDTH_PX - target_width_px) // 2
        # 반전 후 열 값은 (WIDTH-1-원래열) 이므로, 크롭 시작 지점의 값은 그 반전값
        self.assertEqual(result[0, 0, 0], FRAME_WIDTH_PX - 1 - x1_px)

    def test_aspect_ratio_wider_than_frame_clamps_to_original_width(self):
        preprocessor = Preprocessor(make_config(crop_enabled=True, aspect_ratio=10.0))
        result = preprocessor.preprocess_frame(make_frame())
        self.assertEqual(result.shape[:2], (FRAME_HEIGHT_PX, FRAME_WIDTH_PX))


class ApplyCropOverrideTest(unittest.TestCase):
    """2026-07-31 사용자 결정 — 손 모드=16:9(크롭 없음), 헤드트래커 모드=9:16(크롭).
    realtime_loop.py가 매 프레임 apply_crop으로 명시 재정의한다."""

    def test_override_true_crops_even_when_config_disabled(self):
        preprocessor = Preprocessor(make_config(crop_enabled=False))
        result = preprocessor.preprocess_frame(make_frame(), apply_crop=True)
        target_width_px = round(FRAME_HEIGHT_PX * 0.5625)
        self.assertEqual(result.shape[:2], (FRAME_HEIGHT_PX, target_width_px))

    def test_override_false_skips_crop_even_when_config_enabled(self):
        preprocessor = Preprocessor(make_config(crop_enabled=True))
        result = preprocessor.preprocess_frame(make_frame(), apply_crop=False)
        self.assertEqual(result.shape[:2], (FRAME_HEIGHT_PX, FRAME_WIDTH_PX))

    def test_no_override_falls_back_to_config(self):
        preprocessor = Preprocessor(make_config(crop_enabled=True))
        result = preprocessor.preprocess_frame(make_frame())
        target_width_px = round(FRAME_HEIGHT_PX * 0.5625)
        self.assertEqual(result.shape[:2], (FRAME_HEIGHT_PX, target_width_px))


if __name__ == "__main__":
    unittest.main()
