"""preprocessor 단위 테스트 — 카메라 없이 크롭·축소·거울 반전 로직만 검증한다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.inference.preprocessor import Preprocessor


def make_config(proc_width_px=640, proc_height_px=360, mirror=False):
    return {
        "camera": {
            "mirror": mirror,
            "proc_width_px": proc_width_px,
            "proc_height_px": proc_height_px,
        },
    }


class PreprocessFrameTest(unittest.TestCase):
    def test_downscales_720p_to_proc_size(self):
        pre = Preprocessor(make_config())
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = pre.preprocess_frame(frame)
        self.assertEqual(out.shape, (360, 640, 3))

    def test_center_crops_4_3_frame_to_16_9(self):
        # 4:3 카메라(640x480) — 위아래를 중앙 크롭해 16:9로 맞춰야 한다(왜곡 방지).
        # 크롭 없이 리사이즈하면 얼굴이 세로로 눌려 랜드마크 비율·커서 정렬이 틀어진다
        pre = Preprocessor(make_config())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:60, :, :] = 255    # 상단 60px 흰색 — 크롭으로 잘려나가야 함
        frame[-60:, :, :] = 255   # 하단 60px 흰색
        out = pre.preprocess_frame(frame)
        self.assertEqual(out.shape, (360, 640, 3))
        self.assertEqual(int(out.max()), 0)   # 흰 띠가 살아있으면 크롭이 안 된 것

    def test_passes_through_when_already_proc_size(self):
        pre = Preprocessor(make_config())
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        out = pre.preprocess_frame(frame)
        self.assertEqual(out.shape, (360, 640, 3))

    def test_mirror_flips_horizontally(self):
        pre = Preprocessor(make_config(mirror=True))
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :10, :] = 255    # 왼쪽 끝 흰 띠
        out = pre.preprocess_frame(frame)
        self.assertEqual(int(out[:, -10:, :].min()), 255)   # 반전 후 오른쪽 끝으로
        self.assertEqual(int(out[:, :10, :].max()), 0)


if __name__ == "__main__":
    unittest.main()
