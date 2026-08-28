"""roi_zoom 단위 테스트 — 손 앵커 디지털 줌 crop 계산."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.postprocess.roi_zoom import crop_frame, offset_landmarks_xy, resolve_roi_box

ROI_CFG = {"pad_ratio": 2.5, "min_side_px": 320, "move_ratio": 0.15, "resize_ratio": 0.2}


class ResolveRoiBoxTest(unittest.TestCase):
    def test_no_hand_returns_none(self):
        self.assertIsNone(resolve_roi_box(None, None, 1280, 720, ROI_CFG))

    def test_first_activation_centers_on_hand(self):
        box = resolve_roi_box(None, (600, 300, 650, 350), 1280, 720, ROI_CFG)
        self.assertEqual(box, (465, 165, 785, 485))

    def test_small_movement_keeps_previous_window(self):
        prev_box = resolve_roi_box(None, (600, 300, 650, 350), 1280, 720, ROI_CFG)
        box = resolve_roi_box(prev_box, (605, 305, 655, 355), 1280, 720, ROI_CFG)
        self.assertEqual(box, prev_box)

    def test_large_movement_recenters_window(self):
        prev_box = resolve_roi_box(None, (600, 300, 650, 350), 1280, 720, ROI_CFG)
        box = resolve_roi_box(prev_box, (900, 500, 950, 550), 1280, 720, ROI_CFG)
        self.assertIsNotNone(box)
        self.assertNotEqual(box, prev_box)

    def test_near_full_frame_hand_returns_none(self):
        # 손(실제로는 사람 전체 크기 정도)이 프레임 대부분을 덮을 만큼 크면
        # digital zoom이 무의미 -> 전체 프레임 사용
        box = resolve_roi_box(None, (100, 100, 500, 140), 640, 480, ROI_CFG)
        self.assertIsNone(box)


class OffsetLandmarksXyTest(unittest.TestCase):
    def test_shifts_xy_only(self):
        landmarks = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        shifted = offset_landmarks_xy(landmarks, 10.0, 20.0)
        np.testing.assert_allclose(shifted, [[11.0, 22.0, 3.0], [14.0, 25.0, 6.0]])
        # 원본은 변경되지 않아야 한다(복사본 반환)
        np.testing.assert_allclose(landmarks, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


class CropFrameTest(unittest.TestCase):
    def test_crops_expected_region_and_is_contiguous(self):
        frame = np.arange(300).reshape(10, 10, 3).astype(np.uint8)
        cropped = crop_frame(frame, (2, 3, 6, 7))
        np.testing.assert_array_equal(cropped, frame[3:7, 2:6])
        self.assertTrue(cropped.flags["C_CONTIGUOUS"])


if __name__ == "__main__":
    unittest.main()
