"""카메라 이동 보정 단위 테스트."""
import unittest

import cv2
import numpy as np

from src.postprocess.camera_motion import CameraMotionCompensator


class CameraMotionCompensatorTest(unittest.TestCase):
    def _frame(self):
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def test_static_camera_does_not_move_point(self):
        compensator = CameraMotionCompensator({"analysis_scale": 0.5})
        frame = self._frame()
        compensator.update(frame)
        compensator.update(frame.copy())
        result = compensator.compensate((250.0, 170.0))
        self.assertAlmostEqual(result[0], 250.0, delta=0.2)
        self.assertAlmostEqual(result[1], 170.0, delta=0.2)

    def test_translated_camera_keeps_scene_point_stable(self):
        compensator = CameraMotionCompensator({"analysis_scale": 0.5})
        first = self._frame()
        dx, dy = 14.0, -9.0
        second = cv2.warpAffine(first, np.float32([[1, 0, dx], [0, 1, dy]]),
                                (first.shape[1], first.shape[0]))
        compensator.update(first)
        compensator.update(second)
        result = compensator.compensate((250.0 + dx, 170.0 + dy))
        self.assertAlmostEqual(result[0], 250.0, delta=1.0)
        self.assertAlmostEqual(result[1], 170.0, delta=1.0)

    def test_rotated_camera_keeps_scene_point_stable(self):
        compensator = CameraMotionCompensator(
            {"analysis_scale": 0.5, "max_rotation_deg": 8.0})
        first = self._frame()
        center = (first.shape[1] / 2.0, first.shape[0] / 2.0)
        matrix = cv2.getRotationMatrix2D(center, 4.0, 1.03)
        second = cv2.warpAffine(first, matrix, (first.shape[1], first.shape[0]))
        original = np.array([250.0, 170.0, 1.0])
        moved = matrix @ original
        compensator.update(first)
        compensator.update(second)
        result = compensator.compensate((moved[0], moved[1]))
        self.assertAlmostEqual(result[0], original[0], delta=1.5)
        self.assertAlmostEqual(result[1], original[1], delta=1.5)

    def test_disabled_compensation_returns_original_point(self):
        compensator = CameraMotionCompensator({"enabled": False})
        self.assertEqual(compensator.compensate((10.0, 20.0)), (10.0, 20.0))


if __name__ == "__main__":
    unittest.main()
