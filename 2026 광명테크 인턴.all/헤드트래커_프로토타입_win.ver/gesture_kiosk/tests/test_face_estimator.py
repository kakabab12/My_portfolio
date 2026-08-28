"""face_estimator 단위 테스트 — mediapipe 없이 순수 로직(bbox 계산·데이터 접근자)만 검증한다.

FaceEstimator 클래스 자체(모델 로딩·추론)는 mediapipe가 필요해 여기서 다루지 않는다 —
scripts/smoke_test.py가 실기에서 더미 프레임으로 검증한다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.inference.face_estimator import FaceLandmarks, _landmarks_to_bbox_px


class BboxFromLandmarksTest(unittest.TestCase):
    def test_bbox_wraps_all_points_with_padding(self):
        landmarks_px = np.array([[100.0, 100.0], [200.0, 150.0], [150.0, 250.0]], dtype=np.float32)
        x1, y1, x2, y2 = _landmarks_to_bbox_px(landmarks_px, frame_shape=(720, 1280, 3))
        # 패딩(BBOX_PAD_RATIO=0.10)만큼 원래 범위(100~200, 100~250)보다 넓어야 한다
        self.assertLess(x1, 100.0)
        self.assertLess(y1, 100.0)
        self.assertGreater(x2, 200.0)
        self.assertGreater(y2, 250.0)

    def test_bbox_clamps_to_frame_bounds(self):
        # 프레임 가장자리 근처 랜드마크 — 패딩을 더해도 프레임을 벗어나면 안 된다
        landmarks_px = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32)
        x1, y1, x2, y2 = _landmarks_to_bbox_px(landmarks_px, frame_shape=(100, 100, 3))
        self.assertGreaterEqual(x1, 0.0)
        self.assertGreaterEqual(y1, 0.0)
        self.assertLessEqual(x2, 99.0)
        self.assertLessEqual(y2, 99.0)

    def test_bbox_has_minimum_padding_for_tiny_landmark_cluster(self):
        # 랜드마크가 한 점에 몰려 있어도(범위 0) 최소 패딩(20px 기준)은 있어야 한다
        landmarks_px = np.array([[500.0, 500.0], [500.0, 500.0]], dtype=np.float32)
        x1, y1, x2, y2 = _landmarks_to_bbox_px(landmarks_px, frame_shape=(1080, 1920, 3))
        self.assertLess(x1, 500.0)
        self.assertGreater(x2, 500.0)


class FaceLandmarksAccessorTest(unittest.TestCase):
    def _make(self, landmarks_px=None, blendshapes=None):
        if landmarks_px is None:
            landmarks_px = np.zeros((478, 2), dtype=np.float32)
        return FaceLandmarks(
            bbox=(0, 0, 10, 10), conf=1.0, landmarks_px=landmarks_px,
            blendshapes=blendshapes or {},
        )

    def test_landmark_px_returns_float_tuple(self):
        landmarks_px = np.zeros((478, 2), dtype=np.float32)
        landmarks_px[1] = (123.5, 456.5)
        face = self._make(landmarks_px=landmarks_px)
        x, y = face.landmark_px(1)
        self.assertEqual((x, y), (123.5, 456.5))
        self.assertIsInstance(x, float)

    def test_blendshape_returns_score_when_present(self):
        face = self._make(blendshapes={"jawOpen": 0.73})
        self.assertAlmostEqual(face.blendshape("jawOpen"), 0.73)

    def test_blendshape_returns_default_when_missing(self):
        face = self._make(blendshapes={})
        self.assertEqual(face.blendshape("jawOpen"), 0.0)
        self.assertEqual(face.blendshape("jawOpen", default=-1.0), -1.0)


if __name__ == "__main__":
    unittest.main()
