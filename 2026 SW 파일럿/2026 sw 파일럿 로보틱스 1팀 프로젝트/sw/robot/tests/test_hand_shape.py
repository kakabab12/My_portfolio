"""hand_shape 단위 테스트 — 손 모양 판별(주먹/한 손가락/손바닥) 기하 규칙 검증.

카메라·모델 없이 합성 랜드마크(hand_fixtures)로만 검증한다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.postprocess.hand_shape import (
    HandShapeStabilizer, classify_hand_shape, hand_center_point,
)
from tests.hand_fixtures import hand_center_of, make_hand_landmarks

EXTEND_RATIO = 1.35
MIN_VALID_FINGERS = 3
CURL_CONFIRM_RATIO = 0.9


def classify(landmarks):
    return classify_hand_shape(landmarks, EXTEND_RATIO, MIN_VALID_FINGERS,
                               CURL_CONFIRM_RATIO)


class ClassifyHandShapeTest(unittest.TestCase):
    def test_fist_is_classified(self):
        self.assertEqual(classify(make_hand_landmarks("fist")), "fist")

    def test_index_finger_is_classified(self):
        self.assertEqual(classify(make_hand_landmarks("finger")), "finger")

    def test_any_single_finger_counts(self):
        self.assertEqual(classify(make_hand_landmarks("middle_finger")), "finger")

    def test_open_hand_is_open(self):
        self.assertEqual(classify(make_hand_landmarks("open")), "open")

    def test_index_and_middle_extended_is_v(self):
        # 검지·중지만 펴고 약지·새끼를 접으면 명확한 V 제스처.
        landmarks = make_hand_landmarks("open")
        curled = make_hand_landmarks("fist")
        landmarks[13:21] = curled[13:21]   # 약지·새끼는 주먹 자세로 — 폄 2(검지·중지)
        self.assertEqual(classify(landmarks), "v")

    def test_point_at_camera_is_finger(self):
        # 카메라를 가리키는 검지는 화면 투영이 짧지만 z(깊이)로는 길다 —
        # 3D 거리로 "폄" 판정 -> 한 손가락
        self.assertEqual(classify(make_hand_landmarks("point_camera")), "finger")

    def test_bunched_flat_finger_is_not_fist(self):
        # 짧지만 접힘 증거(방향 반전·확실 비율)가 없는 손가락 = 기권 —
        # 기권이 있으면 주먹을 단정하지 않는다(오발 방지)
        self.assertIsNone(classify(make_hand_landmarks("bunched_flat")))

    def test_folded_finger_confirms_curl_by_direction(self):
        # 되접힘(방향 반전)은 비율과 무관하게 굽힘 확인 — curl_confirm을 0으로
        # 줘서 비율 경로를 꺼도 fist가 나오면 방향 반전 경로가 동작한 것
        self.assertEqual(
            classify_hand_shape(make_hand_landmarks("fist"), EXTEND_RATIO,
                                MIN_VALID_FINGERS, 0.0),
            "fist",
        )

    def test_short_landmarks_are_unknown(self):
        self.assertIsNone(classify(np.zeros((17, 3))))
        self.assertIsNone(classify(None))


class HandCenterPointTest(unittest.TestCase):
    def test_center_is_mean_of_landmarks(self):
        landmarks = make_hand_landmarks("fist")
        center = hand_center_point(landmarks)
        expected = hand_center_of(landmarks)
        self.assertAlmostEqual(center[0], expected[0], places=4)
        self.assertAlmostEqual(center[1], expected[1], places=4)

    def test_short_landmarks_return_none(self):
        self.assertIsNone(hand_center_point(np.zeros((17, 3))))
        self.assertIsNone(hand_center_point(None))


class HandShapeStabilizerTest(unittest.TestCase):
    def _confirm(self, stabilizer, shape, start_sec=1.0):
        self.assertIsNone(stabilizer.update(shape, True, start_sec))
        return stabilizer.update(shape, True, start_sec + 0.01)

    def test_action_shapes_require_two_matching_frames(self):
        for shape in ("open", "finger", "v"):
            stabilizer = HandShapeStabilizer(confirm_frames=2)
            self.assertEqual(self._confirm(stabilizer, shape), shape)

    def test_single_frame_false_positive_is_rejected(self):
        stabilizer = HandShapeStabilizer(confirm_frames=2)
        self.assertIsNone(stabilizer.update("finger", True, 1.0))
        self.assertIsNone(stabilizer.update("open", True, 1.01))

    def test_confirmed_shape_remains_stable(self):
        stabilizer = HandShapeStabilizer(confirm_frames=2)
        self._confirm(stabilizer, "open")
        self.assertEqual(stabilizer.update("open", True, 1.02), "open")

    def test_per_shape_confirmation_frames(self):
        stabilizer = HandShapeStabilizer(
            confirm_frames=2, confirm_frames_by_shape={"four": 3})
        self.assertIsNone(stabilizer.update("four", True, 1.0))
        self.assertIsNone(stabilizer.update("four", True, 1.1))
        self.assertEqual(stabilizer.update("four", True, 1.2), "four")

    def test_short_unknown_dropout_keeps_open(self):
        stabilizer = HandShapeStabilizer(hold_open_on_unknown_sec=0.3)
        self.assertEqual(self._confirm(stabilizer, "open"), "open")
        self.assertEqual(stabilizer.update(None, True, 1.2), "open")

    def test_long_unknown_dropout_becomes_unknown(self):
        stabilizer = HandShapeStabilizer(hold_open_on_unknown_sec=0.3)
        self._confirm(stabilizer, "open")
        self.assertIsNone(stabilizer.update(None, True, 1.31))

    def test_fist_stops_immediately_and_finger_requires_confirmation(self):
        stabilizer = HandShapeStabilizer(hold_open_on_unknown_sec=0.3)
        self._confirm(stabilizer, "open")
        self.assertEqual(stabilizer.update("fist", True, 1.1), "fist")
        self.assertIsNone(stabilizer.update("finger", True, 2.0))
        self.assertEqual(stabilizer.update("finger", True, 2.1), "finger")

    def test_hand_loss_clears_open_immediately(self):
        stabilizer = HandShapeStabilizer(hold_open_on_unknown_sec=0.3)
        self._confirm(stabilizer, "open")
        self.assertIsNone(stabilizer.update(None, False, 1.1))
        self.assertIsNone(stabilizer.update(None, True, 1.2))


if __name__ == "__main__":
    unittest.main()
