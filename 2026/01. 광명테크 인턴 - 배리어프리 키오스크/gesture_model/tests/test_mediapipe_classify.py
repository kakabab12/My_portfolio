"""detector_mediapipe.is_back_of_hand 단위 테스트 — mediapipe 설치 없이 실행 가능.

손목->검지MCP, 손목->새끼MCP 벡터의 외적 부호로 손등/손바닥 방향을 구분하는
로직을 합성 랜드마크로 검증한다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.detector_mediapipe import is_back_of_hand, user_side_from_label

WRIST = (100.0, 200.0)


def build_hand(index_mcp, pinky_mcp):
    """21개 랜드마크 합성 — 손목/검지MCP/새끼MCP만 채우고 나머지는 무관하므로 중립값."""
    points = [(100.0, 180.0)] * 21
    points[0] = WRIST
    points[5] = index_mcp
    points[17] = pinky_mcp
    return points


# 오른손이 손바닥을 카메라로 향하고 손가락이 위를 가리킬 때: 검지(엄지쪽)가 왼쪽,
# 새끼가 오른쪽에 온다 (화면 y는 아래로 증가).
RIGHT_PALM_FACING = build_hand(index_mcp=(90.0, 150.0), pinky_mcp=(110.0, 150.0))
# 같은 오른손을 손목 축으로 뒤집으면(손등이 보이면) 검지/새끼 좌우가 바뀐다.
RIGHT_BACK_FACING = build_hand(index_mcp=(110.0, 150.0), pinky_mcp=(90.0, 150.0))
# 왼손은 오른손의 거울상이라 반대.
LEFT_PALM_FACING = build_hand(index_mcp=(110.0, 150.0), pinky_mcp=(90.0, 150.0))
LEFT_BACK_FACING = build_hand(index_mcp=(90.0, 150.0), pinky_mcp=(110.0, 150.0))


class IsBackOfHandTest(unittest.TestCase):
    def test_right_hand_back_facing_is_true(self):
        self.assertTrue(is_back_of_hand(RIGHT_BACK_FACING, "right"))

    def test_right_hand_palm_facing_is_false(self):
        self.assertFalse(is_back_of_hand(RIGHT_PALM_FACING, "right"))

    def test_left_hand_back_facing_is_true(self):
        self.assertTrue(is_back_of_hand(LEFT_BACK_FACING, "left"))

    def test_left_hand_palm_facing_is_false(self):
        self.assertFalse(is_back_of_hand(LEFT_PALM_FACING, "left"))

    def test_unknown_handedness_is_false(self):
        self.assertFalse(is_back_of_hand(RIGHT_BACK_FACING, "unknown"))

    def test_flip_orientation_inverts_result(self):
        self.assertFalse(is_back_of_hand(RIGHT_BACK_FACING, "right", flip_orientation=True))
        self.assertTrue(is_back_of_hand(RIGHT_PALM_FACING, "right", flip_orientation=True))


class UserSideFromLabelTest(unittest.TestCase):
    """handedness 라벨 -> 사용자 좌/우 매핑 — 거울·flip_handedness 4조합 + 무효 라벨."""

    def test_mirror_with_flip_swaps_label(self):
        self.assertEqual(user_side_from_label("left", True, True), "right")
        self.assertEqual(user_side_from_label("right", True, True), "left")

    def test_mirror_without_flip_keeps_label(self):
        self.assertEqual(user_side_from_label("left", True, False), "left")

    def test_no_mirror_without_flip_swaps_label(self):
        self.assertEqual(user_side_from_label("left", False, False), "right")

    def test_no_mirror_with_flip_keeps_label(self):
        self.assertEqual(user_side_from_label("left", False, True), "left")

    def test_unknown_label_returns_none(self):
        self.assertIsNone(user_side_from_label("unknown", True, True))


if __name__ == "__main__":
    unittest.main()
