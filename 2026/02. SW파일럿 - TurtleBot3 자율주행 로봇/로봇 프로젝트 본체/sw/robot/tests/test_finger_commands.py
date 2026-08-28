import unittest

import numpy as np

from src.control.finger_commands import (
    WaveDetector, classify_finger_command, classify_thumb_toggle,
    is_back_of_hand, is_ok_sign, is_thumb_extended,
)
from tests.hand_fixtures import make_hand_landmarks


class WaveDetectorTest(unittest.TestCase):
    def test_entering_frame_once_does_not_hide_four_finger_pose(self):
        detector = WaveDetector({"amplitude_ratio": 0.05, "timeout_sec": 3.0})
        detector.update(make_hand_landmarks("open", root_xy=(400, 300)), 1000, 0.0)
        detector.update(make_hand_landmarks("open", root_xy=(500, 300)), 1000, 0.1)
        self.assertFalse(detector.in_progress)

    def test_three_full_swings_trigger_once(self):
        detector = WaveDetector({"amplitude_ratio": 0.05, "timeout_sec": 3.0})
        points = [500, 560, 440, 560, 440, 560, 440]
        fired = []
        for index, x in enumerate(points):
            landmarks = make_hand_landmarks("open", root_xy=(x, 300))
            fired.append(detector.update(landmarks, 1000, index * 0.2))
        self.assertEqual(fired.count(True), 1)

    def test_slow_wave_resets(self):
        detector = WaveDetector({"amplitude_ratio": 0.05, "timeout_sec": 0.3})
        first = make_hand_landmarks("open", root_xy=(500, 300))
        right = make_hand_landmarks("open", root_xy=(560, 300))
        left = make_hand_landmarks("open", root_xy=(440, 300))
        detector.update(first, 1000, 0.0)
        detector.update(right, 1000, 0.1)
        self.assertFalse(detector.update(left, 1000, 1.0))

    def test_wave_uses_each_previous_extreme_as_reference(self):
        detector = WaveDetector({"amplitude_ratio": 0.05, "timeout_sec": 3.0})
        points = [500, 560, 500, 560, 500, 560, 500]
        fired = []
        for index, x in enumerate(points):
            landmarks = make_hand_landmarks("open", root_xy=(x, 300))
            fired.append(detector.update(landmarks, 1000, index * 0.2))
        self.assertEqual(fired.count(True), 1)
        self.assertTrue(fired[-1])


class BackOfHandTest(unittest.TestCase):
    def test_mirrored_right_palm_is_not_back(self):
        landmarks = np.zeros((21, 3), dtype=float)
        landmarks[5, :2] = (1.0, -1.0)
        landmarks[17, :2] = (-1.0, -1.0)  # cross < 0: 미러된 오른손 손등
        self.assertTrue(is_back_of_hand(landmarks, "right"))
        landmarks[5, :2], landmarks[17, :2] = landmarks[17, :2].copy(), landmarks[5, :2].copy()
        self.assertFalse(is_back_of_hand(landmarks, "right"))


class FingerCommandProjectionTest(unittest.TestCase):
    def test_screen_projection_wins_when_world_depth_is_wrong(self):
        screen = make_hand_landmarks("open")
        screen[4] = screen[3]  # 엄지를 접은 네 손가락 포즈
        bad_world = make_hand_landmarks("fist") * 0.001
        self.assertEqual(classify_finger_command(
            bad_world, screen, "right", 1.03, 0.97), "four")

    def test_all_five_fingers_is_reserved_for_wave(self):
        screen = make_hand_landmarks("open")
        self.assertTrue(is_thumb_extended(screen))
        self.assertEqual(classify_finger_command(
            screen * 0.001, screen, "right", 1.03, 0.97), "open")

    def test_ok_sign_is_reserved_for_joystick_toggle(self):
        screen = make_hand_landmarks("open")
        screen[4] = screen[8]  # 엄지 끝과 검지 끝 접촉
        self.assertEqual(classify_finger_command(
            screen * 0.001, screen, "right", 1.03, 0.97),
            "joystick_toggle")

    def test_thumb_up_wins_over_false_world_index_extension(self):
        screen = make_hand_landmarks("fist")
        screen[0, :2] = (500.0, 400.0)
        screen[9, :2] = (500.0, 350.0)
        screen[2, :2] = (500.0, 370.0)
        screen[3, :2] = (500.0, 335.0)
        screen[4, :2] = (500.0, 300.0)
        bad_world = make_hand_landmarks("finger") * 0.001
        self.assertEqual(classify_finger_command(
            bad_world, screen, "right", 1.03, 0.97), "mode_on")

    def test_folded_thumb_does_not_steal_index_command(self):
        screen = make_hand_landmarks("finger")
        screen[2, :2] = (520.0, 375.0)
        screen[3, :2] = (535.0, 365.0)
        screen[4, :2] = (522.0, 378.0)
        self.assertEqual(classify_finger_command(
            screen * 0.001, screen, "right", 1.03, 0.97), "finger")

    def test_straight_thumb_does_not_steal_index_command(self):
        screen = make_hand_landmarks("finger")
        screen[0, :2] = (500.0, 400.0)
        screen[9, :2] = (500.0, 350.0)
        screen[2, :2] = (500.0, 370.0)
        screen[3, :2] = (500.0, 335.0)
        screen[4, :2] = (500.0, 300.0)
        self.assertEqual(classify_finger_command(
            screen * 0.001, screen, "right", 1.03, 0.97), "finger")

    def test_straight_thumb_does_not_steal_two_finger_command(self):
        screen = make_hand_landmarks("open")
        # 약지와 새끼만 접어 검지+중지 두 손가락 포즈를 만든다.
        screen[16, 1] = 360.0
        screen[20, 1] = 360.0
        screen[0, :2] = (500.0, 400.0)
        screen[9, :2] = (500.0, 350.0)
        screen[2, :2] = (500.0, 370.0)
        screen[3, :2] = (500.0, 335.0)
        screen[4, :2] = (500.0, 300.0)
        self.assertEqual(classify_finger_command(
            screen * 0.001, screen, "right", 1.03, 0.97), "two")


class ThumbToggleTest(unittest.TestCase):
    def _landmarks(self, tip_y):
        landmarks = np.zeros((21, 3), dtype=float)
        landmarks[:, 0] = np.linspace(0.0, 100.0, 21)
        landmarks[0, :2] = (50.0, 90.0)
        landmarks[9, :2] = (50.0, 40.0)
        landmarks[2, :2] = (50.0, 60.0)
        landmarks[3, :2] = (51.0, (60.0 + tip_y) / 2.0)
        landmarks[4, :2] = (52.0, tip_y)
        return landmarks

    def test_thumb_up_turns_mode_on(self):
        self.assertEqual(classify_thumb_toggle(self._landmarks(5.0)), "mode_on")

    def test_thumb_down_turns_mode_off(self):
        self.assertEqual(classify_thumb_toggle(self._landmarks(95.0)), "mode_off")

    def test_sideways_thumb_does_not_toggle(self):
        landmarks = self._landmarks(50.0)
        landmarks[4, 0] = 90.0
        self.assertIsNone(classify_thumb_toggle(landmarks))

    def test_folded_thumb_does_not_toggle(self):
        landmarks = self._landmarks(5.0)
        landmarks[3, :2] = (70.0, 35.0)
        landmarks[4, :2] = (52.0, 58.0)
        self.assertIsNone(classify_thumb_toggle(landmarks))

    def test_thumb_toggle_survives_90_degree_hand_rotation(self):
        landmarks = self._landmarks(5.0)
        center = np.array([50.0, 50.0])
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        landmarks[:, :2] = (landmarks[:, :2] - center) @ rotation.T + center
        self.assertEqual(classify_thumb_toggle(landmarks), "mode_on")


if __name__ == "__main__":
    unittest.main()
