import unittest

from ros2_bridge.cmd_vel_mux import select_source


class SelectSourceTest(unittest.TestCase):
    def test_auto_uses_only_nav(self):
        self.assertEqual(select_source(False, False, False, True, True, True, True), "nav")
        self.assertIsNone(select_source(False, False, False, False, True, True, True))

    def test_gesture_mode_uses_only_gesture(self):
        self.assertEqual(select_source(True, False, False, True, True, True, True), "gesture")
        self.assertIsNone(select_source(True, False, False, True, False, True, True))

    def test_joystick_mode_uses_joystick_when_stick_is_active(self):
        self.assertEqual(select_source(True, True, False, True, True, True, True), "joystick")
        self.assertIsNone(select_source(True, True, False, True, True, False, True))

    def test_controller_mode_falls_back_to_glove_when_stick_is_neutral(self):
        self.assertEqual(
            select_source(True, True, True, True, True, True, True), "glove")

    def test_active_joystick_wins_over_glove_and_stops_when_stale(self):
        self.assertEqual(
            select_source(True, True, True, True, True, True, True,
                          joystick_active=True), "joystick")
        self.assertIsNone(
            select_source(True, True, True, True, True, False, True,
                          joystick_active=True))

    def test_glove_has_highest_priority_and_stops_when_stale(self):
        self.assertEqual(select_source(True, True, True, True, True, True, True), "glove")
        self.assertIsNone(select_source(True, True, True, True, True, True, False))

    def test_auto_ignores_moving_glove_when_glove_is_controller_bound(self):
        self.assertEqual(
            select_source(
                False, False, True, True, True, True, True,
                glove_requires_controller=True),
            "nav")

    def test_controller_bound_glove_still_works_in_controller_mode(self):
        self.assertEqual(
            select_source(
                False, True, True, True, True, True, True,
                glove_requires_controller=True),
            "glove")

    def test_stale_inputs_stop(self):
        self.assertIsNone(select_source(False, False, False, False, False, False, False))
        self.assertIsNone(select_source(True, False, False, False, False, False, False))


if __name__ == "__main__":
    unittest.main()
