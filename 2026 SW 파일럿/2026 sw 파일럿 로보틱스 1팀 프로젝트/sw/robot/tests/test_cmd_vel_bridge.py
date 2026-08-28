import math
import unittest
from types import SimpleNamespace
from unittest import mock

from ros2_bridge.cmd_vel_bridge import (
    CmdVelBridge, GESTURE_INACTIVITY_TIMEOUT_SEC, MODE_OFF_HOLD_SEC, STEP_ANGLE_RAD,
    STEP_DISTANCE_M, STEP_FIST_CANCEL_HOLD_SEC, _angle_distance,
    _gesture_inactivity_expired, _yaw_from_quaternion,
)


class StepGeometryTest(unittest.TestCase):
    @staticmethod
    def _controller_bridge(shape):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "age_sec": 0.0,
            "hand_detected": True,
            "shape": shape,
            "zone": "center",
        }
        logger = SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        return SimpleNamespace(
            _cmd_url="http://127.0.0.1:5000/cmd",
            _request_timeout_sec=0.15,
            _command_stale_sec=0.5,
            _session=SimpleNamespace(get=mock.Mock(return_value=response)),
            _mission_auto_locked=False,
            _mission_hold_locked=False,
            _gesture_mode_enabled=False,
            _joystick_enabled=True,
            _joystick_mode_enabled=True,
            _last_gesture_input_sec=None,
            _mode_on_hold_start_sec=None,
            _mode_on_off_latched=False,
            _joystick_toggle_latched=False,
            _joystick_off_hold_start_sec=None,
            _step_exclusive=True,
            _step_latched=True,
            _cancel_step=mock.Mock(),
            _publish_mode=mock.Mock(),
            _publish_joystick_mode=mock.Mock(),
            _switch_to_controller_if_gesture_inactive=mock.Mock(
                return_value=False),
            get_logger=lambda: logger,
        )

    def test_default_forward_and_reverse_distance_is_one_meter(self):
        self.assertEqual(STEP_DISTANCE_M, 1.0)

    def test_default_turn_is_ninety_degrees(self):
        self.assertAlmostEqual(STEP_ANGLE_RAD, math.pi / 2.0, places=6)

    def test_emergency_fist_requires_short_deliberate_hold(self):
        self.assertEqual(STEP_FIST_CANCEL_HOLD_SEC, 0.25)

    def test_manual_mode_off_hold_is_one_and_a_half_seconds(self):
        self.assertEqual(MODE_OFF_HOLD_SEC, 1.5)

    def test_auto_lock_ignores_all_hand_input_during_a_to_b(self):
        bridge = SimpleNamespace(
            _mission_auto_locked=True,
            _mission_hold_locked=False,
        )

        self.assertIsNone(CmdVelBridge._resolve_twist(bridge))

    def test_gesture_inactivity_switches_at_fifteen_seconds(self):
        self.assertEqual(GESTURE_INACTIVITY_TIMEOUT_SEC, 15.0)
        self.assertFalse(_gesture_inactivity_expired(10.0, 24.999, 15.0))
        self.assertTrue(_gesture_inactivity_expired(10.0, 25.0, 15.0))

    def test_zero_gesture_inactivity_timeout_disables_switch(self):
        self.assertFalse(_gesture_inactivity_expired(10.0, 100.0, 0.0))

    def test_expired_gesture_inactivity_stops_and_enables_controller(self):
        logger = SimpleNamespace(info=mock.Mock())
        bridge = SimpleNamespace(
            _joystick_enabled=True,
            _gesture_mode_enabled=True,
            _joystick_mode_enabled=False,
            _mission_auto_locked=False,
            _mission_hold_locked=False,
            _step_zone=None,
            _nav_active=False,
            _last_gesture_input_sec=10.0,
            _gesture_inactivity_timeout_sec=15.0,
            _cancel_step=mock.Mock(),
            _step_exclusive=True,
            _step_latched=True,
            _mode_on_hold_start_sec=12.0,
            _mode_on_off_latched=True,
            _joystick_toggle_latched=True,
            _joystick_off_hold_start_sec=12.0,
            _publish_mode=mock.Mock(),
            _publish_joystick_mode=mock.Mock(),
            get_logger=lambda: logger,
        )

        switched = CmdVelBridge._switch_to_controller_if_gesture_inactive(
            bridge, 25.0)

        self.assertTrue(switched)
        self.assertFalse(bridge._gesture_mode_enabled)
        self.assertTrue(bridge._joystick_mode_enabled)
        self.assertIsNone(bridge._last_gesture_input_sec)
        bridge._publish_mode.assert_called_once_with()
        bridge._publish_joystick_mode.assert_called_once_with()

    def test_active_one_shot_defers_gesture_inactivity_switch(self):
        bridge = SimpleNamespace(
            _joystick_enabled=True,
            _gesture_mode_enabled=True,
            _joystick_mode_enabled=False,
            _mission_auto_locked=False,
            _mission_hold_locked=False,
            _step_zone="up",
            _nav_active=False,
            _last_gesture_input_sec=10.0,
            _gesture_inactivity_timeout_sec=15.0,
        )

        switched = CmdVelBridge._switch_to_controller_if_gesture_inactive(
            bridge, 30.0)

        self.assertFalse(switched)
        self.assertTrue(bridge._gesture_mode_enabled)
        self.assertFalse(bridge._joystick_mode_enabled)

    def test_controller_ok_hold_returns_to_auto(self):
        bridge = self._controller_bridge("joystick_toggle")

        with mock.patch(
                "ros2_bridge.cmd_vel_bridge.time.monotonic",
                return_value=10.0):
            first_output = CmdVelBridge._resolve_twist(bridge)

        self.assertIsNone(first_output)
        self.assertTrue(bridge._joystick_mode_enabled)
        self.assertEqual(bridge._joystick_off_hold_start_sec, 10.0)

        with mock.patch(
                "ros2_bridge.cmd_vel_bridge.time.monotonic",
                return_value=11.5):
            second_output = CmdVelBridge._resolve_twist(bridge)

        self.assertIsNotNone(second_output)
        self.assertFalse(bridge._joystick_mode_enabled)
        self.assertFalse(bridge._gesture_mode_enabled)
        self.assertTrue(bridge._joystick_toggle_latched)
        bridge._publish_joystick_mode.assert_called_once_with()
        bridge._publish_mode.assert_called_once_with()

    def test_controller_non_ok_gestures_do_not_release_to_auto(self):
        for shape in ("mode_on", "mode_off"):
            with self.subTest(shape=shape):
                bridge = self._controller_bridge(shape)

                with mock.patch(
                        "ros2_bridge.cmd_vel_bridge.time.monotonic",
                        return_value=10.0):
                    output = CmdVelBridge._resolve_twist(bridge)

                self.assertIsNone(output)
                self.assertTrue(bridge._joystick_mode_enabled)
                bridge._publish_joystick_mode.assert_not_called()
                bridge._publish_mode.assert_not_called()

    def test_angle_distance_wraps_across_pi(self):
        self.assertAlmostEqual(
            _angle_distance(math.radians(179), math.radians(-179)),
            math.radians(2), places=6)

    def test_yaw_from_quaternion(self):
        yaw = math.radians(30)
        orientation = SimpleNamespace(
            x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))
        self.assertAlmostEqual(_yaw_from_quaternion(orientation), yaw, places=6)


if __name__ == "__main__":
    unittest.main()
