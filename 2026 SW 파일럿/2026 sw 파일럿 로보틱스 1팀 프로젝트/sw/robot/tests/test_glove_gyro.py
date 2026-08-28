import unittest

from ros2_bridge.glove_gyro import (
    Orientation,
    average_orientation,
    orientation_to_velocity,
    parse_orientation,
    shortest_angle_delta,
)


class GloveGyroParserTest(unittest.TestCase):
    def test_parses_json(self):
        self.assertEqual(
            parse_orientation('{"pitch": 12.5, "roll": -8, "enabled": true}'),
            Orientation(pitch=12.5, roll=-8.0, enabled=True))

    def test_parses_named_text(self):
        self.assertEqual(
            parse_orientation('Pitch: -12.5, Roll=8.0, Yaw: 0'),
            Orientation(pitch=-12.5, roll=8.0, yaw=0.0))

    def test_parses_esp32_test_bundle_message(self):
        self.assertEqual(
            parse_orientation(
                "ROLL: -6.50 | CTRL_ROLL: 6.50 | PITCH: -18.25 | "
                "STATE:FORWARD_RIGHT"),
            Orientation(pitch=-18.25, roll=6.5))

    def test_parses_csv_and_can_swap_order(self):
        self.assertEqual(parse_orientation('12.5,-8,3'), Orientation(12.5, -8.0, 3.0))
        self.assertEqual(
            parse_orientation('-8,12.5', ('roll', 'pitch')),
            Orientation(12.5, -8.0))

    def test_rejects_incomplete_or_non_numeric_data(self):
        self.assertIsNone(parse_orientation('{"pitch": 1}'))
        self.assertIsNone(parse_orientation('hello'))
        self.assertIsNone(parse_orientation('1,nope'))


class GloveGyroMappingTest(unittest.TestCase):
    def test_neutral_is_stop(self):
        velocity = orientation_to_velocity(Orientation(1, -2), Orientation(1, -2))
        self.assertEqual(velocity.linear_x, 0.0)
        self.assertEqual(velocity.angular_z, 0.0)

    def test_forward_and_right_tilt(self):
        velocity = orientation_to_velocity(
            Orientation(30, 30), Orientation(0, 0), deadzone_deg=10,
            linear_per_degree=0.01, angular_per_degree=0.02,
            max_linear=0.3, max_angular=0.7)
        self.assertAlmostEqual(velocity.linear_x, 0.2)
        self.assertAlmostEqual(velocity.angular_z, -0.4)

    def test_speed_is_capped(self):
        velocity = orientation_to_velocity(
            Orientation(100, -100), Orientation(0, 0), max_linear=0.1,
            max_angular=0.2)
        self.assertEqual(velocity.linear_x, 0.1)
        self.assertEqual(velocity.angular_z, 0.2)

    def test_angle_wrap_and_calibration_average(self):
        self.assertAlmostEqual(shortest_angle_delta(-179, 179), 2.0)
        self.assertEqual(
            average_orientation([Orientation(10, -2), Orientation(14, 2)]),
            Orientation(12.0, 0.0))


if __name__ == '__main__':
    unittest.main()
