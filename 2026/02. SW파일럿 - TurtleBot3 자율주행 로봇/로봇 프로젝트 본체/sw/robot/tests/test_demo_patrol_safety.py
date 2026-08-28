"""시연용 Nav2+Safety 무한 순찰 구성의 회귀 테스트."""

import os
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ros2_bridge.cmd_vel_mux_demo_safety import DemoSafetyCmdVelMux
from ros2_bridge.waypoint_patrol_demo_safety import (
    DEMO_AUTO_PHASE,
    DEMO_ROUTE,
    DEMO_WAYPOINTS,
    DemoSafetyPatrolMission,
    demo_arrival_validation_reason,
    demo_path_reaches_waypoint,
)


class DemoPatrolRouteTest(unittest.TestCase):
    def test_route_repeats_b_c_d_a(self):
        self.assertEqual(DEMO_ROUTE, ("B", "C", "D", "A"))
        repeated = tuple(DEMO_ROUTE[index % len(DEMO_ROUTE)] for index in range(10))
        self.assertEqual(
            repeated,
            ("B", "C", "D", "A", "B", "C", "D", "A", "B", "C"),
        )

    def test_every_leg_uses_the_existing_auto_safety_phase(self):
        for route_index, waypoint_name in enumerate(DEMO_ROUTE):
            mission = SimpleNamespace(
                _demo_halted=False,
                _route_index=route_index,
                _checkpoints={name: "대기" for name in DEMO_WAYPOINTS},
                _set_phase=mock.Mock(),
                _send_nav_goal=mock.Mock(),
            )

            DemoSafetyPatrolMission._start_current_leg(mission)

            self.assertEqual(
                mission._set_phase.call_args.args[0], DEMO_AUTO_PHASE)
            self.assertEqual(
                mission._set_phase.call_args.args[2], waypoint_name)
            mission._send_nav_goal.assert_called_once_with(waypoint_name)

    def test_arrival_validation_accepts_each_copied_waypoint(self):
        for waypoint_name, (x, y, _yaw) in DEMO_WAYPOINTS.items():
            self.assertIsNone(demo_arrival_validation_reason(
                waypoint_name,
                (x, y),
                pose_age_sec=0.0,
                covariance_xy=0.01,
                safety_state="monitoring",
                scan_age_sec=0.0,
            ))

    def test_recovery_path_must_end_at_demo_target(self):
        path = Path()
        start = PoseStamped()
        start.pose.position.x = 0.2
        start.pose.position.y = 0.1
        endpoint = PoseStamped()
        endpoint.pose.position.x = DEMO_WAYPOINTS["C"][0]
        endpoint.pose.position.y = DEMO_WAYPOINTS["C"][1]
        path.poses = [start, endpoint]

        self.assertTrue(demo_path_reaches_waypoint(
            path, "C", expected_start=(0.2, 0.1, 0.0)))
        self.assertFalse(demo_path_reaches_waypoint(
            path, "D", expected_start=(0.2, 0.1, 0.0)))


class DemoMuxGateTest(unittest.TestCase):
    def test_nav_is_allowed_only_for_running_status(self):
        mux = SimpleNamespace(
            _demo_nav_allowed=False,
            _nav_sec=10.0,
            _stop_cycles=0,
            get_logger=mock.Mock(return_value=mock.Mock()),
        )
        status = String()
        status.data = "navigating"

        DemoSafetyCmdVelMux._on_demo_nav_status(mux, status)

        self.assertTrue(mux._demo_nav_allowed)
        self.assertIsNone(mux._nav_sec)
        self.assertEqual(mux._stop_cycles, 1)

        mux._nav_sec = 11.0
        status.data = "failed"
        DemoSafetyCmdVelMux._on_demo_nav_status(mux, status)

        self.assertFalse(mux._demo_nav_allowed)
        self.assertIsNone(mux._nav_sec)
        self.assertEqual(mux._stop_cycles, 1)


if __name__ == "__main__":
    unittest.main()
