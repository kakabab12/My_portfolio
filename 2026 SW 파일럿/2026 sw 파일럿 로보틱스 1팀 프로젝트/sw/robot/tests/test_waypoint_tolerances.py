"""A-B-C-D-A 순찰 waypoint 도착 허용오차의 단위 테스트."""
import importlib.util
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ros2_bridge.waypoint_handoff_mission import (
    ARRIVAL_RADIUS_M, INITIAL_POSE_WAIT_TIMEOUT_SEC,
    GoalStatus,
    START_POSE_MIN_ENDPOINTS, START_POSE_MIN_MATCH_RATIO,
    WaypointHandoffMission, scan_map_match_stats,
)


def _load_navigation_launch_module():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ros2_bridge", "navigation_with_mux.launch.py")
    spec = importlib.util.spec_from_file_location("navigation_with_mux_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WaypointToleranceTest(unittest.TestCase):
    def test_b_arrival_waits_for_user_gesture_in_auto_mode(self):
        mission = SimpleNamespace(
            _active_goal_name="B",
            _phase="auto_to_b",
            _checkpoints={"B": "Nav2 이동 중"},
            _set_phase=mock.Mock(),
            _publish_mode_command=mock.Mock(),
            _halt=mock.Mock(),
        )
        future = SimpleNamespace(
            result=lambda: SimpleNamespace(
                status=GoalStatus.STATUS_SUCCEEDED))

        WaypointHandoffMission._on_nav_result(mission, "B", future)

        self.assertEqual(mission._checkpoints["B"], "도착 완료")
        mission._set_phase.assert_called_once_with(
            "wait_gesture",
            "B 도착 완료 — 짧은 따봉으로 제스처 모드를 켜세요.",
            "C")
        mission._publish_mode_command.assert_called_once_with("auto")
        mission._halt.assert_not_called()

    def test_new_thumbs_up_after_b_starts_gesture_leg(self):
        mission = SimpleNamespace(
            _phase="wait_gesture",
            _gesture_mode=False,
            _set_phase=mock.Mock(),
            _maybe_start_return_to_a=mock.Mock(),
        )

        WaypointHandoffMission._on_gesture_mode(
            mission, SimpleNamespace(data=True))

        mission._set_phase.assert_called_once_with(
            "gesture_to_c",
            "제스처 모드 ON — C 지점으로 이동하세요.",
            "C")
        mission._maybe_start_return_to_a.assert_called_once_with()

    def test_gesture_off_at_b_keeps_waiting(self):
        mission = SimpleNamespace(
            _phase="wait_gesture",
            _gesture_mode=False,
            _set_phase=mock.Mock(),
            _maybe_start_return_to_a=mock.Mock(),
        )

        WaypointHandoffMission._on_gesture_mode(
            mission, SimpleNamespace(data=False))

        mission._set_phase.assert_not_called()

    def test_initial_pose_waits_long_enough_for_amcl_activation(self):
        self.assertEqual(INITIAL_POSE_WAIT_TIMEOUT_SEC, 15.0)

    def test_manual_waypoints_accept_positions_within_ten_centimeters(self):
        self.assertEqual(ARRIVAL_RADIUS_M, 0.10)

    def test_controller_transition_forces_c_arrival_without_position_check(self):
        logger = mock.Mock()
        mission = SimpleNamespace(
            _phase="gesture_to_c",
            _joystick_mode=False,
            _complete_manual_arrival=mock.Mock(),
            _maybe_start_return_to_a=mock.Mock(),
            get_logger=mock.Mock(return_value=logger),
        )

        WaypointHandoffMission._on_joystick_mode(
            mission, SimpleNamespace(data=True))

        mission._complete_manual_arrival.assert_called_once_with("C")
        mission._maybe_start_return_to_a.assert_called_once_with()

    def test_auto_transition_forces_d_arrival_before_return_to_a(self):
        logger = mock.Mock()
        def set_phase(phase, _label, _target):
            mission._phase = phase

        mission = SimpleNamespace(
            _phase="controller_to_d",
            _gesture_mode=False,
            _joystick_mode=False,
            _arrival_started_sec=None,
            _checkpoints={"A": "대기", "D": "이동 중"},
            _set_phase=mock.Mock(side_effect=set_phase),
            _send_nav_goal=mock.Mock(),
            get_logger=mock.Mock(return_value=logger),
        )
        mission._complete_manual_arrival = mock.Mock(
            side_effect=lambda waypoint: WaypointHandoffMission._complete_manual_arrival(
                mission, waypoint))

        WaypointHandoffMission._maybe_start_return_to_a(mission)

        mission._complete_manual_arrival.assert_called_once_with("D")
        self.assertEqual(mission._checkpoints["D"], "도착 완료")
        mission._send_nav_goal.assert_called_once_with("A")

    def test_controller_fallback_at_c_continues_directly_to_d(self):
        mission = SimpleNamespace(
            _arrival_started_sec=1.0,
            _checkpoints={"C": "이동 중"},
            _joystick_mode=True,
            _set_phase=mock.Mock(),
        )

        WaypointHandoffMission._complete_manual_arrival(mission, "C")

        self.assertEqual(mission._checkpoints["C"], "도착 완료")
        mission._set_phase.assert_called_once_with(
            "controller_to_d",
            "C 도착 완료 — 현재 컨트롤러 모드로 D 지점 이동을 계속하세요.",
            "D")

    def test_nav2_waypoints_use_same_ten_centimeter_radius_without_yaw_requirement(self):
        # 실제 launch가 만드는 RewrittenYaml까지 열어, Humble의
        # general_goal_checker와 DWB 양쪽에 설정이 들어가는지 검증한다.
        with tempfile.TemporaryDirectory() as log_dir:
            with mock.patch.dict(os.environ, {"ROS_LOG_DIR": log_dir}):
                navigation_launch = _load_navigation_launch_module()
                from launch import LaunchContext
                from launch.actions import IncludeLaunchDescription

                launch_description = navigation_launch.generate_launch_description()
                include = next(
                    entity for entity in launch_description.entities
                    if isinstance(entity, IncludeLaunchDescription))
                params_path = dict(include.launch_arguments)["params_file"].perform(
                    LaunchContext())
        with open(params_path, encoding="utf-8") as params_file:
            all_params = yaml.safe_load(params_file)
            params = all_params["controller_server"]["ros__parameters"]

        self.assertEqual(navigation_launch.WAYPOINT_XY_GOAL_TOLERANCE_M, 0.10)
        self.assertEqual(params["general_goal_checker"]["xy_goal_tolerance"], 0.10)
        self.assertGreater(params["general_goal_checker"]["yaw_goal_tolerance"], 3.14159)
        self.assertFalse(params["general_goal_checker"]["stateful"])
        self.assertEqual(params["FollowPath"]["xy_goal_tolerance"], 0.10)
        self.assertFalse(params["FollowPath"]["stateful"])

        planner_params = all_params["planner_server"]["ros__parameters"]
        self.assertEqual(navigation_launch.NAVFN_GOAL_TOLERANCE_M, 0.10)
        self.assertEqual(planner_params["GridBased"]["tolerance"], 0.10)

        navigator_params = all_params["bt_navigator"]["ros__parameters"]
        self.assertEqual(
            navigator_params["default_bt_xml_filename"],
            navigation_launch.SAFE_NAV_TO_POSE_BT_XML)
        with open(
                navigation_launch.SAFE_NAV_TO_POSE_BT_XML,
                encoding="utf-8") as behavior_tree:
            behavior_tree_text = behavior_tree.read()
        self.assertNotIn("<Spin", behavior_tree_text)
        self.assertNotIn("<BackUp", behavior_tree_text)

        global_costmap = all_params["global_costmap"]["global_costmap"][
            "ros__parameters"]
        self.assertFalse(global_costmap["obstacle_layer"]["enabled"])
        self.assertFalse(global_costmap["voxel_layer"]["enabled"])

        local_costmap = all_params["local_costmap"]["local_costmap"][
            "ros__parameters"]
        self.assertFalse(local_costmap["obstacle_layer"]["enabled"])
        self.assertTrue(local_costmap["voxel_layer"]["enabled"])

    def test_start_pose_scan_map_match_distinguishes_aligned_endpoint(self):
        origin = SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        info = SimpleNamespace(resolution=0.1, width=20, height=20, origin=origin)
        data = [0] * (info.width * info.height)
        data[10 * info.width + 10] = 100
        occupancy_grid = SimpleNamespace(info=info, data=data)
        scan = SimpleNamespace(
            ranges=[0.5], range_min=0.05, range_max=8.0,
            angle_min=0.0, angle_increment=0.1,
        )

        matched = scan_map_match_stats(
            scan, occupancy_grid, (0.5, 1.0, 0.0),
            static_margin=0.0, sample_step=1)
        mismatched = scan_map_match_stats(
            scan, occupancy_grid, (0.5, 0.5, 0.0),
            static_margin=0.0, sample_step=1)

        self.assertEqual(matched, (1, 1, 1.0))
        self.assertEqual(mismatched, (0, 1, 0.0))
        self.assertGreater(START_POSE_MIN_ENDPOINTS, 1)
        self.assertGreater(START_POSE_MIN_MATCH_RATIO, 0.5)


if __name__ == "__main__":
    unittest.main()
