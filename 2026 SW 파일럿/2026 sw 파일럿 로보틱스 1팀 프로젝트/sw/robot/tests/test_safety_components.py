"""Safety 전용 mux와 갑작스러운 장애물 판정의 회귀 테스트."""

import math
import os
import sys
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
import rclpy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFETY_PACKAGE_ROOT = os.path.join(
    os.path.dirname(PROJECT_ROOT),
    "..",
    "turtlebot3_ws",
    "src",
    "turtlebot3_waypoint_patrol",
)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.abspath(SAFETY_PACKAGE_ROOT))

from ros2_bridge.cmd_vel_mux_safety import select_source_safety
from ros2_bridge.waypoint_handoff_mission_safety import (
    SafetyWaypointHandoffMission,
    nav_arrival_validation_reason,
)
from ros2_bridge.waypoint_handoff_mission import (
    NAV_GOAL_RESPONSE_TIMEOUT_SEC,
    WAYPOINTS,
    WaypointHandoffMission,
    claim_goal_response,
)
from turtlebot3_waypoint_patrol.safety_mission_manager import (
    SafetyMissionManager,
    SafetyState,
    backward_progress,
    count_matching_points,
    count_novel_points,
    dynamic_scan_points,
    is_new_dynamic_cluster_seed,
    is_sudden_obstacle_seed,
    map_point_is_static,
    points_in_odom,
    points_in_sector,
)
from ros2_bridge.waypoint_handoff_mission_safety import (
    path_reaches_waypoint,
)


class SafetyMuxSelectionTest(unittest.TestCase):
    def _select(
            self, enabled=True, heartbeat=True,
            active=False, safety_fresh=False):
        return select_source_safety(
            enabled, heartbeat, active, safety_fresh,
            False, False, False,
            True, True, True, True,
        )

    def test_normal_safety_state_preserves_nav_selection(self):
        self.assertEqual(self._select(), "nav")

    def test_active_safety_has_absolute_priority(self):
        self.assertEqual(
            self._select(active=True, safety_fresh=True), "safety")

    def test_stale_safety_command_stops_instead_of_falling_back(self):
        self.assertIsNone(
            self._select(active=True, safety_fresh=False))

    def test_missing_heartbeat_stops_all_sources(self):
        self.assertIsNone(self._select(heartbeat=False))

    def test_manual_gesture_ignores_missing_safety_heartbeat(self):
        self.assertEqual(
            select_source_safety(
                False, False, True, False,
                True, False, False,
                False, True, False, False,
            ),
            "gesture",
        )

    def test_manual_gesture_ignores_stale_safety_active_command(self):
        self.assertEqual(
            select_source_safety(
                False, True, True, False,
                True, False, False,
                False, True, False, False,
            ),
            "gesture",
        )


class SuddenObstacleClassifierTest(unittest.TestCase):
    def _seed(self, **overrides):
        arguments = {
            "have_baseline": True,
            "previous_front_distance": 0.65,
            "current_front_distance": 0.25,
            "close_point_count": 5,
            "novel_point_count": 5,
            "forward_speed": 0.10,
            "angular_speed": 0.0,
            "trigger_distance": 0.30,
            "minimum_distance_drop": 0.10,
            "minimum_points": 3,
            "minimum_forward_speed": 0.02,
            "maximum_angular_speed": 0.30,
        }
        arguments.update(overrides)
        return is_sudden_obstacle_seed(**arguments)

    def test_new_close_cluster_while_driving_forward_is_candidate(self):
        self.assertTrue(self._seed())

    def test_preexisting_or_gradually_approached_obstacle_is_not_candidate(self):
        self.assertFalse(self._seed(
            previous_front_distance=0.29,
            current_front_distance=0.25))

    def test_first_scan_cannot_claim_sudden_appearance(self):
        self.assertFalse(self._seed(have_baseline=False))

    def test_stationary_robot_does_not_start_special_recovery(self):
        self.assertFalse(self._seed(forward_speed=0.0))

    def test_turning_robot_does_not_start_special_recovery(self):
        self.assertFalse(self._seed(angular_speed=0.45))

    def test_scan_noise_without_cluster_is_rejected(self):
        self.assertFalse(self._seed(close_point_count=1, novel_point_count=1))

    def test_new_map_filtered_cluster_can_seed_without_range_jump(self):
        self.assertTrue(is_new_dynamic_cluster_seed(
            have_baseline=True,
            previous_dynamic_point_count=0,
            current_dynamic_point_count=5,
            current_front_distance=0.40,
            forward_speed=0.10,
            angular_speed=0.0,
            trigger_distance=0.45,
            minimum_points=3,
            minimum_forward_speed=0.02,
            maximum_angular_speed=0.30,
        ))

    def test_existing_dynamic_cluster_is_not_a_new_onset(self):
        self.assertFalse(is_new_dynamic_cluster_seed(
            have_baseline=True,
            previous_dynamic_point_count=5,
            current_dynamic_point_count=5,
            current_front_distance=0.40,
            forward_speed=0.10,
            angular_speed=0.0,
            trigger_distance=0.45,
            minimum_points=3,
            minimum_forward_speed=0.02,
            maximum_angular_speed=0.30,
        ))

    def test_four_point_wall_fragment_is_too_small_for_surprise_recovery(self):
        self.assertFalse(is_new_dynamic_cluster_seed(
            have_baseline=True,
            previous_dynamic_point_count=0,
            current_dynamic_point_count=4,
            current_front_distance=0.30,
            forward_speed=0.10,
            angular_speed=0.0,
            trigger_distance=0.35,
            minimum_points=6,
            minimum_forward_speed=0.02,
            maximum_angular_speed=0.30,
        ))

    def test_six_point_wall_fragment_and_goal_creep_are_not_recovery_events(self):
        self.assertFalse(is_new_dynamic_cluster_seed(
            have_baseline=True,
            previous_dynamic_point_count=0,
            current_dynamic_point_count=6,
            current_front_distance=0.31,
            forward_speed=0.046,
            angular_speed=0.128,
            trigger_distance=0.35,
            minimum_points=8,
            minimum_forward_speed=0.06,
            maximum_angular_speed=0.30,
        ))

    def test_odom_projection_matches_static_wall_after_base_translation(self):
        previous = points_in_odom(
            [(0.0, 0.50)], odom_x=0.0, odom_y=0.0, odom_yaw=0.0)
        current = points_in_odom(
            [(0.0, 0.45)], odom_x=0.05, odom_y=0.0, odom_yaw=0.0)
        self.assertEqual(count_novel_points(current, previous, 0.02), 0)

    def test_retreat_is_measured_from_dynamic_obstacle_event_pose(self):
        # Progress is relative to the obstacle event, not an absolute odom
        # coordinate, and works for arbitrary event positions and headings.
        self.assertAlmostEqual(
            backward_progress(0.0, 0.0, 0.0, -0.05, 0.0), 0.05)
        self.assertAlmostEqual(
            backward_progress(1.0, 2.0, math.pi / 2.0, 1.0, 1.75), 0.25)

    def test_odom_projection_matches_static_wall_after_base_rotation(self):
        previous = points_in_odom(
            [(math.pi / 6.0, 0.50)],
            odom_x=0.0, odom_y=0.0, odom_yaw=0.0)
        current = points_in_odom(
            [(0.0, 0.50)],
            odom_x=0.0, odom_y=0.0, odom_yaw=math.pi / 6.0)
        self.assertEqual(count_novel_points(current, previous, 0.02), 0)

    def test_new_world_point_is_novel(self):
        self.assertEqual(
            count_novel_points([(0.20, 0.0)], [(0.60, 0.0)], 0.06), 1)

    def test_only_original_obstacle_cluster_points_match_during_clear_wait(self):
        reference = [(0.30, -0.02), (0.30, 0.00), (0.30, 0.02)]
        current = [(0.31, -0.02), (0.31, 0.01), (0.55, 0.00)]
        self.assertEqual(
            count_matching_points(current, reference, 0.08),
            2,
        )

    def test_recent_nav_forward_intent_survives_immediate_zero_command(self):
        manager = SimpleNamespace(
            _forward_speed=0.0,
            _last_forward_intent_sec=10.0,
            _last_forward_intent_speed=0.12,
            forward_intent_hold_sec=1.0,
        )
        self.assertEqual(
            SafetyMissionManager._effective_forward_intent(manager, 10.5),
            0.12,
        )
        self.assertEqual(
            SafetyMissionManager._effective_forward_intent(manager, 11.1),
            0.0,
        )

    @staticmethod
    def _map():
        occupancy_grid = OccupancyGrid()
        occupancy_grid.info.resolution = 0.05
        occupancy_grid.info.width = 20
        occupancy_grid.info.height = 20
        occupancy_grid.info.origin.position.x = 0.0
        occupancy_grid.info.origin.position.y = -0.50
        occupancy_grid.info.origin.orientation.w = 1.0
        occupancy_grid.data = [0] * 400
        return occupancy_grid

    def test_static_map_wall_is_removed_from_surprise_candidates(self):
        occupancy_grid = self._map()
        # base=(0,0), scan endpoint=(0.50,0) -> grid cell (10,10)
        occupancy_grid.data[10 * 20 + 10] = 100
        dynamic = dynamic_scan_points(
            [(0.0, 0.50)],
            sensor_x=0.0,
            sensor_y=0.0,
            sensor_yaw=0.0,
            occupancy_grid=occupancy_grid,
            static_margin=0.0,
            occupied_threshold=65,
        )
        self.assertEqual(dynamic, [])

    def test_unmapped_temporary_obstacle_remains_a_candidate(self):
        occupancy_grid = self._map()
        dynamic = dynamic_scan_points(
            [(0.0, 0.25)],
            sensor_x=0.0,
            sensor_y=0.0,
            sensor_yaw=0.0,
            occupancy_grid=occupancy_grid,
            static_margin=0.0,
            occupied_threshold=65,
        )
        self.assertEqual(dynamic, [(0.0, 0.25)])

    def test_unknown_or_out_of_map_point_is_not_auto_recovered(self):
        occupancy_grid = self._map()
        occupancy_grid.data[10 * 20 + 10] = -1
        self.assertTrue(map_point_is_static(
            occupancy_grid, 0.50, 0.0, 0.0, 65))
        self.assertTrue(map_point_is_static(
            occupancy_grid, 2.00, 0.0, 0.0, 65))


class SafetyRuntimeTriggerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    @staticmethod
    def _free_map():
        occupancy_grid = OccupancyGrid()
        occupancy_grid.info.resolution = 0.05
        occupancy_grid.info.width = 100
        occupancy_grid.info.height = 100
        occupancy_grid.info.origin.position.x = -2.5
        occupancy_grid.info.origin.position.y = -2.5
        occupancy_grid.info.origin.orientation.w = 1.0
        occupancy_grid.data = [0] * 10000
        return occupancy_grid

    @staticmethod
    def _scan(front_distance=None, side_distance=None, front_point_count=9):
        scan = LaserScan()
        scan.header.frame_id = "base_scan"
        scan.angle_min = -math.pi
        scan.angle_increment = math.radians(1.0)
        scan.range_min = 0.05
        scan.range_max = 12.0
        scan.ranges = [math.inf] * 360
        if front_distance is not None:
            first_index = 180 - int(front_point_count) // 2
            for index in range(first_index, first_index + int(front_point_count)):
                scan.ranges[index] = float(front_distance)
        if side_distance is not None:
            # +25 degrees: inside the detection sector (±30 degrees), but
            # outside the central resume corridor (±15 degrees).
            for index in range(203, 208):
                scan.ranges[index] = float(side_distance)
        return scan

    @staticmethod
    def _odometry(x=0.0, y=0.0, yaw=0.0, linear_x=0.0):
        odometry = Odometry()
        odometry.pose.pose.position.x = float(x)
        odometry.pose.pose.position.y = float(y)
        odometry.pose.pose.orientation.z = math.sin(yaw / 2.0)
        odometry.pose.pose.orientation.w = math.cos(yaw / 2.0)
        odometry.twist.twist.linear.x = float(linear_x)
        return odometry

    @staticmethod
    def _use_exact_scan_tf(node, x=0.0, y=0.0, yaw=0.0):
        node._lookup_scan_pose_in_map = mock.Mock(
            return_value=(float(x), float(y), float(yaw)))

    def test_scan_map_lookup_uses_laser_timestamp_not_latest_pose(self):
        node = SafetyMissionManager()
        try:
            transform = SimpleNamespace(
                transform=SimpleNamespace(
                    translation=SimpleNamespace(x=1.0, y=2.0),
                    rotation=SimpleNamespace(
                        x=0.0,
                        y=0.0,
                        z=math.sin(0.25),
                        w=math.cos(0.25),
                    ),
                ))
            node._tf_buffer = mock.Mock()
            node._tf_buffer.lookup_transform.return_value = transform
            scan = self._scan()
            scan.header.stamp.sec = 12
            scan.header.stamp.nanosec = 345

            pose = node._lookup_scan_pose_in_map(scan)

            self.assertEqual(pose[:2], (1.0, 2.0))
            self.assertAlmostEqual(pose[2], 0.5)
            arguments = node._tf_buffer.lookup_transform.call_args.args
            keyword_arguments = (
                node._tf_buffer.lookup_transform.call_args.kwargs)
            self.assertEqual(arguments[:2], ("map", "base_scan"))
            self.assertEqual(arguments[2].nanoseconds, 12_000_000_345)
            self.assertEqual(
                keyword_arguments["timeout"].nanoseconds,
                80_000_000,
            )
        finally:
            node.destroy_node()

    def test_hard_stop_is_published_before_waiting_for_scan_tf(self):
        node = SafetyMissionManager()
        try:
            node._map_callback(self._free_map())
            node._odom_callback(self._odometry())

            phase = String()
            phase.data = "auto_to_b"
            node._phase_callback(phase)
            nav_status = String()
            nav_status.data = "navigating"
            node._nav_status_callback(nav_status)
            node._scan_callback(self._scan())

            node._publish_stop = mock.Mock(wraps=node._publish_stop)

            def assert_stopped_before_tf(_message):
                self.assertEqual(node.state, SafetyState.PROTECTIVE_STOP)
                self.assertTrue(node._active)
                self.assertTrue(node._publish_stop.called)
                return (0.0, 0.0, 0.0)

            node._lookup_scan_pose_in_map = assert_stopped_before_tf
            node._scan_callback(self._scan(front_distance=0.10))

            self.assertEqual(node.state, SafetyState.PROTECTIVE_STOP)
        finally:
            node.destroy_node()

    def test_nav_brake_does_not_hide_new_obstacle_from_state_machine(self):
        node = SafetyMissionManager()
        try:
            node._map_callback(self._free_map())
            self._use_exact_scan_tf(node)

            node._odom_callback(self._odometry())

            phase = String()
            phase.data = "auto_to_b"
            node._phase_callback(phase)
            nav_status = String()
            nav_status.data = "navigating"
            node._nav_status_callback(nav_status)

            node._scan_callback(self._scan())
            nav_command = Twist()
            nav_command.linear.x = 0.12
            node._nav_cmd_callback(nav_command)

            # Measured odometry is already zero, modeling Nav2's immediate
            # local-costmap brake.  The recent forward-intent latch must keep
            # the surprise detector eligible for these two scan frames.
            node._scan_callback(self._scan(front_distance=0.30))
            self.assertEqual(node.state, SafetyState.CANDIDATE)
            node._scan_callback(self._scan(front_distance=0.30))
            self.assertEqual(node.state, SafetyState.WAIT_NAV_CANCEL)

            canceled = String()
            canceled.data = "canceled"
            node._nav_status_callback(canceled)
            self.assertEqual(node.state, SafetyState.REVERSE)

            node._odom_callback(self._odometry(x=-0.05))
            node._control_tick()
            self.assertEqual(node.state, SafetyState.REVERSE)

            node._odom_callback(self._odometry(x=-0.25))
            node._control_tick()
            self.assertEqual(node.state, SafetyState.CLEAR_WAIT)

            # 후진 뒤에도 처음 감지한 같은 장애물이 지도상 같은 위치에
            # 남아 있으면 raw 거리 증가와 무관하게 제거 대기를 유지한다.
            self._use_exact_scan_tf(node, x=-0.25)
            node._scan_callback(self._scan(front_distance=0.55))
            self.assertFalse(node._front_clear)

            node._scan_callback(self._scan())
            node._front_clear_started_sec -= 1.1
            node._control_tick()
            self.assertEqual(node.state, SafetyState.WAIT_NAV_RESUME)

            resumed = String()
            resumed.data = "resumed"
            node._nav_status_callback(resumed)
            self.assertEqual(node.state, SafetyState.COOLDOWN)
            node._state_started_sec -= 1.1
            node._control_tick()
            self.assertEqual(node.state, SafetyState.MONITORING)
        finally:
            node.destroy_node()

    def test_dynamic_cluster_onset_triggers_when_nearest_range_barely_changes(self):
        node = SafetyMissionManager()
        try:
            node._map_callback(self._free_map())
            self._use_exact_scan_tf(node)
            node._odom_callback(self._odometry())

            phase = String()
            phase.data = "auto_to_b"
            node._phase_callback(phase)
            nav_status = String()
            nav_status.data = "navigating"
            node._nav_status_callback(nav_status)
            node._scan_callback(self._scan())

            nav_command = Twist()
            nav_command.linear.x = 0.12
            node._nav_cmd_callback(nav_command)
            # 다른 전방 특징 때문에 최근 최단거리가 이미 0.32m였다고 가정한다.
            # 새 동적 군집은 0.30m라 거리 급감 기준(0.10m)을 충족하지 못하지만,
            # scan-time map-filtered 군집의 0 -> 7 onset으로 후보가 되어야 한다.
            node._previous_front_distance = 0.32
            node._scan_callback(self._scan(front_distance=0.30))
            self.assertEqual(node.state, SafetyState.CANDIDATE)
            node._scan_callback(self._scan(front_distance=0.30))
            self.assertEqual(node.state, SafetyState.WAIT_NAV_CANCEL)
        finally:
            node.destroy_node()

    def test_twenty_point_mapped_wall_never_starts_recovery(self):
        node = SafetyMissionManager()
        try:
            occupancy_grid = self._free_map()
            wall_cell_x = int((0.30 - occupancy_grid.info.origin.position.x)
                              / occupancy_grid.info.resolution)
            wall_cell_y = int((0.00 - occupancy_grid.info.origin.position.y)
                              / occupancy_grid.info.resolution)
            occupancy_grid.data[
                wall_cell_y * occupancy_grid.info.width + wall_cell_x] = 100
            node._map_callback(occupancy_grid)
            self._use_exact_scan_tf(node)
            node._odom_callback(self._odometry())

            phase = String()
            phase.data = "auto_to_b"
            node._phase_callback(phase)
            nav_status = String()
            nav_status.data = "navigating"
            node._nav_status_callback(nav_status)
            node._scan_callback(self._scan())

            nav_command = Twist()
            nav_command.linear.x = 0.12
            node._nav_cmd_callback(nav_command)
            for _ in range(3):
                node._scan_callback(self._scan(
                    front_distance=0.30,
                    front_point_count=20,
                ))

            self.assertEqual(node.state, SafetyState.MONITORING)
        finally:
            node.destroy_node()

    def test_missing_scan_time_tf_disables_reverse_but_keeps_hard_stop(self):
        node = SafetyMissionManager()
        try:
            node._map_callback(self._free_map())
            node._lookup_scan_pose_in_map = mock.Mock(return_value=None)
            node._odom_callback(self._odometry())

            phase = String()
            phase.data = "auto_to_b"
            node._phase_callback(phase)
            nav_status = String()
            nav_status.data = "navigating"
            node._nav_status_callback(nav_status)
            node._scan_callback(self._scan())

            nav_command = Twist()
            nav_command.linear.x = 0.12
            node._nav_cmd_callback(nav_command)
            for _ in range(3):
                node._scan_callback(self._scan(front_distance=0.30))
            self.assertEqual(node.state, SafetyState.MONITORING)

            node._scan_callback(self._scan(front_distance=0.10))
            self.assertEqual(node.state, SafetyState.PROTECTIVE_STOP)
        finally:
            node.destroy_node()

    def test_six_point_wall_fragment_does_not_start_recovery(self):
        node = SafetyMissionManager()
        try:
            node._map_callback(self._free_map())
            self._use_exact_scan_tf(node)
            node._odom_callback(self._odometry())

            phase = String()
            phase.data = "auto_to_b"
            node._phase_callback(phase)
            nav_status = String()
            nav_status.data = "navigating"
            node._nav_status_callback(nav_status)
            node._scan_callback(self._scan())

            nav_command = Twist()
            nav_command.linear.x = 0.12
            node._nav_cmd_callback(nav_command)
            for _ in range(3):
                node._scan_callback(self._scan(
                    front_distance=0.30,
                    front_point_count=6,
                ))

            self.assertEqual(node.state, SafetyState.MONITORING)
        finally:
            node.destroy_node()

    def test_side_wall_outside_narrow_corridor_does_not_start_recovery(self):
        node = SafetyMissionManager()
        try:
            node._map_callback(self._free_map())
            self._use_exact_scan_tf(node)
            node._odom_callback(self._odometry())

            phase = String()
            phase.data = "auto_to_b"
            node._phase_callback(phase)
            nav_status = String()
            nav_status.data = "navigating"
            node._nav_status_callback(nav_status)
            node._scan_callback(self._scan())

            nav_command = Twist()
            nav_command.linear.x = 0.12
            node._nav_cmd_callback(nav_command)
            for _ in range(3):
                node._scan_callback(self._scan(side_distance=0.30))

            self.assertEqual(node.state, SafetyState.MONITORING)
        finally:
            node.destroy_node()

    def test_side_object_keeps_wide_absolute_protective_stop(self):
        node = SafetyMissionManager()
        try:
            phase = String()
            phase.data = "auto_to_b"
            node._phase_callback(phase)
            node._odom_callback(self._odometry())
            node._scan_callback(self._scan(side_distance=0.10))
            self.assertEqual(node.state, SafetyState.PROTECTIVE_STOP)
        finally:
            node.destroy_node()

    def test_manual_phases_never_invoke_safety_velocity_or_scan_logic(self):
        for phase_name in (
                "wait_gesture", "gesture_to_c", "wait_controller", "controller_to_d",
                "wait_auto", "complete"):
            with self.subTest(phase=phase_name):
                node = SafetyMissionManager()
                try:
                    node._odom_callback(self._odometry())
                    phase = String()
                    phase.data = phase_name
                    node._phase_callback(phase)
                    node._publish_stop = mock.Mock()
                    node._lookup_scan_pose_in_map = mock.Mock()

                    node._scan_callback(self._scan(front_distance=0.10))
                    node._last_scan_sec = None
                    node._last_odom_sec = None
                    node._control_tick()

                    self.assertEqual(node.state, SafetyState.MONITORING)
                    self.assertFalse(node._active)
                    node._publish_stop.assert_not_called()
                    node._lookup_scan_pose_in_map.assert_not_called()
                finally:
                    node.destroy_node()

    def test_entering_manual_phase_releases_autonomous_protective_stop(self):
        node = SafetyMissionManager()
        try:
            auto_phase = String()
            auto_phase.data = "auto_to_b"
            node._phase_callback(auto_phase)
            node._odom_callback(self._odometry())
            node._scan_callback(self._scan(front_distance=0.10))
            self.assertEqual(node.state, SafetyState.PROTECTIVE_STOP)
            self.assertTrue(node._active)

            manual_phase = String()
            manual_phase.data = "gesture_to_c"
            node._phase_callback(manual_phase)

            self.assertEqual(node.state, SafetyState.MONITORING)
            self.assertFalse(node._active)
        finally:
            node.destroy_node()

    def test_zero_clear_wait_timeout_waits_for_operator(self):
        node = SafetyMissionManager()
        try:
            phase = String()
            phase.data = "auto_to_b"
            node._phase_callback(phase)
            node.state = SafetyState.CLEAR_WAIT
            node.clear_wait_timeout_sec = 0.0
            node._state_started_sec -= 3600.0
            node._last_scan_sec = time.monotonic()
            node._last_odom_sec = time.monotonic()
            node._front_clear_started_sec = None

            node._control_tick()

            self.assertEqual(node.state, SafetyState.CLEAR_WAIT)
        finally:
            node.destroy_node()

    def test_side_wall_does_not_block_central_resume_corridor(self):
        node = SafetyMissionManager()
        try:
            phase = String()
            phase.data = "auto_to_b"
            node._phase_callback(phase)
            node._odom_callback(self._odometry())
            node.state = SafetyState.CLEAR_WAIT
            node._scan_callback(self._scan(side_distance=0.65))
            self.assertTrue(node._front_clear)

            node._front_clear_started_sec -= 1.1
            node._control_tick()

            self.assertEqual(node.state, SafetyState.WAIT_NAV_RESUME)
        finally:
            node.destroy_node()

    def test_central_obstacle_still_blocks_resume(self):
        node = SafetyMissionManager()
        try:
            phase = String()
            phase.data = "auto_to_b"
            node._phase_callback(phase)
            node._odom_callback(self._odometry())
            node.state = SafetyState.CLEAR_WAIT
            scan = self._scan(front_distance=0.65)
            node._recovery_obstacle_points_map = points_in_odom(
                points_in_sector(
                    scan, 0.0, node.resume_angle,
                    node.resume_clear_distance),
                0.0, 0.0, 0.0)
            self._use_exact_scan_tf(node)
            node._scan_callback(scan)

            self.assertFalse(node._front_clear)
            node._control_tick()
            self.assertEqual(node.state, SafetyState.CLEAR_WAIT)
        finally:
            node.destroy_node()

    def test_unrelated_front_structure_does_not_block_detected_obstacle_clear(self):
        node = SafetyMissionManager()
        try:
            phase = String()
            phase.data = "auto_to_b"
            node._phase_callback(phase)
            node._map_callback(self._free_map())
            node._odom_callback(self._odometry())
            node.state = SafetyState.CLEAR_WAIT
            # 처음 감지한 임시 장애물은 map x=0.30 부근이었다.
            node._recovery_obstacle_points_map = [
                (0.30, -0.03), (0.30, 0.00), (0.30, 0.03)]
            self._use_exact_scan_tf(node)

            # 제거 뒤 map x=0.65에 보이는 다른 구조물은 원래 장애물
            # 군집과 공간적으로 일치하지 않으므로 CLEAR_WAIT를 막지 않는다.
            node._scan_callback(self._scan(
                front_distance=0.65,
                front_point_count=9,
            ))

            self.assertTrue(node._front_clear)
            node._front_clear_started_sec -= 1.1
            node._control_tick()
            self.assertEqual(node.state, SafetyState.WAIT_NAV_RESUME)
        finally:
            node.destroy_node()


class SafetyMissionHandshakeTest(unittest.TestCase):
    @staticmethod
    def _mission(**overrides):
        statuses = []
        goals = []
        mission = SimpleNamespace(
            _phase="auto_to_b",
            _active_goal_name=None,
            _pending_goal_name="B",
            _active_goal_handle=None,
            _safety_cancel_pending=False,
            _safety_paused=False,
            _safety_resuming=False,
            _interrupted_goal_name=None,
            _label="",
            _post_status=lambda force=False: None,
            _publish_nav_status=statuses.append,
            _cancel_active_goal=mock.Mock(),
            _begin_resume_live_pose_wait=goals.append,
            get_logger=lambda: mock.Mock(),
        )
        mission._clear_pending_goal = lambda: setattr(
            mission, "_pending_goal_name", None)
        for name, value in overrides.items():
            setattr(mission, name, value)
        return mission, statuses, goals

    @staticmethod
    def _live_pose_mission(**overrides):
        mission = SimpleNamespace(
            _resume_waiting_for_live_pose=True,
            _resume_live_pose_deadline_sec=20.0,
            _safety_resuming=True,
            _resume_last_live_pose_diagnostic_sec=0.0,
            _lookup_current_map_pose=lambda: (0.51, -0.10, 0.03),
            _resume_expected_pose=None,
            _resume_target_name="B",
            _label="",
            _post_status=lambda force=False: None,
            _publish_nav_status=mock.Mock(),
            _halt=mock.Mock(),
            _begin_resume_costmap_clear=mock.Mock(),
            get_logger=lambda: mock.Mock(),
        )
        for name, value in overrides.items():
            setattr(mission, name, value)
        return mission

    def test_pending_auto_goal_can_be_safely_paused(self):
        mission, statuses, _ = self._mission()
        SafetyWaypointHandoffMission._handle_safety_cancel(mission)
        self.assertEqual(mission._interrupted_goal_name, "B")
        self.assertTrue(mission._safety_paused)
        self.assertFalse(mission._safety_cancel_pending)
        self.assertEqual(statuses, ["canceling", "canceled"])

    def test_missing_goal_response_halts_instead_of_waiting_forever(self):
        halt = mock.Mock()
        mission = SimpleNamespace(
            _pending_goal_name="B",
            _active_goal_name="B",
            _goal_response_future=object(),
            _goal_response_deadline_sec=9.0,
            _goal_accept_deadline_sec=30.0,
            _next_goal_attempt_sec=0.0,
            _halt=halt,
        )

        def clear_pending_goal():
            mission._pending_goal_name = None
            mission._goal_accept_deadline_sec = None
            mission._next_goal_attempt_sec = None

        mission._clear_pending_goal = clear_pending_goal
        with mock.patch(
                "ros2_bridge.waypoint_handoff_mission.time.monotonic",
                return_value=10.0):
            WaypointHandoffMission._retry_pending_nav_goal(mission)

        self.assertIsNone(mission._pending_goal_name)
        self.assertIsNone(mission._active_goal_name)
        self.assertIsNone(mission._goal_response_future)
        halt.assert_called_once()
        self.assertIn(
            f"{NAV_GOAL_RESPONSE_TIMEOUT_SEC:.0f}초",
            halt.call_args.args[0],
        )

    def test_late_goal_response_after_watchdog_is_ignored(self):
        current_future = object()
        late_future = object()
        logger = mock.Mock()
        mission = SimpleNamespace(
            _goal_response_future=current_future,
            _goal_response_deadline_sec=12.0,
            get_logger=lambda: logger,
        )

        self.assertFalse(claim_goal_response(mission, late_future))
        self.assertIs(mission._goal_response_future, current_future)
        logger.warning.assert_called_once()

        self.assertTrue(claim_goal_response(mission, current_future))
        self.assertIsNone(mission._goal_response_future)
        self.assertIsNone(mission._goal_response_deadline_sec)

    def test_false_nav_success_is_queued_for_same_goal_retry(self):
        statuses = []
        logger = mock.Mock()
        mission = SimpleNamespace(
            _active_goal_name="B",
            _active_goal_handle=object(),
            _pending_goal_name=None,
            _goal_accept_deadline_sec=None,
            _next_goal_attempt_sec=None,
            _invalid_nav_success_waypoint=None,
            _invalid_nav_success_count=0,
            _label="",
            _post_status=mock.Mock(),
            _publish_nav_status=statuses.append,
            get_logger=lambda: logger,
        )

        with mock.patch(
                "ros2_bridge.waypoint_handoff_mission_safety.time.monotonic",
                return_value=10.0):
            SafetyWaypointHandoffMission._queue_invalid_nav_success_retry(
                mission, "B", "현재 위치 기준 목표 밖")

        self.assertIsNone(mission._active_goal_name)
        self.assertIsNone(mission._active_goal_handle)
        self.assertEqual(mission._pending_goal_name, "B")
        self.assertEqual(mission._next_goal_attempt_sec, 10.5)
        self.assertEqual(statuses, ["navigating"])
        self.assertIn("같은 목표 재시도", mission._label)
        self.assertEqual(mission._invalid_nav_success_count, 1)

    def test_repeated_false_nav_success_halts_instead_of_retry_storm(self):
        halt = mock.Mock()
        mission = SimpleNamespace(
            _active_goal_name="B",
            _active_goal_handle=object(),
            _pending_goal_name=None,
            _goal_accept_deadline_sec=None,
            _next_goal_attempt_sec=None,
            _invalid_nav_success_waypoint="B",
            _invalid_nav_success_count=2,
            _label="",
            _post_status=mock.Mock(),
            _publish_nav_status=mock.Mock(),
            _halt=halt,
            get_logger=lambda: mock.Mock(),
        )

        SafetyWaypointHandoffMission._queue_invalid_nav_success_retry(
            mission, "B", "맵 경계에서 경로 생성 실패")

        halt.assert_called_once()
        self.assertIn("연속 3회", halt.call_args.args[0])
        self.assertIsNone(mission._pending_goal_name)
        mission._publish_nav_status.assert_not_called()

    def test_nav_result_uses_live_tf_and_retries_when_tf_is_outside_b(self):
        retry = mock.Mock()
        mission = SimpleNamespace(
            _active_goal_handle=object(),
            _safety_cancel_pending=False,
            _last_position_sec=9.9,
            _last_arrival_scan_sec=9.9,
            # AMCL 필드는 B를 가리켜도 최신 TF가 실제로 B 밖이면 거부한다.
            _position=WAYPOINTS["B"][:2],
            _position_covariance_xy=0.01,
            _safety_state="monitoring",
            _lookup_current_map_pose=lambda: (0.0, 0.0, 0.0),
            _publish_nav_status=mock.Mock(),
            _queue_invalid_nav_success_retry=retry,
        )
        future = SimpleNamespace(
            result=lambda: SimpleNamespace(
                status=GoalStatus.STATUS_SUCCEEDED))

        with mock.patch(
                "ros2_bridge.waypoint_handoff_mission_safety.time.monotonic",
                return_value=10.0):
            SafetyWaypointHandoffMission._on_nav_result(
                mission, "B", future)

        retry.assert_called_once()
        self.assertEqual(retry.call_args.args[0], "B")
        self.assertIn("허용 반경", retry.call_args.args[1])

    def test_active_goal_cancel_is_delegated_to_its_own_goal_handle(self):
        mission, statuses, _ = self._mission(
            _active_goal_name="B",
            _pending_goal_name=None,
            _active_goal_handle=object(),
        )
        SafetyWaypointHandoffMission._handle_safety_cancel(mission)
        mission._cancel_active_goal.assert_called_once_with()
        self.assertEqual(statuses, ["canceling"])

    def test_manual_phase_rejects_automatic_recovery(self):
        mission, statuses, _ = self._mission(_phase="gesture_to_c")
        SafetyWaypointHandoffMission._handle_safety_cancel(mission)
        self.assertEqual(statuses, ["cancel_failed"])
        self.assertFalse(mission._safety_cancel_pending)

    def test_resume_captures_live_tf_before_resending_interrupted_goal(self):
        mission, statuses, goals = self._mission(
            _pending_goal_name=None,
            _safety_paused=True,
            _interrupted_goal_name="A",
        )
        SafetyWaypointHandoffMission._handle_safety_resume(mission)
        self.assertTrue(mission._safety_resuming)
        self.assertEqual(goals, ["A"])
        self.assertEqual(statuses, [])

    def test_resume_without_confirmed_cancel_is_rejected(self):
        mission, statuses, goals = self._mission(_pending_goal_name=None)
        SafetyWaypointHandoffMission._handle_safety_resume(mission)
        self.assertEqual(statuses, ["resume_failed"])
        self.assertEqual(goals, [])

    def test_current_live_tf_starts_costmap_clear(self):
        mission = self._live_pose_mission()
        with mock.patch(
                "ros2_bridge.waypoint_handoff_mission_safety.time.monotonic",
                return_value=10.0):
            SafetyWaypointHandoffMission._tick_resume_live_pose_wait(
                mission)
        self.assertFalse(mission._resume_waiting_for_live_pose)
        self.assertEqual(mission._resume_expected_pose, (0.51, -0.10, 0.03))
        mission._begin_resume_costmap_clear.assert_called_once_with()
        mission._halt.assert_not_called()

    def test_missing_live_tf_waits_without_using_stale_amcl(self):
        mission = self._live_pose_mission(
            _lookup_current_map_pose=lambda: None,
        )
        with mock.patch(
                "ros2_bridge.waypoint_handoff_mission_safety.time.monotonic",
                return_value=10.0):
            SafetyWaypointHandoffMission._tick_resume_live_pose_wait(
                mission)
        self.assertTrue(mission._resume_waiting_for_live_pose)
        mission._begin_resume_costmap_clear.assert_not_called()
        mission._halt.assert_not_called()

    def test_live_tf_timeout_fails_closed(self):
        mission = self._live_pose_mission(
            _resume_live_pose_deadline_sec=9.0,
            _lookup_current_map_pose=lambda: None,
        )
        with mock.patch(
                "ros2_bridge.waypoint_handoff_mission_safety.time.monotonic",
                return_value=10.0):
            SafetyWaypointHandoffMission._tick_resume_live_pose_wait(
                mission)
        self.assertFalse(mission._resume_waiting_for_live_pose)
        mission._begin_resume_costmap_clear.assert_not_called()
        mission._publish_nav_status.assert_called_once_with("resume_failed")
        mission._halt.assert_called_once()

    def test_resume_path_must_reach_requested_waypoint(self):
        path = Path()
        path.poses = [PoseStamped(), PoseStamped()]
        path.poses[-1].pose.position.x = WAYPOINTS["B"][0]
        path.poses[-1].pose.position.y = WAYPOINTS["B"][1]
        self.assertTrue(path_reaches_waypoint(path, "B"))
        self.assertFalse(path_reaches_waypoint(path, "A"))

    def test_resume_path_must_start_at_current_live_pose(self):
        expected_start = (0.46, -0.15, 0.02)
        path = Path()
        path.poses = [PoseStamped(), PoseStamped()]
        path.poses[0].pose.position.x = 0.83
        path.poses[0].pose.position.y = -0.76
        path.poses[-1].pose.position.x = WAYPOINTS["B"][0]
        path.poses[-1].pose.position.y = WAYPOINTS["B"][1]

        self.assertFalse(path_reaches_waypoint(path, "B", expected_start))
        path.poses[0].pose.position.x = expected_start[0]
        path.poses[0].pose.position.y = expected_start[1]
        self.assertTrue(path_reaches_waypoint(path, "B", expected_start))

    def test_resume_sends_goal_only_after_both_costmaps_are_cleared(self):
        goals = []
        global_future = mock.Mock()
        global_future.done.return_value = True
        global_future.result.return_value = object()
        local_future = mock.Mock()
        local_future.done.return_value = True
        local_future.result.return_value = object()
        mission = SimpleNamespace(
            _resume_clearing_costmaps=True,
            _resume_costmap_clear_deadline_sec=20.0,
            _resume_global_clear_future=global_future,
            _resume_local_clear_future=local_future,
            _safety_resuming=True,
            _resume_target_name="A",
            _resume_plan_not_before_sec=None,
            _label="",
            _post_status=lambda force=False: None,
            _send_nav_goal=goals.append,
            _publish_nav_status=mock.Mock(),
            _halt=mock.Mock(),
            get_logger=lambda: mock.Mock(),
        )
        with mock.patch(
                "ros2_bridge.waypoint_handoff_mission_safety.time.monotonic",
                return_value=10.0):
            SafetyWaypointHandoffMission._tick_resume_costmap_clear(mission)

        self.assertFalse(mission._resume_clearing_costmaps)
        self.assertEqual(mission._resume_plan_not_before_sec, 10.0)
        self.assertEqual(goals, ["A"])
        mission._halt.assert_not_called()


class NavArrivalValidationTest(unittest.TestCase):
    def _reason(self, **overrides):
        arguments = {
            "waypoint_name": "B",
            "position": WAYPOINTS["B"][:2],
            "pose_age_sec": 0.1,
            "covariance_xy": 0.02,
            "safety_state": "monitoring",
            "scan_age_sec": 0.1,
        }
        arguments.update(overrides)
        return nav_arrival_validation_reason(**arguments)

    def test_fresh_clear_localized_arrival_is_accepted(self):
        self.assertIsNone(self._reason())

    def test_fresh_scan_does_not_reject_mapped_structure_near_waypoint(self):
        # Raw 전방 거리만으로는 지도상 B/A 구조물과 임시 장애물을 구분할
        # 수 없으므로, 위치·Safety 상태·센서 freshness 검증만 담당한다.
        self.assertIsNone(self._reason())

    def test_nav_success_outside_waypoint_is_rejected(self):
        self.assertIn("허용 반경", self._reason(position=(0.0, 0.0)))

    def test_nav_success_with_invalid_pose_is_rejected(self):
        self.assertIn("유효하지", self._reason(position=(math.nan, 0.0)))

    def test_nav_success_during_safety_recovery_is_rejected(self):
        self.assertIn(
            "Safety 상태",
            self._reason(safety_state="reverse"),
        )


if __name__ == "__main__":
    unittest.main()
